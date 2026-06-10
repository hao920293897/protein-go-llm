"""
GNN模型定义
包括GraphSAGE、RGCN、HGTConv等用于图表示学习
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, HeteroConv, Linear, RGCNConv, HGTConv
from typing import Dict, List, Optional, Tuple
import numpy as np
import math


def load_normal_forms(go_file: str, terms_dict: Dict[str, int]):
    """加载DeepGOZero风格的EL normal forms。"""
    nf1 = []
    nf2 = []
    nf3 = []
    nf4 = []
    relations = {}
    zclasses = {}

    def get_index(go_id):
        if go_id in terms_dict:
            return terms_dict[go_id]
        if go_id not in zclasses:
            zclasses[go_id] = len(terms_dict) + len(zclasses)
        return zclasses[go_id]

    def get_rel_index(rel_id):
        if rel_id not in relations:
            relations[rel_id] = len(relations)
        return relations[rel_id]

    with open(go_file) as f:
        for line in f:
            line = line.strip().replace('_', ':')
            if 'SubClassOf' not in line:
                continue
            left, right = line.split(' SubClassOf ')
            if len(left) == 10 and len(right) == 10:
                nf1.append((get_index(left), get_index(right)))
            elif 'and' in left:
                go1, go2 = left.split(' and ')
                nf2.append((get_index(go1), get_index(go2), get_index(right)))
            elif 'some' in left:
                rel, go1 = left.split(' some ')
                nf3.append((get_rel_index(rel), get_index(go1), get_index(right)))
            elif 'some' in right:
                rel, go2 = right.split(' some ')
                nf4.append((get_index(left), get_rel_index(rel), get_index(go2)))

    return nf1, nf2, nf3, nf4, relations, zclasses


class HeteroGNN(nn.Module):
    """异构图神经网络"""

    def __init__(self,
                 node_types: List[str],
                 edge_types: List[tuple],
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 gnn_type: str = 'sage',
                 num_heads: int = 4,
                 rgcn_num_bases: Optional[int] = None):
        """
        Args:
            node_types: 节点类型列表
            edge_types: 边类型列表 [(src_type, edge_name, dst_type), ...]
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            out_channels: 输出维度
            num_layers: GNN层数
            dropout: Dropout率
            gnn_type: GNN后端，支持'sage'、'rgcn'、'hgt'
            num_heads: HGTConv注意力头数
            rgcn_num_bases: RGCN基分解数量，None表示不使用基分解
        """
        super().__init__()

        self.node_types = node_types
        self.edge_types = edge_types
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type.lower()
        self.num_heads = num_heads
        self.relation_to_idx = {
            edge_type: idx
            for idx, edge_type in enumerate(edge_types)
        }

        if self.gnn_type not in {'sage', 'gat', 'rgcn', 'hgt'}:
            raise ValueError(
                f"Unsupported gnn_type={gnn_type}. "
                "Expected one of: 'sage', 'gat', 'rgcn', 'hgt'."
            )
        if self.gnn_type == 'hgt' and hidden_channels % num_heads != 0:
            raise ValueError("hidden_channels must be divisible by num_heads when gnn_type='hgt'")

        # 输入映射层（统一不同节点类型的特征维度）
        self.input_linears = nn.ModuleDict()
        for node_type in node_types:
            self.input_linears[node_type] = Linear(in_channels, hidden_channels)

        # GNN卷积层。默认使用GraphSAGE以保持旧行为；可切换到RGCN或HGTConv。
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                self._build_conv_layer(
                    hidden_channels=hidden_channels,
                    gnn_type=self.gnn_type,
                    num_heads=num_heads,
                    rgcn_num_bases=rgcn_num_bases
                )
            )

        # 输出映射层
        self.output_linears = nn.ModuleDict()
        for node_type in node_types:
            self.output_linears[node_type] = Linear(hidden_channels, out_channels)

    def _build_conv_layer(self,
                          hidden_channels: int,
                          gnn_type: str,
                          num_heads: int,
                          rgcn_num_bases: Optional[int]):
        if gnn_type in {'sage', 'gat'}:
            conv_dict = {}
            for edge_type in self.edge_types:
                if gnn_type == 'sage':
                    conv_dict[edge_type] = SAGEConv(
                        (hidden_channels, hidden_channels),
                        hidden_channels,
                        aggr='mean'
                    )
                else:
                    conv_dict[edge_type] = GATConv(
                        (hidden_channels, hidden_channels),
                        hidden_channels,
                        heads=1,
                        concat=False,
                        add_self_loops=False
                    )
            return HeteroConv(conv_dict, aggr='sum')

        if gnn_type == 'rgcn':
            return RGCNConv(
                hidden_channels,
                hidden_channels,
                num_relations=len(self.edge_types),
                num_bases=rgcn_num_bases
            )

        return HGTConv(
            hidden_channels,
            hidden_channels,
            metadata=(self.node_types, self.edge_types),
            heads=num_heads
        )

    def _apply_rgcn(self, conv: RGCNConv, x_dict, edge_index_dict):
        node_offsets = {}
        node_sizes = {}
        xs = []
        offset = 0
        for node_type in self.node_types:
            if node_type not in x_dict:
                continue
            node_offsets[node_type] = offset
            node_sizes[node_type] = x_dict[node_type].size(0)
            xs.append(x_dict[node_type])
            offset += x_dict[node_type].size(0)

        if not xs:
            return {}

        x = torch.cat(xs, dim=0)
        edge_indices = []
        edge_type_ids = []

        for edge_type in self.edge_types:
            edge_index = edge_index_dict.get(edge_type)
            if edge_index is None or edge_index.numel() == 0:
                continue

            src_type, _, dst_type = edge_type
            if src_type not in node_offsets or dst_type not in node_offsets:
                continue

            shifted = edge_index.clone()
            shifted[0] += node_offsets[src_type]
            shifted[1] += node_offsets[dst_type]
            edge_indices.append(shifted)
            edge_type_ids.append(torch.full(
                (shifted.size(1),),
                self.relation_to_idx[edge_type],
                dtype=torch.long,
                device=shifted.device
            ))

        if edge_indices:
            edge_index = torch.cat(edge_indices, dim=1)
            edge_type = torch.cat(edge_type_ids, dim=0)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
            edge_type = torch.empty((0,), dtype=torch.long, device=x.device)

        out = conv(x, edge_index, edge_type)
        out_dict = {}
        for node_type, start in node_offsets.items():
            end = start + node_sizes[node_type]
            out_dict[node_type] = out[start:end]
        return out_dict

    def forward(self, x_dict, edge_index_dict):
        """
        前向传播

        Args:
            x_dict: {node_type: features}
            edge_index_dict: {edge_type: edge_index}

        Returns:
            {node_type: embeddings}
        """
        # 输入映射
        for node_type in self.node_types:
            if node_type in x_dict:
                x_dict[node_type] = self.input_linears[node_type](x_dict[node_type])
                x_dict[node_type] = F.relu(x_dict[node_type])

        # GNN卷积
        for i, conv in enumerate(self.convs):
            old_x_dict = x_dict
            if self.gnn_type == 'rgcn':
                conv_out = self._apply_rgcn(conv, x_dict, edge_index_dict)
            else:
                conv_out = conv(x_dict, edge_index_dict)

            # 某些节点类型可能没有入边。保留上一层表示，避免该类型从输出中消失。
            x_dict = {}
            for node_type in self.node_types:
                if node_type in conv_out:
                    x_dict[node_type] = conv_out[node_type]
                    if node_type in old_x_dict and old_x_dict[node_type].shape == x_dict[node_type].shape:
                        x_dict[node_type] = x_dict[node_type] + old_x_dict[node_type]
                elif node_type in old_x_dict:
                    x_dict[node_type] = old_x_dict[node_type]

            # 激活和dropout
            for node_type in x_dict:
                x_dict[node_type] = F.relu(x_dict[node_type])
                x_dict[node_type] = F.dropout(x_dict[node_type],
                                              p=self.dropout,
                                              training=self.training)

        # 输出映射
        out_dict = {}
        for node_type in self.node_types:
            if node_type in x_dict:
                out_dict[node_type] = self.output_linears[node_type](x_dict[node_type])

        return out_dict


class ProteinGONodeClassifier(nn.Module):
    """任务1：蛋白质GO多标签节点分类模型"""

    def __init__(self,
                 node_types: List[str],
                 edge_types: List[tuple],
                 in_channels: int,
                 num_go_terms: int,
                 hidden_channels: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 gnn_type: str = 'sage',
                 num_heads: int = 4,
                 rgcn_num_bases: Optional[int] = None,
                 num_el_classes: Optional[int] = None,
                 num_relations: int = 0,
                 el_margin: float = 0.1):
        """
        Args:
            node_types: 节点类型，节点分类任务通常为protein/gene/interpro
            edge_types: PyG异构边类型
            in_channels: 文本节点特征维度
            num_go_terms: 训练集GO标签数量
            hidden_channels: GNN隐藏维度
            num_layers: GNN层数
            dropout: Dropout率
            gnn_type: GNN后端，支持'sage'、'rgcn'、'hgt'
            num_heads: HGTConv注意力头数
            rgcn_num_bases: RGCN基分解数量
            num_el_classes: EL loss覆盖的GO类数量，包含zero classes
            num_relations: EL normal forms中的关系数量
            el_margin: EL几何约束margin
        """
        super().__init__()

        self.gnn = HeteroGNN(
            node_types=node_types,
            edge_types=edge_types,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            gnn_type=gnn_type,
            num_heads=num_heads,
            rgcn_num_bases=rgcn_num_bases
        )

        self.num_go_terms = num_go_terms
        self.num_el_classes = num_el_classes or num_go_terms
        self.num_relations = num_relations
        self.el_margin = el_margin
        self.go_embed = nn.Embedding(self.num_el_classes, hidden_channels)
        self.go_norm = nn.BatchNorm1d(hidden_channels)
        self.go_rad = nn.Embedding(self.num_el_classes, 1)
        self.rel_embed = nn.Embedding(num_relations + 1, hidden_channels)
        self.register_buffer('all_gos', torch.arange(num_go_terms, dtype=torch.long))
        self.register_buffer('has_func_index', torch.LongTensor([num_relations]))
        k = math.sqrt(1 / hidden_channels)
        nn.init.uniform_(self.go_embed.weight, -k, k)
        nn.init.uniform_(self.go_rad.weight, -k, k)
        nn.init.uniform_(self.rel_embed.weight, -k, k)

    def encode(self, x_dict, edge_index_dict):
        """输出各类型节点的GNN表示。"""
        return self.gnn(x_dict, edge_index_dict)

    def forward(self, x_dict, edge_index_dict, protein_indices: Optional[torch.Tensor] = None):
        emb_dict = self.encode(x_dict, edge_index_dict)
        protein_emb = emb_dict['protein']
        if protein_indices is not None:
            protein_emb = protein_emb[protein_indices]

        go_embed = self._normalize_go(self.all_gos)
        has_func = self.rel_embed(self.has_func_index)
        has_func_go = go_embed + has_func
        go_rad = torch.abs(self.go_rad(self.all_gos).view(1, -1))
        logits = torch.matmul(protein_emb, has_func_go.T) + go_rad
        return logits

    def _zero_el_loss(self):
        return self.go_embed.weight.sum() * 0.0

    def el_loss(self, go_normal_forms):
        """DeepGOZero风格EL loss，约束GO类别与本体关系embedding。"""
        if go_normal_forms is None:
            return self._zero_el_loss()
        nf1, nf2, nf3, nf4 = go_normal_forms
        return (
            self.nf1_loss(nf1)
            + self.nf2_loss(nf2)
            + self.nf3_loss(nf3)
            + self.nf4_loss(nf4)
        )

    def _normalize_go(self, go_ids):
        emb = self.go_embed(go_ids)
        if self.training and emb.size(0) <= 1:
            return F.batch_norm(
                emb,
                self.go_norm.running_mean,
                self.go_norm.running_var,
                self.go_norm.weight,
                self.go_norm.bias,
                training=False,
                eps=self.go_norm.eps
            )
        return self.go_norm(emb)

    def class_dist(self, data):
        c = self._normalize_go(data[:, 0])
        d = self._normalize_go(data[:, 1])
        rc = torch.abs(self.go_rad(data[:, 0]))
        rd = torch.abs(self.go_rad(data[:, 1]))
        return torch.linalg.norm(c - d, dim=1, keepdim=True) + rc - rd

    def nf1_loss(self, data):
        if data.numel() == 0:
            return self._zero_el_loss()
        pos_dist = self.class_dist(data)
        return torch.mean(torch.relu(pos_dist - self.el_margin))

    def nf2_loss(self, data):
        if data.numel() == 0:
            return self._zero_el_loss()
        c = self._normalize_go(data[:, 0])
        d = self._normalize_go(data[:, 1])
        e = self._normalize_go(data[:, 2])
        rc = torch.abs(self.go_rad(data[:, 0]))
        rd = torch.abs(self.go_rad(data[:, 1]))

        sr = rc + rd
        dst = torch.linalg.norm(c - d, dim=1, keepdim=True)
        dst2 = torch.linalg.norm(e - c, dim=1, keepdim=True)
        dst3 = torch.linalg.norm(e - d, dim=1, keepdim=True)
        return torch.mean(
            torch.relu(dst - sr - self.el_margin)
            + torch.relu(dst2 - rc - self.el_margin)
            + torch.relu(dst3 - rd - self.el_margin)
        )

    def nf3_loss(self, data):
        if data.numel() == 0:
            return self._zero_el_loss()
        rE = self.rel_embed(data[:, 0])
        c = self._normalize_go(data[:, 1])
        d = self._normalize_go(data[:, 2])
        rc = torch.abs(self.go_rad(data[:, 1]))
        rd = torch.abs(self.go_rad(data[:, 2]))

        rSomeC = c + rE
        euc = torch.linalg.norm(rSomeC - d, dim=1, keepdim=True)
        return torch.mean(torch.relu(euc + rc - rd - self.el_margin))

    def nf4_loss(self, data):
        if data.numel() == 0:
            return self._zero_el_loss()
        c = self._normalize_go(data[:, 0])
        rE = self.rel_embed(data[:, 1])
        d = self._normalize_go(data[:, 2])

        rc = torch.abs(self.go_rad(data[:, 0]))
        rd = torch.abs(self.go_rad(data[:, 2]))
        sr = rc + rd
        rSomeD = d + rE
        dst = torch.linalg.norm(c - rSomeD, dim=1, keepdim=True)
        return torch.mean(torch.relu(dst - sr - self.el_margin))


class ProteinGOLinkPredictor(nn.Module):
    """蛋白质-GO术语链接预测模型"""

    def __init__(self,
                 node_types: List[str],
                 edge_types: List[tuple],
                 in_channels: int,
                 hidden_channels: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 gnn_type: str = 'sage',
                 num_heads: int = 4,
                 rgcn_num_bases: Optional[int] = None):
        """
        Args:
            node_types: 节点类型列表
            edge_types: 边类型列表
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            num_layers: GNN层数
            dropout: Dropout率
            gnn_type: GNN后端，支持'sage'、'rgcn'、'hgt'
            num_heads: HGTConv注意力头数
            rgcn_num_bases: RGCN基分解数量
        """
        super().__init__()

        # GNN编码器
        self.gnn = HeteroGNN(
            node_types=node_types,
            edge_types=edge_types,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            gnn_type=gnn_type,
            num_heads=num_heads,
            rgcn_num_bases=rgcn_num_bases
        )

        # 链接预测头
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )

    def encode(self, x_dict, edge_index_dict):
        """编码节点"""
        return self.gnn(x_dict, edge_index_dict)

    def decode(self, protein_emb, go_emb):
        """
        解码链接概率

        Args:
            protein_emb: 蛋白质embedding [N, D]
            go_emb: GO术语embedding [M, D]

        Returns:
            链接分数 [N, M]
        """
        # 计算所有蛋白质-GO对的分数
        num_proteins = protein_emb.size(0)
        num_gos = go_emb.size(0)

        # 扩展维度
        protein_emb_expanded = protein_emb.unsqueeze(1).expand(num_proteins, num_gos, -1)
        go_emb_expanded = go_emb.unsqueeze(0).expand(num_proteins, num_gos, -1)

        # 拼接
        pair_emb = torch.cat([protein_emb_expanded, go_emb_expanded], dim=-1)

        # 预测
        scores = self.link_predictor(pair_emb).squeeze(-1)

        return scores

    def forward(self, x_dict, edge_index_dict, protein_indices, go_indices):
        """
        前向传播

        Args:
            x_dict: 节点特征字典
            edge_index_dict: 边索引字典
            protein_indices: 蛋白质节点索引
            go_indices: GO节点索引

        Returns:
            链接分数
        """
        # 编码
        emb_dict = self.encode(x_dict, edge_index_dict)

        # 获取蛋白质和GO的embedding
        protein_emb = emb_dict['protein'][protein_indices]
        go_emb = emb_dict['go_term'][go_indices]

        # 解码
        scores = self.decode(protein_emb, go_emb)

        return scores


class NodeClassificationTrainer:
    """蛋白质GO多标签节点分类训练器"""

    def __init__(self,
                 model: nn.Module,
                 device: Optional[str] = None,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 pos_weight: Optional[torch.Tensor] = None,
                 go_normal_forms: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
                 el_loss_weight: float = 0.0):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight.to(device) if pos_weight is not None else None
        )
        self.go_normal_forms = self._move_normal_forms(go_normal_forms)
        self.el_loss_weight = el_loss_weight
        self.last_loss_parts = {
            'classification_loss': 0.0,
            'el_loss': 0.0,
            'total_loss': 0.0
        }

    @staticmethod
    def make_normal_form_tensors(normal_forms, device: Optional[str] = None):
        """将normal forms列表转为形状稳定的LongTensor。"""
        widths = (2, 3, 3, 3)
        tensors = []
        for values, width in zip(normal_forms, widths):
            if values:
                tensor = torch.LongTensor(values)
            else:
                tensor = torch.empty((0, width), dtype=torch.long)
            if device is not None:
                tensor = tensor.to(device)
            tensors.append(tensor)
        return tuple(tensors)

    def _move_normal_forms(self, go_normal_forms):
        if go_normal_forms is None:
            return None
        return tuple(tensor.to(self.device) for tensor in go_normal_forms)

    @staticmethod
    def compute_pos_weight(labels: torch.Tensor,
                           mask: torch.Tensor,
                           max_weight: float = 50.0) -> torch.Tensor:
        """根据训练集标签稀疏度计算BCE正样本权重。"""
        y = labels[mask]
        pos = y.sum(dim=0)
        neg = y.size(0) - pos
        weight = neg / pos.clamp(min=1.0)
        return weight.clamp(max=max_weight)

    def _move_graph(self, data):
        x_dict = {k: v.to(self.device) for k, v in data.x_dict.items()}
        edge_index_dict = {k: v.to(self.device) for k, v in data.edge_index_dict.items()}
        return x_dict, edge_index_dict

    def train_epoch(self, data) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        x_dict, edge_index_dict = self._move_graph(data)
        labels = data['protein'].y.to(self.device)
        train_mask = data['protein'].train_mask.to(self.device)

        logits = self.model(x_dict, edge_index_dict)
        classification_loss = self.criterion(logits[train_mask], labels[train_mask])
        el_loss = (
            self.model.el_loss(self.go_normal_forms)
            if self.el_loss_weight > 0 and self.go_normal_forms is not None
            else classification_loss * 0.0
        )
        loss = classification_loss + self.el_loss_weight * el_loss
        loss.backward()
        self.optimizer.step()

        self.last_loss_parts = {
            'classification_loss': float(classification_loss.detach().item()),
            'el_loss': float(el_loss.detach().item()),
            'total_loss': float(loss.detach().item())
        }
        return float(loss.item())

    @torch.no_grad()
    def evaluate(self, data, split: str = 'valid', threshold: float = 0.3) -> Dict[str, float]:
        self.model.eval()

        x_dict, edge_index_dict = self._move_graph(data)
        labels = data['protein'].y.to(self.device)
        mask = getattr(data['protein'], f'{split}_mask').to(self.device)

        logits = self.model(x_dict, edge_index_dict)
        loss = self.criterion(logits[mask], labels[mask])
        probs = torch.sigmoid(logits[mask])
        preds = (probs >= threshold).float()
        gold = labels[mask]

        tp = (preds * gold).sum().item()
        pred_pos = preds.sum().item()
        gold_pos = gold.sum().item()
        precision = tp / pred_pos if pred_pos > 0 else 0.0
        recall = tp / gold_pos if gold_pos > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        return {
            'loss': float(loss.item()),
            'threshold': threshold,
            'micro_precision': precision,
            'micro_recall': recall,
            'micro_f1': f1
        }

    @torch.no_grad()
    def predict_proba(self, data, split: Optional[str] = None) -> np.ndarray:
        self.model.eval()
        x_dict, edge_index_dict = self._move_graph(data)
        logits = self.model(x_dict, edge_index_dict)
        probs = torch.sigmoid(logits)

        if split is not None:
            mask = getattr(data['protein'], f'{split}_mask').to(self.device)
            probs = probs[mask]

        return probs.cpu().numpy()

    @torch.no_grad()
    def predict_topk(self,
                     data,
                     go_vocab: List[str],
                     split: str = 'test',
                     k: int = 100) -> Dict[str, List[Tuple[str, float]]]:
        probs = self.predict_proba(data)
        mask = getattr(data['protein'], f'{split}_mask')
        protein_ids = data['protein'].node_ids

        results = {}
        split_indices = mask.nonzero(as_tuple=False).view(-1).tolist()
        for protein_idx in split_indices:
            scores = probs[protein_idx]
            top_idx = np.argsort(-scores)[:k]
            results[protein_ids[protein_idx]] = [
                (go_vocab[idx], float(scores[idx]))
                for idx in top_idx
            ]

        return results


class GNNTrainer:
    """GNN模型训练器"""

    def __init__(self,
                 model: nn.Module,
                 device: str = 'cuda',
                 lr: float = 0.001,
                 weight_decay: float = 5e-4):
        """
        Args:
            model: GNN模型
            device: 设备
            lr: 学习率
            weight_decay: 权重衰减
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.criterion = nn.BCEWithLogitsLoss()

    def train_epoch(self, data, train_protein_indices, train_go_indices, train_labels):
        """
        训练一个epoch

        Args:
            data: PyG HeteroData对象
            train_protein_indices: 训练集蛋白质索引
            train_go_indices: 训练集GO索引
            train_labels: 训练标签 [N, M]

        Returns:
            训练损失
        """
        self.model.train()
        self.optimizer.zero_grad()

        # 将数据移到设备
        x_dict = {k: v.to(self.device) for k, v in data.x_dict.items()}
        edge_index_dict = {k: v.to(self.device) for k, v in data.edge_index_dict.items()}

        protein_indices = train_protein_indices.to(self.device)
        go_indices = train_go_indices.to(self.device)
        labels = train_labels.to(self.device)

        # 前向传播
        scores = self.model(x_dict, edge_index_dict, protein_indices, go_indices)

        # 计算损失
        loss = self.criterion(scores, labels)

        # 反向传播
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self, data, eval_protein_indices, eval_go_indices, eval_labels):
        """
        评估模型

        Args:
            data: PyG HeteroData对象
            eval_protein_indices: 评估集蛋白质索引
            eval_go_indices: 评估集GO索引
            eval_labels: 评估标签

        Returns:
            评估损失和指标
        """
        self.model.eval()

        # 将数据移到设备
        x_dict = {k: v.to(self.device) for k, v in data.x_dict.items()}
        edge_index_dict = {k: v.to(self.device) for k, v in data.edge_index_dict.items()}

        protein_indices = eval_protein_indices.to(self.device)
        go_indices = eval_go_indices.to(self.device)
        labels = eval_labels.to(self.device)

        # 前向传播
        scores = self.model(x_dict, edge_index_dict, protein_indices, go_indices)

        # 计算损失
        loss = self.criterion(scores, labels)

        # 计算预测
        probs = torch.sigmoid(scores)

        return loss.item(), probs.cpu().numpy()

    @torch.no_grad()
    def predict(self, data, protein_indices, go_indices):
        """
        预测蛋白质-GO链接

        Args:
            data: PyG HeteroData对象
            protein_indices: 蛋白质索引
            go_indices: GO索引

        Returns:
            预测分数
        """
        self.model.eval()

        # 将数据移到设备
        x_dict = {k: v.to(self.device) for k, v in data.x_dict.items()}
        edge_index_dict = {k: v.to(self.device) for k, v in data.edge_index_dict.items()}

        protein_indices = protein_indices.to(self.device)
        go_indices = go_indices.to(self.device)

        # 前向传播
        scores = self.model(x_dict, edge_index_dict, protein_indices, go_indices)
        probs = torch.sigmoid(scores)

        return probs.cpu().numpy()


if __name__ == "__main__":
    # 测试
    print("GNN模型定义模块")
