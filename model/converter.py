"""
将NetworkX知识图谱转换为PyTorch Geometric格式
用于GNN模型训练
"""
import torch
import numpy as np
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class KGToPyGConverter:
    """知识图谱到PyG格式的转换器"""

    def __init__(self, kg):
        """
        Args:
            kg: ProteinKnowledgeGraph实例
        """
        self.kg = kg
        self.node_type_to_ids = defaultdict(list)
        self.node_id_to_idx = {}

        # 构建节点映射
        self._build_node_mappings()

    def _build_node_mappings(self):
        """构建节点ID到索引的映射"""
        for node_id in self.kg.graph.nodes():
            node_type = self.kg.graph.nodes[node_id].get('node_type')
            self.node_type_to_ids[node_type].append(node_id)

        # 为每种节点类型创建索引
        for node_type, node_ids in self.node_type_to_ids.items():
            for idx, node_id in enumerate(node_ids):
                self.node_id_to_idx[(node_type, node_id)] = idx

    def convert_to_hetero_data(self,
                               node_feature_dim: Optional[int] = None,
                               use_text_embeddings: bool = True,
                               text_embeddings: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
                               text_embedding_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                               task_mode: str = 'link_prediction',
                               add_reverse_edges: bool = True,
                               add_interpro_onehot_to_protein: bool = False) -> HeteroData:
        """
        转换为PyG的HeteroData格式

        Args:
            node_feature_dim: 节点特征维度
            use_text_embeddings: 是否使用文本embedding作为节点特征
            text_embeddings: 预先计算好的文本embedding，格式为
                {node_type: {node_id: embedding}}
            text_embedding_model_name: 没有传入text_embeddings时使用的SentenceTransformer模型
            task_mode: 'node_classification'或'link_prediction'。
                节点分类时不包含GO节点，链路预测时包含GO节点。
            add_reverse_edges: 是否为异构关系添加反向边，便于目标节点聚合邻居信息
            add_interpro_onehot_to_protein: 是否给蛋白质节点追加InterPro one-hot特征。
                非蛋白节点会追加同长度零向量，以保持所有节点类型输入维度一致。

        Returns:
            HeteroData对象
        """
        data = HeteroData()

        if use_text_embeddings and text_embeddings is None:
            emb_gen = TextEmbeddingGenerator(model_name=text_embedding_model_name)
            text_embeddings = emb_gen.generate_all_embeddings(
                self.kg,
                node_types=self._get_node_types(task_mode)
            )

        if node_feature_dim is None:
            node_feature_dim = self._infer_feature_dim(text_embeddings, default_dim=384)

        # 添加节点特征
        self._add_node_features(
            data=data,
            feature_dim=node_feature_dim,
            use_text_embeddings=use_text_embeddings,
            text_embeddings=text_embeddings,
            task_mode=task_mode
        )

        if add_interpro_onehot_to_protein:
            self._append_interpro_onehot_features(data, task_mode=task_mode)

        # 添加边
        self._add_edges(data, task_mode=task_mode, add_reverse_edges=add_reverse_edges)

        if task_mode == 'node_classification':
            self._add_node_classification_targets(data)

        return data

    def _get_node_types(self, task_mode: str) -> List[str]:
        """根据任务选择节点类型。节点分类不把GO作为图节点，避免标签节点泄漏。"""
        node_types = [
            self.kg.NODE_PROTEIN,
            self.kg.NODE_GENE,
            self.kg.NODE_INTERPRO
        ]
        if task_mode == 'link_prediction':
            node_types.append(self.kg.NODE_GO)
        return node_types

    @staticmethod
    def _infer_feature_dim(text_embeddings: Optional[Dict[str, Dict[str, np.ndarray]]],
                           default_dim: int) -> int:
        if text_embeddings:
            for node_embs in text_embeddings.values():
                for emb in node_embs.values():
                    return int(np.asarray(emb).shape[-1])
        return default_dim

    def _add_node_features(self,
                           data: HeteroData,
                           feature_dim: int,
                           use_text_embeddings: bool,
                           text_embeddings: Optional[Dict[str, Dict[str, np.ndarray]]],
                           task_mode: str):
        """添加节点特征"""
        print("添加节点特征...")

        node_types = self._get_node_types(task_mode)

        for node_type in node_types:
            node_ids = self.node_type_to_ids[node_type]
            num_nodes = len(node_ids)

            if use_text_embeddings:
                features_np = np.zeros((num_nodes, feature_dim), dtype=np.float32)
                node_embs = text_embeddings.get(node_type, {}) if text_embeddings else {}
                missing = 0
                for idx, node_id in enumerate(node_ids):
                    emb = node_embs.get(node_id)
                    if emb is None:
                        missing += 1
                        continue
                    emb = np.asarray(emb, dtype=np.float32)
                    if emb.shape[-1] != feature_dim:
                        raise ValueError(
                            f"{node_type}:{node_id} embedding dim={emb.shape[-1]}, "
                            f"expected {feature_dim}"
                        )
                    features_np[idx] = emb
                features = torch.from_numpy(features_np)
                if missing:
                    print(f"  {node_type}: {missing} 个节点缺少文本，使用零向量")
            else:
                # 使用one-hot或随机初始化
                features = torch.randn(num_nodes, feature_dim)

            data[node_type].x = features
            data[node_type].num_nodes = num_nodes
            data[node_type].node_ids = node_ids

            print(f"  {node_type}: {num_nodes} nodes, feature_dim={feature_dim}")

    def _append_interpro_onehot_features(self, data: HeteroData, task_mode: str):
        """给protein节点追加InterPro one-hot，其他节点类型追加零向量。"""
        print("追加蛋白质InterPro one-hot特征...")

        node_types = self._get_node_types(task_mode)
        interpro_ids = self.node_type_to_ids[self.kg.NODE_INTERPRO]
        interpro_to_idx = {
            interpro_id: idx
            for idx, interpro_id in enumerate(interpro_ids)
        }
        num_interpros = len(interpro_ids)

        if num_interpros == 0:
            print("  InterPro节点为空，跳过one-hot追加")
            return

        protein_ids = self.node_type_to_ids[self.kg.NODE_PROTEIN]
        protein_features = torch.zeros((len(protein_ids), num_interpros), dtype=torch.float)

        for protein_idx, protein_id in enumerate(protein_ids):
            for neighbor_id in self.kg.graph.successors(protein_id):
                edge_datas = self.kg.graph.get_edge_data(protein_id, neighbor_id)
                if not edge_datas:
                    continue
                for edge_data in edge_datas.values():
                    if edge_data.get('edge_type') != self.kg.EDGE_PROTEIN_INTERPRO:
                        continue
                    interpro_idx = interpro_to_idx.get(neighbor_id)
                    if interpro_idx is not None:
                        protein_features[protein_idx, interpro_idx] = 1.0

        for node_type in node_types:
            if node_type == self.kg.NODE_PROTEIN:
                extra = protein_features
            else:
                extra = torch.zeros((data[node_type].num_nodes, num_interpros), dtype=torch.float)
            data[node_type].x = torch.cat([data[node_type].x, extra], dim=1)

        data[self.kg.NODE_PROTEIN].interpro_vocab = interpro_ids
        print(
            f"  追加 {num_interpros} 维InterPro one-hot；"
            f"protein feature_dim={data[self.kg.NODE_PROTEIN].x.size(1)}"
        )

    def _add_edges(self,
                   data: HeteroData,
                   task_mode: str,
                   add_reverse_edges: bool):
        """添加各类边"""
        print("添加边...")

        edge_types = self._get_edge_types(task_mode)

        for src_type, edge_type, dst_type in edge_types:
            edge_index, edge_attr = self._extract_edges(src_type, edge_type, dst_type)

            if edge_index is not None:
                self._store_edge(data, src_type, edge_type, dst_type, edge_index, edge_attr)

                if add_reverse_edges and src_type != dst_type:
                    rev_edge_type = f"rev_{edge_type}"
                    self._store_edge(
                        data,
                        dst_type,
                        rev_edge_type,
                        src_type,
                        edge_index.flip(0),
                        edge_attr
                    )

    def _get_edge_types(self, task_mode: str) -> List[Tuple[str, str, str]]:
        edge_types = [
            (self.kg.NODE_PROTEIN, self.kg.EDGE_PROTEIN_GENE, self.kg.NODE_GENE),
            (self.kg.NODE_PROTEIN, self.kg.EDGE_PROTEIN_PROTEIN, self.kg.NODE_PROTEIN),
            (self.kg.NODE_PROTEIN, self.kg.EDGE_PROTEIN_SEQUENCE_SIMILAR, self.kg.NODE_PROTEIN),
            (self.kg.NODE_PROTEIN, self.kg.EDGE_PROTEIN_INTERPRO, self.kg.NODE_INTERPRO),
        ]

        if task_mode == 'link_prediction':
            edge_types.extend([
                (self.kg.NODE_PROTEIN, self.kg.EDGE_PROTEIN_GO, self.kg.NODE_GO),
                (self.kg.NODE_INTERPRO, self.kg.EDGE_INTERPRO_GO, self.kg.NODE_GO),
                (self.kg.NODE_GO, self.kg.EDGE_GO_GO, self.kg.NODE_GO),
            ])

        return edge_types

    @staticmethod
    def _store_edge(data: HeteroData,
                    src_type: str,
                    edge_type: str,
                    dst_type: str,
                    edge_index: torch.Tensor,
                    edge_attr: Optional[torch.Tensor]):
        data[src_type, edge_type, dst_type].edge_index = edge_index
        if edge_attr is not None:
            data[src_type, edge_type, dst_type].edge_attr = edge_attr

        num_edges = edge_index.shape[1]
        print(f"  ({src_type}, {edge_type}, {dst_type}): {num_edges} edges")

    def _extract_edges(self, src_type: str, edge_type: str, dst_type: str) -> Tuple:
        """
        提取特定类型的边

        Returns:
            (edge_index, edge_attr)
        """
        edge_list = []
        edge_weights = []

        for src_id in self.node_type_to_ids[src_type]:
            for dst_id in self.kg.graph.successors(src_id):
                dst_node_type = self.kg.graph.nodes[dst_id].get('node_type')

                if dst_node_type == dst_type:
                    # 检查边类型
                    edges = self.kg.graph.get_edge_data(src_id, dst_id)
                    if edges:
                        for edge_data in edges.values():
                            if edge_data.get('edge_type') == edge_type:
                                src_idx = self.node_id_to_idx[(src_type, src_id)]
                                dst_idx = self.node_id_to_idx[(dst_type, dst_id)]

                                edge_list.append([src_idx, dst_idx])

                                # 提取边权重
                                weight = edge_data.get('score', 1.0)
                                if 'support' in edge_data:
                                    weight = edge_data['support']
                                edge_weights.append(weight)

        if not edge_list:
            return None, None

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

        return edge_index, edge_attr

    def _add_node_classification_targets(self, data: HeteroData):
        """将任务1的标签矩阵和split mask挂到protein节点上。"""
        task_data = self.kg.supervision.get('node_classification', {})
        if not task_data:
            raise ValueError("kg.supervision['node_classification']为空，请先构建任务1监督数据")

        protein_ids = self.node_type_to_ids[self.kg.NODE_PROTEIN]
        num_proteins = len(protein_ids)
        num_labels = len(task_data['go_vocab'])
        labels = torch.zeros((num_proteins, num_labels), dtype=torch.float)

        masks = {
            'train': torch.zeros(num_proteins, dtype=torch.bool),
            'valid': torch.zeros(num_proteins, dtype=torch.bool),
            'test': torch.zeros(num_proteins, dtype=torch.bool)
        }

        protein_idx = {
            protein_id: idx
            for idx, protein_id in enumerate(protein_ids)
        }

        for split, split_protein_ids in task_data.get('protein_ids', {}).items():
            split_y = task_data['y'][split]
            for row_idx, protein_id in enumerate(split_protein_ids):
                idx = protein_idx.get(protein_id)
                if idx is None:
                    continue
                labels[idx] = torch.from_numpy(split_y[row_idx]).float()
                if split in masks:
                    masks[split][idx] = True

        data[self.kg.NODE_PROTEIN].y = labels
        data[self.kg.NODE_PROTEIN].train_mask = masks['train']
        data[self.kg.NODE_PROTEIN].valid_mask = masks['valid']
        data[self.kg.NODE_PROTEIN].test_mask = masks['test']
        data[self.kg.NODE_PROTEIN].go_vocab = task_data['go_vocab']

    def get_node_idx_mapping(self) -> Dict:
        """获取节点ID到索引的映射"""
        return self.node_id_to_idx

    def get_protein_indices(self, protein_ids: List[str]) -> torch.Tensor:
        """
        获取蛋白质ID对应的索引

        Args:
            protein_ids: 蛋白质ID列表

        Returns:
            索引tensor
        """
        indices = []
        for protein_id in protein_ids:
            if (self.kg.NODE_PROTEIN, protein_id) in self.node_id_to_idx:
                idx = self.node_id_to_idx[(self.kg.NODE_PROTEIN, protein_id)]
                indices.append(idx)

        return torch.tensor(indices, dtype=torch.long)

    def get_go_indices(self, go_ids: List[str]) -> torch.Tensor:
        """
        获取GO术语对应的索引

        Args:
            go_ids: GO术语ID列表

        Returns:
            索引tensor
        """
        indices = []
        for go_id in go_ids:
            if (self.kg.NODE_GO, go_id) in self.node_id_to_idx:
                idx = self.node_id_to_idx[(self.kg.NODE_GO, go_id)]
                indices.append(idx)

        return torch.tensor(indices, dtype=torch.long)


class TextEmbeddingGenerator:
    """为节点生成文本embedding"""

    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Args:
            model_name: 句子embedding模型名称
        """
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        print(f"文本embedding模型已加载: {model_name}")
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def generate_node_embeddings(self,
                                 kg,
                                 node_type: str,
                                 node_ids: List[str],
                                 batch_size: int = 64,
                                 max_chars: int = 1024) -> Dict[str, np.ndarray]:
        """为任意节点类型生成文本embedding。"""
        embeddings = {}
        texts = []
        valid_ids = []

        for node_id in node_ids:
            if hasattr(kg, 'get_text_for_node'):
                text = kg.get_text_for_node(node_id)
            else:
                text = str(kg.get_node_features(node_id))

            text = " ".join(text.split())
            if text:
                texts.append(text[:max_chars])
                valid_ids.append(node_id)

        if texts:
            embs = self.model.encode(
                texts,
                show_progress_bar=True,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            for node_id, emb in zip(valid_ids, embs):
                embeddings[node_id] = emb.astype(np.float32)

        return embeddings

    def generate_protein_embeddings(self, kg, protein_ids: List[str]) -> Dict[str, np.ndarray]:
        """
        为蛋白质生成文本embedding

        Args:
            kg: ProteinKnowledgeGraph实例
            protein_ids: 蛋白质ID列表

        Returns:
            {protein_id: embedding}
        """
        embeddings = {}
        texts = []
        valid_ids = []

        for protein_id in protein_ids:
            attrs = kg.protein_attrs.get(protein_id, {})
            description = attrs.get('description', '')

            if description:
                texts.append(description[:512])  # 截断长文本
                valid_ids.append(protein_id)

        if texts:
            embs = self.model.encode(texts, show_progress_bar=True, batch_size=32)
            for protein_id, emb in zip(valid_ids, embs):
                embeddings[protein_id] = emb

        return embeddings

    def generate_go_embeddings(self, kg, go_ids: List[str]) -> Dict[str, np.ndarray]:
        """
        为GO术语生成文本embedding

        Args:
            kg: ProteinKnowledgeGraph实例
            go_ids: GO术语ID列表

        Returns:
            {go_id: embedding}
        """
        embeddings = {}
        texts = []
        valid_ids = []

        for go_id in go_ids:
            attrs = kg.go_attrs.get(go_id, {})
            name = attrs.get('name', '')
            definition = attrs.get('definition', '')

            text = f"{name}. {definition}"
            if text.strip():
                texts.append(text[:512])
                valid_ids.append(go_id)

        if texts:
            embs = self.model.encode(texts, show_progress_bar=True, batch_size=32)
            for go_id, emb in zip(valid_ids, embs):
                embeddings[go_id] = emb

        return embeddings

    def generate_all_embeddings(self,
                                kg,
                                node_types: Optional[List[str]] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """
        为所有节点类型生成embedding

        Args:
            kg: ProteinKnowledgeGraph实例

        Returns:
            {node_type: {node_id: embedding}}
        """
        print("生成文本embeddings...")

        all_embeddings = {}

        node_id_sources = {
            'protein': list(kg.protein_attrs.keys()),
            'gene': list(kg.gene_attrs.keys()),
            'interpro': list(kg.interpro_attrs.keys()),
            'go_term': list(kg.go_attrs.keys())
        }

        if node_types is None:
            node_types = list(node_id_sources.keys())

        for node_type in node_types:
            node_ids = node_id_sources.get(node_type, [])
            all_embeddings[node_type] = self.generate_node_embeddings(kg, node_type, node_ids)
            print(f"  {node_type} embeddings: {len(all_embeddings[node_type])}")

        return all_embeddings


if __name__ == "__main__":
    # 示例用法
    print("PyG格式转换模块")
