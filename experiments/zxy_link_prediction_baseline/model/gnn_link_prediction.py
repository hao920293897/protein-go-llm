"""
Multi-relation heterogenous GNN link prediction baseline.
This module avoids LLMs and focuses on representation learning over the KG.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch_geometric.nn import HeteroConv, SAGEConv


EdgeType = Tuple[str, str, str]


DEFAULT_TARGET_RELATIONS: List[EdgeType] = [
    ('protein', 'annotated_with', 'go_term'),
    ('protein', 'has_domain', 'interpro'),
    ('interpro', 'associated_with', 'go_term'),
    ('gene', 'gene_interaction', 'gene'),
]


def relation_key(edge_type: EdgeType) -> str:
    return '__'.join(edge_type)


class HeteroRelationLinkPredictor(nn.Module):
    """Learn node embeddings and relation-specific link predictors."""

    def __init__(
        self,
        node_types: List[str],
        edge_types: List[EdgeType],
        num_nodes_dict: Dict[str, int],
        hidden_channels: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        self.node_embeddings = nn.ModuleDict({
            node_type: nn.Embedding(num_nodes_dict[node_type], hidden_channels)
            for node_type in node_types
            if node_type in num_nodes_dict
        })

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                conv_dict[edge_type] = SAGEConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels,
                    aggr='mean',
                )
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))

        self.relation_heads = nn.ModuleDict({
            relation_key(edge_type): nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, 1),
            )
            for edge_type in edge_types
        })

        self.reset_parameters()

    def reset_parameters(self):
        for embedding in self.node_embeddings.values():
            nn.init.xavier_uniform_(embedding.weight)
        for head in self.relation_heads.values():
            for module in head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    def encode(self, edge_index_dict: Dict[EdgeType, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x_dict = {
            node_type: embedding.weight
            for node_type, embedding in self.node_embeddings.items()
        }
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            for node_type in x_dict:
                x_dict[node_type] = F.relu(x_dict[node_type])
                x_dict[node_type] = F.dropout(
                    x_dict[node_type],
                    p=self.dropout,
                    training=self.training,
                )
        return x_dict

    def score_edges(
        self,
        embeddings: Dict[str, torch.Tensor],
        edge_type: EdgeType,
        src_index: torch.Tensor,
        dst_index: torch.Tensor,
    ) -> torch.Tensor:
        src_type, _, dst_type = edge_type
        src_emb = embeddings[src_type][src_index]
        dst_emb = embeddings[dst_type][dst_index]
        pair_emb = torch.cat([src_emb, dst_emb], dim=-1)
        logits = self.relation_heads[relation_key(edge_type)](pair_emb).squeeze(-1)
        return logits


class MultiRelationLinkPredictionRunner:
    """Prepare splits, train a GNN LP model, and report relation-wise metrics."""

    def __init__(
        self,
        kg,
        converter,
        train_proteins,
        valid_proteins,
        test_proteins,
        config: Dict,
        device: Optional[str] = None,
    ):
        self.kg = kg
        self.converter = converter
        self.train_proteins = train_proteins
        self.valid_proteins = valid_proteins
        self.test_proteins = test_proteins
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = int(config.get('gnn_seed', 42))
        self.rng = np.random.default_rng(self.seed)
        self.annotation_field = config.get(
            'gnn_annotation_field',
            config.get('go_annotation_field', 'exp_annotations'),
        )
        self.allowed_go_terms = self._load_allowed_go_terms(
            config.get('gnn_terms_file', config.get('go_terms_file'))
        )
        self.eval_go_terms = self._load_allowed_go_terms(
            config.get('gnn_eval_terms_file', config.get('go_eval_terms_file'))
        )

        self.target_relations = self._resolve_target_relations(
            config.get('gnn_target_relations', None)
        )
        self.active_relations: List[EdgeType] = []
        self.train_pos_limit = int(config.get('gnn_train_pos_limit_per_relation', 5000))
        self.eval_pos_limit = int(config.get('gnn_eval_pos_limit_per_relation', 20000))
        self.num_epochs = int(config.get('gnn_epochs', 20))
        self.learning_rate = float(config.get('gnn_lr', 1e-3))
        self.weight_decay = float(config.get('gnn_weight_decay', 1e-4))
        self.dropout = float(config.get('gnn_dropout', 0.2))
        self.hidden_channels = int(config.get('gnn_hidden_channels', 128))
        self.num_layers = int(config.get('gnn_num_layers', 2))
        self.neg_multiplier = float(config.get('gnn_negative_multiplier', 1.0))
        self.eval_every = int(config.get('gnn_eval_every', 1))

    def run(self, output_dir: Path) -> Dict[str, Dict]:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        hetero_data = self.converter.convert_to_hetero_data(
            node_feature_dim=self.hidden_channels,
            use_text_embeddings=False,
        )
        relation_splits = self._build_relation_splits(hetero_data)
        self.active_relations = [
            edge_type for edge_type in self.target_relations
            if edge_type in relation_splits and len(relation_splits[edge_type]['train_pos']) > 0
        ]
        if not self.active_relations:
            raise RuntimeError('No active relations available for GNN link prediction.')
        train_edge_index_dict = self._build_train_graph_edges(hetero_data, relation_splits)

        model = HeteroRelationLinkPredictor(
            node_types=list(hetero_data.node_types),
            edge_types=list(train_edge_index_dict.keys()),
            num_nodes_dict={node_type: hetero_data[node_type].num_nodes for node_type in hetero_data.node_types},
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()

        best_state = None
        best_metric = -float('inf')
        history: List[Dict] = []

        for epoch in range(1, self.num_epochs + 1):
            model.train()
            optimizer.zero_grad()
            train_edge_index_gpu = {
                k: v.to(self.device) for k, v in train_edge_index_dict.items()
            }
            embeddings = model.encode(train_edge_index_gpu)

            total_loss = 0.0
            relation_train_stats = {}
            for edge_type in self.active_relations:
                split = relation_splits[edge_type]
                pos_edges = split['train_pos']
                sampled_pos = self._sample_positive_subset(pos_edges, self.train_pos_limit)
                sampled_neg = self._sample_negative_edges(
                    edge_type,
                    sampled_pos,
                    self._negative_sample_count(len(sampled_pos)),
                    relation_splits,
                )
                if len(sampled_pos) == 0 or len(sampled_neg) == 0:
                    continue

                pos_src = torch.tensor(sampled_pos[:, 0], dtype=torch.long, device=self.device)
                pos_dst = torch.tensor(sampled_pos[:, 1], dtype=torch.long, device=self.device)
                neg_src = torch.tensor(sampled_neg[:, 0], dtype=torch.long, device=self.device)
                neg_dst = torch.tensor(sampled_neg[:, 1], dtype=torch.long, device=self.device)

                pos_logits = model.score_edges(embeddings, edge_type, pos_src, pos_dst)
                neg_logits = model.score_edges(embeddings, edge_type, neg_src, neg_dst)

                pos_labels = torch.ones_like(pos_logits)
                neg_labels = torch.zeros_like(neg_logits)
                loss = criterion(pos_logits, pos_labels) + criterion(neg_logits, neg_labels)
                total_loss = total_loss + loss
                relation_train_stats[relation_key(edge_type)] = {
                    'train_pos': int(len(sampled_pos)),
                    'train_neg': int(len(sampled_neg)),
                }

            total_loss.backward()
            optimizer.step()

            epoch_record = {
                'epoch': epoch,
                'train_loss': float(total_loss.item()),
                'train_relations': relation_train_stats,
            }

            if epoch % self.eval_every == 0:
                valid_metrics = self.evaluate(model, train_edge_index_dict, relation_splits, split_name='valid')
                epoch_record['valid'] = valid_metrics
                protein_go_key = relation_key(('protein', 'annotated_with', 'go_term'))
                if protein_go_key in valid_metrics:
                    score = valid_metrics[protein_go_key].get('ap', -1.0)
                else:
                    score = np.mean([m.get('ap', 0.0) for m in valid_metrics.values()]) if valid_metrics else -1.0
                if score > best_metric:
                    best_metric = score
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

                print(f"  Epoch {epoch}/{self.num_epochs} | loss={total_loss.item():.4f} | best_valid_ap={best_metric:.4f}")
            else:
                print(f"  Epoch {epoch}/{self.num_epochs} | loss={total_loss.item():.4f}")

            history.append(epoch_record)

        if best_state is not None:
            model.load_state_dict(best_state)

        test_metrics_full = self.evaluate(
            model,
            train_edge_index_dict,
            relation_splits,
            split_name='test',
        )
        test_metrics_zero10 = None
        if self.eval_go_terms is not None:
            test_metrics_zero10 = self.evaluate(
                model,
                train_edge_index_dict,
                relation_splits,
                split_name='test',
                go_term_filter=self.eval_go_terms,
            )
        valid_metrics = self.evaluate(model, train_edge_index_dict, relation_splits, split_name='valid')

        results = {
            'config': {
                'target_relations': [relation_key(r) for r in self.active_relations],
                'device': self.device,
                'train_pos_limit_per_relation': self.train_pos_limit,
                'eval_pos_limit_per_relation': self.eval_pos_limit,
                'negative_multiplier': self.neg_multiplier,
                'epochs': self.num_epochs,
                'annotation_field': self.annotation_field,
                'terms_file': self.config.get('gnn_terms_file', self.config.get('go_terms_file')),
                'eval_terms_file': self.config.get('gnn_eval_terms_file', self.config.get('go_eval_terms_file')),
            },
            'valid_metrics': valid_metrics,
            'test_metrics': test_metrics_full,
            'test_metrics_full': test_metrics_full,
            'test_metrics_zero10': test_metrics_zero10,
            'history': history,
        }

        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(results, f, indent=2)
        torch.save(model.state_dict(), output_dir / 'model.pt')
        with open(output_dir / 'relation_summaries.json', 'w') as f:
            json.dump(self._relation_summary(relation_splits), f, indent=2)

        return results

    def evaluate(
        self,
        model,
        train_edge_index_dict,
        relation_splits,
        split_name: str,
        go_term_filter: Optional[set] = None,
    ) -> Dict[str, Dict[str, float]]:
        model.eval()
        edge_index_gpu = {k: v.to(self.device) for k, v in train_edge_index_dict.items()}
        with torch.no_grad():
            embeddings = model.encode(edge_index_gpu)

        metrics = {}
        for edge_type in self.active_relations:
            split = relation_splits[edge_type]
            pos_edges = split[f'{split_name}_pos']
            if (
                go_term_filter is not None
                and edge_type == ('protein', 'annotated_with', 'go_term')
            ):
                pos_edges = self._filter_go_edges_by_allowed_terms(pos_edges, go_term_filter)
            if len(pos_edges) == 0:
                continue
            pos_edges = self._truncate_edges(pos_edges, self.eval_pos_limit)
            neg_edges = split.get(f'{split_name}_neg')
            if neg_edges is None or len(neg_edges) == 0:
                neg_edges = self._sample_negative_edges(
                    edge_type,
                    pos_edges,
                    self._negative_sample_count(len(pos_edges)),
                    relation_splits,
                )
            neg_edges = self._truncate_edges(neg_edges, self._negative_sample_count(len(pos_edges)))
            if len(neg_edges) == 0:
                continue

            pos_src = torch.tensor(pos_edges[:, 0], dtype=torch.long, device=self.device)
            pos_dst = torch.tensor(pos_edges[:, 1], dtype=torch.long, device=self.device)
            neg_src = torch.tensor(neg_edges[:, 0], dtype=torch.long, device=self.device)
            neg_dst = torch.tensor(neg_edges[:, 1], dtype=torch.long, device=self.device)

            with torch.no_grad():
                pos_scores = torch.sigmoid(model.score_edges(embeddings, edge_type, pos_src, pos_dst)).cpu().numpy()
                neg_scores = torch.sigmoid(model.score_edges(embeddings, edge_type, neg_src, neg_dst)).cpu().numpy()

            labels = np.concatenate([
                np.ones(len(pos_scores), dtype=np.int32),
                np.zeros(len(neg_scores), dtype=np.int32),
            ])
            scores = np.concatenate([pos_scores, neg_scores])
            preds = (scores >= 0.5).astype(np.int32)

            try:
                auc = float(roc_auc_score(labels, scores))
            except ValueError:
                auc = 0.0
            try:
                ap = float(average_precision_score(labels, scores))
            except ValueError:
                ap = 0.0

            metrics[relation_key(edge_type)] = {
                'num_pos': int(len(pos_scores)),
                'num_neg': int(len(neg_scores)),
                'roc_auc': auc,
                'ap': ap,
                'f1': float(f1_score(labels, preds, zero_division=0)),
                'accuracy': float((preds == labels).mean()),
                'mean_pos_score': float(np.mean(pos_scores)),
                'mean_neg_score': float(np.mean(neg_scores)),
            }
        return metrics

    def _build_relation_splits(self, hetero_data) -> Dict[EdgeType, Dict[str, np.ndarray]]:
        relation_splits: Dict[EdgeType, Dict[str, np.ndarray]] = {}
        edge_index_dict = dict(hetero_data.edge_index_dict)
        node_count_dict = {node_type: hetero_data[node_type].num_nodes for node_type in hetero_data.node_types}

        for edge_type in self.target_relations:
            if edge_type == ('protein', 'annotated_with', 'go_term'):
                train_pos = self._protein_go_edges_from_df(self.train_proteins)
                valid_pos = self._protein_go_edges_from_df(self.valid_proteins)
                test_pos = self._protein_go_edges_from_df(self.test_proteins)
            else:
                if edge_type not in edge_index_dict:
                    continue
                all_pos = edge_index_dict[edge_type].t().cpu().numpy()
                train_pos, valid_pos, test_pos = self._random_split_edges(all_pos)

            train_pos = self._dedup_edges(train_pos)
            valid_pos = self._dedup_edges(valid_pos)
            test_pos = self._dedup_edges(test_pos)

            all_pos = np.concatenate([arr for arr in [train_pos, valid_pos, test_pos] if len(arr) > 0], axis=0) if any(len(arr) > 0 for arr in [train_pos, valid_pos, test_pos]) else np.zeros((0, 2), dtype=np.int64)
            forbidden = set(map(tuple, all_pos.tolist())) if len(all_pos) > 0 else set()
            src_type, _, dst_type = edge_type
            src_size = node_count_dict[src_type]
            dst_size = node_count_dict[dst_type]

            relation_splits[edge_type] = {
                'train_pos': train_pos,
                'valid_pos': valid_pos,
                'test_pos': test_pos,
                'forbidden': forbidden,
                'src_size': src_size,
                'dst_size': dst_size,
                'train_neg': self._sample_negative_edges(edge_type, train_pos, len(train_pos), {}, forbidden=forbidden, src_size=src_size, dst_size=dst_size),
                'valid_neg': self._sample_negative_edges(edge_type, valid_pos, len(valid_pos), {}, forbidden=forbidden, src_size=src_size, dst_size=dst_size),
                'test_neg': self._sample_negative_edges(edge_type, test_pos, len(test_pos), {}, forbidden=forbidden, src_size=src_size, dst_size=dst_size),
            }
        return relation_splits

    def _build_train_graph_edges(self, hetero_data, relation_splits):
        train_edge_index_dict = {}
        for edge_type, edge_index in hetero_data.edge_index_dict.items():
            if edge_type in relation_splits:
                train_pos = relation_splits[edge_type]['train_pos']
                if len(train_pos) == 0:
                    continue
                train_edge_index_dict[edge_type] = torch.tensor(train_pos.T, dtype=torch.long)
            else:
                train_edge_index_dict[edge_type] = edge_index.clone().detach().cpu()
        return train_edge_index_dict

    def _resolve_target_relations(self, configured_relations) -> List[EdgeType]:
        if not configured_relations:
            return DEFAULT_TARGET_RELATIONS
        parsed = []
        for relation in configured_relations:
            if isinstance(relation, (list, tuple)) and len(relation) == 3:
                parsed.append(tuple(relation))
            elif isinstance(relation, str):
                parts = relation.split('|')
                if len(parts) == 3:
                    parsed.append((parts[0], parts[1], parts[2]))
        return parsed or DEFAULT_TARGET_RELATIONS

    def _protein_go_edges_from_df(self, df) -> np.ndarray:
        edges = []
        for _, row in df.iterrows():
            protein_id = row['proteins']
            go_terms = row.get(self.annotation_field, [])
            if not isinstance(go_terms, list):
                continue
            protein_key = (self.kg.NODE_PROTEIN, protein_id)
            if protein_key not in self.converter.node_id_to_idx:
                continue
            protein_idx = self.converter.node_id_to_idx[protein_key]
            for go_id in go_terms:
                if self.allowed_go_terms is not None and go_id not in self.allowed_go_terms:
                    continue
                go_key = (self.kg.NODE_GO, go_id)
                if go_key in self.converter.node_id_to_idx:
                    go_idx = self.converter.node_id_to_idx[go_key]
                    edges.append((protein_idx, go_idx))
        if not edges:
            return np.zeros((0, 2), dtype=np.int64)
        return np.array(sorted(set(edges)), dtype=np.int64)

    def _load_allowed_go_terms(self, terms_file: Optional[str]) -> Optional[set]:
        if not terms_file:
            return None

        terms_df = pd.read_pickle(terms_file)
        for column in ('terms', 'gos'):
            if column in terms_df.columns:
                return set(terms_df[column].dropna().astype(str).tolist())
        first_column = terms_df.columns[0]
        return set(terms_df[first_column].dropna().astype(str).tolist())

    def _negative_sample_count(self, num_positive: int) -> int:
        if num_positive <= 0:
            return 0
        return max(1, int(np.ceil(num_positive * self.neg_multiplier)))

    def _filter_go_edges_by_allowed_terms(
        self,
        edges: np.ndarray,
        allowed_go_terms: set,
    ) -> np.ndarray:
        if len(edges) == 0:
            return edges

        go_ids = self.converter.node_type_to_ids.get(self.kg.NODE_GO, [])
        filtered_edges = []
        for src_idx, dst_idx in edges:
            if 0 <= int(dst_idx) < len(go_ids) and go_ids[int(dst_idx)] in allowed_go_terms:
                filtered_edges.append((int(src_idx), int(dst_idx)))

        if not filtered_edges:
            return np.zeros((0, 2), dtype=np.int64)
        return np.array(filtered_edges, dtype=np.int64)

    def _random_split_edges(self, edges: np.ndarray, train_ratio: float = 0.8, valid_ratio: float = 0.1):
        if len(edges) == 0:
            empty = np.zeros((0, 2), dtype=np.int64)
            return empty, empty, empty
        edges = edges.copy()
        self.rng.shuffle(edges)
        n = len(edges)
        n_train = max(1, int(n * train_ratio)) if n >= 3 else max(1, n - 2)
        n_valid = max(1, int(n * valid_ratio)) if n >= 3 else 1 if n >= 2 else 0
        if n_train + n_valid >= n:
            n_valid = max(1, n - n_train - 1) if n - n_train > 1 else max(0, n - n_train)
        n_test = max(0, n - n_train - n_valid)
        train_edges = edges[:n_train]
        valid_edges = edges[n_train:n_train + n_valid]
        test_edges = edges[n_train + n_valid:n_train + n_valid + n_test]
        return train_edges, valid_edges, test_edges

    def _sample_positive_subset(self, edges: np.ndarray, max_edges: int) -> np.ndarray:
        if len(edges) <= max_edges:
            return edges
        indices = self.rng.choice(len(edges), size=max_edges, replace=False)
        return edges[np.sort(indices)]

    def _truncate_edges(self, edges: np.ndarray, max_edges: int) -> np.ndarray:
        if len(edges) <= max_edges:
            return edges
        return edges[:max_edges]

    def _dedup_edges(self, edges: np.ndarray) -> np.ndarray:
        if len(edges) == 0:
            return np.zeros((0, 2), dtype=np.int64)
        return np.array(sorted(set(map(tuple, edges.tolist()))), dtype=np.int64)

    def _sample_negative_edges(
        self,
        edge_type: EdgeType,
        positive_edges: np.ndarray,
        num_samples: int,
        relation_splits: Dict[EdgeType, Dict[str, np.ndarray]],
        forbidden: Optional[set] = None,
        src_size: Optional[int] = None,
        dst_size: Optional[int] = None,
    ) -> np.ndarray:
        if num_samples <= 0:
            return np.zeros((0, 2), dtype=np.int64)

        if forbidden is None:
            rel_info = relation_splits[edge_type]
            forbidden = rel_info['forbidden']
            src_size = rel_info['src_size']
            dst_size = rel_info['dst_size']
        assert src_size is not None and dst_size is not None

        if len(positive_edges) == 0:
            anchors = np.column_stack([
                self.rng.integers(0, src_size, size=max(1, num_samples)),
                self.rng.integers(0, dst_size, size=max(1, num_samples)),
            ])
        else:
            anchors = positive_edges

        src_pop = Counter(anchors[:, 0].tolist())
        dst_pop = Counter(anchors[:, 1].tolist())
        popular_src = [node for node, _ in src_pop.most_common(min(512, len(src_pop)))]
        popular_dst = [node for node, _ in dst_pop.most_common(min(512, len(dst_pop)))]

        negatives = set()
        attempts = 0
        max_attempts = max(num_samples * 80, 1000)
        while len(negatives) < num_samples and attempts < max_attempts:
            attempts += 1
            anchor = anchors[self.rng.integers(0, len(anchors))]
            src_idx, dst_idx = int(anchor[0]), int(anchor[1])
            mode = self.rng.choice(['tail_random', 'head_random', 'tail_popular'])
            if mode == 'head_random':
                src_idx = int(self.rng.integers(0, src_size))
            elif mode == 'tail_popular' and popular_dst:
                dst_idx = int(popular_dst[self.rng.integers(0, len(popular_dst))])
            else:
                dst_idx = int(self.rng.integers(0, dst_size))

            candidate = (src_idx, dst_idx)
            if candidate in forbidden or candidate in negatives:
                continue
            negatives.add(candidate)

        if not negatives:
            return np.zeros((0, 2), dtype=np.int64)
        return np.array(sorted(negatives), dtype=np.int64)

    def _relation_summary(self, relation_splits):
        summary = {}
        for edge_type, split in relation_splits.items():
            summary[relation_key(edge_type)] = {
                'train_pos': int(len(split['train_pos'])),
                'valid_pos': int(len(split['valid_pos'])),
                'test_pos': int(len(split['test_pos'])),
                'train_neg': int(len(split['train_neg'])),
                'valid_neg': int(len(split['valid_neg'])),
                'test_neg': int(len(split['test_neg'])),
            }
        return summary
