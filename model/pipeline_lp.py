"""
Standalone pipeline for protein-GO link prediction.

This file keeps the link-prediction workflow separate from pipeline.py while
reusing protein_kg.py, converter.py, and gnn_link_prediction.py.
"""
import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from data_loader import DeepGOZeroDataLoader
from protein_kg import ProteinKnowledgeGraph, build_knowledge_graph
from converter import KGToPyGConverter, TextEmbeddingGenerator
from gnn_link_prediction import ProteinGOLinkPredictor, GNNTrainer


class ProteinGOLinkPredictionPipeline:
    """Protein-GO link prediction pipeline."""

    def __init__(self, data_dir: str, output_dir: str, config: Dict):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

        self.loader = DeepGOZeroDataLoader(data_dir)
        self.train_proteins = self.loader.load_proteins('train')
        self.valid_proteins = self.loader.load_proteins('valid')
        self.test_proteins = self.loader.load_proteins('test')

        self.kg = None
        self.node_embeddings = None
        self.hetero_data = None
        self.link_predictor = None
        self.link_prediction_trainer = None
        self.link_prediction_cache = {}
        self.final_predictions = {}

        print("=" * 80)
        print("蛋白质GO链路预测Pipeline")
        print("=" * 80)
        print(f"训练集: {len(self.train_proteins)} 蛋白质")
        print(f"验证集: {len(self.valid_proteins)} 蛋白质")
        print(f"测试集: {len(self.test_proteins)} 蛋白质")

    def _kg_cfg(self, key: str, default=None):
        kg_cfg = self.config.get('knowledge_graph', {})
        if isinstance(kg_cfg, dict) and key in kg_cfg:
            return kg_cfg[key]
        return self.config.get(key, default)

    def _lp_cfg(self, key: str, default=None):
        lp_cfg = self.config.get('link_prediction', {})
        if isinstance(lp_cfg, dict) and key in lp_cfg:
            return lp_cfg[key]
        return self.config.get(key, default)

    def _eval_cfg(self, key: str, default=None):
        eval_cfg = self.config.get('evaluation', {})
        if isinstance(eval_cfg, dict) and key in eval_cfg:
            return eval_cfg[key]
        return self.config.get(key, default)

    @staticmethod
    def _safe_torch_load(path: Path):
        try:
            return torch.load(path, weights_only=False)
        except TypeError:
            return torch.load(path)

    def stage_kg_construction(self):
        print("\n" + "=" * 80)
        print("阶段1: 构建链路预测知识图谱")
        print("=" * 80)

        lp_dir = self.output_dir / "link_prediction"
        lp_dir.mkdir(exist_ok=True)
        kg_path = lp_dir / "knowledge_graph_link_prediction.pkl"

        if kg_path.exists() and not self._kg_cfg('rebuild', False):
            print("加载已有链路预测知识图谱...")
            self.kg = ProteinKnowledgeGraph.load(kg_path)
        else:
            print("构建新的链路预测知识图谱...")
            self.kg = build_knowledge_graph(
                self.loader,
                go_obo_file=self._kg_cfg('go_obo_file', None),
                task_mode='link_prediction',
                add_train_go_edges=True,
                add_valid_go_edges=False,
                min_interpro_go_support=self._kg_cfg('min_interpro_go_support', 3),
                terms_file=self._lp_cfg('terms_file', None),
                few_shot_terms_file=self._lp_cfg('few_shot_terms_file', None),
                add_sequence_similarity_edges=self._kg_cfg('add_sequence_similarity_edges', False),
                sequence_similarity_top_k=self._kg_cfg('sequence_similarity_top_k', 5),
                sequence_similarity_threshold=self._kg_cfg('sequence_similarity_threshold', 0.75),
                sequence_similarity_kmer_size=self._kg_cfg('sequence_similarity_kmer_size', 3),
                sequence_similarity_max_features=self._kg_cfg('sequence_similarity_max_features', None),
            )
            self.kg.save(kg_path)

        self.kg.print_statistics()

    def stage_text_features(self):
        print("\n" + "=" * 80)
        print("阶段2: 生成/加载文本节点特征")
        print("=" * 80)

        if self.kg is None:
            raise RuntimeError("请先构建知识图谱")

        lp_dir = self.output_dir / "link_prediction"
        lp_dir.mkdir(exist_ok=True)
        model_name = self._lp_cfg('text_embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        model_tag = model_name.replace('/', '__')
        emb_path = lp_dir / f"text_embeddings_{model_tag}.pkl"

        if emb_path.exists() and not self._lp_cfg('rebuild_text_embeddings', False):
            print(f"加载已有文本特征: {emb_path}")
            with open(emb_path, 'rb') as f:
                payload = pickle.load(f)
            self.node_embeddings = payload['embeddings']
        else:
            print(f"生成文本特征，模型: {model_name}")
            emb_gen = TextEmbeddingGenerator(model_name=model_name)
            self.node_embeddings = emb_gen.generate_all_embeddings(
                self.kg,
                node_types=['protein', 'gene', 'interpro', 'go_term'],
            )
            with open(emb_path, 'wb') as f:
                pickle.dump({
                    'model_name': model_name,
                    'embedding_dim': emb_gen.embedding_dim,
                    'embeddings': self.node_embeddings,
                }, f)
            print(f"文本特征已缓存: {emb_path}")

    def stage_pyg_data(self):
        print("\n" + "=" * 80)
        print("阶段3: 构建PyG链路预测数据")
        print("=" * 80)

        if self.kg is None or self.node_embeddings is None:
            raise RuntimeError("请先构建知识图谱并生成文本特征")

        lp_dir = self.output_dir / "link_prediction"
        interpro_suffix = "_with_interpro_onehot" if self._lp_cfg('add_interpro_onehot_to_protein', False) else ""
        data_path = lp_dir / f"hetero_data_link_prediction{interpro_suffix}.pt"

        if data_path.exists() and not self._lp_cfg('rebuild_pyg_data', False):
            print(f"加载已有PyG数据: {data_path}")
            self.hetero_data = self._safe_torch_load(data_path)
        else:
            converter = KGToPyGConverter(self.kg)
            self.hetero_data = converter.convert_to_hetero_data(
                use_text_embeddings=True,
                text_embeddings=self.node_embeddings,
                task_mode='link_prediction',
                add_reverse_edges=True,
                add_interpro_onehot_to_protein=self._lp_cfg('add_interpro_onehot_to_protein', False),
            )
            torch.save(self.hetero_data, data_path)
            print(f"PyG数据已保存: {data_path}")

        print(self.hetero_data)

    def _prepare_cache(self):
        lp_data = self.kg.supervision.get('link_prediction', {})
        if not lp_data:
            raise RuntimeError("kg.supervision['link_prediction']为空")

        protein_to_idx = {pid: idx for idx, pid in enumerate(self.hetero_data['protein'].node_ids)}
        go_to_idx = {go_id: idx for idx, go_id in enumerate(self.hetero_data['go_term'].node_ids)}

        positive_edge_index = {}
        positive_go_lookup = defaultdict(set)
        split_protein_ids = {}
        split_protein_indices = {}

        for split, edges in lp_data.get('positive_edges', {}).items():
            indexed_edges = []
            protein_ids = []
            for protein_id, go_id in edges:
                if protein_id not in protein_to_idx or go_id not in go_to_idx:
                    continue
                protein_idx = protein_to_idx[protein_id]
                go_idx = go_to_idx[go_id]
                indexed_edges.append((protein_idx, go_idx))
                positive_go_lookup[protein_idx].add(go_idx)
                protein_ids.append(protein_id)

            unique_protein_ids = list(dict.fromkeys(protein_ids))
            split_protein_ids[split] = unique_protein_ids
            split_protein_indices[split] = torch.tensor(
                [protein_to_idx[protein_id] for protein_id in unique_protein_ids],
                dtype=torch.long,
            )

            if indexed_edges:
                positive_edge_index[split] = torch.tensor(indexed_edges, dtype=torch.long).t().contiguous()
            else:
                positive_edge_index[split] = torch.empty((2, 0), dtype=torch.long)

        candidate_go_ids = [go_id for go_id in lp_data.get('candidate_go_ids', self.kg.go_vocab) if go_id in go_to_idx]
        few_shot_go_ids = [go_id for go_id in lp_data.get('few_shot_go_vocab', []) if go_id in go_to_idx]

        self.link_prediction_cache = {
            'positive_edge_index': positive_edge_index,
            'positive_go_lookup': dict(positive_go_lookup),
            'split_protein_ids': split_protein_ids,
            'split_protein_indices': split_protein_indices,
            'candidate_go_ids': candidate_go_ids,
            'candidate_go_indices': torch.tensor([go_to_idx[go_id] for go_id in candidate_go_ids], dtype=torch.long),
            'few_shot_go_ids': few_shot_go_ids,
            'few_shot_go_indices': torch.tensor([go_to_idx[go_id] for go_id in few_shot_go_ids], dtype=torch.long),
        }

    def stage_train(self):
        print("\n" + "=" * 80)
        print("阶段4: 训练GNN链路预测模型")
        print("=" * 80)

        if self.hetero_data is None:
            raise RuntimeError("请先构建PyG数据")
        self._prepare_cache()

        lp_dir = self.output_dir / "link_prediction"
        model_path = lp_dir / "link_predictor_best.pt"
        history_path = lp_dir / "training_history.json"

        data = self.hetero_data
        fixed_feature_dim = data['protein'].x.size(1)
        node_feature_mode = self._lp_cfg('node_feature_mode', 'text')
        hidden_channels = self._lp_cfg('hidden_channels', 256)
        if node_feature_mode == 'text':
            in_channels = fixed_feature_dim
        elif node_feature_mode == 'learnable':
            in_channels = hidden_channels
        elif node_feature_mode == 'text+learnable':
            in_channels = fixed_feature_dim + hidden_channels
        else:
            raise ValueError(
                f"Unsupported link_prediction.node_feature_mode={node_feature_mode}. "
                "Expected one of: text, learnable, text+learnable."
            )

        num_nodes_dict = {node_type: data[node_type].num_nodes for node_type in data.node_types}
        model = ProteinGOLinkPredictor(
            node_types=data.node_types,
            edge_types=data.edge_types,
            in_channels=in_channels,
            num_go_terms=data['go_term'].num_nodes,
            num_nodes_dict=num_nodes_dict,
            node_feature_mode=node_feature_mode,
            hidden_channels=hidden_channels,
            num_layers=self._lp_cfg('num_layers', 2),
            dropout=self._lp_cfg('dropout', 0.5),
            gnn_type=self._lp_cfg('gnn_type', 'sage'),
            num_heads=self._lp_cfg('num_heads', 4),
            rgcn_num_bases=self._lp_cfg('rgcn_num_bases', None),
        )
        trainer = GNNTrainer(
            model,
            lr=self._lp_cfg('learning_rate', 1e-3),
            weight_decay=self._lp_cfg('weight_decay', 1e-4),
            negative_ratio=self._lp_cfg('negative_ratio', 3),
            edge_batch_size=self._lp_cfg('edge_batch_size', 65536),
            protein_batch_size=self._lp_cfg('protein_batch_size', 64),
        )

        train_pos = self.link_prediction_cache['positive_edge_index']['train']
        valid_protein_indices = self.link_prediction_cache['split_protein_indices']['valid']
        valid_protein_ids = self.link_prediction_cache['split_protein_ids']['valid']
        candidate_go_indices = self.link_prediction_cache['candidate_go_indices']
        candidate_go_ids = self.link_prediction_cache['candidate_go_ids']

        epochs = self._lp_cfg('epochs', 50)
        patience = self._lp_cfg('patience', 50)
        eval_every = self._lp_cfg('eval_every', 1)
        top_k_values = self._eval_cfg('top_k_values', [10, 20, 50, 100])
        score_k = min(20, max(top_k_values))

        best_score = -1.0
        best_epoch = 0
        bad_epochs = 0
        history = []

        for epoch in range(1, epochs + 1):
            train_loss = trainer.train_epoch(
                data,
                train_positive_edge_index=train_pos,
                train_positive_go_lookup=self.link_prediction_cache['positive_go_lookup'],
                num_go_nodes=len(candidate_go_ids),
            )
            row = {'epoch': epoch, 'train_loss': train_loss}

            if epoch % eval_every == 0:
                valid_predictions = trainer.predict_topk(
                    data,
                    protein_indices=valid_protein_indices,
                    protein_ids=valid_protein_ids,
                    go_indices=candidate_go_indices,
                    go_ids=candidate_go_ids,
                    k=max(top_k_values),
                )
                valid_metrics = self._evaluate_topk_predictions(valid_predictions, 'valid', top_k_values)
                row.update({
                    f"valid_{name}_{metric}": value
                    for name, values in valid_metrics.items()
                    for metric, value in values.items()
                })
                score = valid_metrics[f'top_{score_k}']['f1']
                print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, valid_top_{score_k}_f1={score:.4f}")

                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    bad_epochs = 0
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'node_types': data.node_types,
                        'edge_types': data.edge_types,
                        'in_channels': in_channels,
                        'candidate_go_ids': candidate_go_ids,
                        'node_feature_mode': node_feature_mode,
                        'hidden_channels': hidden_channels,
                        'config': self.config,
                    }, model_path)
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        print(f"早停触发: best_epoch={best_epoch}, best_valid_top_{score_k}_f1={best_score:.4f}")
                        history.append(row)
                        break
            else:
                print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f}")

            history.append(row)

        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        checkpoint = self._safe_torch_load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        self.link_predictor = model
        self.link_prediction_trainer = trainer
        self.link_prediction_trainer.model = model.to(trainer.device)
        print(f"最佳模型已保存: {model_path}")

    def _true_labels_for_split(self, split: str) -> Dict[str, set]:
        df = {'train': self.train_proteins, 'valid': self.valid_proteins, 'test': self.test_proteins}[split]
        allowed_go = set(self.kg.go_vocab) if self.kg is not None and getattr(self.kg, 'go_vocab', None) else None
        true_labels = {}
        for _, row in df.iterrows():
            anns = row.get('prop_annotations', row.get('exp_annotations', []))
            labels = set(anns if isinstance(anns, list) else [])
            if allowed_go is not None:
                labels &= allowed_go
            true_labels[row['proteins']] = labels
        return true_labels

    def _link_label_matrix(self, split: str, protein_ids: List[str], go_ids: List[str]) -> np.ndarray:
        true_labels = self._true_labels_for_split(split)
        go_to_col = {go_id: col_idx for col_idx, go_id in enumerate(go_ids)}
        labels = np.zeros((len(protein_ids), len(go_ids)), dtype=np.float32)
        for row_idx, protein_id in enumerate(protein_ids):
            for go_id in true_labels.get(protein_id, set()):
                col_idx = go_to_col.get(go_id)
                if col_idx is not None:
                    labels[row_idx, col_idx] = 1.0
        return labels

    @staticmethod
    def _evaluate_score_matrix_thresholds(score_matrix: np.ndarray,
                                          label_matrix: np.ndarray,
                                          thresholds: List[float]) -> Dict[str, Dict[str, float]]:
        metrics = {}
        gold = label_matrix.astype(bool)
        gold_pos = int(gold.sum())
        for threshold in thresholds:
            preds = score_matrix >= threshold
            tp = int(np.logical_and(preds, gold).sum())
            pred_pos = int(preds.sum())
            precision = tp / pred_pos if pred_pos > 0 else 0.0
            recall = tp / gold_pos if gold_pos > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            metrics[f"threshold_{threshold:.2f}"] = {
                'threshold': float(threshold),
                'micro_precision': float(precision),
                'micro_recall': float(recall),
                'micro_f1': float(f1),
                'true_positive': tp,
                'predicted_positive': pred_pos,
                'gold_positive': gold_pos,
            }
        if metrics:
            best_name, best_values = max(metrics.items(), key=lambda item: item[1]['micro_f1'])
            metrics['best'] = {'name': best_name, **best_values}
        return metrics

    @staticmethod
    def _topk_from_score_matrix(score_matrix: np.ndarray,
                                protein_ids: List[str],
                                go_ids: List[str],
                                k: int) -> Dict[str, List[Tuple[str, float]]]:
        predictions = {}
        for row_idx, protein_id in enumerate(protein_ids):
            scores = score_matrix[row_idx]
            top_idx = np.argsort(-scores)[:k]
            predictions[protein_id] = [(go_ids[col_idx], float(scores[col_idx])) for col_idx in top_idx]
        return predictions

    def _evaluate_topk_predictions(self,
                                   predictions: Dict[str, List[Tuple[str, float]]],
                                   split: str,
                                   top_k_values: Optional[List[int]] = None) -> Dict[str, Dict[str, float]]:
        if top_k_values is None:
            top_k_values = self._eval_cfg('top_k_values', [10, 20, 50, 100])
        true_labels = self._true_labels_for_split(split)
        metrics = {}
        for k in top_k_values:
            precisions = []
            recalls = []
            for protein_id, labels in true_labels.items():
                if protein_id not in predictions or not labels:
                    continue
                pred_labels = set(go_id for go_id, _ in predictions[protein_id][:k])
                tp = len(pred_labels & labels)
                precisions.append(tp / len(pred_labels) if pred_labels else 0.0)
                recalls.append(tp / len(labels))
            avg_precision = float(np.mean(precisions)) if precisions else 0.0
            avg_recall = float(np.mean(recalls)) if recalls else 0.0
            f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if avg_precision + avg_recall > 0 else 0.0
            metrics[f'top_{k}'] = {'precision': avg_precision, 'recall': avg_recall, 'f1': f1}
        return metrics

    def stage_evaluate(self):
        print("\n" + "=" * 80)
        print("阶段5: 评估与保存预测")
        print("=" * 80)

        if self.link_predictor is None or self.link_prediction_trainer is None:
            raise RuntimeError("请先训练链路预测模型")
        if not self.link_prediction_cache:
            self._prepare_cache()

        lp_dir = self.output_dir / "link_prediction"
        top_k_values = self._eval_cfg('top_k_values', [10, 20, 50, 100])
        max_k = max(top_k_values)
        test_protein_indices = self.link_prediction_cache['split_protein_indices']['test']
        test_protein_ids = self.link_prediction_cache['split_protein_ids']['test']
        candidate_go_indices = self.link_prediction_cache['candidate_go_indices']
        candidate_go_ids = self.link_prediction_cache['candidate_go_ids']

        score_matrix = self.link_prediction_trainer.predict_score_matrix(
            self.hetero_data,
            protein_indices=test_protein_indices,
            go_indices=candidate_go_indices,
        )
        label_matrix = self._link_label_matrix('test', test_protein_ids, candidate_go_ids)
        thresholds = self._lp_cfg('eval_thresholds', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        threshold_metrics = self._evaluate_score_matrix_thresholds(score_matrix, label_matrix, thresholds)
        predictions = self._topk_from_score_matrix(score_matrix, test_protein_ids, candidate_go_ids, max_k)
        topk_metrics = self._evaluate_topk_predictions(predictions, 'test', top_k_values)

        few_shot_predictions = {}
        few_shot_threshold_metrics = {}
        few_shot_topk_metrics = {}
        few_shot_go_indices = self.link_prediction_cache.get('few_shot_go_indices')
        few_shot_go_ids = self.link_prediction_cache.get('few_shot_go_ids', [])
        if few_shot_go_indices is not None and few_shot_go_indices.numel() > 0:
            few_shot_score_matrix = self.link_prediction_trainer.predict_score_matrix(
                self.hetero_data,
                protein_indices=test_protein_indices,
                go_indices=few_shot_go_indices,
            )
            few_shot_label_matrix = self._link_label_matrix('test', test_protein_ids, few_shot_go_ids)
            few_shot_threshold_metrics = self._evaluate_score_matrix_thresholds(few_shot_score_matrix, few_shot_label_matrix, thresholds)
            few_shot_predictions = self._topk_from_score_matrix(
                few_shot_score_matrix,
                test_protein_ids,
                few_shot_go_ids,
                min(max_k, len(few_shot_go_ids)),
            )
            few_shot_topk_values = [k for k in top_k_values if k <= len(few_shot_go_ids)] or [min(max_k, len(few_shot_go_ids))]
            few_shot_topk_metrics = self._evaluate_topk_predictions(few_shot_predictions, 'test', few_shot_topk_values)

        with open(lp_dir / "link_predictor_predictions.pkl", 'wb') as f:
            pickle.dump(predictions, f)
        with open(lp_dir / "link_predictor_predictions.json", 'w') as f:
            json.dump({
                protein_id: [{'go_id': go_id, 'score': float(score)} for go_id, score in preds]
                for protein_id, preds in predictions.items()
            }, f, indent=2)
        if few_shot_predictions:
            with open(lp_dir / "link_predictor_few_shot_predictions.pkl", 'wb') as f:
                pickle.dump(few_shot_predictions, f)
            with open(lp_dir / "link_predictor_few_shot_predictions.json", 'w') as f:
                json.dump({
                    protein_id: [{'go_id': go_id, 'score': float(score)} for go_id, score in preds]
                    for protein_id, preds in few_shot_predictions.items()
                }, f, indent=2)
        with open(lp_dir / "link_predictor_metrics.json", 'w') as f:
            json.dump({
                'threshold_metrics': threshold_metrics,
                'topk_metrics': topk_metrics,
                'few_shot_threshold_metrics': few_shot_threshold_metrics,
                'few_shot_topk_metrics': few_shot_topk_metrics,
                'candidate_go_count': len(candidate_go_ids),
                'few_shot_go_count': len(few_shot_go_ids),
            }, f, indent=2)

        self._print_metrics("测试集全GO阈值指标", threshold_metrics)
        self._print_topk("测试集Top-K指标", topk_metrics)
        if few_shot_threshold_metrics:
            self._print_metrics("测试集few-shot GO阈值指标", few_shot_threshold_metrics)
        if few_shot_topk_metrics:
            self._print_topk("测试集few-shot GO Top-K指标", few_shot_topk_metrics)

        self.final_predictions = predictions

    @staticmethod
    def _print_metrics(title: str, metrics: Dict[str, Dict[str, float]]):
        print(title + ":")
        for name, values in metrics.items():
            if name == 'best':
                continue
            print(
                f"  threshold={values['threshold']:.2f}: "
                f"P={values['micro_precision']:.4f}, "
                f"R={values['micro_recall']:.4f}, "
                f"F1={values['micro_f1']:.4f}"
            )
        if 'best' in metrics:
            best = metrics['best']
            print(
                f"  best={best['name']}: "
                f"P={best['micro_precision']:.4f}, "
                f"R={best['micro_recall']:.4f}, "
                f"F1={best['micro_f1']:.4f}"
            )

    @staticmethod
    def _print_topk(title: str, metrics: Dict[str, Dict[str, float]]):
        print(title + ":")
        for name, values in metrics.items():
            print(f"  {name}: P={values['precision']:.4f}, R={values['recall']:.4f}, F1={values['f1']:.4f}")

    def run(self):
        self.stage_kg_construction()
        self.stage_text_features()
        self.stage_pyg_data()
        self.stage_train()
        self.stage_evaluate()
        print("\n" + "=" * 80)
        print("链路预测Pipeline完成!")
        print("=" * 80)


def default_config() -> Dict:
    return {
        'knowledge_graph': {
            'rebuild': False,
            'go_obo_file': None,
            'min_interpro_go_support': 3,
            'add_sequence_similarity_edges': False,
            'sequence_similarity_top_k': 5,
            'sequence_similarity_threshold': 0.75,
            'sequence_similarity_kmer_size': 3,
            'sequence_similarity_max_features': None,
        },
        'link_prediction': {
            'terms_file': None,
            'few_shot_terms_file': None,
            'text_embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'rebuild_text_embeddings': False,
            'rebuild_pyg_data': False,
            'add_interpro_onehot_to_protein': False,
            'node_feature_mode': 'text',
            'hidden_channels': 256,
            'num_layers': 2,
            'dropout': 0.5,
            'gnn_type': 'sage',
            'num_heads': 4,
            'rgcn_num_bases': None,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'epochs': 50,
            'patience': 50,
            'eval_every': 1,
            'negative_ratio': 3,
            'edge_batch_size': 65536,
            'protein_batch_size': 64,
            'eval_thresholds': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        },
        'evaluation': {
            'top_k_values': [10, 20, 50, 100],
        },
    }


def load_config(config_path: Optional[str]) -> Dict:
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            loaded = json.load(f)
        config = default_config()
        for key, value in loaded.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key].update(value)
            else:
                config[key] = value
        return config
    return default_config()


def main():
    parser = argparse.ArgumentParser(description='蛋白质GO链路预测Pipeline')
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录')
    parser.add_argument('--output_dir', type=str, default='./pipeline_lp_output', help='输出目录')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    args = parser.parse_args()

    pipeline = ProteinGOLinkPredictionPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config=load_config(args.config),
    )
    pipeline.run()


if __name__ == "__main__":
    main()
