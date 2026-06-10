"""
端到端Pipeline
整合所有模块的完整流程
"""
import sys
import argparse
import pickle
import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch

# sys.path.append('/home/claude/protein_go_prediction')

from data_loader import DeepGOZeroDataLoader, DataPreprocessor
from baseline import DeepGOZeroPredictor, ESMBasedPredictor, ensemble_predictions
from protein_kg import  ProteinKnowledgeGraph, build_knowledge_graph
from converter import KGToPyGConverter, TextEmbeddingGenerator
from kg_retrieval import KGRetriever, ToGStyleReasoner
from gnn_models import (
    ProteinGOLinkPredictor,
    GNNTrainer,
    ProteinGONodeClassifier,
    NodeClassificationTrainer,
    load_normal_forms,
)
from llm_reranker import LLMReranker, LLMRerankingTrainer

class ProteinGOPredictionPipeline:
    """蛋白质GO预测完整Pipeline"""

    def __init__(self,
                 data_dir: str,
                 output_dir: str,
                 config: Dict):
        """
        Args:
            data_dir: 数据目录
            output_dir: 输出目录
            config: 配置字典
         """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

        # 数据加载器
        self.loader = DeepGOZeroDataLoader(data_dir)

        # 加载基础数据
        print("=" * 80)
        print("加载数据...")
        print("=" * 80)
        self.train_proteins = self.loader.load_proteins('train')
        self.valid_proteins = self.loader.load_proteins('valid')
        self.test_proteins = self.loader.load_proteins('test')
        self.go_terms = self.loader.load_go_terms()
        self.train_go_set = self.loader.get_train_go_terms()

        print(f"训练集: {len(self.train_proteins)} 蛋白质")
        print(f"验证集: {len(self.valid_proteins)} 蛋白质")
        print(f"测试集: {len(self.test_proteins)} 蛋白质")
        print(f"GO术语: {len(self.train_go_set)} (训练集)")

        # 存储中间结果
        self.baseline_predictions = {}
        self.kg = None
        self.node_embeddings = None
        self.hetero_data = None
        self.node_classifier = None
        self.kg_enhanced_predictions = {}
        self.final_predictions = {}

    def _cfg(self, key: str, default=None):
        """读取兼容 flat config 和 default_config.json nested config 的配置项。"""
        if key in self.config:
            return self.config[key]

        aliases = {
            'rebuild_kg': ('knowledge_graph', 'rebuild'),
            'go_obo_file': ('knowledge_graph', 'go_obo_file'),
            'min_interpro_go_support': ('knowledge_graph', 'min_interpro_go_support'),
            'add_sequence_similarity_edges': ('knowledge_graph', 'add_sequence_similarity_edges'),
            'sequence_similarity_top_k': ('knowledge_graph', 'sequence_similarity_top_k'),
            'sequence_similarity_threshold': ('knowledge_graph', 'sequence_similarity_threshold'),
            'sequence_similarity_kmer_size': ('knowledge_graph', 'sequence_similarity_kmer_size'),
            'sequence_similarity_max_features': ('knowledge_graph', 'sequence_similarity_max_features'),
            'text_embedding_model': ('node_classification', 'text_embedding_model'),
            'rebuild_text_embeddings': ('node_classification', 'rebuild_text_embeddings'),
            'rebuild_pyg_data': ('node_classification', 'rebuild_pyg_data'),
            'add_interpro_onehot_to_protein': ('node_classification', 'add_interpro_onehot_to_protein'),
            'node_feature_mode': ('node_classification', 'node_feature_mode'),
            'learnable_node_dim': ('node_classification', 'learnable_node_dim'),
            'hidden_channels': ('node_classification', 'hidden_channels'),
            'num_layers': ('node_classification', 'num_layers'),
            'dropout': ('node_classification', 'dropout'),
            'gnn_type': ('node_classification', 'gnn_type'),
            'num_heads': ('node_classification', 'num_heads'),
            'rgcn_num_bases': ('node_classification', 'rgcn_num_bases'),
            'use_el_loss': ('node_classification', 'use_el_loss'),
            'el_loss_weight': ('node_classification', 'el_loss_weight'),
            'go_norm_file': ('node_classification', 'go_norm_file'),
            'el_margin': ('node_classification', 'el_margin'),
            'learning_rate': ('node_classification', 'learning_rate'),
            'weight_decay': ('node_classification', 'weight_decay'),
            'epochs': ('node_classification', 'epochs'),
            'patience': ('node_classification', 'patience'),
            'use_pos_weight': ('node_classification', 'use_pos_weight'),
            'top_k_values': ('evaluation', 'top_k_values'),
        }

        path = aliases.get(key)
        if not path:
            return default

        value = self.config
        for part in path:
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value

    @staticmethod
    def _safe_torch_load(path: Path):
        try:
            return torch.load(path, weights_only=False)
        except TypeError:
            return torch.load(path)

    def stage1_baseline_recall(self):
        """阶段1: Baseline召回"""
        print("\n" + "=" * 80)
        print("阶段1: Baseline召回")
        print("=" * 80)

        baseline_dir = self.output_dir / "baseline_results"
        baseline_dir.mkdir(exist_ok=True)

        # 1.1 DeepGOZero
        print("\n[1/3] 运行DeepGOZero Baseline...")
        deepgozero_path = baseline_dir / "deepgozero_predictions.pkl"

        if deepgozero_path.exists() and not self.config.get('rerun_baseline', False):
            print("  加载已有结果...")
            with open(deepgozero_path, 'rb') as f:
                deepgozero_preds = pickle.load(f)
        else:
            # 构建InterPro词汇表
            all_interpros = set()
            for df in [self.train_proteins, self.valid_proteins, self.test_proteins]:
                for interpros in df['interpros']:
                    if isinstance(interpros, list):
                        all_interpros.update(interpros)
            interpro_vocab = sorted(list(all_interpros))

            # 训练和预测
            predictor = DeepGOZeroPredictor(interpro_vocab, sorted(list(self.train_go_set)))
            predictor.build_interpro_go_matrix(self.train_proteins)
            deepgozero_preds = predictor.batch_predict(self.test_proteins, top_k=200)

            with open(deepgozero_path, 'wb') as f:
                pickle.dump(deepgozero_preds, f)
            print(f"  结果已保存: {deepgozero_path}")

        self.baseline_predictions['deepgozero'] = deepgozero_preds

        # 1.2 ESM-2 (可选，计算密集)
        if self.config.get('use_esm', False):
            print("\n[2/3] 运行ESM-2 Baseline...")
            esm_path = baseline_dir / "esm_predictions.pkl"

            if esm_path.exists() and not self.config.get('rerun_baseline', False):
                print("  加载已有结果...")
                with open(esm_path, 'rb') as f:
                    esm_preds = pickle.load(f)
            else:
                predictor = ESMBasedPredictor()
                predictor.load_model()

                # 构建训练集embeddings
                train_embeddings, train_go_labels = predictor.build_train_embeddings(
                    self.train_proteins
                )

                # 预测测试集
                esm_preds = {}
                for idx, row in self.test_proteins.iterrows():
                    protein_id = row['proteins']
                    sequence = row['sequence']
                    query_emb = predictor.get_sequence_embedding(sequence)
                    predictions = predictor.predict_by_similarity(
                        query_emb, train_embeddings, train_go_labels, top_k=200
                    )
                    esm_preds[protein_id] = predictions

                    if (idx + 1) % 100 == 0:
                        print(f"  已处理 {idx + 1}/{len(self.test_proteins)}")

                with open(esm_path, 'wb') as f:
                    pickle.dump(esm_preds, f)

            self.baseline_predictions['esm'] = esm_preds

        # 1.3 融合baseline
        print("\n[3/3] 融合Baseline结果...")
        if len(self.baseline_predictions) > 1:
            ensemble_preds = ensemble_predictions(
                list(self.baseline_predictions.values()),
                weights=self.config.get('baseline_weights', None)
            )
            self.baseline_predictions['ensemble'] = ensemble_preds

            # 截断到top-k
            for protein_id in ensemble_preds:
                ensemble_preds[protein_id] = ensemble_preds[protein_id][:200]
        else:
            self.baseline_predictions['ensemble'] = self.baseline_predictions['deepgozero']

        print(f"\n✓ 阶段1完成，召回了 {len(self.baseline_predictions['ensemble'])} 个蛋白质的预测")

    def stage2_kg_construction(self):
        """阶段2: 知识图谱构建"""
        print("\n" + "=" * 80)
        print("阶段2: 知识图谱构建")
        print("=" * 80)

        kg_path = self.output_dir / "knowledge_graph.pkl"

        if kg_path.exists() and not self.config.get('rebuild_kg', False):
            print("加载已有知识图谱...")
            self.kg = ProteinKnowledgeGraph.load(kg_path)
        else:
            print("构建新的知识图谱...")
            self.kg = build_knowledge_graph(
                self.loader,
                go_obo_file=self.config.get('go_obo_file', None)
            )
            self.kg.save(kg_path)

        self.kg.print_statistics()

        print("\n✓ 阶段2完成")

    def stage_node_classification_kg_construction(self):
        """节点分类任务：构建不含GO标签节点/标签边泄漏的训练图。"""
        print("\n" + "=" * 80)
        print("节点分类阶段1: 构建知识图谱")
        print("=" * 80)

        gnn_dir = self.output_dir / "node_classification"
        gnn_dir.mkdir(exist_ok=True)
        kg_path = gnn_dir / "knowledge_graph_node_classification.pkl"

        if kg_path.exists() and not self._cfg('rebuild_kg', False):
            print("加载已有节点分类知识图谱...")
            self.kg = ProteinKnowledgeGraph.load(kg_path)
        else:
            print("构建新的节点分类知识图谱...")
            self.kg = build_knowledge_graph(
                self.loader,
                go_obo_file=self._cfg('go_obo_file', None),
                task_mode='node_classification',
                add_train_go_edges=False,
                add_valid_go_edges=False,
                min_interpro_go_support=self._cfg('min_interpro_go_support', 3),
                add_sequence_similarity_edges=self._cfg('add_sequence_similarity_edges', False),
                sequence_similarity_top_k=self._cfg('sequence_similarity_top_k', 5),
                sequence_similarity_threshold=self._cfg('sequence_similarity_threshold', 0.75),
                sequence_similarity_kmer_size=self._cfg('sequence_similarity_kmer_size', 3),
                sequence_similarity_max_features=self._cfg('sequence_similarity_max_features', None)
            )
            self.kg.save(kg_path)

        self.kg.print_statistics()
        print("\n✓ 节点分类知识图谱完成")

    def stage_node_text_features(self):
        """节点分类任务：生成并离线缓存protein/gene/interpro文本特征。"""
        print("\n" + "=" * 80)
        print("节点分类阶段2: 生成/加载文本节点特征")
        print("=" * 80)

        if self.kg is None:
            raise RuntimeError("请先运行stage_node_classification_kg_construction")

        gnn_dir = self.output_dir / "node_classification"
        gnn_dir.mkdir(exist_ok=True)

        model_name = self._cfg(
            'text_embedding_model',
            'sentence-transformers/all-MiniLM-L6-v2'
        )
        model_tag = model_name.replace('/', '__')
        emb_path = gnn_dir / f"text_embeddings_{model_tag}.pkl"

        if emb_path.exists() and not self._cfg('rebuild_text_embeddings', False):
            print(f"加载已有文本特征: {emb_path}")
            with open(emb_path, 'rb') as f:
                payload = pickle.load(f)
            self.node_embeddings = payload['embeddings']
        else:
            print(f"生成文本特征，模型: {model_name}")
            emb_gen = TextEmbeddingGenerator(model_name=model_name)
            self.node_embeddings = emb_gen.generate_all_embeddings(
                self.kg,
                node_types=['protein', 'gene', 'interpro']
            )
            payload = {
                'model_name': model_name,
                'embedding_dim': emb_gen.embedding_dim,
                'embeddings': self.node_embeddings
            }
            with open(emb_path, 'wb') as f:
                pickle.dump(payload, f)
            print(f"文本特征已缓存: {emb_path}")

        print("\n✓ 文本节点特征完成")

    def stage_node_classification_pyg_data(self):
        """节点分类任务：将KG和离线文本特征转换为PyG HeteroData。"""
        print("\n" + "=" * 80)
        print("节点分类阶段3: 构建PyG训练数据")
        print("=" * 80)

        if self.kg is None:
            raise RuntimeError("请先构建知识图谱")
        if self.node_embeddings is None:
            raise RuntimeError("请先生成或加载文本节点特征")

        gnn_dir = self.output_dir / "node_classification"
        interpro_suffix = "_with_interpro_onehot" if self._cfg('add_interpro_onehot_to_protein', False) else ""
        data_path = gnn_dir / f"hetero_data_node_classification{interpro_suffix}.pt"

        if data_path.exists() and not self._cfg('rebuild_pyg_data', False):
            print(f"加载已有PyG数据: {data_path}")
            self.hetero_data = self._safe_torch_load(data_path)
        else:
            converter = KGToPyGConverter(self.kg)
            self.hetero_data = converter.convert_to_hetero_data(
                use_text_embeddings=True,
                text_embeddings=self.node_embeddings,
                task_mode='node_classification',
                add_reverse_edges=True,
                add_interpro_onehot_to_protein=self._cfg('add_interpro_onehot_to_protein', False)
            )
            torch.save(self.hetero_data, data_path)
            print(f"PyG数据已保存: {data_path}")

        print(self.hetero_data)
        print("\n✓ PyG训练数据完成")

    def stage_train_node_classifier(self):
        """节点分类任务：训练异构GraphSAGE多标签分类器。"""
        print("\n" + "=" * 80)
        print("节点分类阶段4: 训练GNN节点分类模型")
        print("=" * 80)

        if self.hetero_data is None:
            raise RuntimeError("请先构建PyG训练数据")

        gnn_dir = self.output_dir / "node_classification"
        model_path = gnn_dir / "node_classifier_best.pt"
        history_path = gnn_dir / "training_history.json"

        data = self.hetero_data
        fixed_feature_dim = data['protein'].x.size(1)
        node_feature_mode = self._cfg('node_feature_mode', 'text+learnable')
        learnable_node_dim = self._cfg('learnable_node_dim', 128)
        if node_feature_mode == 'text':
            in_channels = fixed_feature_dim
        elif node_feature_mode == 'learnable':
            in_channels = learnable_node_dim
        elif node_feature_mode == 'text+learnable':
            in_channels = fixed_feature_dim + learnable_node_dim
        else:
            raise ValueError(
                f"Unsupported node_feature_mode={node_feature_mode}. "
                "Expected one of: text, learnable, text+learnable."
            )
        num_nodes_dict = {
            node_type: data[node_type].num_nodes
            for node_type in data.node_types
        }
        num_go_terms = data['protein'].y.size(1)
        go_normal_forms = None
        num_el_classes = num_go_terms
        num_relations = 0
        el_loss_weight = float(self._cfg('el_loss_weight', 0.0))
        use_el_loss = bool(self._cfg('use_el_loss', False)) or el_loss_weight > 0

        if use_el_loss:
            if el_loss_weight <= 0:
                el_loss_weight = 1.0

            go_norm_file = self._cfg('go_norm_file', None)
            if go_norm_file is None:
                project_root = Path(__file__).resolve().parents[2]
                candidates = [
                    self.data_dir / 'go.norm',
                    self.data_dir.parent / 'go.norm',
                    project_root / 'deepgozero-main' / 'data' / 'go.norm'
                ]
                go_norm_path = next((path for path in candidates if path.exists()), candidates[-1])
            else:
                go_norm_path = Path(go_norm_file)

            if not go_norm_path.exists():
                raise FileNotFoundError(f"EL loss需要go.norm文件，但未找到: {go_norm_path}")

            terms_dict = {
                go_id: idx
                for idx, go_id in enumerate(data['protein'].go_vocab)
            }
            nf1, nf2, nf3, nf4, relations, zero_classes = load_normal_forms(
                str(go_norm_path),
                terms_dict
            )
            go_normal_forms = NodeClassificationTrainer.make_normal_form_tensors(
                (nf1, nf2, nf3, nf4)
            )
            num_el_classes = num_go_terms + len(zero_classes)
            num_relations = len(relations)
            print(
                "EL loss已启用: "
                f"go_norm={go_norm_path}, "
                f"nf1={len(nf1)}, nf2={len(nf2)}, nf3={len(nf3)}, nf4={len(nf4)}, "
                f"relations={num_relations}, zero_classes={len(zero_classes)}, "
                f"weight={el_loss_weight}"
            )

        model = ProteinGONodeClassifier(
            node_types=data.node_types,
            edge_types=data.edge_types,
            in_channels=in_channels,
            num_go_terms=num_go_terms,
            num_nodes_dict=num_nodes_dict,
            node_feature_mode=node_feature_mode,
            learnable_node_dim=learnable_node_dim,
            hidden_channels=self._cfg('hidden_channels', 256),
            num_layers=self._cfg('num_layers', 2),
            dropout=self._cfg('dropout', 0.5),
            gnn_type=self._cfg('gnn_type', 'sage'),
            num_heads=self._cfg('num_heads', 4),
            rgcn_num_bases=self._cfg('rgcn_num_bases', None),
            num_el_classes=num_el_classes,
            num_relations=num_relations,
            el_margin=self._cfg('el_margin', 0.1)
        )

        pos_weight = None
        if self._cfg('use_pos_weight', True):
            pos_weight = NodeClassificationTrainer.compute_pos_weight(
                data['protein'].y,
                data['protein'].train_mask
            )

        trainer = NodeClassificationTrainer(
            model,
            lr=self._cfg('learning_rate', 1e-3),
            weight_decay=self._cfg('weight_decay', 1e-4),
            pos_weight=pos_weight,
            go_normal_forms=go_normal_forms,
            el_loss_weight=el_loss_weight if use_el_loss else 0.0
        )

        epochs = self._cfg('epochs', 50)
        patience = self._cfg('patience', 10)
        best_score = -1.0
        best_epoch = 0
        bad_epochs = 0
        history = []

        for epoch in range(1, epochs + 1):
            train_loss = trainer.train_epoch(data)
            valid_metrics = trainer.evaluate(data, split='valid')
            score = valid_metrics['micro_f1']

            row = {
                'epoch': epoch,
                'train_loss': train_loss,
                **{f"train_{k}": v for k, v in trainer.last_loss_parts.items()},
                **{f"valid_{k}": v for k, v in valid_metrics.items()}
            }
            history.append(row)

            loss_msg = f"Epoch {epoch:03d}: train_loss={train_loss:.4f}"
            if use_el_loss:
                loss_msg += (
                    f", cls_loss={trainer.last_loss_parts['classification_loss']:.4f}, "
                    f"el_loss={trainer.last_loss_parts['el_loss']:.4f}"
                )
            print(
                f"{loss_msg}, valid_loss={valid_metrics['loss']:.4f}, "
                f"valid_micro_f1={valid_metrics['micro_f1']:.4f}"
            )

            if score > best_score:
                best_score = score
                best_epoch = epoch
                bad_epochs = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'node_types': data.node_types,
                    'edge_types': data.edge_types,
                    'in_channels': in_channels,
                    'fixed_feature_dim': fixed_feature_dim,
                    'node_feature_mode': node_feature_mode,
                    'learnable_node_dim': learnable_node_dim,
                    'num_nodes_dict': num_nodes_dict,
                    'num_go_terms': num_go_terms,
                    'go_vocab': data['protein'].go_vocab,
                    'config': self.config
                }, model_path)
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"早停触发: best_epoch={best_epoch}, best_valid_micro_f1={best_score:.4f}")
                    break

        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        checkpoint = self._safe_torch_load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        self.node_classifier = model
        self.node_classification_trainer = trainer
        self.node_classification_trainer.model = model.to(trainer.device)

        print(f"最佳模型已保存: {model_path}")
        print("\n✓ GNN节点分类训练完成")

    def _true_labels_for_split(self, split: str) -> Dict[str, set]:
        df = {
            'train': self.train_proteins,
            'valid': self.valid_proteins,
            'test': self.test_proteins
        }[split]

        true_labels = {}
        for _, row in df.iterrows():
            anns = row.get('exp_annotations', [])
            true_labels[row['proteins']] = set(anns if isinstance(anns, list) else [])
        return true_labels

    def _evaluate_topk_predictions(self,
                                   predictions: Dict[str, List[Tuple[str, float]]],
                                   split: str,
                                   top_k_values: Optional[List[int]] = None) -> Dict[str, Dict[str, float]]:
        if top_k_values is None:
            top_k_values = self._cfg('top_k_values', [10, 20, 50, 100])

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
            f1 = (
                2 * avg_precision * avg_recall / (avg_precision + avg_recall)
                if avg_precision + avg_recall > 0
                else 0.0
            )
            metrics[f'top_{k}'] = {
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': f1
            }
        return metrics

    def stage_evaluate_node_classifier(self):
        """节点分类任务：评估测试集Top-K召回并保存预测。"""
        print("\n" + "=" * 80)
        print("节点分类阶段5: 评估与保存预测")
        print("=" * 80)

        if self.node_classifier is None or self.hetero_data is None:
            raise RuntimeError("请先训练节点分类模型")

        gnn_dir = self.output_dir / "node_classification"
        trainer = self.node_classification_trainer
        data = self.hetero_data
        go_vocab = data['protein'].go_vocab
        top_k_values = self._cfg('top_k_values', [10, 20, 50, 100])
        max_k = max(top_k_values)

        test_metrics = trainer.evaluate(data, split='test')
        test_metrics_2 = trainer.evaluate(data, split='test', threshold=0.3)
        predictions = trainer.predict_topk(
            data,
            go_vocab=go_vocab,
            split='test',
            k=max_k
        )
        topk_metrics = self._evaluate_topk_predictions(predictions, split='test', top_k_values=top_k_values)

        with open(gnn_dir / "node_classifier_predictions.pkl", 'wb') as f:
            pickle.dump(predictions, f)
        with open(gnn_dir / "node_classifier_predictions.json", 'w') as f:
            json.dump({
                protein_id: [
                    {'go_id': go_id, 'score': float(score)}
                    for go_id, score in preds
                ]
                for protein_id, preds in predictions.items()
            }, f, indent=2)
        with open(gnn_dir / "node_classifier_metrics.json", 'w') as f:
            json.dump({
                'threshold_metrics': test_metrics,
                'topk_metrics': topk_metrics
            }, f, indent=2)

        # print("测试集阈值=0.5指标:")
        # for key, value in test_metrics.items():
        #     print(f"  {key}: {value:.4f}")
        #
        # print("测试集阈值=0.3指标:")
        # for key, value in test_metrics_2.items():
        #     print(f"  {key}: {value:.4f}")

        print("测试集Top-K指标:")
        for name, values in topk_metrics.items():
            print(
                f"  {name}: P={values['precision']:.4f}, "
                f"R={values['recall']:.4f}, F1={values['f1']:.4f}"
            )

        self.final_predictions = predictions
        print("\n✓ 节点分类评估完成")

    def run_node_classification(self):
        """运行节点分类任务的逐步Pipeline。"""
        print("\n" + "=" * 80)
        print("蛋白质GO节点分类Pipeline")
        print("=" * 80)

        self.stage_node_classification_kg_construction()
        self.stage_node_text_features()
        self.stage_node_classification_pyg_data()
        self.stage_train_node_classifier()
        self.stage_evaluate_node_classifier()

        print("\n" + "=" * 80)
        print("节点分类Pipeline完成!")
        print("=" * 80)

    def stage3_kg_rag_enhancement(self):
        """阶段3: KG-RAG增强"""
        print("\n" + "=" * 80)
        print("阶段3: KG-RAG增强")
        print("=" * 80)

        rag_dir = self.output_dir / "kg_rag_results"
        rag_dir.mkdir(exist_ok=True)

        # 3.1 知识图谱检索
        print("\n[1/3] 知识图谱检索...")
        retriever = KGRetriever(self.kg)

        evidence_path = rag_dir / "kg_evidence.pkl"

        if evidence_path.exists() and not self.config.get('rerun_retrieval', False):
            print("  加载已有证据...")
            with open(evidence_path, 'rb') as f:
                evidence_dict = pickle.load(f)
        else:
            print("  检索证据路径...")
            # 为每个蛋白质-GO对检索证据
            baseline_preds = self.baseline_predictions['ensemble']

            protein_go_pairs = []
            for protein_id, predictions in baseline_preds.items():
                for go_id, score in predictions[:100]:  # 只为top-100检索证据
                    protein_go_pairs.append((protein_id, go_id))

            print(f"  需要检索 {len(protein_go_pairs)} 个蛋白质-GO对...")
            evidence_dict = retriever.batch_retrieve_evidence(
                protein_go_pairs,
                max_paths=5
            )

            with open(evidence_path, 'wb') as f:
                pickle.dump(evidence_dict, f)
            print(f"  证据已保存: {evidence_path}")

        # 3.2 图推理增强
        print("\n[2/3] ToG风格图推理...")
        reasoner = ToGStyleReasoner(self.kg, retriever)

        reasoning_path = rag_dir / "reasoning_enhanced_predictions.pkl"

        if reasoning_path.exists() and not self.config.get('rerun_reasoning', False):
            print("  加载已有推理结果...")
            with open(reasoning_path, 'rb') as f:
                reasoning_enhanced = pickle.load(f)
        else:
            print("  执行图推理...")
            baseline_preds = self.baseline_predictions['ensemble']
            reasoning_enhanced = reasoner.enhance_baseline_predictions(
                baseline_preds,
                alpha=self.config.get('reasoning_alpha', 0.6),
                top_k=100
            )

            with open(reasoning_path, 'wb') as f:
                pickle.dump(reasoning_enhanced, f)
            print(f"  推理结果已保存: {reasoning_path}")

        self.kg_enhanced_predictions = reasoning_enhanced

        # 3.3 GNN增强（可选）
        if self.config.get('use_gnn', False):
            print("\n[3/3] GNN模型增强...")
            self._train_gnn_model(rag_dir)
        else:
            print("\n[3/3] 跳过GNN增强")

        print("\n✓ 阶段3完成")

    def _train_gnn_model(self, rag_dir: Path):
        """训练GNN模型（可选）"""
        print("  转换为PyG格式...")
        converter = KGToPyGConverter(self.kg)

        # 生成文本embeddings
        if self.config.get('use_text_embeddings', False):
            emb_gen = TextEmbeddingGenerator()
            all_embeddings = emb_gen.generate_all_embeddings(self.kg)
            # TODO: 将embeddings加载到PyG数据中

        hetero_data = converter.convert_to_hetero_data(
            node_feature_dim=768,
            use_text_embeddings=False
        )

        print("  初始化GNN模型...")
        node_types = ['protein', 'gene', 'go_term', 'interpro']
        edge_types = []
        for edge_type in hetero_data.edge_types:
            edge_types.append(edge_type)

        model = ProteinGOLinkPredictor(
            node_types=node_types,
            edge_types=edge_types,
            in_channels=768,
            hidden_channels=256,
            num_layers=2,
            dropout=0.5
        )

        trainer = GNNTrainer(
            model,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            lr=0.001
        )

        # 准备训练数据
        # TODO: 构建训练标签矩阵
        print("  GNN训练需要额外实现标签准备逻辑...")
        print("  跳过实际训练步骤")

    def stage4_llm_reranking(self):
        """阶段4: LLM重排序"""
        print("\n" + "=" * 80)
        print("阶段4: LLM重排序")
        print("=" * 80)

        if not self.config.get('use_llm_reranking', True):
            print("跳过LLM重排序，使用KG增强结果作为最终预测")
            self.final_predictions = self.kg_enhanced_predictions
            return

        llm_dir = self.output_dir / "llm_reranking_results"
        llm_dir.mkdir(exist_ok=True)

        # 加载证据
        evidence_path = self.output_dir / "kg_rag_results" / "kg_evidence.pkl"
        with open(evidence_path, 'rb') as f:
            evidence_dict = pickle.load(f)

        # 加载蛋白质描述
        protein_descriptions = self.loader.load_protein_descriptions()

        # 初始化LLM重排序器
        print("\n初始化LLM重排序器...")
        reranker = LLMReranker(
            model_name=self.config.get('llm_model', 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'),
            device='cuda' if torch.cuda.is_available() else 'cpu',
            max_length=512
        )

        # 如果需要训练
        if self.config.get('train_llm_reranker', False):
            print("\n训练LLM重排序器...")
            trainer = LLMRerankingTrainer(reranker, learning_rate=1e-5)

            # 准备训练数据（使用验证集）
            train_data = trainer.prepare_training_data(
                self.valid_proteins,
                self.kg_enhanced_predictions,
                self.kg.go_attrs,
                evidence_dict,
                protein_descriptions
            )

            print(f"  训练样本数: {len(train_data)}")

            # 训练多个epoch
            num_epochs = self.config.get('llm_epochs', 3)
            for epoch in range(num_epochs):
                loss = trainer.train_epoch(train_data, batch_size=16)
                print(f"  Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

            # 保存模型
            torch.save(reranker.classifier.state_dict(), llm_dir / "classifier.pt")
        else:
            # 加载已训练的模型
            classifier_path = llm_dir / "classifier.pt"
            if classifier_path.exists():
                reranker.classifier.load_state_dict(torch.load(classifier_path))
                print("已加载训练好的分类器")

        # 重排序测试集预测
        print("\n重排序测试集预测...")
        reranked_path = llm_dir / "final_predictions.pkl"

        if reranked_path.exists() and not self.config.get('rerun_reranking', False):
            print("  加载已有重排序结果...")
            with open(reranked_path, 'rb') as f:
                final_predictions = pickle.load(f)
        else:
            print("  执行重排序...")

            # 筛选测试集蛋白质
            test_protein_ids = set(self.test_proteins['proteins'].tolist())
            test_predictions = {
                pid: preds for pid, preds in self.kg_enhanced_predictions.items()
                if pid in test_protein_ids
            }
            test_protein_info = {
                pid: desc for pid, desc in protein_descriptions.items()
                if pid in test_protein_ids
            }

            final_predictions = reranker.batch_rerank(
                test_protein_info,
                test_predictions,
                self.kg.go_attrs,
                evidence_dict,
                top_k=100
            )

            with open(reranked_path, 'wb') as f:
                pickle.dump(final_predictions, f)
            print(f"  重排序结果已保存: {reranked_path}")

        self.final_predictions = final_predictions

        print("\n✓ 阶段4完成")

    def evaluate_results(self):
        """评估所有阶段的结果"""
        print("\n" + "=" * 80)
        print("评估结果")
        print("=" * 80)

        from sklearn.metrics import precision_score, recall_score, f1_score

        # 准备真实标签
        true_labels_dict = {}
        for _, row in self.test_proteins.iterrows():
            protein_id = row['proteins']
            true_labels = set(row['exp_annotations']) if isinstance(row['exp_annotations'], list) else set()
            true_labels_dict[protein_id] = true_labels

        # 评估各阶段
        stages = {
            'Baseline (Ensemble)': self.baseline_predictions.get('ensemble', {}),
            'KG-Enhanced': self.kg_enhanced_predictions,
            'Final (LLM-Reranked)': self.final_predictions
        }

        for stage_name, predictions in stages.items():
            if not predictions:
                continue

            print(f"\n{stage_name}:")

            # 计算top-k精确率、召回率
            for k in [10, 20, 50, 100]:
                precisions = []
                recalls = []

                for protein_id, true_labels in true_labels_dict.items():
                    if protein_id not in predictions:
                        continue

                    pred_labels = set([go for go, _ in predictions[protein_id][:k]])

                    if len(pred_labels) > 0 and len(true_labels) > 0:
                        tp = len(pred_labels & true_labels)
                        precision = tp / len(pred_labels)
                        recall = tp / len(true_labels)

                        precisions.append(precision)
                        recalls.append(recall)

                if precisions:
                    avg_precision = np.mean(precisions)
                    avg_recall = np.mean(recalls)
                    f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (
                                                                                                      avg_precision + avg_recall) > 0 else 0

                    print(f"  Top-{k:3d}: P={avg_precision:.4f}, R={avg_recall:.4f}, F1={f1:.4f}")

    def save_final_results(self):
        """保存最终结果"""
        print("\n" + "=" * 80)
        print("保存最终结果")
        print("=" * 80)

        # 保存为JSON格式
        results_json = {}
        for protein_id, predictions in self.final_predictions.items():
            results_json[protein_id] = [
                {'go_id': go_id, 'score': float(score)}
                for go_id, score in predictions
            ]

        output_path = self.output_dir / "final_predictions.json"
        with open(output_path, 'w') as f:
            json.dump(results_json, f, indent=2)

        print(f"最终结果已保存: {output_path}")

        # 保存配置
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        print(f"配置已保存: {config_path}")

    def run(self):
        """运行完整Pipeline"""
        print("\n" + "=" * 80)
        print("蛋白质GO预测Pipeline")
        print("=" * 80)

        # 阶段1: Baseline召回
        # self.stage1_baseline_recall()

        # 阶段2: 知识图谱构建
        self.stage2_kg_construction()

        # 阶段3: KG-RAG增强
        self.stage3_kg_rag_enhancement()

        # 阶段4: LLM重排序
        self.stage4_llm_reranking()

        # 评估
        self.evaluate_results()

        # 保存结果
        self.save_final_results()

        print("\n" + "=" * 80)
        print("Pipeline完成!")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='蛋白质GO预测Pipeline')
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录')
    parser.add_argument('--output_dir', type=str, default='./pipeline_output', help='输出目录')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument(
        '--task',
        type=str,
        default='node_classification',
        choices=['full', 'node_classification'],
        help='运行完整旧流程或新的GNN节点分类流程'
    )

    args = parser.parse_args()

    # 加载配置
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # 默认配置
        config = {
            # Baseline设置
            'use_esm': False,  # ESM-2计算密集，默认关闭
            'baseline_weights': None,  # 等权重融合
            'rerun_baseline': False,

            # 知识图谱设置
            'rebuild_kg': False,
            'go_obo_file': None,  # 可选：GO OBO文件路径
            'add_sequence_similarity_edges': False,
            'sequence_similarity_top_k': 5,
            'sequence_similarity_threshold': 0.75,
            'sequence_similarity_kmer_size': 3,
            'sequence_similarity_max_features': None,

            # KG-RAG设置
            'rerun_retrieval': False,
            'rerun_reasoning': False,
            'reasoning_alpha': 0.6,  # baseline权重
            'use_gnn': False,  # GNN增强（计算密集）
            'use_text_embeddings': False,

            # LLM重排序设置
            'use_llm_reranking': True,
            'llm_model': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
            'train_llm_reranker': False,  # 是否训练（否则使用预训练）
            'llm_epochs': 3,
            'rerun_reranking': False,

            # 节点分类GNN设置
            'text_embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'use_el_loss': False,
            'rebuild_text_embeddings': False,
            'rebuild_pyg_data': False,
            'add_interpro_onehot_to_protein': False,
            'node_feature_mode': 'text+learnable',
            'learnable_node_dim': 128,
            'hidden_channels': 256,
            'num_layers': 2,
            'dropout': 0.5,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'epochs': 50,
            'patience': 10,
            'use_pos_weight': True,
            'top_k_values': [10, 20, 50, 100],
        }

    # 运行Pipeline
    pipeline = ProteinGOPredictionPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config=config
    )

    if args.task == 'node_classification':
        pipeline.run_node_classification()
    else:
        pipeline.run()


if __name__ == "__main__":
    # 示例运行命令：
    # python pipeline.py --data_dir /path/to/data --output_dir ./results
    main()
