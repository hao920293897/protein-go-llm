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
from typing import Dict, List, Tuple
import torch

sys.path.append('/home/claude/protein_go_prediction')

from data_loader import DeepGOZeroDataLoader, DataPreprocessor
from baseline import DeepGOZeroPredictor, ESMBasedPredictor, ensemble_predictions
from protein_kg import ProteinKnowledgeGraph, build_knowledge_graph
from converter import KGToPyGConverter, TextEmbeddingGenerator
from kg_retrieval import KGRetriever, ToGStyleReasoner
from gnn_models import ProteinGOLinkPredictor, GNNTrainer
from gnn_link_prediction import MultiRelationLinkPredictionRunner
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
        self.loader.annotation_field = self.config.get(
            'gnn_annotation_field',
            self.config.get('go_annotation_field', 'exp_annotations'),
        )
        self.loader.terms_file = self.config.get(
            'gnn_terms_file',
            self.config.get('go_terms_file'),
        )
        self.label_field = self.loader.annotation_field

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
        self.kg_enhanced_predictions = {}
        self.final_predictions = {}
        self.gnn_results = {}

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
                go_obo_file=self.config.get('go_obo_file', None),
                ppi_min_score=self.config.get('ppi_min_score', 0.0),
                ppi_min_dscore_or_escore=self.config.get('ppi_min_dscore_or_escore', 0.0),
            )
            self.kg.save(kg_path)

        self.kg.print_statistics()

        print("\n✓ 阶段2完成")

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
        reasoner = ToGStyleReasoner(self.kg, retriever, evidence_cache=evidence_dict)

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
            print(f"  Evidence cache stats: hit={reasoner.cache_hits}, miss={reasoner.cache_misses}")

            with open(evidence_path, 'wb') as f:
                pickle.dump(reasoner.evidence_cache, f)
            print(f"  更新后的证据缓存已保存: {evidence_path}")

        self.kg_enhanced_predictions = reasoning_enhanced

        # 3.3 GNN增强（可选）
        if self.config.get('use_gnn', False):
            print("\n[3/3] GNN模型增强...")
            self._train_gnn_model(rag_dir / 'gnn_link_prediction')
        else:
            print("\n[3/3] 跳过GNN增强")

        print("\n✓ 阶段3完成")

    def _train_gnn_model(self, gnn_dir: Path):
        """训练纯GNN多关系链路预测baseline"""
        print("  构建异构图并准备多关系链路预测任务...")
        converter = KGToPyGConverter(self.kg)
        runner = MultiRelationLinkPredictionRunner(
            kg=self.kg,
            converter=converter,
            train_proteins=self.train_proteins,
            valid_proteins=self.valid_proteins,
            test_proteins=self.test_proteins,
            config=self.config,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
        self.gnn_results = runner.run(gnn_dir)

        print("  GNN测试集结果摘要 (full labels):")
        for relation, metrics in self.gnn_results.get('test_metrics_full', self.gnn_results.get('test_metrics', {})).items():
            print(
                f"    {relation}: AP={metrics['ap']:.4f}, AUC={metrics['roc_auc']:.4f}, "
                f"F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}"
            )
        zero10_metrics = self.gnn_results.get('test_metrics_zero10')
        if zero10_metrics:
            print("  GNN测试集结果摘要 (zero_10 labels):")
            for relation, metrics in zero10_metrics.items():
                print(
                    f"    {relation}: AP={metrics['ap']:.4f}, AUC={metrics['roc_auc']:.4f}, "
                    f"F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}"
                )

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
            labels = row.get(self.label_field, [])
            true_labels = set(labels) if isinstance(labels, list) else set()
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

        if self.config.get('gnn_only_mode', False):
            self.stage2_kg_construction()
            print("\n" + "=" * 80)
            print("阶段3: 纯GNN链路预测Baseline")
            print("=" * 80)
            self._train_gnn_model(self.output_dir / 'gnn_link_prediction')
            print("\n" + "=" * 80)
            print("GNN-only Pipeline完成!")
            print("=" * 80)
            return

        # 阶段1: Baseline召回
        self.stage1_baseline_recall()

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
            'ppi_min_score': 0.9,
            'ppi_min_dscore_or_escore': 0.7,
            'go_annotation_field': 'prop_annotations',
            'go_terms_file': '/home/zhangluhao/workspace/deepgozero-main/data/mf/terms.pkl',
            'go_eval_terms_file': '/home/zhangluhao/workspace/deepgozero-main/data/mf/terms_zero_10.pkl',

            # GNN-only设置
            'gnn_only_mode': False,
            'gnn_epochs': 10,
            'gnn_hidden_channels': 128,
            'gnn_num_layers': 2,
            'gnn_dropout': 0.2,
            'gnn_lr': 1e-3,
            'gnn_weight_decay': 1e-4,
            'gnn_seed': 42,
            'gnn_train_pos_limit_per_relation': 5000,
            'gnn_eval_pos_limit_per_relation': 20000,
            'gnn_negative_multiplier': 1.5,
            'gnn_eval_every': 1,
            'gnn_annotation_field': 'prop_annotations',
            'gnn_terms_file': '/home/zhangluhao/workspace/deepgozero-main/data/mf/terms.pkl',
            'gnn_eval_terms_file': '/home/zhangluhao/workspace/deepgozero-main/data/mf/terms_zero_10.pkl',
            'gnn_target_relations': [
                'protein|annotated_with|go_term',
                'protein|has_domain|interpro',
                'interpro|associated_with|go_term',
                'gene|gene_interaction|gene'
            ],

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
        }

    # 运行Pipeline
    pipeline = ProteinGOPredictionPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config=config
    )

    pipeline.run()


if __name__ == "__main__":
    # 示例运行命令：
    # python pipeline.py --data_dir /path/to/data --output_dir ./results
    main()
