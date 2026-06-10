"""
单元测试脚本
测试各个模块的基本功能
"""
import sys
import unittest
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from data_loader import DataPreprocessor
sys.path.append('/home/claude/protein_go_prediction')


class TestDataLoader(unittest.TestCase):
    """测试数据加载模块"""

    def test_data_preprocessor(self):
        """测试数据预处理器"""


        # 测试PPI解析
        interactions = {
            'P1': {
                'interactions': [
                    {'preferredName_A': 'G1', 'preferredName_B': 'G2', 'score': 0.8}
                ]
            }
        }

        edges = DataPreprocessor.parse_ppi_network(interactions)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0], ('G1', 'G2', 0.8))

        print("✓ DataPreprocessor测试通过")


class TestBaselineModels(unittest.TestCase):
    """测试Baseline模型"""

    def test_deepgozero_predictor(self):
        """测试DeepGOZero预测器"""
        from baseline.deepgozero_baseline import DeepGOZeroPredictor

        interpro_vocab = ['IPR001', 'IPR002', 'IPR003']
        go_vocab = ['GO:0001', 'GO:0002', 'GO:0003']

        predictor = DeepGOZeroPredictor(interpro_vocab, go_vocab)

        # 创建模拟训练数据
        train_data = pd.DataFrame({
            'proteins': ['P1', 'P2'],
            'interpros': [['IPR001', 'IPR002'], ['IPR002', 'IPR003']],
            'exp_annotations': [['GO:0001', 'GO:0002'], ['GO:0002', 'GO:0003']]
        })

        predictor.build_interpro_go_matrix(train_data)

        # 测试预测
        test_protein = {'interpros': ['IPR001']}
        predictions = predictor.predict(test_protein, top_k=2)

        self.assertIsInstance(predictions, list)
        self.assertGreater(len(predictions), 0)

        print("✓ DeepGOZero测试通过")

    def test_ensemble(self):
        """测试融合方法"""
        from baseline.deepgozero_baseline import ensemble_predictions

        pred1 = {
            'P1': [('GO:0001', 0.8), ('GO:0002', 0.6)]
        }
        pred2 = {
            'P1': [('GO:0001', 0.7), ('GO:0003', 0.5)]
        }

        ensembled = ensemble_predictions([pred1, pred2], weights=[0.6, 0.4])

        self.assertIn('P1', ensembled)
        self.assertGreater(len(ensembled['P1']), 0)

        print("✓ Ensemble测试通过")


class TestKnowledgeGraph(unittest.TestCase):
    """测试知识图谱"""

    def test_kg_construction(self):
        """测试KG构建"""
        from kg_construction.knowledge_graph import ProteinKnowledgeGraph

        kg = ProteinKnowledgeGraph()

        # 添加节点
        proteins_df = pd.DataFrame({
            'proteins': ['P1', 'P2'],
            'accessions': ['ACC1', 'ACC2'],
            'sequence': ['ACGT', 'TGCA'],
            'interpros': [['IPR001'], ['IPR002']],
            'genes': [['G1'], ['G2']],
            'exp_annotations': [['GO:0001'], ['GO:0002']]
        })

        protein_desc = {'P1': 'Protein 1', 'P2': 'Protein 2'}
        gene_desc = {'G1': {'symbol': 'Gene1'}, 'G2': {'symbol': 'Gene2'}}
        go_terms = {'GO:0001': {'name': 'GO1'}, 'GO:0002': {'name': 'GO2'}}
        interpro_desc = {'IPR001': 'Domain1', 'IPR002': 'Domain2'}

        kg.add_protein_nodes(proteins_df, protein_desc)
        kg.add_gene_nodes(gene_desc)
        kg.add_go_nodes(go_terms, {'GO:0001', 'GO:0002'})
        kg.add_interpro_nodes(interpro_desc)

        self.assertEqual(kg.stats['protein_nodes'], 2)
        self.assertEqual(kg.stats['gene_nodes'], 2)
        self.assertEqual(kg.stats['go_nodes'], 2)

        # 添加边
        kg.add_protein_gene_edges(proteins_df)
        kg.add_protein_interpro_edges(proteins_df)
        kg.add_protein_go_edges(proteins_df, is_train=True)

        self.assertGreater(kg.graph.number_of_edges(), 0)

        print("✓ KnowledgeGraph测试通过")

    def test_kg_retrieval(self):
        """测试KG检索"""
        from kg_construction.knowledge_graph import ProteinKnowledgeGraph
        from kg_rag.kg_retrieval import KGRetriever

        # 创建简单的测试图
        kg = ProteinKnowledgeGraph()
        kg.graph.add_node('P1', node_type='protein')
        kg.graph.add_node('IPR1', node_type='interpro')
        kg.graph.add_node('GO1', node_type='go_term')

        kg.graph.add_edge('P1', 'IPR1', edge_type='has_domain')
        kg.graph.add_edge('IPR1', 'GO1', edge_type='associated_with')

        kg.protein_attrs = {'P1': {}}
        kg.interpro_attrs = {'IPR1': {'description': 'Test domain'}}
        kg.go_attrs = {'GO1': {'name': 'Test GO'}}

        # 测试检索
        retriever = KGRetriever(kg)
        paths = retriever.find_paths('P1', 'GO1', max_length=3, max_paths=10)

        self.assertGreater(len(paths), 0)
        self.assertEqual(paths[0][0], 'P1')
        self.assertEqual(paths[0][-1], 'GO1')

        print("✓ KGRetrieval测试通过")


class TestPyGConverter(unittest.TestCase):
    """测试PyG转换器"""

    def test_conversion(self):
        """测试转换为PyG格式"""
        from kg_construction.knowledge_graph import ProteinKnowledgeGraph
        from kg_construction.pyg_converter import KGToPyGConverter

        # 创建测试KG
        kg = ProteinKnowledgeGraph()

        kg.graph.add_node('P1', node_type='protein')
        kg.graph.add_node('P2', node_type='protein')
        kg.graph.add_node('GO1', node_type='go_term')

        kg.graph.add_edge('P1', 'GO1', edge_type='annotated_with')

        kg.protein_attrs = {'P1': {}, 'P2': {}}
        kg.go_attrs = {'GO1': {}}
        kg.gene_attrs = {}
        kg.interpro_attrs = {}

        # 转换
        converter = KGToPyGConverter(kg)
        hetero_data = converter.convert_to_hetero_data(
            node_feature_dim=128,
            use_text_embeddings=False
        )

        self.assertIn('protein', hetero_data.node_types)
        self.assertIn('go_term', hetero_data.node_types)

        print("✓ PyGConverter测试通过")


class TestLLMReranker(unittest.TestCase):
    """测试LLM重排序"""

    def test_prompt_construction(self):
        """测试prompt构造"""
        from llm_reranking.llm_reranker import RerankingInput, PromptBasedReranker

        rerank_input = RerankingInput(
            protein_id='P1',
            protein_description='Test protein',
            go_id='GO:0001',
            go_name='Test GO',
            go_definition='Test definition',
            baseline_score=0.8,
            kg_evidence='Test evidence',
            reasoning_paths=['P1 -> GO1']
        )

        reranker = PromptBasedReranker()
        prompt = reranker.construct_reranking_prompt(
            'Test protein',
            'Test GO',
            'Test definition',
            ['Path1'],
            0.8
        )

        self.assertIn('Protein Description', prompt)
        self.assertIn('GO Term', prompt)
        self.assertIn('Evidence', prompt)

        print("✓ LLMReranker prompt测试通过")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("运行单元测试")
    print("=" * 80 + "\n")

    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestBaselineModels))
    suite.addTests(loader.loadTestsFromTestCase(TestKnowledgeGraph))
    suite.addTests(loader.loadTestsFromTestCase(TestPyGConverter))
    suite.addTests(loader.loadTestsFromTestCase(TestLLMReranker))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"运行测试: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✓ 所有测试通过!")
    else:
        print("\n✗ 部分测试失败")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)