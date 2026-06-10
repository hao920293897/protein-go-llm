"""
知识图谱构建模块
整合蛋白质、基因、GO术语、InterPro域等信息
"""
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional
import json
import pickle
from collections import defaultdict
from pathlib import Path


class ProteinKnowledgeGraph:
    """蛋白质知识图谱"""

    # 节点类型
    NODE_PROTEIN = 'protein'
    NODE_GENE = 'gene'
    NODE_GO = 'go_term'
    NODE_INTERPRO = 'interpro'

    # 边类型
    EDGE_PROTEIN_GENE = 'protein_to_gene'
    EDGE_GENE_PROTEIN = 'gene_to_protein'
    EDGE_GENE_GENE = 'gene_interaction'  # PPI
    EDGE_PROTEIN_INTERPRO = 'has_domain'
    EDGE_PROTEIN_GO = 'annotated_with'
    EDGE_GO_GO = 'is_a'  # GO层级关系
    EDGE_INTERPRO_GO = 'associated_with'  # InterPro域与GO的统计关联

    def __init__(self):
        """初始化空图"""
        self.graph = nx.MultiDiGraph()

        # 存储节点属性
        self.protein_attrs = {}  # {protein_id: {sequence, description, ...}}
        self.gene_attrs = {}  # {gene_id: {symbol, description, ...}}
        self.go_attrs = {}  # {go_id: {name, definition, namespace}}
        self.interpro_attrs = {}  # {interpro_id: {description}}

        # 统计信息
        self.stats = defaultdict(int)

    def add_protein_nodes(self, proteins_df: pd.DataFrame, protein_descriptions: Dict[str, str]):
        """
        添加蛋白质节点

        Args:
            proteins_df: 蛋白质DataFrame
            protein_descriptions: 蛋白质描述字典
        """
        print("添加蛋白质节点...")

        for _, row in proteins_df.iterrows():
            protein_id = row['proteins']

            # 添加节点
            self.graph.add_node(protein_id, node_type=self.NODE_PROTEIN)

            # 存储属性
            self.protein_attrs[protein_id] = {
                'accession': row.get('accessions', ''),
                'sequence': row.get('sequence', row.get('sequences', '')),
                'string_id': row.get('string_id', row.get('string_ids', '')),
                'organism': row.get('orgs', ''),
                'description': protein_descriptions.get(protein_id, '')
            }

            self.stats['protein_nodes'] += 1

        print(f"  添加 {self.stats['protein_nodes']} 个蛋白质节点")

    def add_gene_nodes(self, gene_descriptions: Dict[str, Dict]):
        """
        添加基因节点

        Args:
            gene_descriptions: 基因描述字典
        """
        print("添加基因节点...")

        for gene_id, attrs in gene_descriptions.items():
            self.graph.add_node(gene_id, node_type=self.NODE_GENE)
            self.gene_attrs[gene_id] = {
                'symbol': attrs.get('symbol', ''),
                'description': attrs.get('description', ''),
                'summary': attrs.get('summary', '')
            }
            self.stats['gene_nodes'] += 1

        print(f"  添加 {self.stats['gene_nodes']} 个基因节点")

    def add_go_nodes(self, go_terms: Dict[str, Dict], train_go_set: Set[str]):
        """
        添加GO术语节点（仅训练集相关的）

        Args:
            go_terms: GO术语字典
            train_go_set: 训练集中出现的GO术语集合
        """
        print("添加GO术语节点...")

        for go_id, attrs in go_terms.items():
            if go_id in train_go_set:
                self.graph.add_node(go_id, node_type=self.NODE_GO)
                self.go_attrs[go_id] = {
                    'name': attrs.get('name', ''),
                    'definition': attrs.get('definition', ''),
                    'namespace': attrs.get('namespace', '')
                }
                self.stats['go_nodes'] += 1

        print(f"  添加 {self.stats['go_nodes']} 个GO术语节点")

    def add_interpro_nodes(self, interpro_descriptions: Dict[str, str]):
        """
        添加InterPro域节点

        Args:
            interpro_descriptions: InterPro描述字典
        """
        print("添加InterPro域节点...")

        for interpro_id, description in interpro_descriptions.items():
            self.graph.add_node(interpro_id, node_type=self.NODE_INTERPRO)
            self.interpro_attrs[interpro_id] = {
                'description': description
            }
            self.stats['interpro_nodes'] += 1

        print(f"  添加 {self.stats['interpro_nodes']} 个InterPro域节点")

    def add_protein_gene_edges(self, proteins_df: pd.DataFrame):
        """
        添加蛋白质-基因边

        Args:
            proteins_df: 蛋白质DataFrame
        """
        print("添加蛋白质-基因边...")

        for _, row in proteins_df.iterrows():
            protein_id = row['proteins']
            genes = row.get('genes', [])

            if isinstance(genes, str):
                genes = [genes]
            elif not isinstance(genes, list):
                genes = []

            for gene_id in genes:
                if self.graph.has_node(gene_id):
                    self.graph.add_edge(
                        protein_id,
                        gene_id,
                        edge_type=self.EDGE_PROTEIN_GENE
                    )
                    self.graph.add_edge(
                        gene_id,
                        protein_id,
                        edge_type=self.EDGE_GENE_PROTEIN
                    )
                    self.stats['protein_gene_edges'] += 1
                    self.stats['gene_protein_edges'] += 1

        print(f"  添加 {self.stats['protein_gene_edges']} 条蛋白质-基因边")

    def add_gene_interaction_edges(self, ppi_edges: List[Tuple[str, str, float]]):
        """
        添加基因相互作用边（PPI网络）

        Args:
            ppi_edges: [(gene_a, gene_b, score), ...]
        """
        print("添加基因相互作用边...")

        for gene_a, gene_b, score in ppi_edges:
            if self.graph.has_node(gene_a) and self.graph.has_node(gene_b):
                self.graph.add_edge(
                    gene_a,
                    gene_b,
                    edge_type=self.EDGE_GENE_GENE,
                    score=score
                )
                self.stats['gene_gene_edges'] += 1

        print(f"  添加 {self.stats['gene_gene_edges']} 条基因相互作用边")

    def add_protein_interpro_edges(self, proteins_df: pd.DataFrame):
        """
        添加蛋白质-InterPro域边

        Args:
            proteins_df: 蛋白质DataFrame
        """
        print("添加蛋白质-InterPro域边...")

        for _, row in proteins_df.iterrows():
            protein_id = row['proteins']
            interpros = row.get('interpros', [])

            if not isinstance(interpros, list):
                interpros = []

            for interpro_id in interpros:
                if self.graph.has_node(interpro_id):
                    self.graph.add_edge(
                        protein_id,
                        interpro_id,
                        edge_type=self.EDGE_PROTEIN_INTERPRO
                    )
                    self.stats['protein_interpro_edges'] += 1

        print(f"  添加 {self.stats['protein_interpro_edges']} 条蛋白质-InterPro边")

    def add_protein_go_edges(
        self,
        proteins_df: pd.DataFrame,
        annotation_field: str = 'exp_annotations',
        allowed_go_terms: Optional[Set[str]] = None,
        is_train: bool = True,
    ):
        """
        添加蛋白质-GO术语边

        Args:
            proteins_df: 蛋白质DataFrame
            is_train: 是否为训练集（测试集不添加此边）
        """
        if not is_train:
            print("跳过测试集的蛋白质-GO边")
            return

        print("添加蛋白质-GO术语边...")

        for _, row in proteins_df.iterrows():
            protein_id = row['proteins']
            go_terms = row.get(annotation_field, [])

            if not isinstance(go_terms, list):
                go_terms = []

            for go_id in go_terms:
                if allowed_go_terms is not None and go_id not in allowed_go_terms:
                    continue
                if self.graph.has_node(go_id):
                    self.graph.add_edge(
                        protein_id,
                        go_id,
                        edge_type=self.EDGE_PROTEIN_GO
                    )
                    self.stats['protein_go_edges'] += 1

        print(f"  添加 {self.stats['protein_go_edges']} 条蛋白质-GO边")

    def add_interpro_go_edges(
        self,
        train_proteins: pd.DataFrame,
        annotation_field: str = 'exp_annotations',
        allowed_go_terms: Optional[Set[str]] = None,
        min_support: int = 3,
    ):
        """
        添加InterPro-GO关联边（基于共现统计）

        Args:
            train_proteins: 训练集DataFrame
            min_support: 最小共现次数
        """
        print("添加InterPro-GO关联边...")

        # 统计共现
        co_occurrence = defaultdict(int)

        for _, row in train_proteins.iterrows():
            interpros = row.get('interpros', [])
            go_terms = row.get(annotation_field, [])

            if not isinstance(interpros, list):
                interpros = []
            if not isinstance(go_terms, list):
                go_terms = []

            for ipr in interpros:
                for go in go_terms:
                    if allowed_go_terms is not None and go not in allowed_go_terms:
                        continue
                    if self.graph.has_node(ipr) and self.graph.has_node(go):
                        co_occurrence[(ipr, go)] += 1

        # 添加边
        for (ipr, go), count in co_occurrence.items():
            if count >= min_support:
                self.graph.add_edge(
                    ipr,
                    go,
                    edge_type=self.EDGE_INTERPRO_GO,
                    support=count
                )
                self.stats['interpro_go_edges'] += 1

        print(f"  添加 {self.stats['interpro_go_edges']} 条InterPro-GO关联边")

    def add_go_hierarchy_edges(self, go_obo_file: Optional[str] = None):
        """
        添加GO层级关系边（is_a关系）

        Args:
            go_obo_file: GO OBO文件路径（可选）
        """
        print("添加GO层级关系边...")

        if go_obo_file is None:
            print("  未提供GO OBO文件，跳过层级关系构建")
            return

        # 解析GO OBO文件
        from goatools.obo_parser import GODag

        go_dag = GODag(go_obo_file)

        for go_id in self.go_attrs.keys():
            if go_id in go_dag:
                go_term = go_dag[go_id]
                # 添加父节点关系
                for parent in go_term.parents:
                    parent_id = parent.id
                    if self.graph.has_node(parent_id):
                        self.graph.add_edge(
                            go_id,
                            parent_id,
                            edge_type=self.EDGE_GO_GO
                        )
                        self.stats['go_go_edges'] += 1

        print(f"  添加 {self.stats['go_go_edges']} 条GO层级边")

    def get_node_features(self, node_id: str) -> Dict:
        """获取节点的特征和描述"""
        node_type = self.graph.nodes[node_id].get('node_type')

        if node_type == self.NODE_PROTEIN:
            return self.protein_attrs.get(node_id, {})
        elif node_type == self.NODE_GENE:
            return self.gene_attrs.get(node_id, {})
        elif node_type == self.NODE_GO:
            return self.go_attrs.get(node_id, {})
        elif node_type == self.NODE_INTERPRO:
            return self.interpro_attrs.get(node_id, {})
        else:
            return {}

    def get_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[str]:
        """
        获取节点的邻居

        Args:
            node_id: 节点ID
            edge_type: 边类型过滤（可选）

        Returns:
            邻居节点ID列表
        """
        neighbors = []
        for neighbor in self.graph.successors(node_id):
            if edge_type is None:
                neighbors.append(neighbor)
            else:
                # 检查边类型
                edges = self.graph.get_edge_data(node_id, neighbor)
                if edges:
                    for edge_data in edges.values():
                        if edge_data.get('edge_type') == edge_type:
                            neighbors.append(neighbor)
                            break
        return neighbors

    def print_statistics(self):
        """打印图统计信息"""
        print("\n=== 知识图谱统计 ===")
        print(f"总节点数: {self.graph.number_of_nodes()}")
        print(f"总边数: {self.graph.number_of_edges()}")
        print("\n节点统计:")
        print(f"  蛋白质: {self.stats['protein_nodes']}")
        print(f"  基因: {self.stats['gene_nodes']}")
        print(f"  GO术语: {self.stats['go_nodes']}")
        print(f"  InterPro域: {self.stats['interpro_nodes']}")
        print("\n边统计:")
        print(f"  蛋白质-基因: {self.stats['protein_gene_edges']}")
        print(f"  基因-蛋白质(反向): {self.stats['gene_protein_edges']}")
        print(f"  基因-基因(PPI): {self.stats['gene_gene_edges']}")
        print(f"  蛋白质-InterPro: {self.stats['protein_interpro_edges']}")
        print(f"  蛋白质-GO: {self.stats['protein_go_edges']}")
        print(f"  InterPro-GO: {self.stats['interpro_go_edges']}")
        print(f"  GO-GO(层级): {self.stats['go_go_edges']}")

    def save(self, output_path: str):
        """保存知识图谱"""
        save_data = {
            'graph': self.graph,
            'protein_attrs': self.protein_attrs,
            'gene_attrs': self.gene_attrs,
            'go_attrs': self.go_attrs,
            'interpro_attrs': self.interpro_attrs,
            'stats': dict(self.stats)
        }

        with open(output_path, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"\n知识图谱已保存: {output_path}")

    @classmethod
    def load(cls, input_path: str) -> 'ProteinKnowledgeGraph':
        """加载知识图谱"""
        with open(input_path, 'rb') as f:
            save_data = pickle.load(f)

        kg = cls()
        kg.graph = save_data['graph']
        kg.protein_attrs = save_data['protein_attrs']
        kg.gene_attrs = save_data['gene_attrs']
        kg.go_attrs = save_data['go_attrs']
        kg.interpro_attrs = save_data['interpro_attrs']
        kg.stats = defaultdict(int, save_data['stats'])

        print(f"知识图谱已加载: {input_path}")
        return kg


def build_knowledge_graph(
    data_loader,
    go_obo_file: Optional[str] = None,
    ppi_min_score: float = 0.0,
    ppi_min_dscore_or_escore: float = 0.0,
) -> ProteinKnowledgeGraph:
    """
    构建完整的蛋白质知识图谱

    Args:
        data_loader: DeepGOZeroDataLoader实例
        go_obo_file: GO OBO文件路径（可选）

    Returns:
        构建好的知识图谱
    """
    print("开始构建知识图谱...")
    # 'train_data.pkl', 'test_data.pkl', 'valid.pkl'
    # 加载数据
    train_proteins = data_loader.load_proteins('train')
    valid_proteins = data_loader.load_proteins('valid')
    test_proteins = data_loader.load_proteins('test')

    protein_interactions = data_loader.load_protein_interactions()
    interpro_desc = data_loader.load_interpro_descriptions()
    gene_desc = data_loader.load_gene_descriptions()
    go_terms = data_loader.load_go_terms()
    protein_desc = data_loader.load_protein_descriptions()

    annotation_field = getattr(data_loader, 'annotation_field', 'exp_annotations')
    terms_file = getattr(data_loader, 'terms_file', None)
    train_go_set = data_loader.get_train_go_terms(
        annotation_field=annotation_field,
        terms_file=terms_file,
    )
    all_proteins = pd.concat([train_proteins, valid_proteins, test_proteins], ignore_index=True)

    # 预处理PPI网络
    from data_loader import DataPreprocessor
    preprocessor = DataPreprocessor()
    ppi_edges = preprocessor.parse_ppi_network(
        protein_interactions,
        gene_descriptions=gene_desc,
        proteins_df=all_proteins,
        min_score=ppi_min_score,
        min_dscore_or_escore=ppi_min_dscore_or_escore,
    )
    print(f"PPI映射后可用边数: {len(ppi_edges)}")

    # 初始化知识图谱
    kg = ProteinKnowledgeGraph()

    # 添加节点
    kg.add_protein_nodes(all_proteins, protein_desc)
    kg.add_gene_nodes(gene_desc)
    kg.add_go_nodes(go_terms, train_go_set)
    kg.add_interpro_nodes(interpro_desc)

    # 添加边
    kg.add_protein_gene_edges(all_proteins)
    kg.add_gene_interaction_edges(ppi_edges)
    kg.add_protein_interpro_edges(all_proteins)

    # 训练集添加protein-go边，测试集不添加
    kg.add_protein_go_edges(
        train_proteins,
        annotation_field=annotation_field,
        allowed_go_terms=train_go_set,
        is_train=True,
    )
    kg.add_protein_go_edges(
        valid_proteins,
        annotation_field=annotation_field,
        allowed_go_terms=train_go_set,
        is_train=True,
    )
    kg.add_protein_go_edges(
        test_proteins,
        annotation_field=annotation_field,
        allowed_go_terms=train_go_set,
        is_train=False,
    )

    # 添加关联边
    kg.add_interpro_go_edges(
        train_proteins,
        annotation_field=annotation_field,
        allowed_go_terms=train_go_set,
        min_support=3,
    )
    kg.add_go_hierarchy_edges(go_obo_file)

    # 打印统计
    kg.print_statistics()

    return kg


if __name__ == "__main__":
    # 示例用法
    print("知识图谱构建模块")
