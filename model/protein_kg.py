"""
知识图谱构建模块
整合蛋白质、基因、GO术语、InterPro域等信息
"""
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional, Any
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
    EDGE_GENE_GENE = 'gene_interaction'  # PPI
    EDGE_PROTEIN_PROTEIN = 'protein_interaction'  # gene-gene PPI mapped to proteins
    EDGE_PROTEIN_SEQUENCE_SIMILAR = 'sequence_similar'
    EDGE_PROTEIN_INTERPRO = 'has_domain'
    EDGE_PROTEIN_GO = 'annotated_with'
    EDGE_GO_GO = 'is_a'  # GO层级关系
    EDGE_INTERPRO_GO = 'associated_with'  # InterPro域与GO的统计关联

    SPLITS = ('train', 'valid', 'test')

    def __init__(self):
        """初始化空图"""
        self.graph = nx.MultiDiGraph()

        # 存储节点属性
        self.protein_attrs = {}  # {protein_id: {sequence, description, ...}}
        self.gene_attrs = {}  # {gene_id: {symbol, description, ...}}
        self.go_attrs = {}  # {go_id: {name, definition, namespace}}
        self.interpro_attrs = {}  # {interpro_id: {description}}

        # 任务相关的监督数据。图结构只保存允许进入message passing的边；
        # 标签矩阵和评估边单独保存，避免验证/测试标签泄漏。
        self.protein_splits = {}  # {protein_id: train|valid|test}
        self.go_vocab = []  # terms_zero_10.pkl中的GO标签，固定分类/召回空间
        self.go_to_idx = {}  # {go_id: index}，顺序与terms_zero_10.pkl一致
        self.gene_id_name = {}
        self.supervision = {
            'node_classification': {},
            'link_prediction': {}
        }

        # 统计信息
        self.stats = defaultdict(int)

    @staticmethod
    def _as_list(value: Any) -> List:
        """将DataFrame单元格中的标量、数组或缺失值统一成list。"""
        if isinstance(value, list):
            return value
        if isinstance(value, tuple) or isinstance(value, set):
            return list(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if value is None:
            return []
        try:
            if pd.isna(value):
                return []
        except (TypeError, ValueError):
            pass
        if isinstance(value, str):
            return [value] if value else []
        return []

    def add_protein_nodes(self,
                          proteins_df: pd.DataFrame,
                          protein_descriptions: Dict[str, str],
                          split: Optional[str] = None):
        """
        添加蛋白质节点

        Args:
            proteins_df: 蛋白质DataFrame
            protein_descriptions: 蛋白质描述字典
            split: 数据划分名称；传入后写入protein_splits和节点属性
        """
        print("添加蛋白质节点...")

        for _, row in proteins_df.iterrows():
            protein_id = row['proteins']
            row_split = row.get('split', split)

            # 添加节点
            self.graph.add_node(protein_id, node_type=self.NODE_PROTEIN, split=row_split)

            # 存储属性
            self.protein_attrs[protein_id] = {
                'accession': row.get('accessions', ''),
                'sequence': row.get('sequence', ''),
                'string_id': row.get('string_id', ''),
                'organism': row.get('orgs', ''),
                'description': protein_descriptions.get(protein_id, row.get('uniprot_text', '')),
                'split': row_split
            }
            if row_split:
                self.protein_splits[protein_id] = row_split

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
            gene_name = attrs.get('symbol', '')
            if gene_name is not None:
                if self.gene_id_name.get(gene_id) is None:
                    self.gene_id_name[gene_id] = gene_name
                self.graph.add_node(gene_name, node_type=self.NODE_GENE)
                self.gene_attrs[gene_id] = {
                    'symbol': attrs.get('symbol', ''),
                    'description': attrs.get('description', ''),
                    'summary': attrs.get('summary', '')
                }
                self.stats['gene_nodes'] += 1

        print(f"  添加 {self.stats['gene_nodes']} 个基因节点")

    def add_go_nodes(self, go_terms: Dict[str, Dict], prediction_go_terms: List[str]):
        """
        添加GO术语节点（仅训练集相关的）

        Args:
            go_terms: GO术语字典
            train_go_set: 训练集中出现的GO术语集合
        """
        print("添加GO术语节点...")

        for go_id in prediction_go_terms:
            attrs = go_terms.get(go_id, {})
            self.graph.add_node(go_id, node_type=self.NODE_GO)
            self.go_attrs[go_id] = {
                'name': attrs.get('name', ''),
                'definition': attrs.get('definition', ''),
                'namespace': attrs.get('namespace', '')}
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

            genes = self._as_list(genes)
            for gene_id in genes:
                gene_name = self.gene_id_name.get(gene_id)
                if gene_name is not None and self.graph.has_node(gene_name):
                    self.graph.add_edge(
                        protein_id,
                        gene_id,
                        edge_type=self.EDGE_PROTEIN_GENE
                    )
                    self.stats['protein_gene_edges'] += 1

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

    def add_protein_interaction_edges_from_gene_interactions(
            self,
            ppi_edges: List[Tuple[str, str, float]],
            proteins_df: pd.DataFrame):
        """
        将基因相互作用边通过蛋白质-基因关系映射为蛋白质-蛋白质边。

        Args:
            ppi_edges: [(gene_a, gene_b, score), ...]
            proteins_df: 含 proteins 和 genes 列的蛋白质DataFrame
        """
        print("根据蛋白质-基因关系映射基因相互作用为蛋白质相互作用边...")

        gene_to_proteins = defaultdict(set)
        for _, row in proteins_df.iterrows():
            protein_id = row['proteins']
            if not self.graph.has_node(protein_id):
                continue

            for gene_id in self._as_list(row.get('genes', [])):
                if gene_id:
                    gene_name = self.gene_id_name.get(gene_id)
                    if gene_name is not None:
                        gene_to_proteins[gene_name].add(protein_id)

        mapped_edges = {}
        skipped_gene_edges = 0
        self_interactions = 0

        for gene_a, gene_b, score in ppi_edges:
            # gene_a_id, gene_b_id  = self.gene_id_name.get(gene_a), self.gene_id_name.get(gene_b)
            proteins_a = gene_to_proteins.get(gene_a, set())
            proteins_b = gene_to_proteins.get(gene_b, set())

            if not proteins_a or not proteins_b:
                skipped_gene_edges += 1
                continue

            for protein_a in proteins_a:
                for protein_b in proteins_b:
                    if protein_a == protein_b:
                        self_interactions += 1
                        continue

                    edge_key = (protein_a, protein_b)
                    edge_info = mapped_edges.setdefault(
                        edge_key,
                        {
                            'score': float(score) if score is not None else 0.0,
                            'support': 0
                        }
                    )
                    edge_info['score'] = max(
                        edge_info['score'],
                        float(score) if score is not None else 0.0
                    )
                    edge_info['support'] += 1

        for (protein_a, protein_b), edge_info in mapped_edges.items():
            self.graph.add_edge(
                protein_a,
                protein_b,
                edge_type=self.EDGE_PROTEIN_PROTEIN,
                score=edge_info['score'],
                support=edge_info['support']
            )
            self.stats['protein_protein_edges'] += 1

        self.stats['mapped_gene_gene_edges'] += len(ppi_edges) - skipped_gene_edges
        self.stats['skipped_gene_gene_edges'] += skipped_gene_edges
        self.stats['self_protein_interactions'] += self_interactions

        print(f"  添加 {self.stats['protein_protein_edges']} 条蛋白质相互作用边")
        print(f"  已映射基因互作: {self.stats['mapped_gene_gene_edges']} 条")
        if skipped_gene_edges:
            print(f"  跳过无蛋白映射的基因互作: {skipped_gene_edges} 条")
        if self_interactions:
            print(f"  跳过自相互作用映射: {self_interactions} 条")

    def add_protein_sequence_similarity_edges(
            self,
            proteins_df: pd.DataFrame,
            top_k: int = 5,
            similarity_threshold: float = 0.75,
            kmer_size: int = 3,
            bidirectional: bool = True,
            max_features: Optional[int] = None):
        """
        基于蛋白质氨基酸序列k-mer TF-IDF余弦相似度添加蛋白质-蛋白质边。

        该边只使用sequence字段，不使用任何GO标签，因此可连接train/valid/test
        中所有蛋白质，适合transductive节点分类设置。

        Args:
            proteins_df: 包含proteins和sequence列的DataFrame
            top_k: 每个蛋白保留的近邻数
            similarity_threshold: 最小cosine相似度
            kmer_size: amino-acid k-mer大小，默认3
            bidirectional: 是否写入双向边
            max_features: TF-IDF最大特征数；None表示不限制
        """
        print("添加蛋白质-蛋白质序列相似边...")

        if top_k <= 0:
            print("  top_k<=0，跳过序列相似边")
            return

        protein_ids = []
        sequences = []
        for _, row in proteins_df.iterrows():
            protein_id = row['proteins']
            sequence = row.get('sequences', '')
            if not self.graph.has_node(protein_id):
                continue
            if not isinstance(sequence, str) or not sequence:
                continue
            protein_ids.append(protein_id)
            sequences.append(sequence.upper())

        if len(protein_ids) <= 1:
            print("  有效蛋白序列不足，跳过序列相似边")
            return

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neighbors import NearestNeighbors

        vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(kmer_size, kmer_size),
            lowercase=False,
            norm='l2',
            max_features=max_features
        )
        seq_features = vectorizer.fit_transform(sequences)

        n_neighbors = min(top_k + 1, len(protein_ids))
        nn = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric='cosine',
            algorithm='brute',
            n_jobs=-1
        )
        nn.fit(seq_features)
        distances, indices = nn.kneighbors(seq_features, return_distance=True)

        edges = {}
        for src_idx, (neighbor_indices, neighbor_distances) in enumerate(zip(indices, distances)):
            src_id = protein_ids[src_idx]
            for dst_idx, distance in zip(neighbor_indices, neighbor_distances):
                if src_idx == dst_idx:
                    continue
                similarity = 1.0 - float(distance)
                if similarity < similarity_threshold:
                    continue

                dst_id = protein_ids[dst_idx]
                edge_key = (src_id, dst_id)
                if edge_key not in edges or similarity > edges[edge_key]:
                    edges[edge_key] = similarity

                if bidirectional:
                    rev_key = (dst_id, src_id)
                    if rev_key not in edges or similarity > edges[rev_key]:
                        edges[rev_key] = similarity

        for (src_id, dst_id), similarity in edges.items():
            self.graph.add_edge(
                src_id,
                dst_id,
                edge_type=self.EDGE_PROTEIN_SEQUENCE_SIMILAR,
                score=similarity
            )
            self.stats['protein_sequence_similarity_edges'] += 1

        self.stats['sequence_similarity_top_k'] = top_k
        self.stats['sequence_similarity_threshold'] = similarity_threshold
        self.stats['sequence_similarity_kmer_size'] = kmer_size
        print(
            f"  添加 {self.stats['protein_sequence_similarity_edges']} 条蛋白质序列相似边 "
            f"(top_k={top_k}, threshold={similarity_threshold}, kmer={kmer_size})"
        )

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

            interpros = self._as_list(interpros)

            for interpro_id in interpros:
                if self.graph.has_node(interpro_id):
                    self.graph.add_edge(
                        protein_id,
                        interpro_id,
                        edge_type=self.EDGE_PROTEIN_INTERPRO
                    )
                    self.stats['protein_interpro_edges'] += 1

        print(f"  添加 {self.stats['protein_interpro_edges']} 条蛋白质-InterPro边")

    def add_protein_go_edges(self,
                             proteins_df: pd.DataFrame,
                             split: str = 'train',
                             add_edges: bool = True,
                             is_train: Optional[bool] = None):
        """
        添加蛋白质-GO术语边

        Args:
            proteins_df: 蛋白质DataFrame
            split: 边所属数据划分
            add_edges: 是否真正加入图；False时只打印跳过信息
            is_train: 兼容旧接口；False等价于add_edges=False
        """
        if is_train is not None:
            add_edges = is_train
            if not is_train and split == 'train':
                split = 'test'

        if not add_edges:
            print(f"跳过{split}集的蛋白质-GO边")
            return

        print(f"添加{split}集蛋白质-GO术语边...")

        for _, row in proteins_df.iterrows():
            protein_id = row['proteins']
            go_terms = row.get('prop_annotations', [])

            go_terms = self._as_list(go_terms)

            for go_id in go_terms:
                if self.graph.has_node(go_id):
                    self.graph.add_edge(
                        protein_id,
                        go_id,
                        edge_type=self.EDGE_PROTEIN_GO,
                        split=split
                    )
                    self.stats['protein_go_edges'] += 1

        print(f"  添加 {self.stats['protein_go_edges']} 条蛋白质-GO边")

    def add_interpro_go_edges(self, train_proteins: pd.DataFrame, min_support: int = 3):
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
            go_terms = row.get('prop_annotations', [])

            interpros = self._as_list(interpros)
            go_terms = self._as_list(go_terms)

            for ipr in interpros:
                for go in go_terms:
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

    def set_go_vocab(self, terms: List[str]):
        """固定GO预测空间。默认只预测训练集中出现过的GO术语。"""
        # self.go_vocab = sorted([go_id for go_id in train_go_set if go_id in self.go_attrs])
        self.go_vocab = list(terms)
        self.go_to_idx = {go_id: idx for idx, go_id in enumerate(self.go_vocab)}
        self.stats['go_vocab_size'] = len(self.go_vocab)

    def build_node_classification_data(self, split_dfs: Dict[str, pd.DataFrame]):
        """
        构建任务1：蛋白质节点多标签分类监督数据。

        Returns保存在self.supervision['node_classification']中：
            go_vocab: List[str]
            protein_ids: Dict[str, List[str]]
            y: Dict[str, np.ndarray]，shape=[num_proteins, num_go_terms]
        """
        print("构建任务1节点分类标签矩阵...")
        go_to_idx = self.go_to_idx
        task_data = {
            'go_vocab': self.go_vocab,
            'go_to_idx': go_to_idx,
            'protein_ids': {},
            'y': {}
        }

        for split in self.SPLITS:
            df = split_dfs.get(split)
            if df is None:
                continue

            protein_ids = []
            labels = np.zeros((len(df), len(self.go_vocab)), dtype=np.float32)

            for row_idx, (_, row) in enumerate(df.iterrows()):
                protein_id = row['proteins']
                protein_ids.append(protein_id)
                for go_id in self._as_list(row.get('prop_annotations', [])):
                    go_idx = go_to_idx.get(go_id)
                    if go_idx is not None:
                        labels[row_idx, go_idx] = 1.0

            task_data['protein_ids'][split] = protein_ids
            task_data['y'][split] = labels
            self.stats[f'task1_{split}_proteins'] = len(protein_ids)
            self.stats[f'task1_{split}_positive_labels'] = int(labels.sum())

        self.supervision['node_classification'] = task_data
        print(f"  GO标签空间: {len(self.go_vocab)}")

    def build_link_prediction_data(self, split_dfs: Dict[str, pd.DataFrame]):
        """
        构建任务2：蛋白质-GO链路预测监督数据。

        图中只应加入训练集蛋白质-GO边；valid/test边保存在这里用于验证、
        测试和Top-N召回评估。
        """
        print("构建任务2链路预测边划分...")
        go_to_idx = {go_id: idx for idx, go_id in enumerate(self.go_vocab)}
        task_data = {
            'go_vocab': self.go_vocab,
            'go_to_idx': go_to_idx,
            'positive_edges': {},
            'candidate_go_ids': self.go_vocab
        }

        for split in self.SPLITS:
            df = split_dfs.get(split)
            if df is None:
                continue

            positive_edges = []
            for _, row in df.iterrows():
                protein_id = row['proteins']
                if not self.graph.has_node(protein_id):
                    continue
                for go_id in self._as_list(row.get('prop_annotations', [])):
                    if go_id in go_to_idx and self.graph.has_node(go_id):
                        positive_edges.append((protein_id, go_id))

            task_data['positive_edges'][split] = positive_edges
            self.stats[f'task2_{split}_positive_edges'] = len(positive_edges)

        self.supervision['link_prediction'] = task_data
        print(
            "  正样本边: "
            + ", ".join(
                f"{split}={len(task_data['positive_edges'].get(split, []))}"
                for split in self.SPLITS
            )
        )

    def get_text_for_node(self, node_id: str) -> str:
        """获取可直接送入文本编码器的节点文本属性。"""
        attrs = self.get_node_features(node_id)
        node_type = self.graph.nodes[node_id].get('node_type')

        if node_type == self.NODE_PROTEIN:
            fields = [
                attrs.get('description', ''),
                attrs.get('sequence', '')
            ]
        elif node_type == self.NODE_GENE:
            fields = [
                attrs.get('symbol', ''),
                attrs.get('description', ''),
                attrs.get('summary', '')
            ]
        elif node_type == self.NODE_GO:
            fields = [
                attrs.get('name', ''),
                attrs.get('namespace', ''),
                attrs.get('definition', '')
            ]
        elif node_type == self.NODE_INTERPRO:
            fields = [attrs.get('description', '')]
        else:
            fields = []

        return " ".join(str(field) for field in fields if field)

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
        print(f"  基因-基因(PPI): {self.stats['gene_gene_edges']}")
        print(f"  蛋白质-蛋白质(PPI): {self.stats['protein_protein_edges']}")
        print(f"  蛋白质-蛋白质(序列相似): {self.stats['protein_sequence_similarity_edges']}")
        print(f"  蛋白质-InterPro: {self.stats['protein_interpro_edges']}")
        print(f"  蛋白质-GO: {self.stats['protein_go_edges']}")
        print(f"  InterPro-GO: {self.stats['interpro_go_edges']}")
        print(f"  GO-GO(层级): {self.stats['go_go_edges']}")
        if self.go_vocab:
            print(f"\n任务GO标签空间: {len(self.go_vocab)}")
            for split in self.SPLITS:
                if f'task1_{split}_positive_labels' in self.stats:
                    print(
                        f"  任务1 {split}: proteins={self.stats[f'task1_{split}_proteins']}, "
                        f"labels={self.stats[f'task1_{split}_positive_labels']}"
                    )
            for split in self.SPLITS:
                if f'task2_{split}_positive_edges' in self.stats:
                    print(f"  任务2 {split}: positive_edges={self.stats[f'task2_{split}_positive_edges']}")

    def save(self, output_path: str):
        """保存知识图谱"""
        save_data = {
            'graph': self.graph,
            'protein_attrs': self.protein_attrs,
            'gene_attrs': self.gene_attrs,
            'go_attrs': self.go_attrs,
            'interpro_attrs': self.interpro_attrs,
            'protein_splits': self.protein_splits,
            'go_vocab': self.go_vocab,
            'go_to_idx': self.go_to_idx,
            'gene_id_name': self.gene_id_name,
            'supervision': self.supervision,
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
        kg.protein_splits = save_data.get('protein_splits', {})
        kg.go_vocab = save_data.get('go_vocab', [])
        kg.go_to_idx = save_data.get(
            'go_to_idx',
            {go_id: idx for idx, go_id in enumerate(kg.go_vocab)}
        )
        kg.gene_id_name = save_data.get('gene_id_name', {})
        kg.supervision = save_data.get('supervision', {
            'node_classification': {},
            'link_prediction': {}
        })
        kg.stats = defaultdict(int, save_data['stats'])

        print(f"知识图谱已加载: {input_path}")
        return kg

def build_knowledge_graph(data_loader,
                          go_obo_file: Optional[str] = None,
                          task_mode: str = 'both',
                          add_train_go_edges: bool = True,
                          add_valid_go_edges: bool = False,
                          min_interpro_go_support: int = 3,
                          terms_file: Optional[str] = None,
                          add_sequence_similarity_edges: bool = False,
                          sequence_similarity_top_k: int = 5,
                          sequence_similarity_threshold: float = 0.75,
                          sequence_similarity_kmer_size: int = 3,
                          sequence_similarity_max_features: Optional[int] = None) -> ProteinKnowledgeGraph:
    """
    构建完整的蛋白质知识图谱

    Args:
        data_loader: DeepGOZeroDataLoader实例
        go_obo_file: GO OBO文件路径（可选）
        task_mode: 'node_classification'、'link_prediction'或'both'
        add_train_go_edges: 是否把训练集蛋白质-GO边加入图中。
            任务1纯节点分类可设为False，避免直接使用标签边做message passing；
            任务2链路预测通常设为True。
        add_valid_go_edges: 是否把验证集蛋白质-GO边加入图中。默认False，避免泄漏。
        min_interpro_go_support: InterPro-GO训练集共现边的最小支持数
        terms_file: DeepGOZero terms_zero_10.pkl路径。默认使用
            ../deepgozero-main/data/{ont}/terms_zero_10.pkl
        add_sequence_similarity_edges: 是否添加蛋白质-蛋白质序列相似边
        sequence_similarity_top_k: 每个蛋白保留的序列近邻数
        sequence_similarity_threshold: 序列相似边的最小cosine相似度
        sequence_similarity_kmer_size: k-mer TF-IDF的k值
        sequence_similarity_max_features: TF-IDF最大特征数；None表示不限制

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
    # protein_desc = data_loader.load_protein_descriptions()
    all_data = pd.concat([
        train_proteins[['proteins', 'uniprot_text']],
        valid_proteins[['proteins', 'uniprot_text']],
        test_proteins[['proteins', 'uniprot_text']]
    ])

    protein_desc = dict(
        zip(all_data['proteins'], all_data['uniprot_text'])
    )

    ont = getattr(data_loader, 'ont', 'mf')
    if terms_file is None:
        terms_file = f'../deepgozero-main/data/{ont}/terms_zero_10.pkl'
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.tolist()
    terms_dict = {v: k for k, v in enumerate(terms)}

    # 预处理PPI网络
    try:
        from utils.data_loader import DataPreprocessor
    except ImportError:
        from data_loader import DataPreprocessor
    preprocessor = DataPreprocessor()
    ppi_edges = preprocessor.parse_ppi_network(protein_interactions)

    split_dfs = {
        'train': train_proteins,
        'valid': valid_proteins,
        'test': test_proteins
    }

    # 初始化知识图谱
    kg = ProteinKnowledgeGraph()

    # 添加节点
    split_tagged = []
    for split, df in split_dfs.items():
        tagged = df.copy()
        tagged['split'] = split
        split_tagged.append(tagged)
    all_proteins = pd.concat(split_tagged, ignore_index=True)

    kg.add_protein_nodes(all_proteins, protein_desc)
    kg.add_gene_nodes(gene_desc)
    kg.add_go_nodes(go_terms, terms)
    kg.set_go_vocab(terms)
    kg.add_interpro_nodes(interpro_desc)

    # 添加边
    kg.add_protein_gene_edges(all_proteins)
    kg.add_protein_interaction_edges_from_gene_interactions(ppi_edges, all_proteins)
    if add_sequence_similarity_edges:
        kg.add_protein_sequence_similarity_edges(
            all_proteins,
            top_k=sequence_similarity_top_k,
            similarity_threshold=sequence_similarity_threshold,
            kmer_size=sequence_similarity_kmer_size,
            max_features=sequence_similarity_max_features
        )
    kg.add_protein_interpro_edges(all_proteins)

    # 只把允许用于训练图消息传递的protein-go边加入图。
    kg.add_protein_go_edges(train_proteins, split='train', add_edges=add_train_go_edges)
    kg.add_protein_go_edges(valid_proteins, split='valid', add_edges=add_valid_go_edges)
    kg.add_protein_go_edges(test_proteins, split='test', add_edges=False)

    # 添加关联边
    kg.add_interpro_go_edges(train_proteins, min_support=min_interpro_go_support)
    kg.add_go_hierarchy_edges(go_obo_file)

    # 构建两类任务的监督数据。监督数据不等于图结构，valid/test标签只在这里出现。
    if task_mode in ('node_classification', 'both'):
        kg.build_node_classification_data(split_dfs)
    if task_mode in ('link_prediction', 'both'):
        kg.build_link_prediction_data(split_dfs)

    # 打印统计
    kg.print_statistics()

    return kg

if __name__ == "__main__":
    # 示例用法
    print("知识图谱构建模块")
