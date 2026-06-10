"""
数据加载和预处理模块
"""
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pickle


class DeepGOZeroDataLoader:
    """加载DeepGOZero数据集"""

    def __init__(
        self,
        data_dir: str,
        ont: str = 'mf',
        annotation_field: str = 'exp_annotations',
        terms_file: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.ont = ont
        self.annotation_field = annotation_field
        self.terms_file = terms_file
        self.data_split = {"train": "train_data.pkl", "valid": "valid_data.pkl", "test": "test_data.pkl"}

    def load_proteins(self, split: str = 'train') -> pd.DataFrame:
        """
        加载蛋白质数据

        Args:
            split: 'train', 'valid', 'test'

        Returns:
            DataFrame with columns: proteins, accessions, sequence, string_id,
                                   orgs, genes, interpros, exp_annotations
        """
        # file_path = self.data_dir / f"{split}_proteins.pkl"
        file_path = f"{self.data_dir}/{self.ont}/{self.data_split[split]}"
        # with open(file_path, 'rb') as f:
        #     data = pickle.load(f)
        data = pd.read_pickle(file_path)
        return data


    def load_protein_interactions(self) -> Dict:
        """
        加载蛋白质相互作用数据

        Returns:
            {uniprot_id: {'interactions': [...]}}
        """
        file_path = self.data_dir / "protein_interactions.json"
        with open(file_path, 'r') as f:
            interactions = json.load(f)
        return interactions

    def load_interpro_descriptions(self) -> Dict[str, str]:
        """
        加载InterPro域描述

        Returns:
            {interpro_id: description}
        """
        file_path = self.data_dir / "interpro_descriptions.json"
        with open(file_path, 'r') as f:
            interpro_desc = json.load(f)
        # 解析格式: "name#@@#description"
        parsed = {}
        for ipr_id, text in interpro_desc.items():
            if '#@@#' in text:
                name, desc = text.split('#@@#', 1)
                parsed[ipr_id] = f"{name}: {desc}"
            else:
                parsed[ipr_id] = text
        return parsed

    def load_gene_descriptions(self) -> Dict[str, Dict]:
        """
        加载基因描述

        Returns:
            {gene_id: {'symbol': ..., 'description': ..., 'summary': ...}}
        """
        file_path = self.data_dir / "gene_descriptions.json"
        with open(file_path, 'r') as f:
            gene_desc = json.load(f)
        return gene_desc

    def load_go_terms(self) -> Dict[str, Dict]:
        """
        加载GO术语本体

        Returns:
            {go_id: {'name': ..., 'definition': ..., 'namespace': ...}}
        """
        file_path = self.data_dir / "go_terms.json"
        with open(file_path, 'r') as f:
            go_terms = json.load(f)
        return go_terms

    def load_protein_descriptions(self) -> Dict[str, str]:
        """
        加载蛋白质详细描述

        Returns:
            {protein_id: full_description}
        """
        file_path = self.data_dir / "protein_descriptions.txt"
        proteins_desc = {}

        with open(file_path, 'r') as f:
            current_id = None
            current_text = []

            for line in f:
                line = line.strip()
                if line.startswith("Protein ID:"):
                    # 保存上一个蛋白质
                    if current_id:
                        proteins_desc[current_id] = ' '.join(current_text)
                    # 解析新蛋白质
                    parts = line.split('|')
                    current_id = parts[0].replace("Protein ID:", "").strip()
                    current_text = [line]
                elif line and current_id:
                    current_text.append(line)

            # 保存最后一个
            if current_id:
                proteins_desc[current_id] = ' '.join(current_text)

        return proteins_desc

    def load_term_vocab(self, terms_file: Optional[str] = None) -> Optional[List[str]]:
        """加载DeepGOZero约束使用的GO标签词表。"""
        resolved_terms_file = terms_file or self.terms_file
        if not resolved_terms_file:
            return None

        terms_path = Path(resolved_terms_file)
        if not terms_path.exists():
            raise FileNotFoundError(f"Terms file not found: {terms_path}")

        terms_df = pd.read_pickle(terms_path)
        for column in ('terms', 'gos'):
            if column in terms_df.columns:
                return terms_df[column].dropna().astype(str).tolist()

        first_column = terms_df.columns[0]
        return terms_df[first_column].dropna().astype(str).tolist()

    def get_train_go_terms(
        self,
        annotation_field: Optional[str] = None,
        terms_file: Optional[str] = None,
    ) -> set:
        """获取训练集中出现的所有GO terms，并可选约束到固定标签空间。"""
        train_df = self.load_proteins('train')
        annotation_column = annotation_field or self.annotation_field
        allowed_terms = self.load_term_vocab(terms_file)
        allowed_term_set = set(allowed_terms) if allowed_terms is not None else None
        all_go_terms = set()
        for annotations in train_df[annotation_column]:
            if isinstance(annotations, list):
                if allowed_term_set is None:
                    all_go_terms.update(annotations)
                else:
                    all_go_terms.update(go_id for go_id in annotations if go_id in allowed_term_set)
        return all_go_terms


class DataPreprocessor:
    """数据预处理工具"""

    @staticmethod
    def filter_go_terms(go_terms: Dict, train_go_set: set) -> Dict:
        """只保留训练集相关的GO术语"""
        return {k: v for k, v in go_terms.items() if k in train_go_set}

    @staticmethod
    def parse_ppi_network(
        interactions: Dict,
        gene_descriptions: Optional[Dict[str, Dict]] = None,
        proteins_df: Optional[pd.DataFrame] = None,
        min_score: float = 0.0,
        min_dscore_or_escore: float = 0.0,
    ) -> List[Tuple[str, str, float]]:
        """
        解析PPI网络为边列表，并尽量映射回gene_id。

        优先级:
        1. STRING protein id -> dataset中的gene_id
        2. gene symbol -> gene_info中的唯一gene_id

        过滤规则:
        - score >= min_score
        - dscore >= min_dscore_or_escore 或 escore >= min_dscore_or_escore

        Returns:
            [(gene_a, gene_b, score), ...]
        """
        from collections import defaultdict

        string_id_to_genes = defaultdict(set)
        if proteins_df is not None:
            for _, row in proteins_df.iterrows():
                genes = row.get('genes', [])
                if isinstance(genes, str):
                    genes = [genes]
                elif not isinstance(genes, list):
                    genes = []

                string_ids = row.get('string_ids', row.get('string_id', []))
                if isinstance(string_ids, str):
                    string_ids = [string_ids]
                elif not isinstance(string_ids, list):
                    string_ids = []

                for string_id in string_ids:
                    for gene_id in genes:
                        if string_id and gene_id:
                            string_id_to_genes[string_id].add(gene_id)

        exact_symbol_to_genes = defaultdict(set)
        lower_symbol_to_genes = defaultdict(set)
        if gene_descriptions is not None:
            for gene_id, attrs in gene_descriptions.items():
                symbol = attrs.get('symbol', '')
                if symbol:
                    exact_symbol_to_genes[symbol].add(gene_id)
                    lower_symbol_to_genes[symbol.lower()].add(gene_id)

        def resolve_interactor(interaction: Dict, side: str) -> List[str]:
            string_id = interaction.get(f'stringId_{side}')
            if string_id and string_id in string_id_to_genes:
                return sorted(string_id_to_genes[string_id])

            symbol = interaction.get(f'preferredName_{side}')
            if not symbol:
                return []

            if gene_descriptions is not None and symbol in gene_descriptions:
                return [symbol]

            exact_matches = exact_symbol_to_genes.get(symbol, set())
            if len(exact_matches) == 1:
                return sorted(exact_matches)

            lower_matches = lower_symbol_to_genes.get(symbol.lower(), set())
            if len(lower_matches) == 1:
                return sorted(lower_matches)

            return []

        best_scores = {}
        for _, data in interactions.items():
            for interaction in data.get('interactions', []):
                score = interaction.get('score', 0.0)
                dscore = interaction.get('dscore', 0.0)
                escore = interaction.get('escore', 0.0)
                if score < min_score:
                    continue
                if max(dscore, escore) < min_dscore_or_escore:
                    continue
                gene_as = resolve_interactor(interaction, 'A')
                gene_bs = resolve_interactor(interaction, 'B')

                for gene_a in gene_as:
                    for gene_b in gene_bs:
                        if not gene_a or not gene_b or gene_a == gene_b:
                            continue
                        edge = (gene_a, gene_b)
                        best_scores[edge] = max(score, best_scores.get(edge, 0.0))

        return [(gene_a, gene_b, score) for (gene_a, gene_b), score in best_scores.items()]

    @staticmethod
    def extract_protein_gene_mapping(proteins_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        提取蛋白质到基因的映射

        Returns:
            {protein_id: [gene_ids]}
        """
        mapping = {}
        for _, row in proteins_df.iterrows():
            protein_id = row['proteins']
            genes = row['genes']
            if isinstance(genes, list):
                mapping[protein_id] = genes
            elif isinstance(genes, str):
                mapping[protein_id] = [genes]
            else:
                mapping[protein_id] = []
        return mapping


if __name__ == "__main__":
    # 测试数据加载
    loader = DeepGOZeroDataLoader("/path/to/data")

    # 加载各类数据
    train_proteins = loader.load_proteins('train')
    print(f"训练蛋白质数量: {len(train_proteins)}")

    go_terms = loader.load_go_terms()
    print(f"GO术语数量: {len(go_terms)}")

    train_go_set = loader.get_train_go_terms()
    print(f"训练集GO术语数量: {len(train_go_set)}")
