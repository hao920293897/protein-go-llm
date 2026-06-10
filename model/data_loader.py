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

    def __init__(self, data_dir: str, ont:str = 'mf'):
        self.data_dir = Path(data_dir)
        self.ont = ont
        self.data_split = {"train": 'train_data.pkl', "valid": 'test_data.pkl', "test": 'valid.pkl'}

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
        proteins_desc = dict(zip(data['proteins'], data['uniport_text']))
        return data


    def load_protein_interactions(self) -> Dict:
        """
        加载蛋白质相互作用数据

        Returns:
            {uniprot_id: {'interactions': [...]}}
        """
        file_path = f"{self.data_dir}/{self.ont}/mf_protein_interactions.json"
        protein_interactions = {}
        with open(file_path, 'r') as f:
            for line in f:
                uniprot_id, interactions = line['uniprot_id'], line['interactions']
                new_interactions = []
                for interaction in interactions:
                    score, escore, dscore = interaction['score'], interaction['escore'], interaction['dscore']
                    if score>=0.9 and (escore>=0.7 or dscore>=0.7):
                        new_interactions.append(interaction)
                protein_interactions[uniprot_id] = {'interactions': new_interactions}
        return protein_interactions

    def load_interpro_descriptions(self) -> Dict[str, str]:
        """
        加载InterPro域描述

        Returns:
            {interpro_id: description}
        """
        file_path = f"{self.data_dir}/interpro_descriptions.json"
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
        file_path = f"{self.data_dir}/gene_descriptions.json"
        with open(file_path, 'r') as f:
            gene_desc = json.load(f)
        return gene_desc

    def load_go_terms(self) -> Dict[str, Dict]:
        """
        加载GO术语本体

        Returns:
            {go_id: {'name': ..., 'definition': ..., 'namespace': ...}}
        """
        file_path = f"{self.data_dir}/go_definitions_obo.json"
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

    def get_train_go_terms(self) -> set:
        """获取训练集中出现的所有GO terms"""
        train_df = self.load_proteins('train')
        all_go_terms = set()
        for annotations in train_df['exp_annotations']:
            if isinstance(annotations, list):
                all_go_terms.update(annotations)
        return all_go_terms


class DataPreprocessor:
    """数据预处理工具"""

    @staticmethod
    def filter_go_terms(go_terms: Dict, train_go_set: set) -> Dict:
        """只保留训练集相关的GO术语"""
        return {k: v for k, v in go_terms.items() if k in train_go_set}

    @staticmethod
    def parse_ppi_network(interactions: Dict) -> List[Tuple[str, str, float]]:
        """
        解析PPI网络为边列表

        Returns:
            [(gene_a, gene_b, score), ...]
        """
        edges = []
        for protein_id, data in interactions.items():
            for interaction in data.get('interactions', []):
                gene_a = interaction.get('preferredName_A')
                gene_b = interaction.get('preferredName_B')
                score = interaction.get('score', 0.0)
                if gene_a and gene_b:
                    edges.append((gene_a, gene_b, score))
        return edges

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