"""
DeepGOZero Baseline实现
基于功能相似性的GO预测
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.metrics import pairwise_distances
from collections import defaultdict


class DeepGOZeroPredictor:
    """
    DeepGOZero预测器
    使用InterPro域作为特征进行GO预测
    """

    def __init__(self, interpro_vocab: List[str], go_vocab: List[str]):
        """
        Args:
            interpro_vocab: InterPro域ID列表
            go_vocab: GO术语ID列表
        """
        self.interpro_vocab = interpro_vocab
        self.go_vocab = go_vocab
        self.interpro_to_idx = {ipr: idx for idx, ipr in enumerate(interpro_vocab)}
        self.go_to_idx = {go: idx for idx, go in enumerate(go_vocab)}

        # 统计InterPro-GO共现矩阵
        self.interpro_go_matrix = None
        self.interpro_go_scores = None

    def build_interpro_go_matrix(self, train_proteins: pd.DataFrame):
        """
        构建InterPro域到GO术语的映射矩阵

        Args:
            train_proteins: 训练集DataFrame
        """
        n_interpro = len(self.interpro_vocab)
        n_go = len(self.go_vocab)

        # 共现计数
        co_occurrence = np.zeros((n_interpro, n_go))
        interpro_counts = np.zeros(n_interpro)

        for _, row in train_proteins.iterrows():
            interpros = row['interpros'] if isinstance(row['interpros'], list) else []
            go_terms = row['exp_annotations'] if isinstance(row['exp_annotations'], list) else []

            for ipr in interpros:
                if ipr in self.interpro_to_idx:
                    ipr_idx = self.interpro_to_idx[ipr]
                    interpro_counts[ipr_idx] += 1

                    for go in go_terms:
                        if go in self.go_to_idx:
                            go_idx = self.go_to_idx[go]
                            co_occurrence[ipr_idx, go_idx] += 1

        # 计算条件概率 P(GO|InterPro)
        self.interpro_go_matrix = co_occurrence.copy()
        for i in range(n_interpro):
            if interpro_counts[i] > 0:
                self.interpro_go_matrix[i, :] /= interpro_counts[i]

        self.interpro_go_scores = self.interpro_go_matrix
        print(f"InterPro-GO矩阵构建完成: {self.interpro_go_matrix.shape}")

    def predict(self, protein: Dict, top_k: int = 100) -> List[Tuple[str, float]]:
        """
        预测单个蛋白质的GO术语

        Args:
            protein: 包含'interpros'字段的字典
            top_k: 返回top-k个预测

        Returns:
            [(go_term, score), ...]
        """
        interpros = protein.get('interpros', [])
        if not isinstance(interpros, list):
            interpros = []

        # 聚合InterPro域的预测分数
        go_scores = np.zeros(len(self.go_vocab))
        valid_interpros = 0

        for ipr in interpros:
            if ipr in self.interpro_to_idx:
                ipr_idx = self.interpro_to_idx[ipr]
                go_scores += self.interpro_go_scores[ipr_idx, :]
                valid_interpros += 1

        # 归一化
        if valid_interpros > 0:
            go_scores /= valid_interpros

        # 获取top-k
        top_indices = np.argsort(go_scores)[::-1][:top_k]
        predictions = [(self.go_vocab[idx], float(go_scores[idx]))
                       for idx in top_indices if go_scores[idx] > 0]

        return predictions

    def batch_predict(self, proteins_df: pd.DataFrame, top_k: int = 100) -> Dict[str, List[Tuple[str, float]]]:
        """批量预测"""
        predictions = {}
        for _, row in proteins_df.iterrows():
            protein_id = row['proteins']
            protein_dict = row.to_dict()
            predictions[protein_id] = self.predict(protein_dict, top_k)
        return predictions


class ESMBasedPredictor:
    """
    基于ESM-2序列embedding的GO预测
    使用预训练的蛋白质语言模型
    """

    def __init__(self, model_name: str = "esm2_t33_650M_UR50D"):
        """
        Args:
            model_name: ESM模型名称
        """
        self.model_name = model_name
        self.model = None
        self.alphabet = None
        self.batch_converter = None

    def load_model(self):
        """加载ESM-2模型"""
        import esm

        # 加载模型
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()

        # 移到GPU
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        print(f"ESM模型加载完成: {self.model_name}")

    def get_sequence_embedding(self, sequence: str) -> np.ndarray:
        """
        获取序列的embedding

        Args:
            sequence: 氨基酸序列

        Returns:
            embedding向量
        """
        data = [("protein", sequence)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)

        if torch.cuda.is_available():
            batch_tokens = batch_tokens.cuda()

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33])
            embedding = results["representations"][33].mean(1)  # 平均池化

        return embedding.cpu().numpy()[0]

    def build_train_embeddings(self, train_proteins: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建训练集的序列embedding和GO标签矩阵

        Args:
            train_proteins: 训练集DataFrame

        Returns:
            (embeddings, go_labels)
        """
        embeddings = []
        go_labels = []

        for idx, row in train_proteins.iterrows():
            sequence = row['sequence']
            go_terms = row['exp_annotations']

            # 获取embedding
            emb = self.get_sequence_embedding(sequence)
            embeddings.append(emb)

            # GO标签(多标签)
            go_labels.append(go_terms if isinstance(go_terms, list) else [])

            if (idx + 1) % 100 == 0:
                print(f"已处理 {idx + 1} 个蛋白质")

        return np.array(embeddings), go_labels

    def predict_by_similarity(self,
                              query_embedding: np.ndarray,
                              train_embeddings: np.ndarray,
                              train_go_labels: List[List[str]],
                              top_k: int = 10,
                              top_neighbors: int = 50) -> List[Tuple[str, float]]:
        """
        基于embedding相似度预测GO术语

        Args:
            query_embedding: 查询蛋白质的embedding
            train_embeddings: 训练集embeddings
            train_go_labels: 训练集GO标签
            top_k: 返回top-k个预测
            top_neighbors: 使用top-n个最相似邻居

        Returns:
            [(go_term, score), ...]
        """
        # 计算余弦相似度
        query_emb = query_embedding.reshape(1, -1)
        distances = pairwise_distances(query_emb, train_embeddings, metric='cosine')[0]
        similarities = 1 - distances

        # 获取最相似的邻居
        top_neighbor_indices = np.argsort(similarities)[::-1][:top_neighbors]

        # 聚合邻居的GO术语
        go_scores = defaultdict(float)
        total_similarity = 0

        for idx in top_neighbor_indices:
            sim = similarities[idx]
            for go_term in train_go_labels[idx]:
                go_scores[go_term] += sim
            total_similarity += sim

        # 归一化
        if total_similarity > 0:
            go_scores = {go: score / total_similarity for go, score in go_scores.items()}

        # 排序并返回top-k
        sorted_predictions = sorted(go_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return sorted_predictions


def ensemble_predictions(predictions_list: List[Dict[str, List[Tuple[str, float]]]],
                         weights: List[float] = None) -> Dict[str, List[Tuple[str, float]]]:
    """
    融合多个baseline的预测结果

    Args:
        predictions_list: 多个预测器的结果列表
        weights: 各预测器的权重

    Returns:
        融合后的预测结果
    """
    if weights is None:
        weights = [1.0] * len(predictions_list)

    # 归一化权重
    weights = np.array(weights) / sum(weights)

    # 获取所有蛋白质ID
    all_proteins = set()
    for predictions in predictions_list:
        all_proteins.update(predictions.keys())

    # 融合预测
    ensembled = {}
    for protein_id in all_proteins:
        go_scores = defaultdict(float)

        for pred_dict, weight in zip(predictions_list, weights):
            if protein_id in pred_dict:
                for go_term, score in pred_dict[protein_id]:
                    go_scores[go_term] += weight * score

        # 排序
        sorted_predictions = sorted(go_scores.items(), key=lambda x: x[1], reverse=True)
        ensembled[protein_id] = sorted_predictions

    return ensembled


if __name__ == "__main__":
    # 示例用法
    print("DeepGOZero Baseline模块")