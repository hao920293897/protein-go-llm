"""
LLM重排序模块
使用大语言模型对召回的GO预测结果进行打分和调整
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass
import json


@dataclass
class RerankingInput:
    """重排序输入"""
    protein_id: str
    protein_description: str
    go_id: str
    go_name: str
    go_definition: str
    baseline_score: float
    kg_evidence: str
    reasoning_paths: List[str]


class LLMReranker:
    """基于LLM的重排序器"""

    def __init__(self,
                 model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                 device: str = 'cuda',
                 max_length: int = 512):
        """
        Args:
            model_name: 预训练模型名称
            device: 设备
            max_length: 最大序列长度
        """
        self.device = device
        self.max_length = max_length

        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        # 添加分类头
        hidden_size = self.model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ).to(device)

        print(f"LLM Reranker初始化完成: {model_name}")

    def construct_prompt(self, rerank_input: RerankingInput) -> str:
        """
        构造重排序的输入prompt

        Args:
            rerank_input: 重排序输入数据

        Returns:
            prompt文本
        """
        prompt_parts = []

        # 蛋白质描述
        prompt_parts.append(f"Protein: {rerank_input.protein_description[:200]}")

        # GO术语信息
        prompt_parts.append(f"GO Term: {rerank_input.go_name} ({rerank_input.go_id})")
        prompt_parts.append(f"Definition: {rerank_input.go_definition[:200]}")

        # 知识图谱证据
        if rerank_input.kg_evidence:
            prompt_parts.append(f"Evidence: {rerank_input.kg_evidence[:200]}")

        # 推理路径（只取前两条）
        if rerank_input.reasoning_paths:
            paths_text = " | ".join(rerank_input.reasoning_paths[:2])
            prompt_parts.append(f"Reasoning Paths: {paths_text[:200]}")

        # Baseline分数
        prompt_parts.append(f"Initial Score: {rerank_input.baseline_score:.4f}")

        return " [SEP] ".join(prompt_parts)

    def encode_text(self, text: str) -> torch.Tensor:
        """
        编码文本为embedding

        Args:
            text: 输入文本

        Returns:
            embedding tensor
        """
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用[CLS] token的embedding
            embedding = outputs.last_hidden_state[:, 0, :]

        return embedding

    def score_single(self, rerank_input: RerankingInput) -> float:
        """
        对单个蛋白质-GO对打分

        Args:
            rerank_input: 重排序输入

        Returns:
            调整后的分数
        """
        # 构造prompt
        prompt = self.construct_prompt(rerank_input)

        # 编码
        embedding = self.encode_text(prompt)

        # 分类得分
        with torch.no_grad():
            score = self.classifier(embedding).item()

        return score

    def rerank_predictions(self,
                           protein_id: str,
                           protein_description: str,
                           predictions: List[Tuple[str, float]],
                           go_attrs: Dict[str, Dict],
                           evidence_dict: Dict[Tuple[str, str], Dict],
                           top_k: int = 100) -> List[Tuple[str, float]]:
        """
        重排序预测结果

        Args:
            protein_id: 蛋白质ID
            protein_description: 蛋白质描述
            predictions: [(go_id, score), ...]
            go_attrs: GO术语属性字典
            evidence_dict: 证据字典
            top_k: 返回top-k

        Returns:
            重排序后的预测
        """
        reranked = []

        for go_id, baseline_score in predictions[:top_k * 2]:  # 多处理一些候选
            # 获取GO信息
            go_info = go_attrs.get(go_id, {})
            go_name = go_info.get('name', go_id)
            go_definition = go_info.get('definition', '')

            # 获取证据
            evidence = evidence_dict.get((protein_id, go_id), {})
            kg_evidence = ' '.join(evidence.get('descriptions', [])[:2])
            reasoning_paths = evidence.get('descriptions', [])

            # 构造输入
            rerank_input = RerankingInput(
                protein_id=protein_id,
                protein_description=protein_description,
                go_id=go_id,
                go_name=go_name,
                go_definition=go_definition,
                baseline_score=baseline_score,
                kg_evidence=kg_evidence,
                reasoning_paths=reasoning_paths
            )

            # 打分
            llm_score = self.score_single(rerank_input)

            # 融合baseline和LLM分数
            final_score = 0.6 * baseline_score + 0.4 * llm_score

            reranked.append((go_id, final_score))

        # 排序
        reranked.sort(key=lambda x: x[1], reverse=True)

        return reranked[:top_k]

    def batch_rerank(self,
                     proteins_info: Dict[str, str],
                     predictions_dict: Dict[str, List[Tuple[str, float]]],
                     go_attrs: Dict[str, Dict],
                     evidence_dict: Dict[Tuple[str, str], Dict],
                     top_k: int = 100) -> Dict[str, List[Tuple[str, float]]]:
        """
        批量重排序

        Args:
            proteins_info: {protein_id: description}
            predictions_dict: {protein_id: [(go_id, score), ...]}
            go_attrs: GO术语属性
            evidence_dict: 证据字典
            top_k: 返回top-k

        Returns:
            {protein_id: [(go_id, final_score), ...]}
        """
        reranked_dict = {}

        for protein_id, predictions in predictions_dict.items():
            protein_desc = proteins_info.get(protein_id, '')

            reranked = self.rerank_predictions(
                protein_id,
                protein_desc,
                predictions,
                go_attrs,
                evidence_dict,
                top_k
            )

            reranked_dict[protein_id] = reranked

        return reranked_dict


class LLMRerankingTrainer:
    """LLM重排序训练器"""

    def __init__(self,
                 reranker: LLMReranker,
                 learning_rate: float = 1e-5,
                 weight_decay: float = 0.01):
        """
        Args:
            reranker: LLMReranker实例
            learning_rate: 学习率
            weight_decay: 权重衰减
        """
        self.reranker = reranker

        # 只优化分类头
        self.optimizer = torch.optim.AdamW(
            reranker.classifier.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.criterion = nn.BCELoss()

    def prepare_training_data(self,
                              proteins_df,
                              predictions_dict: Dict[str, List[Tuple[str, float]]],
                              go_attrs: Dict[str, Dict],
                              evidence_dict: Dict[Tuple[str, str], Dict],
                              proteins_info: Dict[str, str]) -> List[Tuple[RerankingInput, float]]:
        """
        准备训练数据

        Args:
            proteins_df: 蛋白质DataFrame（包含真实标签）
            predictions_dict: 预测结果
            go_attrs: GO术语属性
            evidence_dict: 证据字典
            proteins_info: 蛋白质描述

        Returns:
            [(RerankingInput, label), ...]
        """
        training_data = []

        for _, row in proteins_df.iterrows():
            protein_id = row['proteins']
            true_labels = set(row['exp_annotations']) if isinstance(row['exp_annotations'], list) else set()

            if protein_id not in predictions_dict:
                continue

            predictions = predictions_dict[protein_id]
            protein_desc = proteins_info.get(protein_id, '')

            # 为每个预测的GO构造样本
            for go_id, baseline_score in predictions[:50]:  # 限制样本数
                # 标签：是否为真实标注
                label = 1.0 if go_id in true_labels else 0.0

                # 获取GO信息
                go_info = go_attrs.get(go_id, {})
                go_name = go_info.get('name', go_id)
                go_definition = go_info.get('definition', '')

                # 获取证据
                evidence = evidence_dict.get((protein_id, go_id), {})
                kg_evidence = ' '.join(evidence.get('descriptions', [])[:2])
                reasoning_paths = evidence.get('descriptions', [])

                # 构造输入
                rerank_input = RerankingInput(
                    protein_id=protein_id,
                    protein_description=protein_desc,
                    go_id=go_id,
                    go_name=go_name,
                    go_definition=go_definition,
                    baseline_score=baseline_score,
                    kg_evidence=kg_evidence,
                    reasoning_paths=reasoning_paths
                )

                training_data.append((rerank_input, label))

        return training_data

    def train_epoch(self, training_data: List[Tuple[RerankingInput, float]], batch_size: int = 16) -> float:
        """
        训练一个epoch

        Args:
            training_data: 训练数据
            batch_size: 批大小

        Returns:
            平均损失
        """
        self.reranker.classifier.train()

        # 随机打乱
        np.random.shuffle(training_data)

        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i + batch_size]

            # 构造batch
            prompts = []
            labels = []

            for rerank_input, label in batch:
                prompt = self.reranker.construct_prompt(rerank_input)
                prompts.append(prompt)
                labels.append(label)

            # Tokenize
            inputs = self.reranker.tokenizer(
                prompts,
                padding='max_length',
                truncation=True,
                max_length=self.reranker.max_length,
                return_tensors='pt'
            )

            inputs = {k: v.to(self.reranker.device) for k, v in inputs.items()}
            labels_tensor = torch.tensor(labels, dtype=torch.float).to(self.reranker.device)

            # 前向传播
            self.optimizer.zero_grad()

            with torch.no_grad():
                outputs = self.reranker.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]

            scores = self.reranker.classifier(embeddings).squeeze()

            # 计算损失
            loss = self.criterion(scores, labels_tensor)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def evaluate(self, eval_data: List[Tuple[RerankingInput, float]], batch_size: int = 16) -> Dict[str, float]:
        """
        评估模型

        Args:
            eval_data: 评估数据
            batch_size: 批大小

        Returns:
            评估指标
        """
        self.reranker.classifier.eval()

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for i in range(0, len(eval_data), batch_size):
                batch = eval_data[i:i + batch_size]

                prompts = []
                labels = []

                for rerank_input, label in batch:
                    prompt = self.reranker.construct_prompt(rerank_input)
                    prompts.append(prompt)
                    labels.append(label)

                # Tokenize
                inputs = self.reranker.tokenizer(
                    prompts,
                    padding='max_length',
                    truncation=True,
                    max_length=self.reranker.max_length,
                    return_tensors='pt'
                )

                inputs = {k: v.to(self.reranker.device) for k, v in inputs.items()}

                # 前向传播
                outputs = self.reranker.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]
                scores = self.reranker.classifier(embeddings).squeeze()

                all_predictions.extend(scores.cpu().numpy().tolist())
                all_labels.extend(labels)

        # 计算指标
        from sklearn.metrics import roc_auc_score, average_precision_score

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        metrics = {
            'auc_roc': roc_auc_score(all_labels, all_predictions),
            'auc_pr': average_precision_score(all_labels, all_predictions)
        }

        return metrics


class PromptBasedReranker:
    """
    基于提示词的重排序器（使用API调用Claude/GPT）
    适用于小规模数据或需要更强推理能力的场景
    """

    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-20250514"):
        """
        Args:
            api_key: API密钥
            model: 模型名称
        """
        self.api_key = api_key
        self.model = model

    def construct_reranking_prompt(self,
                                   protein_desc: str,
                                   go_name: str,
                                   go_definition: str,
                                   evidence_paths: List[str],
                                   baseline_score: float) -> str:
        """构造重排序提示词"""

        prompt = f"""You are a protein function annotation expert. Given the following information, evaluate how likely this GO term annotation is correct for this protein.

Protein Description:
{protein_desc[:500]}

GO Term: {go_name}
GO Definition: {go_definition}

Evidence from Knowledge Graph:
"""

        if evidence_paths:
            for i, path in enumerate(evidence_paths[:3], 1):
                prompt += f"{i}. {path}\n"
        else:
            prompt += "No direct evidence paths found.\n"

        prompt += f"""
Baseline Prediction Score: {baseline_score:.4f}

Based on the protein description, GO term definition, and knowledge graph evidence, provide:
1. A confidence score (0-1) for this annotation
2. A brief reasoning (1-2 sentences)

Output in JSON format:
{{"score": <float>, "reasoning": "<string>"}}
"""

        return prompt

    def rerank_with_llm(self,
                        protein_desc: str,
                        predictions: List[Tuple[str, float]],
                        go_attrs: Dict[str, Dict],
                        evidence_dict: Dict[Tuple[str, str], Dict],
                        protein_id: str,
                        top_k: int = 20) -> List[Tuple[str, float, str]]:
        """
        使用LLM API重排序（仅处理top-k个候选）

        Args:
            protein_desc: 蛋白质描述
            predictions: 预测列表
            go_attrs: GO术语属性
            evidence_dict: 证据字典
            protein_id: 蛋白质ID
            top_k: 处理的候选数量

        Returns:
            [(go_id, final_score, reasoning), ...]
        """
        # 注意: 这里需要实际的API调用实现
        # 这是一个占位符实现

        reranked = []

        for go_id, baseline_score in predictions[:top_k]:
            go_info = go_attrs.get(go_id, {})
            go_name = go_info.get('name', go_id)
            go_definition = go_info.get('definition', '')

            evidence = evidence_dict.get((protein_id, go_id), {})
            evidence_paths = evidence.get('descriptions', [])

            # 构造prompt
            prompt = self.construct_reranking_prompt(
                protein_desc,
                go_name,
                go_definition,
                evidence_paths,
                baseline_score
            )

            # TODO: 实际API调用
            # response = call_llm_api(prompt)
            # llm_score = parse_response(response)

            # 占位符：直接使用baseline分数
            llm_score = baseline_score
            reasoning = "Placeholder reasoning"

            # 融合分数
            final_score = 0.5 * baseline_score + 0.5 * llm_score

            reranked.append((go_id, final_score, reasoning))

        # 保留未处理的低分候选
        for go_id, baseline_score in predictions[top_k:]:
            reranked.append((go_id, baseline_score, "Not evaluated by LLM"))

        # 排序
        reranked.sort(key=lambda x: x[1], reverse=True)

        return reranked


if __name__ == "__main__":
    print("LLM重排序模块")