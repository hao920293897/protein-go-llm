"""
知识图谱检索和路径推理模块
实现G-Retriever和ToG风格的检索策略
"""
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict, deque
import heapq


class KGRetriever:
    """知识图谱检索器"""

    def __init__(self, kg):
        """
        Args:
            kg: ProteinKnowledgeGraph实例
        """
        self.kg = kg

    def find_paths(self,
                   start_node: str,
                   end_node: str,
                   max_length: int = 4,
                   max_paths: int = 10) -> List[List[str]]:
        """
        查找两个节点之间的路径

        Args:
            start_node: 起始节点
            end_node: 终止节点
            max_length: 最大路径长度
            max_paths: 最多返回路径数

        Returns:
            路径列表 [[node1, node2, ...], ...]
        """
        if not self.kg.graph.has_node(start_node) or not self.kg.graph.has_node(end_node):
            return []

        try:
            # 使用NetworkX的简单路径查找
            paths = []
            for path in nx.all_simple_paths(self.kg.graph, start_node, end_node, cutoff=max_length):
                paths.append(path)
                if len(paths) >= max_paths:
                    break
            return paths
        except nx.NetworkXNoPath:
            return []

    def find_metapaths(self,
                       start_node: str,
                       end_node: str,
                       metapath_schema: List[str],
                       max_paths: int = 10) -> List[List[str]]:
        """
        按照元路径模式查找路径

        Args:
            start_node: 起始节点
            end_node: 终止节点
            metapath_schema: 元路径模式，例如 ['protein', 'interpro', 'go_term']
            max_paths: 最多返回路径数

        Returns:
            符合模式的路径列表
        """
        if len(metapath_schema) < 2:
            return []

        # BFS搜索符合元路径的路径
        paths = []
        queue = deque([([start_node], 0)])  # (current_path, schema_index)

        while queue and len(paths) < max_paths:
            current_path, schema_idx = queue.popleft()
            current_node = current_path[-1]

            # 检查当前节点类型是否匹配
            current_type = self.kg.graph.nodes[current_node].get('node_type')
            if current_type != metapath_schema[schema_idx]:
                continue

            # 到达终点
            if schema_idx == len(metapath_schema) - 1:
                if current_node == end_node:
                    paths.append(current_path)
                continue

            # 扩展到下一层
            next_type = metapath_schema[schema_idx + 1]
            for neighbor in self.kg.graph.successors(current_node):
                neighbor_type = self.kg.graph.nodes[neighbor].get('node_type')
                if neighbor_type == next_type:
                    if neighbor not in current_path:  # 避免环路
                        new_path = current_path + [neighbor]
                        queue.append((new_path, schema_idx + 1))

        return paths

    def find_k_hop_subgraph(self,
                            center_node: str,
                            k: int = 2) -> nx.DiGraph:
        """
        提取k跳子图

        Args:
            center_node: 中心节点
            k: 跳数

        Returns:
            子图
        """
        if not self.kg.graph.has_node(center_node):
            return nx.DiGraph()

        # BFS提取k跳邻居
        visited = {center_node}
        current_layer = {center_node}

        for _ in range(k):
            next_layer = set()
            for node in current_layer:
                for neighbor in self.kg.graph.successors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_layer.add(neighbor)
                for neighbor in self.kg.graph.predecessors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_layer.add(neighbor)
            current_layer = next_layer

        # 提取子图
        subgraph = self.kg.graph.subgraph(visited).copy()

        return subgraph

    def retrieve_evidence_for_protein_go(self,
                                         protein_id: str,
                                         go_id: str,
                                         max_paths: int = 5) -> Dict:
        """
        检索蛋白质到GO术语的证据路径

        Args:
            protein_id: 蛋白质ID
            go_id: GO术语ID
            max_paths: 最多返回路径数

        Returns:
            证据字典，包含路径和描述
        """
        evidence = {
            'protein_id': protein_id,
            'go_id': go_id,
            'paths': [],
            'descriptions': [],
            'confidence': 0.0
        }

        # 定义常见的元路径模式
        metapath_schemas = [
            ['protein', 'interpro', 'go_term'],  # 通过InterPro域
            ['protein', 'gene', 'gene', 'protein', 'go_term'],  # 通过PPI网络
            ['protein', 'go_term'],  # 直接关联（训练集）
        ]

        all_paths = []

        # 尝试各种元路径
        for schema in metapath_schemas:
            paths = self.find_metapaths(protein_id, go_id, schema, max_paths)
            all_paths.extend(paths)

            if len(all_paths) >= max_paths:
                break

        # 如果没有找到元路径，尝试一般路径
        if not all_paths:
            all_paths = self.find_paths(protein_id, go_id, max_length=5, max_paths=max_paths)

        # 提取路径描述
        for path in all_paths[:max_paths]:
            path_desc = self._describe_path(path)
            evidence['paths'].append(path)
            evidence['descriptions'].append(path_desc)

        # 计算置信度（基于路径数量和长度）
        if all_paths:
            avg_length = np.mean([len(p) for p in all_paths[:max_paths]])
            evidence['confidence'] = min(1.0, len(all_paths) / max_paths * (1.0 / avg_length))

        return evidence

    def _describe_path(self, path: List[str]) -> str:
        """
        描述一条路径

        Args:
            path: 节点ID列表

        Returns:
            路径的文本描述
        """
        descriptions = []

        for i, node_id in enumerate(path):
            node_type = self.kg.graph.nodes[node_id].get('node_type')
            attrs = self.kg.get_node_features(node_id)

            # 根据节点类型提取关键信息
            if node_type == 'protein':
                desc = f"Protein {node_id}"
            elif node_type == 'gene':
                symbol = attrs.get('symbol', node_id)
                desc = f"Gene {symbol}"
            elif node_type == 'go_term':
                name = attrs.get('name', node_id)
                desc = f"GO: {name}"
            elif node_type == 'interpro':
                # 提取InterPro描述的前50个字符
                full_desc = attrs.get('description', '')
                if '#@@#' in full_desc:
                    name = full_desc.split('#@@#')[0]
                    desc = f"Domain: {name}"
                else:
                    desc = f"InterPro {node_id}"
            else:
                desc = node_id

            descriptions.append(desc)

            # 添加边的描述
            if i < len(path) - 1:
                next_node = path[i + 1]
                edge_data = self.kg.graph.get_edge_data(node_id, next_node)
                if edge_data:
                    edge_type = list(edge_data.values())[0].get('edge_type', '')
                    descriptions.append(f"--[{edge_type}]-->")

        return ' '.join(descriptions)

    def batch_retrieve_evidence(self,
                                protein_go_pairs: List[Tuple[str, str]],
                                max_paths: int = 5) -> Dict[Tuple[str, str], Dict]:
        """
        批量检索证据

        Args:
            protein_go_pairs: [(protein_id, go_id), ...]
            max_paths: 每对最多返回路径数

        Returns:
            {(protein_id, go_id): evidence_dict}
        """
        evidence_dict = {}

        for protein_id, go_id in protein_go_pairs:
            evidence = self.retrieve_evidence_for_protein_go(
                protein_id, go_id, max_paths
            )
            evidence_dict[(protein_id, go_id)] = evidence

        return evidence_dict


class ToGStyleReasoner:
    """ToG (Think-on-Graph) 风格的推理器"""

    def __init__(self, kg, retriever: KGRetriever):
        """
        Args:
            kg: ProteinKnowledgeGraph实例
            retriever: KGRetriever实例
        """
        self.kg = kg
        self.retriever = retriever

    def reason_protein_to_go(self,
                             protein_id: str,
                             candidate_go_ids: List[str],
                             top_k: int = 10) -> List[Tuple[str, float, Dict]]:
        """
        对候选GO术语进行推理和打分

        Args:
            protein_id: 蛋白质ID
            candidate_go_ids: 候选GO术语列表
            top_k: 返回top-k个结果

        Returns:
            [(go_id, score, evidence), ...]
        """
        results = []

        for go_id in candidate_go_ids:
            # 检索证据
            evidence = self.retriever.retrieve_evidence_for_protein_go(
                protein_id, go_id, max_paths=5
            )

            # 计算推理分数
            score = self._calculate_reasoning_score(protein_id, go_id, evidence)

            results.append((go_id, score, evidence))

        # 排序并返回top-k
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def _calculate_reasoning_score(self,
                                   protein_id: str,
                                   go_id: str,
                                   evidence: Dict) -> float:
        """
        计算推理分数

        基于:
        1. 路径数量
        2. 路径长度
        3. 路径质量（边类型权重）

        Args:
            protein_id: 蛋白质ID
            go_id: GO术语ID
            evidence: 证据字典

        Returns:
            推理分数
        """
        if not evidence['paths']:
            return 0.0

        # 路径数量分数
        num_paths = len(evidence['paths'])
        path_count_score = min(1.0, num_paths / 5.0)

        # 路径长度分数（较短路径得分更高）
        avg_length = np.mean([len(p) for p in evidence['paths']])
        length_score = 1.0 / avg_length

        # 路径质量分数（基于边类型）
        edge_type_weights = {
            'annotated_with': 1.0,  # 直接标注最可靠
            'has_domain': 0.8,  # InterPro域关联
            'associated_with': 0.7,  # InterPro-GO统计关联
            'gene_interaction': 0.6,  # PPI传播
            'protein_to_gene': 0.5,
            'is_a': 0.9,  # GO层级关系
        }

        quality_scores = []
        for path in evidence['paths']:
            path_quality = 0.0
            for i in range(len(path) - 1):
                node1, node2 = path[i], path[i + 1]
                edge_data = self.kg.graph.get_edge_data(node1, node2)
                if edge_data:
                    edge_type = list(edge_data.values())[0].get('edge_type', '')
                    weight = edge_type_weights.get(edge_type, 0.5)
                    path_quality += weight

            if len(path) > 1:
                path_quality /= (len(path) - 1)
            quality_scores.append(path_quality)

        avg_quality = np.mean(quality_scores) if quality_scores else 0.0

        # 综合分数
        final_score = (0.3 * path_count_score +
                       0.3 * length_score +
                       0.4 * avg_quality)

        return final_score

    def enhance_baseline_predictions(self,
                                     baseline_predictions: Dict[str, List[Tuple[str, float]]],
                                     alpha: float = 0.5,
                                     top_k: int = 100) -> Dict[str, List[Tuple[str, float]]]:
        """
        使用图推理增强baseline预测

        Args:
            baseline_predictions: {protein_id: [(go_id, score), ...]}
            alpha: baseline权重（1-alpha为推理权重）
            top_k: 返回top-k预测

        Returns:
            增强后的预测
        """
        enhanced_predictions = {}

        for protein_id, predictions in baseline_predictions.items():
            candidate_go_ids = [go_id for go_id, _ in predictions[:top_k * 2]]  # 扩展候选集

            # 对候选进行推理
            reasoning_results = self.reason_protein_to_go(
                protein_id, candidate_go_ids, top_k=top_k * 2
            )

            # 融合baseline分数和推理分数
            baseline_scores = {go_id: score for go_id, score in predictions}
            reasoning_scores = {go_id: score for go_id, score, _ in reasoning_results}

            # 归一化
            if baseline_scores:
                max_baseline = max(baseline_scores.values())
                baseline_scores = {k: v / max_baseline for k, v in baseline_scores.items()}

            if reasoning_scores:
                max_reasoning = max(reasoning_scores.values())
                reasoning_scores = {k: v / max_reasoning for k, v in reasoning_scores.items()}

            # 组合分数
            combined_scores = {}
            all_go_ids = set(baseline_scores.keys()) | set(reasoning_scores.keys())

            for go_id in all_go_ids:
                base_score = baseline_scores.get(go_id, 0.0)
                reason_score = reasoning_scores.get(go_id, 0.0)
                combined_scores[go_id] = alpha * base_score + (1 - alpha) * reason_score

            # 排序
            sorted_predictions = sorted(
                combined_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]

            enhanced_predictions[protein_id] = sorted_predictions

        return enhanced_predictions


if __name__ == "__main__":
    print("KG检索和推理模块")