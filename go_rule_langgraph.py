"""
基于 LangGraph 的蛋白质功能预测智能体
核心特性:
1. 状态机管理证据收集流程
2. 条件路由避免盲目工具调用
3. 并行节点提升效率
4. 自适应决策终止
"""

import numpy as np
import pandas as pd
from typing import List, Dict, TypedDict, Annotated, Literal
from dataclasses import dataclass, field
import operator
import json
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# ================================
# 状态定义
# ================================

class ProteinRefinementState(TypedDict):
    """全局状态 - 在所有节点间共享"""
    # 蛋白质基础信息
    protein_id: str
    gene_id: str
    sequence: str
    interpros: List[str]
    organism: str
    uniprot_text: str

    # GO Term 信息
    go_term: str
    initial_score: float
    diamond_score: float
    current_score: float

    # 证据累积 (使用 Annotated 实现累加)
    evidence: Annotated[List[Dict], operator.add]

    # 决策状态
    confidence_level: str  # "high" | "medium" | "low" | "conflicting"
    tools_called: Annotated[List[str], operator.add]
    evidence_count: int

    # 更新历史
    update_history: Annotated[List[Dict], operator.add]

    # 控制流
    next_action: str
    reasoning: str
    final_decision: Dict


# ================================
# 证据数据类
# ================================

@dataclass
class Evidence:
    source: str
    content: str
    confidence: float  # 0-1
    support_direction: str  # "positive" | "negative" | "neutral"
    weight: float = 1.0

    def to_dict(self):
        return {
            "source": self.source,
            "content": self.content,
            "confidence": self.confidence,
            "support_direction": self.support_direction,
            "weight": self.weight
        }


# ================================
# 数据管理器 (复用你的代码)
# ================================

class DataManager:
    """管理所有共享数据"""

    def __init__(self, data_dir="data"):
        self.gene_info = self._load_json(f"{data_dir}/gene_info.json")
        self.interpro_info = self._load_json(f"{data_dir}/interpro_descriptions.json")
        self.interpro_to_go = pd.read_csv(
            f"{data_dir}/interpro2go_mapping.tsv",
            sep="\t"
        )

    @staticmethod
    def _load_json(path):
        try:
            with open(path) as f:
                return json.load(f)
        except:
            return {}


# ================================
# 工具函数 (转换为纯函数)
# ================================

class EvidenceTools:
    """所有证据收集工具"""

    def __init__(self, data_manager: DataManager, ontology):
        self.data_manager = data_manager
        self.ontology = ontology

    def get_interpro_evidence(self, state: ProteinRefinementState) -> Evidence:
        """检查 InterPro 域与 GO Term 的关联"""
        interpros = state["interpros"]
        go_term = state["go_term"]

        if not interpros:
            return Evidence(
                source="interpro",
                content="No InterPro domains found",
                confidence=0.0,
                support_direction="neutral"
            )

        # 查找关联
        mask = self.data_manager.interpro_to_go["interpro_id"].isin(interpros)
        associated_gos = self.data_manager.interpro_to_go.loc[
            mask, "go_id"
        ].unique().tolist()

        if go_term in associated_gos:
            # 找到直接证据
            matching_interpros = self.data_manager.interpro_to_go[
                (self.data_manager.interpro_to_go["interpro_id"].isin(interpros)) &
                (self.data_manager.interpro_to_go["go_id"] == go_term)
                ]["interpro_id"].tolist()

            content = f"Direct match: InterPro domains {matching_interpros} are annotated with {go_term}"
            return Evidence(
                source="interpro",
                content=content,
                confidence=0.9,
                support_direction="positive",
                weight=1.5  # InterPro 是强证据
            )
        else:
            # 检查父子关系
            parent_child_match = self._check_go_hierarchy(go_term, associated_gos)
            if parent_child_match:
                return Evidence(
                    source="interpro",
                    content=f"Hierarchical match: {parent_child_match}",
                    confidence=0.6,
                    support_direction="positive",
                    weight=1.0
                )

            return Evidence(
                source="interpro",
                content=f"No association found between InterPro domains and {go_term}",
                confidence=0.7,
                support_direction="negative",
                weight=0.8
            )

    def get_diamond_evidence(self, state: ProteinRefinementState) -> Evidence:
        """检查序列相似性证据"""
        diamond_score = state["diamond_score"]
        go_term = state["go_term"]

        if diamond_score is None:
            return Evidence(
                source="diamond",
                content="No DIAMOND similarity data available",
                confidence=0.0,
                support_direction="neutral"
            )

        if diamond_score > 0.7:
            return Evidence(
                source="diamond",
                content=f"High sequence similarity (score={diamond_score:.3f}) to proteins with {go_term}",
                confidence=0.85,
                support_direction="positive",
                weight=1.2
            )
        elif diamond_score > 0.4:
            return Evidence(
                source="diamond",
                content=f"Moderate sequence similarity (score={diamond_score:.3f})",
                confidence=0.6,
                support_direction="positive",
                weight=0.8
            )
        else:
            return Evidence(
                source="diamond",
                content=f"Low sequence similarity (score={diamond_score:.3f})",
                confidence=0.7,
                support_direction="negative",
                weight=0.6
            )

    def get_gene_evidence(self, state: ProteinRefinementState) -> Evidence:
        """检查基因功能描述"""
        gene_id = state["gene_id"]
        go_term = state["go_term"]

        gene_info = self.data_manager.gene_info.get(gene_id)

        if gene_info is None:
            return Evidence(
                source="gene_info",
                content="No gene information available",
                confidence=0.0,
                support_direction="neutral"
            )

        summary = gene_info.get("summary", "")
        go_name = self.ontology.get_term_name(go_term)

        # 简单的文本匹配 (可替换为语义相似度)
        keywords = self._extract_keywords(go_name)
        matches = sum(1 for kw in keywords if kw.lower() in summary.lower())

        if matches > 0:
            return Evidence(
                source="gene_info",
                content=f"Gene summary mentions {matches} keywords related to {go_name}",
                confidence=0.7,
                support_direction="positive",
                weight=1.0
            )
        else:
            return Evidence(
                source="gene_info",
                content=f"Gene summary does not mention {go_name}",
                confidence=0.5,
                support_direction="neutral",
                weight=0.5
            )

    def get_taxon_evidence(self, state: ProteinRefinementState) -> Evidence:
        """检查分类学约束"""
        organism = state["organism"]
        go_term = state["go_term"]

        if organism not in self.ontology.taxon_map:
            return Evidence(
                source="taxon",
                content="No taxonomic constraints available",
                confidence=0.0,
                support_direction="neutral"
            )

        in_taxon, never_in_taxon = self.ontology.taxon_map[organism]

        if go_term in never_in_taxon:
            return Evidence(
                source="taxon",
                content=f"{go_term} is explicitly NEVER found in {organism}",
                confidence=0.95,
                support_direction="negative",
                weight=2.0  # 分类学约束是强否定证据
            )
        elif go_term in in_taxon:
            return Evidence(
                source="taxon",
                content=f"{go_term} is commonly found in {organism}",
                confidence=0.8,
                support_direction="positive",
                weight=1.2
            )
        else:
            return Evidence(
                source="taxon",
                content=f"No specific taxonomic constraint for {go_term} in {organism}",
                confidence=0.5,
                support_direction="neutral"
            )

    def _check_go_hierarchy(self, go_term, candidate_gos):
        """检查 GO 层级关系"""
        # 简化实现 - 你可以用 ontology.get_ancestors() 等
        return None

    def _extract_keywords(self, go_name):
        """从 GO 名称提取关键词"""
        stopwords = {"activity", "process", "component", "the", "of", "and"}
        words = go_name.lower().split()
        return [w for w in words if w not in stopwords and len(w) > 3]


# ================================
# 节点函数
# ================================

class GraphNodes:
    """所有图节点"""

    def __init__(self, evidence_tools: EvidenceTools):
        self.tools = evidence_tools

    # -------------------- 入口节点 --------------------

    def triage_node(self, state: ProteinRefinementState) -> ProteinRefinementState:
        """
        快速预判节点 - 决定是否需要深度检索
        """
        initial_score = state["initial_score"]
        diamond_score = state["diamond_score"] if state["diamond_score"] else 0.0

        # 规则1: 双高分 - 直接通过
        if initial_score > 0.85 and diamond_score > 0.8:
            state["confidence_level"] = "high"
            state["reasoning"] = "Both prediction and DIAMOND scores are very high"
            state["next_action"] = "finalize"
            return state

        # 规则2: 双低分 - 直接拒绝
        if initial_score < 0.15 and diamond_score < 0.2:
            state["confidence_level"] = "high"
            state["reasoning"] = "Both scores are very low, unlikely to be true"
            state["next_action"] = "finalize"
            state["current_score"] = initial_score * 0.5  # 进一步降低
            return state

        # 规则3: 分数严重冲突 - 需要深度调查
        if abs(initial_score - diamond_score) > 0.5:
            state["confidence_level"] = "conflicting"
            state["reasoning"] = f"Conflicting scores: prediction={initial_score:.3f}, DIAMOND={diamond_score:.3f}"
            state["next_action"] = "deep_investigation"
            return state

        # 规则4: 常规情况 - 需要证据
        state["confidence_level"] = "medium"
        state["reasoning"] = "Scores are uncertain, need evidence gathering"
        state["next_action"] = "gather_evidence"

        return state

    # -------------------- 证据收集节点 --------------------

    def interpro_node(self, state: ProteinRefinementState) -> ProteinRefinementState:
        """收集 InterPro 证据"""
        evidence = self.tools.get_interpro_evidence(state)

        state["evidence"].append(evidence.to_dict())
        state["tools_called"].append("interpro")
        state["evidence_count"] = len(state["evidence"])

        print(f"  [InterPro] {evidence.content} (confidence={evidence.confidence:.2f})")

        return state

    def diamond_node(self, state: ProteinRefinementState) -> ProteinRefinementState:
        """收集 DIAMOND 证据"""
        evidence = self.tools.get_diamond_evidence(state)

        state["evidence"].append(evidence.to_dict())
        state["tools_called"].append("diamond")
        state["evidence_count"] = len(state["evidence"])

        print(f"  [DIAMOND] {evidence.content} (confidence={evidence.confidence:.2f})")

        return state

    def gene_node(self, state: ProteinRefinementState) -> ProteinRefinementState:
        """收集基因信息证据"""
        evidence = self.tools.get_gene_evidence(state)

        state["evidence"].append(evidence.to_dict())
        state["tools_called"].append("gene_info")
        state["evidence_count"] = len(state["evidence"])

        print(f"  [Gene] {evidence.content} (confidence={evidence.confidence:.2f})")

        return state

    def taxon_node(self, state: ProteinRefinementState) -> ProteinRefinementState:
        """收集分类学证据"""
        evidence = self.tools.get_taxon_evidence(state)

        state["evidence"].append(evidence.to_dict())
        state["tools_called"].append("taxon")
        state["evidence_count"] = len(state["evidence"])

        print(f"  [Taxon] {evidence.content} (confidence={evidence.confidence:.2f})")

        return state

    # -------------------- 决策节点 --------------------

    def evaluate_evidence_node(self, state: ProteinRefinementState) -> ProteinRefinementState:
        """
        评估已收集的证据 - 决定是否需要更多证据
        """
        evidence_list = state["evidence"]

        if len(evidence_list) == 0:
            state["next_action"] = "gather_evidence"
            return state

        # 计算证据一致性
        positive_count = sum(1 for e in evidence_list if e["support_direction"] == "positive")
        negative_count = sum(1 for e in evidence_list if e["support_direction"] == "negative")
        neutral_count = sum(1 for e in evidence_list if e["support_direction"] == "neutral")

        total_confidence = np.mean([e["confidence"] for e in evidence_list])

        # 判断1: 强一致性 - 可以决策
        if positive_count >= 2 and negative_count == 0:
            state["confidence_level"] = "high"
            state["reasoning"] = f"Strong positive evidence ({positive_count} sources agree)"
            state["next_action"] = "finalize"
            return state

        if negative_count >= 2 and positive_count == 0:
            state["confidence_level"] = "high"
            state["reasoning"] = f"Strong negative evidence ({negative_count} sources agree)"
            state["next_action"] = "finalize"
            return state

        # 判断2: 证据冲突 - 需要更多证据
        if positive_count > 0 and negative_count > 0:
            if len(state["tools_called"]) >= 4:  # 已经调用了所有工具
                state["confidence_level"] = "conflicting"
                state["reasoning"] = "Evidence is conflicting even after deep investigation"
                state["next_action"] = "finalize"
            else:
                state["confidence_level"] = "conflicting"
                state["reasoning"] = "Evidence is conflicting, need more data"
                state["next_action"] = "deep_investigation"
            return state

        # 判断3: 证据不足 - 继续收集
        if len(state["tools_called"]) < 2:
            state["next_action"] = "gather_evidence"
            state["reasoning"] = "Not enough evidence yet"
            return state

        # 判断4: 中等置信度 - 可以决策
        state["confidence_level"] = "medium"
        state["reasoning"] = f"Moderate evidence ({len(evidence_list)} sources, avg confidence={total_confidence:.2f})"
        state["next_action"] = "finalize"

        return state

    # -------------------- 最终决策节点 --------------------

    def finalize_node(self, state: ProteinRefinementState) -> ProteinRefinementState:
        """
        最终决策 - 计算更新后的分数
        """
        evidence_list = state["evidence"]
        initial_score = state["initial_score"]

        if len(evidence_list) == 0:
            # 没有证据,保持原分数
            state["current_score"] = initial_score
            state["final_decision"] = {
                "old_score": initial_score,
                "new_score": initial_score,
                "change": 0.0,
                "reasoning": "No evidence collected, score unchanged"
            }
            return state

        # 计算加权分数调整
        total_adjustment = 0.0

        for evidence in evidence_list:
            direction = evidence["support_direction"]
            confidence = evidence["confidence"]
            weight = evidence["weight"]

            if direction == "positive":
                adjustment = 0.2 * confidence * weight
            elif direction == "negative":
                adjustment = -0.2 * confidence * weight
            else:
                adjustment = 0.0

            total_adjustment += adjustment

        # 归一化调整 (避免过度变化)
        total_adjustment = np.clip(total_adjustment, -0.5, 0.5)

        new_score = np.clip(initial_score + total_adjustment, 0.0, 1.0)

        state["current_score"] = new_score
        state["final_decision"] = {
            "old_score": float(initial_score),
            "new_score": float(new_score),
            "change": float(new_score - initial_score),
            "reasoning": state["reasoning"],
            "evidence_count": len(evidence_list),
            "confidence_level": state["confidence_level"]
        }

        # 记录更新历史
        update_record = {
            "go_term": state["go_term"],
            "protein_id": state["protein_id"],
            "old_score": float(initial_score),
            "new_score": float(new_score),
            "change": float(new_score - initial_score),
            "tools_used": state["tools_called"],
            "timestamp": datetime.now().isoformat()
        }
        state["update_history"].append(update_record)

        print(
            f"\n  [FINAL] {state['go_term']}: {initial_score:.3f} → {new_score:.3f} (Δ={new_score - initial_score:+.3f})")
        print(f"  Reasoning: {state['reasoning']}")

        return state


# ================================
# 路由函数
# ================================

def route_after_triage(state: ProteinRefinementState) -> str:
    """根据 triage 结果路由"""
    action = state["next_action"]

    if action == "finalize":
        return "finalize"
    elif action == "deep_investigation":
        return "deep"
    else:
        return "gather"


def route_after_evaluation(state: ProteinRefinementState) -> str:
    """根据证据评估结果路由"""
    action = state["next_action"]
    tools_called = state["tools_called"]

    if action == "finalize":
        return "finalize"
    elif action == "deep_investigation":
        # 选择下一个未调用的工具
        if "gene_info" not in tools_called:
            return "gene"
        elif "taxon" not in tools_called:
            return "taxon"
        else:
            return "finalize"
    else:  # gather_evidence
        # 选择基础工具
        if "interpro" not in tools_called:
            return "interpro"
        elif "diamond" not in tools_called:
            return "diamond"
        else:
            return "evaluate"


# ================================
# 构建图
# ================================

def build_refinement_graph(evidence_tools: EvidenceTools) -> StateGraph:
    """构建完整的 LangGraph"""

    nodes = GraphNodes(evidence_tools)

    # 创建图
    workflow = StateGraph(ProteinRefinementState)

    # 添加所有节点
    workflow.add_node("triage", nodes.triage_node)
    workflow.add_node("interpro", nodes.interpro_node)
    workflow.add_node("diamond", nodes.diamond_node)
    workflow.add_node("gene", nodes.gene_node)
    workflow.add_node("taxon", nodes.taxon_node)
    workflow.add_node("evaluate", nodes.evaluate_evidence_node)
    workflow.add_node("finalize", nodes.finalize_node)

    # 设置入口
    workflow.set_entry_point("triage")

    # 添加条件边
    workflow.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "finalize": "finalize",
            "deep": "gene",  # 冲突时先查基因
            "gather": "interpro"  # 常规情况先查 InterPro
        }
    )

    # 证据收集后都进入评估
    workflow.add_edge("interpro", "evaluate")
    workflow.add_edge("diamond", "evaluate")
    workflow.add_edge("gene", "evaluate")
    workflow.add_edge("taxon", "evaluate")

    # 评估后的条件路由
    workflow.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
        {
            "finalize": "finalize",
            "interpro": "interpro",
            "diamond": "diamond",
            "gene": "gene",
            "taxon": "taxon",
            "evaluate": "evaluate"  # 自循环 (不应该发生)
        }
    )

    # 最终节点连接到 END
    workflow.add_edge("finalize", END)

    return workflow.compile()


# ================================
# 主执行类
# ================================

class LangGraphProteinRefinement:
    """LangGraph 蛋白质功能预测精炼器"""

    def __init__(self, ontology, data_manager: DataManager):
        self.ontology = ontology
        self.data_manager = data_manager

        # 初始化工具
        self.evidence_tools = EvidenceTools(data_manager, ontology)

        # 构建图
        self.graph = build_refinement_graph(self.evidence_tools)

    def refine_single_go_term(
            self,
            protein_data: Dict,
            go_term: str,
            initial_score: float,
            diamond_score: float
    ) -> Dict:
        """
        为单个 GO term 执行精炼流程

        Args:
            protein_data: 蛋白质数据字典
            go_term: GO term ID
            initial_score: 初始预测分数
            diamond_score: DIAMOND 相似度分数

        Returns:
            包含最终决策的字典
        """

        # 初始化状态
        initial_state = ProteinRefinementState(
            protein_id=protein_data["proteins"],
            gene_id=protein_data["genes"],
            sequence=protein_data["sequences"],
            interpros=protein_data["interpros"],
            organism=protein_data["orgs"],
            uniprot_text=protein_data.get("uniprot_text", ""),
            go_term=go_term,
            initial_score=initial_score,
            diamond_score=diamond_score,
            current_score=initial_score,
            evidence=[],
            confidence_level="unknown",
            tools_called=[],
            evidence_count=0,
            update_history=[],
            next_action="",
            reasoning="",
            final_decision={}
        )

        # 执行图
        print(f"\n{'=' * 60}")
        print(f"Refining: {protein_data['proteins']} - {go_term}")
        print(f"Initial Score: {initial_score:.3f}, DIAMOND Score: {diamond_score if diamond_score else 'N/A'}")
        print(f"{'=' * 60}")

        final_state = self.graph.invoke(initial_state)

        return final_state["final_decision"]

    def refine_all_predictions(
            self,
            protein_data: Dict,
            terms: List[str],
            terms_dict: Dict[str, int],
            score_threshold: float = 0.1
    ) -> np.ndarray:
        """
        批量精炼一个蛋白质的所有 GO term 预测

        Args:
            protein_data: 蛋白质数据
            terms: GO term 列表
            terms_dict: GO term 到索引的映射
            score_threshold: 只精炼分数 >= threshold 的 term

        Returns:
            更新后的预测分数数组
        """

        predictions = protein_data["preds"].copy()
        diam_preds = protein_data.get("diam_preds", {})

        # 筛选需要精炼的 GO terms
        high_score_indices = np.where(predictions >= 0.05 and predictions < 0.6)[0]
        go_terms_to_refine = [terms[i] for i in high_score_indices]

        print(f"\n{'#' * 60}")
        print(f"Protein: {protein_data['proteins']}")
        print(f"Total GO terms to refine: {len(go_terms_to_refine)}")
        print(f"{'#' * 60}")

        update_count = 0

        for go_term in go_terms_to_refine:
            idx = terms_dict[go_term]
            initial_score = predictions[idx]
            diamond_score = diam_preds.get(go_term)

            # 执行精炼
            decision = self.refine_single_go_term(
                protein_data,
                go_term,
                initial_score,
                diamond_score
            )

            # 更新分数
            new_score = decision["new_score"]
            if abs(new_score - initial_score) > 0.01:
                predictions[idx] = new_score
                update_count += 1
                update_record = {
                    'proteins': protein_data["proteins"],
                    'go_term': go_term,
                    'old_score': initial_score,
                    'new_score': new_score,
                    'timestamp': datetime.now().isoformat()
                }
                # 实时写文件（append）
                with open("./data/mf/update_history_langgraph.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(update_record, ensure_ascii=False) + "\n")

        print(f"\n{'#' * 60}")
        print(f"Refinement complete: {update_count}/{len(go_terms_to_refine)} terms updated")
        print(f"{'#' * 60}\n")
        return predictions


# ================================
# 简化版 Ontology (用于演示)
# ================================

class SimpleOntology:
    """简化版本体类"""

    def __init__(self):
        self.taxon_map = {}  # 在实际使用中加载真实数据

    def get_term_name(self, go_id):
        # 简化实现
        return f"Function of {go_id}"

    def get_term_info(self, go_id):
        return f"Definition of {go_id}"


# ================================
# 使用示例
# ================================

def main():
    """演示如何使用 LangGraph 框架"""

    # 初始化
    ontology = SimpleOntology()
    data_manager = DataManager(data_dir="data")

    refiner = LangGraphProteinRefinement(ontology, data_manager)

    # 模拟蛋白质数据
    protein_data = {
        "proteins": "P12345",
        "genes": "BRCA1",
        "sequences": "MKTAYIAKQRQISFVKSHFSRQLE...",
        "interpros": ["IPR001664", "IPR002048"],
        "orgs": "9606",  # Human
        "uniprot_text": "Tumor suppressor involved in DNA repair...",
        "preds": np.random.rand(2041),  # 模拟预测分数
        "diam_preds": {
            "GO:0003824": 0.85,
            "GO:0005515": 0.62
        }
    }

    # 模拟 GO terms
    terms = [f"GO:{1000000 + i:07d}" for i in range(2041)]
    terms[0] = "GO:0003824"  # catalytic activity
    terms[1] = "GO:0005515"  # protein binding

    terms_dict = {v: k for k, v in enumerate(terms)}

    # 执行精炼
    updated_predictions = refiner.refine_all_predictions(
        protein_data,
        terms,
        terms_dict,
        score_threshold=0.5
    )

    print("\n" + "=" * 60)
    print("Refinement Summary:")
    print(f"Original predictions shape: {protein_data['preds'].shape}")
    print(f"Updated predictions shape: {updated_predictions.shape}")
    print(f"Number of changes: {np.sum(updated_predictions != protein_data['preds'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()