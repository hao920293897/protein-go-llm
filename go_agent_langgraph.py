"""
LLM驱动的 LangGraph 蛋白质功能预测智能体
核心特性:
1. 每个节点都使用 LLM 进行推理
2. LLM 自主决定下一步行动
3. 状态机管理整体流程
4. 工具调用由 LLM 主动触发
"""

import numpy as np
import pandas as pd
from typing import List, Dict, TypedDict, Annotated, Literal
import operator
import json
from datetime import datetime
from data_process.ontology import Ontology
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# LLM 相关导入
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool

# # 如果使用你的 CAMEL 框架
# try:
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from agents.models import qwen_vllm_model

USE_CAMEL = True
# except:
#     USE_CAMEL = False
#     print("CAMEL not available, using LangChain models")


# ================================
# 状态定义
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

class ProteinRefinementState(TypedDict):
    """全局状态"""
    # 蛋白质信息
    protein_id: str
    gene_id: str
    sequence: str
    interpros: List[str]
    organism: str
    uniprot_text: str

    # GO Term
    go_term: str
    initial_score: float
    diamond_score: float
    current_score: float

    # 证据和历史
    evidence: Annotated[List[Dict], operator.add]
    tools_called: Annotated[List[str], operator.add]

    # LLM 对话历史
    messages: Annotated[List, operator.add]

    # 决策
    next_action: str  # "call_tool" | "evaluate" | "finalize" | "stop"
    tool_to_call: str  # "interpro" | "diamond" | "gene" | "taxon"
    confidence_level: str
    reasoning: str
    final_decision: Dict


# ================================
# LLM 包装器
# ================================

class LLMWrapper:
    """统一的 LLM 接口"""

    def __init__(self, model_type="camel", model_name="gpt-4"):
        self.model_type = model_type
        self.model = qwen_vllm_model
        self.use_camel = True

    def invoke(self, messages):
        """调用 LLM"""
        if self.use_camel:
            # CAMEL 框架调用方式
            role_map = {
                "human": "user",
                "ai": "assistant",
                "system": "system"
            }
            openai_messages = []
            for m in messages:
                role = role_map.get(m.type, "user")
                openai_messages.append({
                    "role": role,
                    "content": str(m.content)
                })
            response = self.model.run(openai_messages)

            return AIMessage(
                content=response.msg.content
            )
        else:
            # LangChain 调用方式
            return self.model.invoke(messages)


# ================================
# 工具定义 (供 LLM 调用)
# ================================

class EvidenceToolsLLM:
    """LLM 可调用的工具"""

    def __init__(self, data_manager, ontology):
        self.data_manager = data_manager
        self.ontology = ontology
        self.current_state = None  # 将在运行时注入

    @property
    def get_interpro_evidence_tool(self):
        @tool
        def wrapped():
            """
            Obtain evidence of association between InterPro domains and GO terms.
            This tool examines whether protein domains are related to a target function.
            :return: GO term IDs associated with InterPro domains
            """
            return self._get_interpro_evidence_impl()

        return wrapped
    def _get_interpro_evidence_impl(self) -> str:
        """
        Obtain evidence of association between InterPro domains and GO terms.
        This tool examines whether protein domains are related to a target function.
        """
        if self.current_state is None:
            return "Error: State not initialized"

        interpros = self.current_state["interpros"]
        go_term = self.current_state["go_term"]

        if not interpros:
            return "No InterPro domains found for this protein."

        # 查找关联
        mask = self.data_manager.interpro_to_go["interpro_id"].isin(interpros)
        associated_gos = self.data_manager.interpro_to_go.loc[
            mask, "go_id"
        ].unique().tolist()

        if go_term in associated_gos:
            matching_interpros = self.data_manager.interpro_to_go[
                (self.data_manager.interpro_to_go["interpro_id"].isin(interpros)) &
                (self.data_manager.interpro_to_go["go_id"] == go_term)
                ]["interpro_id"].tolist()

            # 获取域的详细信息
            domain_details = []
            for ipr in matching_interpros:
                info = self.data_manager.interpro_info.get(ipr, "No description")
                domain_details.append(f"  - {ipr}: {info}")

            result = f"✓ STRONG EVIDENCE: InterPro domains directly annotated with {go_term}:\n"
            result += "\n".join(domain_details)

            return result
        else:
            return f"✗ No direct association found between InterPro domains {interpros} and {go_term}"

    @property
    def get_diamond_evidence(self):
        @tool
        def wrapped():
            """
            Obtain evidence of sequence similarity.
            Check if there are proteins with similar sequences that have this function.
            :return: The diamond score for given hypothesis function.
            """
            return self._get_diamond_evidence_impl()
        return wrapped
    def _get_diamond_evidence_impl(self) -> str:
        """
        Obtain evidence of sequence similarity.
        Check if there are proteins with similar sequences that have this function.
        """
        if self.current_state is None:
            return "Error: State not initialized"

        diamond_score = self.current_state["diamond_score"]
        go_term = self.current_state["go_term"]

        if diamond_score is None:
            return "No DIAMOND similarity data available."

        if diamond_score > 0.7:
            return f"✓ STRONG EVIDENCE: High sequence similarity (score={diamond_score:.3f}) to proteins with {go_term}. This suggests strong functional conservation."
        elif diamond_score > 0.4:
            return f"○ MODERATE EVIDENCE: Moderate sequence similarity (score={diamond_score:.3f}) to proteins with {go_term}. Supports but doesn't confirm the annotation."
        else:
            return f"✗ WEAK EVIDENCE: Low sequence similarity (score={diamond_score:.3f}). Similar proteins do not have {go_term}."

    @property
    def get_gene_info(self):
        @tool
        def wrapped():
            """
            Obtain the gene function description.
            Check if the gene's known function supports the GO annotation.
            :return: The information associated with the gene id.
            """
            return self._get_gene_info_impl()

        return wrapped

    def _get_gene_info_impl(self) -> str:
        """
        Obtain the gene function description.
        Check if the gene's known function supports the GO annotation.
        """
        if self.current_state is None:
            return "Error: State not initialized"

        gene_id = self.current_state["gene_id"]
        go_term = self.current_state["go_term"]

        gene_info = self.data_manager.gene_info.get(gene_id)

        if gene_info is None:
            return "No gene information available in database."

        summary = gene_info.get("summary", "")
        go_name = self.ontology.get_term_name(go_term)

        return f"Gene {gene_id} Summary:\n{summary}\n\nTarget GO term: {go_name}\n\nAnalyze whether the gene function description supports this GO annotation."

    @property
    def get_taxon_constraints(self):
        @tool
        def wrapped():
            """
            Obtain taxonomic constraints.
            Check if the function is reasonably present in this species.
            :return:  a demonstration about if the function is reasonably present in this species.
            """
            return self._get_taxon_constraints_impl()

        return wrapped

    def _get_taxon_constraints_impl(self) -> str:
        """
        获取分类学约束。
        检查该功能是否在该物种中合理存在。
        """
        if self.current_state is None:
            return "Error: State not initialized"

        organism = self.current_state["organism"]
        go_term = self.current_state["go_term"]

        if organism not in self.ontology.taxon_map:
            return "No taxonomic constraints available for this organism."

        in_taxon, never_in_taxon = self.ontology.taxon_map[organism]

        if go_term in never_in_taxon:
            return f"✗ CRITICAL: {go_term} is NEVER found in organism {organism}. This is a strong constraint that should heavily penalize the prediction."
        elif go_term in in_taxon:
            return f"✓ {go_term} is commonly found in organism {organism}. This supports the annotation."
        else:
            return f"○ No specific taxonomic constraint for {go_term} in {organism}."


# ================================
# LLM 驱动的节点
# ================================

class LLMNodes:
    """所有节点都使用 LLM 进行推理"""

    def __init__(self, llm: LLMWrapper, tools: EvidenceToolsLLM):
        self.llm = llm
        self.tools = tools

    def coordinator_node(self, state: ProteinRefinementState) -> ProteinRefinementState:
        """
        协调者节点 - LLM 决定下一步行动
        """

        # 构建 prompt
        system_prompt = """You are an expert GO annotation curator. Your task is to refine protein function predictions by gathering evidence and making informed decisions.

You have access to these tools:
1. get_interpro_evidence() - Check protein domains
2. get_diamond_evidence() - Check sequence similarity
3. get_gene_info() - Check gene function description
4. get_taxon_constraints() - Check taxonomic constraints

Your workflow:
1. Analyze the current situation
2. Decide if you need more evidence
3. If yes, choose which tool to call next
4. If no, make a final decision

Respond in JSON format:
{
  "reasoning": "Your step-by-step reasoning",
  "next_action": "call_tool" or "finalize",
  "tool_to_call": "interpro" or "diamond" or "gene" or "taxon" (if next_action is call_tool),
  "confidence_level": "high" or "medium" or "low"
}
"""

        # 构建上下文
        evidence_summary = self._format_evidence(state["evidence"])
        tools_used = state["tools_called"]

        user_prompt = f"""
PROTEIN: {state['protein_id']} (Gene: {state['gene_id']})
GO TERM: {state['go_term']}
INITIAL SCORE: {state['initial_score']:.3f}
DIAMOND SCORE: {state['diamond_score'] if state['diamond_score'] else 'N/A'}

TOOLS ALREADY USED: {', '.join(tools_used) if tools_used else 'None'}

EVIDENCE COLLECTED SO FAR:
{evidence_summary if evidence_summary else 'No evidence yet'}

What should we do next? Should we call another tool, or do we have enough evidence to make a decision?
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        # LLM 推理
        response = self.llm.invoke(messages)

        # 解析响应
        try:
            decision = json.loads(response.content)
        except:
            # 如果不是 JSON，尝试提取
            decision = {
                "reasoning": response.content,
                "next_action": "finalize",
                "confidence_level": "medium"
            }

        # 更新状态
        state["reasoning"] = decision.get("reasoning", "")
        state["next_action"] = decision.get("next_action", "finalize")
        state["tool_to_call"] = decision.get("tool_to_call", "")
        state["confidence_level"] = decision.get("confidence_level", "medium")
        state["messages"].append(response)

        print(f"\n[LLM COORDINATOR]")
        print(f"  Reasoning: {state['reasoning'][:200]}...")
        print(f"  Next Action: {state['next_action']}")
        if state['next_action'] == 'call_tool':
            print(f"  Tool to Call: {state['tool_to_call']}")

        return state

    def tool_calling_node(self, state: ProteinRefinementState) -> ProteinRefinementState:
        """
        工具调用节点 - 执行 LLM 选择的工具
        """
        tool_name = state["tool_to_call"]

        # 注入当前状态到工具
        self.tools.current_state = state

        # 调用工具
        if tool_name == "interpro":
            result = self.tools.get_interpro_evidence_tool.invoke({})
        elif tool_name == "diamond":
            result = self.tools.get_diamond_evidence()
        elif tool_name == "gene":
            result = self.tools.get_gene_info()
        elif tool_name == "taxon":
            result = self.tools.get_taxon_constraints()
        else:
            result = f"Unknown tool: {tool_name}"

        # 记录证据
        evidence = {
            "source": tool_name,
            "content": result,
            "timestamp": datetime.now().isoformat()
        }

        state["evidence"].append(evidence)
        state["tools_called"].append(tool_name)

        print(f"\n[TOOL RESULT: {tool_name}]")
        print(f"  {result[:200]}...")

        return state

    def finalize_node(self, state: ProteinRefinementState) -> ProteinRefinementState:
        """
        最终决策节点 - LLM 综合所有证据做出决策
        """

        system_prompt = """You are making the final decision on a GO annotation refinement.

Based on all the evidence collected, decide:
1. Should the score be increased, decreased, or kept the same?
2. By how much? (score change between -0.5 and +0.5)

Scoring rules:
- Strong positive evidence: +0.2 to +0.3
- Moderate positive evidence: +0.1 to +0.15
- Weak or neutral evidence: -0.05 to +0.05
- Contradictory evidence: -0.1 to -0.2
- Strong negative evidence (e.g., taxon constraint): -0.3 to -0.5

Respond in JSON format:
{
  "score_adjustment": <float between -0.5 and 0.5>,
  "reasoning": "Your detailed reasoning for this decision",
  "confidence": "high" or "medium" or "low"
}
"""

        evidence_summary = self._format_evidence(state["evidence"])

        user_prompt = f"""
PROTEIN: {state['protein_id']}
GO TERM: {state['go_term']}
INITIAL SCORE: {state['initial_score']:.3f}

ALL EVIDENCE COLLECTED:
{evidence_summary}

Based on this evidence, what should the final score be?
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = self.llm.invoke(messages)

        # 解析决策
        try:
            decision = json.loads(response.content)
            score_adjustment = decision.get("score_adjustment", 0.0)
            reasoning = decision.get("reasoning", "")
            confidence = decision.get("confidence", "medium")
        except:
            # Fallback
            score_adjustment = 0.0
            reasoning = response.content
            confidence = "low"

        # 计算新分数
        new_score = np.clip(state['initial_score'] + score_adjustment, 0.0, 1.0)

        state["current_score"] = new_score
        state["final_decision"] = {
            "old_score": float(state['initial_score']),
            "new_score": float(new_score),
            "change": float(score_adjustment),
            "reasoning": reasoning,
            "confidence": confidence,
            "evidence_count": len(state["evidence"])
        }

        print(f"\n[FINAL DECISION]")
        print(f"  {state['go_term']}: {state['initial_score']:.3f} → {new_score:.3f} (Δ={score_adjustment:+.3f})")
        print(f"  Confidence: {confidence}")
        print(f"  Reasoning: {reasoning[:200]}...")

        return state

    def _format_evidence(self, evidence_list):
        """格式化证据列表"""
        if not evidence_list:
            return "No evidence collected yet."

        formatted = []
        for i, evidence in enumerate(evidence_list, 1):
            formatted.append(f"{i}. [{evidence['source']}] {evidence['content']}")

        return "\n\n".join(formatted)


# ================================
# 路由函数
# ================================

def route_after_coordinator(state: ProteinRefinementState) -> str:
    """根据 LLM 决策路由"""
    action = state["next_action"]

    if action == "call_tool":
        return "call_tool"
    elif action == "finalize":
        return "finalize"
    else:
        return "finalize"  # 默认


# ================================
# 构建 LLM 驱动的图
# ================================

def build_llm_graph(llm: LLMWrapper, tools: EvidenceToolsLLM) -> StateGraph:
    """构建 LLM 驱动的 LangGraph"""

    nodes = LLMNodes(llm, tools)

    workflow = StateGraph(ProteinRefinementState)

    # 添加节点
    workflow.add_node("coordinator", nodes.coordinator_node)
    workflow.add_node("call_tool", nodes.tool_calling_node)
    workflow.add_node("finalize", nodes.finalize_node)

    # 设置入口
    workflow.set_entry_point("coordinator")

    # 条件路由
    workflow.add_conditional_edges(
        "coordinator",
        route_after_coordinator,
        {
            "call_tool": "call_tool",
            "finalize": "finalize"
        }
    )

    # 工具调用后回到协调者
    workflow.add_edge("call_tool", "coordinator")

    # 最终节点
    workflow.add_edge("finalize", END)

    return workflow.compile()


# ================================
# 主执行类
# ================================

class LLMDrivenProteinRefinement:
    """LLM 驱动的蛋白质功能预测精炼器"""

    def __init__(
            self,
            ontology,
            data_manager,
            model_type="camel",
            model_name="qwen"
    ):
        self.ontology = ontology
        self.data_manager = data_manager

        # 初始化 LLM
        self.llm = LLMWrapper(model_type, model_name)

        # 初始化工具
        self.tools = EvidenceToolsLLM(data_manager, ontology)

        # 构建图
        self.graph = build_llm_graph(self.llm, self.tools)

    def refine_single_go_term(
            self,
            protein_data: Dict,
            go_term: str,
            initial_score: float,
            diamond_score: float
    ) -> Dict:
        """为单个 GO term 执行精炼"""

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
            tools_called=[],
            messages=[],
            next_action="",
            tool_to_call="",
            confidence_level="",
            reasoning="",
            final_decision={}
        )

        print(f"\n{'=' * 60}")
        print(f"LLM-Driven Refinement: {protein_data['proteins']} - {go_term}")
        print(f"{'=' * 60}")

        final_state = self.graph.invoke(initial_state)

        return final_state["final_decision"]

    def refine_all_predictions(
            self,
            protein_data: Dict,
            terms: List[str],
            terms_dict: Dict[str, int],
            score_threshold: float = 0.1,
            max_terms: int = None
    ) -> np.ndarray:
        """批量精炼"""

        predictions = protein_data["preds"].copy()
        diam_preds = protein_data.get("diam_preds", {})

        high_score_indices = np.where(predictions>=0.1 and predictions<0.6)[0]
        go_terms_to_refine = [terms[i] for i in high_score_indices]

        # if max_terms:
        #     go_terms_to_refine = go_terms_to_refine[:max_terms]

        print(f"\nProtein: {protein_data['proteins']}")
        print(f"GO terms to refine: {len(go_terms_to_refine)}")

        for go_term in go_terms_to_refine:
            idx = terms_dict[go_term]
            initial_score = predictions[idx]
            diamond_score = diam_preds.get(go_term)

            decision = self.refine_single_go_term(
                protein_data,
                go_term,
                initial_score,
                diamond_score
            )

            update_record = {
                'proteins': protein_data["proteins"],
                'go_term': go_term,
                'old_score': initial_score,
                'new_score': decision["new_score"],
                'timestamp': datetime.now().isoformat()
            }
            # 实时写文件（append）
            with open("./data/mf/update_history_langgraph.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(update_record, ensure_ascii=False) + "\n")

            predictions[idx] = decision["new_score"]

        return predictions


# ================================
# 使用示例
# ================================

if __name__ == "__main__":

    # 初始化
    ontology = Ontology()
    data_manager = DataManager(data_dir="data")

    # 创建 LLM 驱动的精炼器
    refiner = LLMDrivenProteinRefinement(
        ontology,
        data_manager,
        model_type="openai",
        model_name="gpt-4"
    )

    # 模拟数据
    protein_data = {
        "proteins": "P12345",
        "genes": "BRCA1",
        "sequences": "MKTAYIAKQRQISFVKSHFSRQLE...",
        "interpros": ["IPR001664", "IPR002048"],
        "orgs": "9606",
        "uniprot_text": "Tumor suppressor involved in DNA repair...",
        "preds": np.random.rand(100),
        "diam_preds": {"GO:0003824": 0.85}
    }

    # 执行精炼
    updated = refiner.refine_all_predictions(
        protein_data,
        [f"GO:{i:07d}" for i in range(100)],
        {f"GO:{i:07d}": i for i in range(100)},
        score_threshold=0.5,
        max_terms=5  # 限制数量用于演示
    )

    print("\nRefinement complete!")