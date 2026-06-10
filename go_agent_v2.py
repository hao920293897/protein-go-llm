import numpy as np
import pandas as pd
from typing import List, Dict

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.agents import ChatAgent
from camel.toolkits import FunctionTool:
from tqdm import tqdm
import click as ck
import json
from data_process.ontology import Ontology
from agents.models import qwen_vllm_model
from datetime import datetime

# ================================
# GO TERM OBJECT
# ================================

class GOTerm:

    def __init__(self, go_id, info, predicted_score,diamond_score, frequency=None):

        self.go_id = go_id
        self.info = info
        self.predicted_score = predicted_score
        self.frequency = frequency
        self.diamond_score = diamond_score

    def __repr__(self):
        return f"Start of information for {self.go_id}:\n Predicted_score={self.predicted_score}, diamond_score={self.diamond_score}, annotation_frequency={self.frequency}, info={self.info}\nEnd of information for {self.go_id}.\n"

    def __str__(self):
        return f"Start of information for {self.go_id}:\n Predicted Score: {self.predicted_score}, diamond Score: {self.diamond_score}. Annotation Frequency: {self.frequency if self.frequency is not None else 'N/A'}, {self.info}\nEnd of information for {self.go_id}.\n"


# ================================
# PROTEIN AGENT
# ================================

class DataManager:
    """管理所有共享数据"""

    def __init__(self, data_dir="data"):
        self.gene_info = self._load_json(f"{data_dir}/gene_info.json")
        self.interpro_info = self._load_json(f"{data_dir}/interpro_descriptions.json")
        self.interpro_to_go = pd.read_csv(f"{data_dir}/interpro2go_mapping.tsv", sep="\t")

    @staticmethod
    def _load_json(path):
        with open(path) as f:
            return json.load(f)

class ProteinAgent(ChatAgent):

    def __init__(
        self,
        model_name,
        ontology,
        ont,
        terms_dict,
        data_row,
        data_manager
    ):

        # ================================
        # MODEL
        # ================================

        model = qwen_vllm_model

        # ================================
        # DATA
        # ================================

        self.ont = ont
        self.go = ontology
        self.data_row = data_row
        self.terms_dict = terms_dict

        self.prot_id = data_row["proteins"]
        self.gene_id = data_row["genes"]
        self.data_manager = data_manager
        # self.gene_info = json.load(open(f"data/gene_info.json"))
        # self.interpro_info = json.load(open(f"data/interpro_descriptions.json"))

        self.sequence = data_row["sequences"]
        self.interpros = data_row["interpros"]
        self.update_history = []

        # self.interpro_to_go = pd.read_csv(
        #     "data/interpro2go_mapping.tsv",
        #     sep="\t"
        # )

        # ================================
        # TOOLS
        # ================================

        interpro_tool = FunctionTool(
            func=self.get_interpro_annotations
        )

        go_info_tool = FunctionTool(
            func=self.get_go_term_info
        )

        taxon_tool = FunctionTool(
            func=self.get_taxon_constraints
        )

        update_tool = FunctionTool(
            func=self.update_predictions
        )

        gene_tool = FunctionTool(
            func=self.get_gene_info
        )

        interpro_info_tool = FunctionTool(
            func=self.get_interpro_info
        )

        tools = [
            interpro_tool,
            go_info_tool,
            taxon_tool,
            gene_tool,
            interpro_info_tool,
            update_tool
        ]

        # ================================
        # SYSTEM PROMPT
        # ================================
        uniprot_info = self.get_uniprot_information()
        context =  f"""You are an expert GO annotation curator refining predictions for protein {self.prot_id}.
        PROTEIN INFORMATION:
        - Protein ID: {self.prot_id}
        - Gene ID: {self.gene_id}
        - InterPro domains: {self.interpros}
        - Background: {uniprot_info}
        YOUR TASK:
        Review and refine GO term predictions by systematically checking multiple evidence sources, such as the
interpro annotations or diamond score similarity.
        REQUIRED WORKFLOW (follow this order):
        1. get_go_term_info(go_term) - Get GO term definition , current score and diamond score similarity.
        2. get_interpro_annotations() - Check which GO terms are associated with this protein's InterPro domains
        3. get_interpro_info(interpro_id) - Get detailed descriptions of each InterPro domain
        4. get_gene_info(gene_id) - Get gene function summary
        5. get_taxon_constraints() - Check taxonomic constraints
        6. update_predictions(go_term, new_score) - Update score based on evidence
        SCORING RULES:
        - Strong supporting evidence (direct match in InterPro→GO, gene function aligns): +0.2
        - Weak supporting evidence (indirect match, partial alignment): +0.1
        - Contradictory evidence (conflicts with gene function or taxon constraints): -0.2
        - No change if evidence is ambiguous: 0.0
        IMPORTANT RULES:
        - ALWAYS call tools to gather evidence BEFORE making updates
        - Do NOT update scores without checking ALL available evidence
        - Explicitly state your reasoning for each score change
        - If a GO term violates taxon constraints (in never_in_taxon), penalize heavily
"""

        prompt = f"""You are an expert GO annotation curator refining predictions for protein {self.prot_id}.
        PROTEIN INFORMATION:
        - Protein ID: {self.prot_id}
        - Gene ID: {self.gene_id}
        - InterPro domains: {self.interpros}
        - Background: {uniprot_info}
        YOUR TASK:
        Review and refine GO term predictions by systematically checking multiple evidence sources. You have access to multiple biological evidence tools. Tool usage must be adaptive. Do NOT call every tool by default.
        TOOL USAGE POLICY:
        - Use get_go_term_info to understand the term.
        - Use get_interpro_annotations for domain evidence.
        - Use get_interpro_info only if domain meaning is unclear.
        - Use get_gene_info if domain evidence is insufficient.
        - Use get_taxon_constraints only if species validity is uncertain.
        - Use update_predictions only after sufficient evidence.
        DECISION RULES:
        - Collect only necessary evidence.
        - Do not call all tools blindly.
        - Stop when confidence is high.
        - After each tool call, decide whether another tool is needed.
        UPDATE RULES:
        - Strong supporting evidence (direct match in InterPro→GO, gene function aligns): +0.2
        - Weak supporting evidence (indirect match, partial alignment): +0.1
        - Contradictory evidence (conflicts with gene function or taxon constraints): -0.2
        - No change if evidence is ambiguous: +0.0
        Before updating:
        1. Collect evidence
        2. Explain reasoning
        3. Update score
        """
        super().__init__(
            system_message=prompt,
            model=model,
            tools=tools
        )

    # ================================
    # TOOLS
    # ================================
    def get_uniprot_information(self) -> str:
        """
        Retrieve UniProt information for the current sequence.
        Returns:
            str: A string containing the UniProt information.
        """
        uniprot_info = self.data_row['uniprot_text']
        return uniprot_info

    def get_interpro_annotations(self):
        """
        Retrieve GO terms associated with this protein's InterPro domains.
        Returns:
            list: Unique GO term IDs associated with InterPro domains.
        """
        interpros = self.data_row["interpros"]
        if not interpros:
            return []
        # 一次性过滤
        mask = self.data_manager.interpro_to_go["interpro_id"].isin(interpros)
        gos = self.data_manager.interpro_to_go.loc[mask, "go_id"].unique().tolist()

        return gos

    def get_interpro_info(self, interpro_id: str):
        """
        Retrieve the information of interpro_id for a given GO term.
        Args:
            interpro_id (str): The interpro_id to retrieve information.
        Returns:
            str: The information associated with interpro.
        """
        interpro_info = self.data_manager.interpro_info.get(interpro_id,  "No interpro info available")
        return interpro_info

    def get_gene_info(self, gene_id: str) -> str:
        """
        Retrieve the information for a given gene ID of the protein.
        Args:
            gene_id (str): The gene id to  retrieve information.
        Returns:
            str: The information associated with the gene id.
        """
        gene_info = self.data_manager.gene_info.get(gene_id)
        if gene_info is None:
            return "No gene information available"
        return str(gene_info["summary"])

    def get_diamond_score(self, go_term: str) -> float:
        """
        Retrieve the diamond score for a given sequence and hypothesis function.

        Args:
            go_term (str): The GO term to analyze.
        Returns:
            float: The diamond score for given hypothesis function. If the GO term is not found, returns None.
        """
        preds = self.data_row['diam_preds']

        if go_term not in preds:
            return None
        else:
            return float(preds[go_term])

    def get_go_term_info(self, go_term: str):
        """
        Retrieve the information for a given GO term.
        Args:
            go_term (str): The GO term to retrieve information for.
        Returns:
            str: The information associated with the GO term.
        """
        if go_term not in self.terms_dict:
            return "GO term not found"

        return str(self.create_go_term(go_term))

    def get_taxon_constraints(self):

        """
        Retrieve taxon constraints for the current sequence.
        Returns:
            dict: A dictionary containing 'in_taxon' and 'never_in_taxon' lists.
        """
        org = self.data_row["orgs"]

        if org not in self.go.taxon_map:
            return {}

        taxa = self.go.taxon_map[org]

        return {
            "in_taxon": taxa[0],
            "never_in_taxon": taxa[1]
        }

    def create_go_term(self, go_term):

        return GOTerm(
            go_id=go_term,
            info=self.go.get_term_info(go_term),
            predicted_score=self.query_score(go_term),
            diamond_score=self.get_diamond_score(go_term),
        )

    def query_score(self, go_term):
        """
        Query the initial score for a specific GO term.
        Args:
            go_term (str): The GO term to query.
        Returns:
            float: The score for the given GO term.
        """

        idx = self.terms_dict[go_term]

        return self.data_row[f"preds"][idx]

    def update_predictions(self, go_term: str, score: float):
        """
        Update the predictions dictionary with a new score for a GO term.
        Args:
            go_term (str): The GO term identifier to update.
            score (float): The new score for the GO term (must be between 0 and 1).
        Returns:
            str: Confirmation message with old and new scores.
        """
        if go_term not in self.terms_dict:
            return f"Error: GO term {go_term} not found in vocabulary"

        # 限制分数范围
        score = max(0.0, min(1.0, score))

        idx = self.terms_dict[go_term]
        old_score = self.data_row[f"preds"][idx]
        self.data_row[f"preds"][idx] = score

        change = score - old_score

        # ... 更新逻辑
        update_record = {
            'go_term': go_term,
            'old_score': float(old_score),
            'new_score': float(score),
            'timestamp': datetime.now().isoformat()
        }

        # 内存保存
        self.update_history.append(update_record)

        # 实时写文件（append）
        with open("update_history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(update_record, ensure_ascii=False) + "\n")

        # 日志
        print(f"Execute update. Updated {go_term}: {old_score:.3f} → {score:.3f}")
        self.logger.info(
            f"Updated {go_term}: {old_score:.3f} → {score:.3f}"
        )
        # return f"Updated {go_term}: {old_score:.3f} → {score:.3f} (change: {change:+.3f})"

    # def update_predictions(self, go_term: str, score: float):
    #     """
    #     Update the predictions dictionary with a new score for a GO term.
    #     Args:
    #         go_term (str): The GO term identifier to update.
    #         score (float): The new score for the GO term.
    #     """
    #     if go_term in self.terms_dict:
    #
    #         idx = self.terms_dict[go_term]
    #         self.data_row[f"preds"][idx] = score


# refine_predictions(model_name, go, ont, terms, terms_dict, row)


def refine_predictions(model_name, go, ont, terms, terms_dict, row, data_manager):

    agent = ProteinAgent(
        model_name=model_name,
        ontology=go,
        ont=ont,
        terms_dict=terms_dict,
        data_row=row,
        data_manager=data_manager
    )

    go_terms = []
    for i, score in enumerate(row[f'preds']):
        if score >= 0.1:
            go_terms.append(terms[i])
    go_terms_info = [agent.get_go_term_info(go_term) for go_term in go_terms]
    go_terms_info = "\n".join(go_terms_info)
    analysis_prompt = f""" You have now this information about the GO terms you discussed before {go_terms_info}.

    For each relevant GO term suggested:
    - Adaptively calling tools and analyze supporting evidence (descriptions of InterPro / Gene or diamond score similarity) for each plausible term.
    - If there is conflicting evidence, provide your resolution
    - Provide Current score vs. recommended score. update by incrementing or decrementing the score with the confidence level, Strong evidence +0.2, Weak evidence +0.1, Contradiction  -0.2.
    - Confidence level (Strong/Weak/Contradiction)
    Output your report with all the points above in a structured format. Annotation frequency is based on the training data and is an important factor in your analysis.
    """
    response = agent.step(analysis_prompt)
    print(response.msgs[0].content)
    updating_prompt = f""" Apply your analysis to update GO term scores with update tool. Perform the update and also provide a rationale for each
    change. If no changes are needed, return 'No changes needed'.  """
    agent.step(updating_prompt)

def load_initial_predictions(ont):
    # df = pd.read_pickle('data/test_predictions_abstracts.pkl')
    df = pd.read_pickle(f'data/{ont}/test_data_diam_with_text.pkl')
    return df

@ck.command()
@ck.option('--run_number', type=int, default=0, help='Run number for output file naming.')
@ck.option('--model_name', type=ck.Choice(['gemini', 'gpt', 'qwen']), default='gemini', help='Model name to use for the agent.')
def main(run_number, model_name):
    data_manager = DataManager()
    print(f"Running refinement with model {model_name} for run number {run_number}")
    go = Ontology('../deepgozero-main/data/go.obo', with_rels=True)
    for ont in ['mf', 'cc', 'bp']:
        df = load_initial_predictions(ont)
        terms_df = pd.read_pickle(f'../deepgozero-main/data/{ont}/terms_zero_10.pkl')
        terms = terms_df['terms'].values.tolist()
        terms_dict = {v: k for k, v in enumerate(terms)}
        skipped = 0
        for i in tqdm(range(len(df))):
            try:
                row = df.iloc[i]
                prop_annotations = row['prop_annotations']
                terms_in_ont = [t for t in prop_annotations if t in terms]
                if len(terms_in_ont) == 0:
                    print(f"Skipping protein {i} as it has no prop_annotations in {ont}")
                    skipped += 1
                    continue

                old_preds = row[f'preds'].copy()
                refine_predictions(model_name, go, ont, terms, terms_dict, row, data_manager)                # 回写到DataFrame
                df.at[i, f'preds'] = row[f'preds'].copy()

                new_preds = row[f'preds']
                c = np.sum(old_preds != new_preds)
                print(f'Protein {i}: Updated {c} predictions')
            except Exception as e:
                print(f"Error processing protein {i}: {e}")

        processed = len(df) - skipped
        print(f"Processed {processed} proteins for ontology {ont}. Skipped {skipped} proteins.")
        df.to_pickle(f'data/{ont}/test_predictions_refined_{model_name}_run{run_number}.pkl', protocol=4)


if __name__ == "__main__":
    main()
