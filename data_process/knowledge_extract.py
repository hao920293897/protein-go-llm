"""
生物医学知识图谱信息抽取模块
从InterPro、Gene、GO、Protein描述中提取三元组
"""
import json
import re
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
import logging
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import pandas as pd

# ================================
# 数据结构定义
# ================================
@dataclass
class Triple:
    """三元组数据结构"""
    head: str
    relation: str
    tail: str

    def to_tuple(self) -> Tuple[str, str, str]:
        return (self.head, self.relation, self.tail)

    def __hash__(self):
        return hash(self.to_tuple())

    def __eq__(self, other):
        return self.to_tuple() == other.to_tuple()


# ================================
# Prompt模板定义
# ================================
class PromptTemplates:
    """各类文本的信息抽取Prompt模板"""

    # 通用系统提示
    SYSTEM_PROMPT = """You are a biomedical knowledge extraction expert. Your task is to extract structured knowledge triples from biological texts.

CRITICAL RULES:
1. Output ONLY valid JSON - no markdown, no explanations, no preambles
2. Entity names must be in English
3. Entity and Relation names must be standardized (use predefined entities and relations when possible)
4. Extract factual information only - no inferences
5. Keep entity names concise but specific
6. Format: {"triples": [["entity1", "relation", "entity2"], ...]}

STANDARD RELATIONS (use these when applicable):
- has_function: entity performs a function
- located_in: subcellular/cellular location
- part_of: component relationship
- interacts_with: physical interaction
- regulates: regulatory relationship
- associated_with: general association
- involved_in: participates in a process
- has_activity: biochemical activity
- binds_to: binding interaction
- expressed_in: expression pattern
- codes_for: genetic relationship
- belongs_to: classification/taxonomy"""

    # InterPro域提取
    INTERPRO_PROMPT = """Extract knowledge triples from this InterPro domain description.

InterPro ID: {interpro_id}
Description: {description}

Focus on extracting:
1. Domain name and its functions
2. Structural components
3. Cellular locations
4. Biological processes it's involved in
5. Molecular interactions
6. Diseases or pathways

Example output format:
{{"triples": [
    ["IPR042473", "has_name", "V-type immunoglobulin domain"],
    ["VISTA", "is_type_of", "transmembrane protein"],
    ["VISTA", "has_component", "Ig V domain"],
    ["VISTA", "expressed_in", "hematopoietic cells"],
    ["VISTA", "has_function", "immune checkpoint"],
    ["VISTA", "regulates", "anti-tumor immunity"],
    ["VISTA", "associated_with", "autoimmune diseases"]
]}}

Now extract triples from the provided description. Output ONLY valid JSON."""

    # Gene描述提取
    GENE_PROMPT = """Extract knowledge triples from this gene description.

Gene ID: {gene_id}
Symbol: {symbol}
Description: {description}
Summary: {summary}

Focus on extracting:
1. Gene symbol and its full name
2. Molecular functions (GTPase, kinase, binding activities)
3. Biological processes
4. Cellular components/locations
5. Expression patterns
6. Orthology relationships

Example output format:
{{"triples": [
    ["56212", "has_symbol", "Rhog"],
    ["Rhog", "has_name", "ras homolog family member G"],
    ["Rhog", "has_activity", "GTP binding activity"],
    ["Rhog", "has_activity", "GTPase activity"],
    ["Rhog", "located_in", "cytoplasmic vesicle"],
    ["Rhog", "expressed_in", "cerebral cortex"],
    ["Rhog", "orthologous_to", "human RHOG"]
]}}

Now extract triples. Output ONLY valid JSON."""

    # GO术语提取
    GO_PROMPT = """Extract knowledge triples from this GO term.

GO ID: {go_id}
Name: {name}
Definition: {definition}
Namespace: {namespace}

Focus on extracting:
1. GO term and its name
2. The biological concept it represents
3. Key processes or components mentioned
4. Relationships described in the definition

Example output format:
{{"triples": [
    ["GO:0000001", "has_name", "mitochondrion inheritance"],
    ["GO:0000001", "belongs_to", "biological_process"],
    ["mitochondrion inheritance", "involves", "mitochondria"],
    ["mitochondrion inheritance", "involves", "daughter cells"],
    ["mitochondrion inheritance", "occurs_during", "mitosis"],
    ["mitochondrion inheritance", "mediated_by", "cytoskeleton"]
]}}

Now extract triples. Output ONLY valid JSON."""

    # 蛋白质描述提取
    PROTEIN_PROMPT = """Extract knowledge triples from this protein description.

Protein ID: {protein_id}
Description: {description}

Focus on extracting:
1. Protein name and gene
2. Organism
3. Molecular functions and activities
4. Substrates and products
5. Subcellular locations
6. Biological processes
7. Catalytic activities
8. Regulatory roles

Example output format:
{{"triples": [
    ["ALKB1_MOUSE", "has_uniprot_id", "P0CB42"],
    ["ALKB1_MOUSE", "codes_by_gene", "Alkbh1"],
    ["ALKB1_MOUSE", "from_organism", "Mus musculus"],
    ["ALKB1_MOUSE", "has_function", "dioxygenase activity"],
    ["ALKB1_MOUSE", "acts_on", "DNA"],
    ["ALKB1_MOUSE", "acts_on", "tRNA"],
    ["ALKB1_MOUSE", "located_in", "nucleus"],
    ["ALKB1_MOUSE", "has_activity", "tRNA demethylase"],
    ["ALKB1_MOUSE", "regulates", "translation initiation"],
    ["ALKB1_MOUSE", "requires", "molecular oxygen"],
    ["ALKB1_MOUSE", "requires", "alpha-ketoglutarate"]
]}}

Now extract triples. Output ONLY valid JSON."""


# ================================
# 信息抽取器
# ================================
class KnowledgeExtractor:
    """知识抽取器"""

    def __init__(
            self,
            model_name: str = "Qwen/Qwen2.5-7B-Instruct",
            api_base: str = "http://localhost:8000/v1",
            api_key: str = "EMPTY"
    ):
        """
        初始化抽取器

        Args:
            model_name: 模型名称
            api_base: API地址
            api_key: API密钥
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )
        self.model_name = model_name
        self.templates = PromptTemplates()

        logger.info(f"Initialized KnowledgeExtractor with model: {model_name}")

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """调用LLM"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=2048
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return '{"triples": []}'

    def _parse_response(self, response: str) -> List[Triple]:
        """解析LLM响应"""
        try:
            # 移除可能的markdown标记
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            # 解析JSON
            data = json.loads(response)

            if "triples" not in data:
                logger.warning("No 'triples' key in response")
                return []

            # 转换为Triple对象
            triples = []
            for triple_list in data["triples"]:
                if len(triple_list) != 3:
                    logger.warning(f"Invalid triple format: {triple_list}")
                    continue

                head, relation, tail = triple_list

                # 清理和标准化
                head = str(head).strip()
                relation = str(relation).strip().lower().replace(" ", "_")
                tail = str(tail).strip()

                if head and relation and tail:
                    triples.append(Triple(head, relation, tail))

            return triples

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.debug(f"Response was: {response[:200]}")
            return []
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return []

    def extract_from_interpro(self, interpro_id: str, description: str) -> List[Triple]:
        """从InterPro描述中提取三元组"""
        # 分离名称和描述
        if "#@@#" in description:
            name, desc = description.split("#@@#", 1)
            full_desc = f"Name: {name}\nDescription: {desc}"
        else:
            full_desc = description

        prompt = self.templates.INTERPRO_PROMPT.format(
            interpro_id=interpro_id,
            description=full_desc
        )

        response = self._call_llm(self.templates.SYSTEM_PROMPT, prompt)
        triples = self._parse_response(response)

        logger.info(f"Extracted {len(triples)} triples from InterPro {interpro_id}")
        return triples

    def extract_from_gene(self, gene_data: Dict) -> List[Triple]:
        """从Gene描述中提取三元组"""
        prompt = self.templates.GENE_PROMPT.format(
            gene_id=gene_data.get("gene_id", ""),
            symbol=gene_data.get("symbol", ""),
            description=gene_data.get("description", ""),
            summary=gene_data.get("summary", "")
        )

        response = self._call_llm(self.templates.SYSTEM_PROMPT, prompt)
        triples = self._parse_response(response)

        logger.info(f"Extracted {len(triples)} triples from Gene {gene_data.get('symbol', 'unknown')}")
        return triples

    def extract_from_go(self, go_data: Dict, go_id: str) -> List[Triple]:
        """从GO术语中提取三元组"""
        prompt = self.templates.GO_PROMPT.format(
            go_id=go_id,
            name=go_data.get("name", ""),
            definition=go_data.get("definition", ""),
            namespace=go_data.get("namespace", "")
        )

        response = self._call_llm(self.templates.SYSTEM_PROMPT, prompt)
        triples = self._parse_response(response)

        logger.info(f"Extracted {len(triples)} triples from GO {go_id}")
        return triples

    def extract_from_protein(self, protein_id: str, description: str) -> List[Triple]:
        """从蛋白质描述中提取三元组"""
        # 截断过长的描述
        if len(description) > 2000:
            description = description[:2000] + "..."

        prompt = self.templates.PROTEIN_PROMPT.format(
            protein_id=protein_id,
            description=description
        )

        response = self._call_llm(self.templates.SYSTEM_PROMPT, prompt)
        triples = self._parse_response(response)

        logger.info(f"Extracted {len(triples)} triples from Protein {protein_id}")
        return triples


# ================================
# 批量处理器
# ================================
class BatchExtractor:
    """批量信息抽取"""

    def __init__(self, extractor: KnowledgeExtractor):
        self.extractor = extractor
        self.all_triples: Set[Triple] = set()
        self.go_triples = []
        self.protein_triples = []

    def process_interpro_file(self, file_path: str) -> List[Tuple[str, str, str]]:
        """处理InterPro文件"""
        logger.info(f"Processing InterPro file: {file_path}")

        with open(file_path, 'r') as f:
            interpro_data = json.load(f)

        for interpro_id, description in tqdm(interpro_data.items(), desc="InterPro"):
            triples = self.extractor.extract_from_interpro(interpro_id, description)
            self.all_triples.update(triples)

        logger.info(f"Total unique triples so far: {len(self.all_triples)}")
        return [t.to_tuple() for t in self.all_triples]

    def process_gene_file(self, file_path: str) -> List[Tuple[str, str, str]]:
        """处理Gene文件"""
        logger.info(f"Processing Gene file: {file_path}")

        with open(file_path, 'r') as f:
            gene_data = json.load(f)

        # 处理每个基因
        for gene_id, gene_info in tqdm(gene_data.items(), desc="Genes"):
            if isinstance(gene_info, dict):
                gene_info['gene_id'] = gene_id
                triples = self.extractor.extract_from_gene(gene_info)
                self.all_triples.update(triples)

        logger.info(f"Total unique triples so far: {len(self.all_triples)}")
        return [t.to_tuple() for t in self.all_triples]

    def process_go_file(self, file_path: str) -> List[Tuple[str, str, str]]:
        """处理GO文件"""
        logger.info(f"Processing GO file: {file_path}")

        with open(file_path, 'r') as f:
            go_data = json.load(f)

        for go_id, go_info in tqdm(go_data.items(), desc="GO terms"):
            triples = self.extractor.extract_from_go(go_info, go_id)
            self.go_triples.append({"go_id":go_id, "triples":triples})
            self.all_triples.update(triples)

        logger.info(f"Total unique triples so far: {len(self.go_triples)}")
        return [t.to_tuple() for t in self.go_triples]

    def process_protein_file(self, file_path: str) -> List[Tuple[str, str, str]]:
        """处理蛋白质描述文件"""
        logger.info(f"Processing Protein file: {file_path}")
        test_text = pd.read_pickle(file_path)

        # 提取 proteins 和 uniport_text 两列，转为字典
        proteins = dict(zip(test_text['proteins'], test_text['uniport_text']))

        # 提取三元组
        for protein_id, description in tqdm(proteins.items(), desc="Proteins"):
            triples = self.extractor.extract_from_protein(protein_id, description)
            self.protein_triples.append({'proteiin_id': protein_id, 'triples': triples})
            self.all_triples.update(triples)

        logger.info(f"Total unique triples so far: {len(self.protein_triples)}")
        return [t.to_tuple() for t in self.all_triples]

    def get_all_triples(self) -> List[Tuple[str, str, str]]:
        """获取所有提取的三元组"""
        return [t.to_tuple() for t in self.all_triples]

    def save_sep_triples(self, triples_list, file_name) -> None:
        with open(file_name, 'w') as f:
            for line in tqdm(triples_list, desc=f"saving triples to {file_name}", total=len(triples_list)):
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(triples_list)} triples to {file_name}")

    def save_triples(self, output_path: str):
        """保存三元组到文件"""
        triples_list = self.get_all_triples()

        # 保存为JSON
        with open(output_path, 'w') as f:
            json.dump(triples_list, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(triples_list)} triples to {output_path}")

        # 同时保存为TSV格式（方便查看）
        tsv_path = output_path.replace('.json', '.tsv')
        with open(tsv_path, 'w') as f:
            f.write("head\trelation\ttail\n")
            for head, rel, tail in triples_list:
                f.write(f"{head}\t{rel}\t{tail}\n")

        logger.info(f"Saved triples to {tsv_path}")

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        triples = list(self.all_triples)

        # 统计实体
        entities = set()
        relations = set()

        for triple in triples:
            entities.add(triple.head)
            entities.add(triple.tail)
            relations.add(triple.relation)

        # 统计关系频率
        relation_counts = {}
        for triple in triples:
            relation_counts[triple.relation] = relation_counts.get(triple.relation, 0) + 1

        return {
            "total_triples": len(triples),
            "unique_entities": len(entities),
            "unique_relations": len(relations),
            "relation_distribution": dict(sorted(
                relation_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20])  # Top 20关系
        }


if __name__ == "__main__":
    # 示例用法
    print("Knowledge Extraction Module for Biomedical KG")