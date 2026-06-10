"""
测试脚本 - 验证信息抽取效果
"""
import json
from data_process.knowledge_extract import KnowledgeExtractor, Triple


def test_interpro_extraction():
    """测试InterPro域抽取"""
    print("\n" + "=" * 80)
    print("Testing InterPro Domain Extraction")
    print("=" * 80)

    extractor = KnowledgeExtractor(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        api_base="http://localhost:8000/v1"
    )

    # 测试数据
    interpro_id = "IPR042473"
    description = """V-type immunoglobulin domain-containing suppressor of T-cell activation#@@#V-type immunoglobulin domain-containing suppressor of T-cell activation (VISTA) is a type I transmembrane protein consisting of a single N-terminal immunoglobulin (Ig) V domain, an approximately 30 amino acid (aa) stalk, a transmembrane domain, and a 95 aa cytoplasmic tail [[cite:PUB00093009]]. It is primarily expressed on hematopoietic cells and functions as both a ligand and a receptor [[cite:PUB00093008]]. It serves as a immune-checkpoint protein that hampers the generation of effective anti-tumor immunity. Besides cancer, VISTA also regulates the development of autoimmune and inflammatory diseases [[cite:PUB00093010], [cite:PUB00093011]]."""

    triples = extractor.extract_from_interpro(interpro_id, description)

    print(f"\nExtracted {len(triples)} triples:")
    for triple in triples:
        print(f"  ({triple.head}, {triple.relation}, {triple.tail})")

    return triples


def test_gene_extraction():
    """测试Gene抽取"""
    print("\n" + "=" * 80)
    print("Testing Gene Extraction")
    print("=" * 80)

    extractor = KnowledgeExtractor(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        api_base="http://localhost:8000/v1"
    )

    # 测试数据
    gene_data = {
        "gene_id": "56212",
        "symbol": "Rhog",
        "description": "ras homolog family member G",
        "summary": "Predicted to enable GTP binding activity; GTPase activity; and protein kinase binding activity. Acts upstream of or within regulation of ruffle assembly. Predicted to be active in several cellular components, including cytoplasmic vesicle; glutamatergic synapse; and postsynapse. Is expressed in cerebral cortex ventricular layer and meninges. Orthologous to human RHOG (ras homolog family member G). [provided by Alliance of Genome Resources, Apr 2025]"
    }

    triples = extractor.extract_from_gene(gene_data)

    print(f"\nExtracted {len(triples)} triples:")
    for triple in triples:
        print(f"  ({triple.head}, {triple.relation}, {triple.tail})")

    return triples


def test_go_extraction():
    """测试GO术语抽取"""
    print("\n" + "=" * 80)
    print("Testing GO Term Extraction")
    print("=" * 80)

    extractor = KnowledgeExtractor(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        api_base="http://localhost:8000/v1"
    )

    # 测试数据
    go_id = "GO:0000001"
    go_data = {
        "name": "mitochondrion inheritance",
        "definition": "The distribution of mitochondria, including the mitochondrial genome, into daughter cells after mitosis or meiosis, mediated by interactions between mitochondria and the cytoskeleton. [GOC:mcc, PMID:10873824, PMID:11389764]",
        "namespace": "biological_process"
    }

    triples = extractor.extract_from_go(go_data, go_id)

    print(f"\nExtracted {len(triples)} triples:")
    for triple in triples:
        print(f"  ({triple.head}, {triple.relation}, {triple.tail})")

    return triples


def test_protein_extraction():
    """测试蛋白质抽取"""
    print("\n" + "=" * 80)
    print("Testing Protein Extraction")
    print("=" * 80)

    extractor = KnowledgeExtractor(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        api_base="http://localhost:8000/v1"
    )

    # 测试数据
    protein_id = "ALKB1_MOUSE"
    description = """Protein ID: ALKB1_MOUSE | UniProt: P0CB42;
Description: Protein: Nucleic acid dioxygenase ALKBH1 | Gene: Alkbh1 | Organism: Mus musculus | Function: Dioxygenase that acts on nucleic acids, such as DNA and tRNA (PubMed:27027282, PubMed:27745969). Requires molecular oxygen, alpha-ketoglutarate and iron (PubMed:27027282). Mainly acts as a tRNA demethylase by removing N(1)-methyladenine from various tRNAs. Acts as a regulator of translation initiation and elongation in response to glucose deprivation. Located in nucleus."""

    triples = extractor.extract_from_protein(protein_id, description)

    print(f"\nExtracted {len(triples)} triples:")
    for triple in triples:
        print(f"  ({triple.head}, {triple.relation}, {triple.tail})")

    return triples


def validate_triples(triples):
    """验证三元组质量"""
    print("\n" + "=" * 80)
    print("Validating Triple Quality")
    print("=" * 80)

    issues = []

    for triple in triples:
        # 检查空值
        if not triple.head or not triple.relation or not triple.tail:
            issues.append(f"Empty field in triple: {triple}")

        # 检查过长的实体名
        if len(triple.head) > 200:
            issues.append(f"Head too long: {triple.head[:50]}...")
        if len(triple.tail) > 200:
            issues.append(f"Tail too long: {triple.tail[:50]}...")

        # 检查关系名格式
        if ' ' in triple.relation:
            issues.append(f"Relation contains spaces: {triple.relation}")

    if issues:
        print(f"\n⚠️  Found {len(issues)} issues:")
        for issue in issues[:10]:  # 只显示前10个
            print(f"  - {issue}")
    else:
        print("\n✅ All triples passed validation!")

    return len(issues) == 0


def analyze_relation_types(triples):
    """分析关系类型分布"""
    print("\n" + "=" * 80)
    print("Analyzing Relation Types")
    print("=" * 80)

    relation_counts = {}
    for triple in triples:
        rel = triple.relation
        relation_counts[rel] = relation_counts.get(rel, 0) + 1

    print(f"\nFound {len(relation_counts)} unique relations:")
    sorted_rels = sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)

    for rel, count in sorted_rels:
        print(f"  {rel}: {count}")

    return relation_counts


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("Knowledge Extraction Test Suite")
    print("=" * 80)

    all_triples = []

    try:
        # 测试各类抽取
        triples = test_interpro_extraction()
        all_triples.extend(triples)

        triples = test_gene_extraction()
        all_triples.extend(triples)

        triples = test_go_extraction()
        all_triples.extend(triples)

        triples = test_protein_extraction()
        all_triples.extend(triples)

        # 验证和分析
        validate_triples(all_triples)
        analyze_relation_types(all_triples)

        # 保存测试结果
        output = {
            "total_triples": len(all_triples),
            "triples": [t.to_tuple() for t in all_triples]
        }

        with open("test_output.json", 'w') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 80)
        print(f"✅ Test Complete! Extracted {len(all_triples)} triples")
        print("Results saved to: test_output.json")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()