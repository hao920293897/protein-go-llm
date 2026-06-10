"""
主运行脚本 - 从生物医学文本中提取知识三元组
"""
import argparse
import json
from pathlib import Path
import logging
from data_process.knowledge_extract import KnowledgeExtractor, BatchExtractor

# from knowledge_extractor import KnowledgeExtractor, BatchExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Extract knowledge triples from biomedical texts'
    )

    # 模型配置
    parser.add_argument(
        '--model_name',
        type=str,
        default='Qwen/Qwen2.5-7B-Instruct',
        help='Model name (default: Qwen/Qwen2.5-7B-Instruct)'
    )
    parser.add_argument(
        '--api_base',
        type=str,
        default='http://localhost:8000/v1',
        help='API base URL'
    )
    parser.add_argument(
        '--api_key',
        type=str,
        default='EMPTY',
        help='API key'
    )

    # 数据文件
    parser.add_argument(
        '--interpro_file',
        type=str,
        default='data/interpro_descriptions.json',
        help='InterPro descriptions file'
    )
    parser.add_argument(
        '--gene_file',
        type=str,
        default='data/gene_info.json',
        help='Gene information file'
    )
    parser.add_argument(
        '--go_file',
        type=str,
        default='data/go_terms.json',
        help='GO terms file'
    )
    parser.add_argument(
        '--protein_file',
        type=str,
        default='data/protein_descriptions.txt',
        help='Protein descriptions file'
    )

    # 输出配置
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output',
        help='Output directory'
    )

    # 处理选项
    parser.add_argument(
        '--sources',
        nargs='+',
        default=['interpro', 'gene', 'go', 'protein'],
        choices=['interpro', 'gene', 'go', 'protein'],
        help='Which sources to process'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of items to process (for testing)'
    )

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化抽取器
    logger.info("=" * 80)
    logger.info("Biomedical Knowledge Graph Triple Extraction")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"API Base: {args.api_base}")
    logger.info(f"Sources: {', '.join(args.sources)}")
    logger.info("=" * 80)

    extractor = KnowledgeExtractor(
        model_name=args.model_name,
        api_base=args.api_base,
        api_key=args.api_key
    )

    batch_extractor = BatchExtractor(extractor)

    # 处理各类数据源
    if 'interpro' in args.sources and Path(args.interpro_file).exists():
        logger.info("\n" + "=" * 80)
        logger.info("Processing InterPro Domains")
        logger.info("=" * 80)
        batch_extractor.process_interpro_file(args.interpro_file)

        # 保存中间结果
        batch_extractor.save_triples( f"{args.output}/triples_interpro.json")

    if 'gene' in args.sources and Path(args.gene_file).exists():
        logger.info("\n" + "=" * 80)
        logger.info("Processing Genes")
        logger.info("=" * 80)
        batch_extractor.process_gene_file(args.gene_file)

        # 保存中间结果
        # batch_extractor.save_triples(f"{args.output}/triples_gene.json")

    if 'go' in args.sources and Path(args.go_file).exists():
        logger.info("\n" + "=" * 80)
        logger.info("Processing GO Terms")
        logger.info("=" * 80)
        batch_extractor.process_go_file(args.go_file)

        # 保存中间结果
        batch_extractor.save_sep_triples(batch_extractor.go_triples , f"{args.output}/triples_go.json")

        # batch_extractor.save_triples(output_dir / "triples_go.json")

    if 'protein' in args.sources and Path(args.protein_file).exists():
        logger.info("\n" + "=" * 80)
        logger.info("Processing Proteins")
        logger.info("=" * 80)
        batch_extractor.process_protein_file(args.protein_file)

        # 保存中间结果
        # batch_extractor.save_triples(output_dir / "triples_protein.json")
        batch_extractor.save_sep_triples(batch_extractor.protein_triples , f"{args.output}/triples_protein.json")


    # 保存最终结果
    logger.info("\n" + "=" * 80)
    logger.info("Saving Final Results")
    logger.info("=" * 80)

    final_output = output_dir / "knowledge_graph_triples.json"
    batch_extractor.save_triples(final_output)

    # 打印统计信息
    stats = batch_extractor.get_statistics()

    logger.info("\n" + "=" * 80)
    logger.info("Extraction Statistics")
    logger.info("=" * 80)
    logger.info(f"Total Triples: {stats['total_triples']}")
    logger.info(f"Unique Entities: {stats['unique_entities']}")
    logger.info(f"Unique Relations: {stats['unique_relations']}")
    logger.info("\nTop Relations:")
    for relation, count in list(stats['relation_distribution'].items())[:10]:
        logger.info(f"  {relation}: {count}")

    # 保存统计信息
    stats_path = output_dir / "statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"\nStatistics saved to: {stats_path}")
    logger.info(f"Final triples saved to: {final_output}")
    logger.info("=" * 80)
    logger.info("Extraction Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()