"""
与原有 CAMEL 代码集成的接口
直接替换原有的 refine_predictions 函数
"""

import numpy as np
import pandas as pd
from typing import List, Dict
import click as ck
from tqdm import tqdm
import json

from go_agent_langgraph import (
    LLMDrivenProteinRefinement,
    DataManager,
)
from data_process.ontology import Ontology


def refine_predictions_langgraph(
        model_name: str,
        go: Ontology,
        ont: str,
        terms: List[str],
        terms_dict: Dict[str, int],
        row: pd.Series,
        data_manager: DataManager
) -> None:
    """
    使用 LangGraph 精炼预测 - 直接替换原有函数

    这个函数签名与原有的 refine_predictions 完全一致,
    可以无缝替换
    """

    # 初始化 LangGraph 精炼器
    refiner = LLMDrivenProteinRefinement(go, data_manager)

    # 转换数据格式
    protein_data = {
        "proteins": row["proteins"],
        "genes": row["genes"],
        "sequences": row["sequences"],
        "interpros": row["interpros"],
        "orgs": row["orgs"],
        "uniprot_text": row.get("uniprot_text", ""),
        "preds": row["preds"].copy(),
        "diam_preds": row.get("diam_preds", {})
    }

    # 执行精炼
    updated_predictions = refiner.refine_all_predictions(
        protein_data,
        terms,
        terms_dict,
        score_threshold=0.1  # 只精炼分数 >= 0.1 的 term
    )

    # 更新原始数据
    row["preds"] = updated_predictions


@ck.command()
@ck.option('--run_number', type=int, default=0, help='Run number for output file naming.')
@ck.option('--model_name', type=ck.Choice(['gemini', 'gpt', 'qwen']), default='qwen',
           help='Model name (unused in LangGraph version).')
@ck.option('--data_dir', type=str, default='data', help='Data directory path.')
@ck.option('--score_threshold', type=float, default=0.1, help='Score threshold for refinement.')
def main_langgraph(run_number, model_name, data_dir, score_threshold):
    """
    使用 LangGraph 的主函数 - 完全替换原有 main
    """

    print(f"Running LangGraph refinement (run {run_number})")
    print(f"Score threshold: {score_threshold}")

    # 初始化
    data_manager = DataManager(data_dir=data_dir)
    go = Ontology('../deepgozero-main/data/go.obo', with_rels=True)

    for ont in ['mf', 'cc', 'bp']:
        print(f"\n{'=' * 60}")
        print(f"Processing ontology: {ont}")
        print(f"{'=' * 60}")

        # 加载数据
        df = pd.read_pickle(f'{data_dir}/{ont}/test_data_diam_with_text.pkl')
        terms_df = pd.read_pickle(f'../deepgozero-main/data/{ont}/terms_zero_10.pkl')
        terms = terms_df['terms'].values.tolist()
        terms_dict = {v: k for k, v in enumerate(terms)}

        # 初始化精炼器
        refiner = LLMDrivenProteinRefinement(go, data_manager)

        skipped = 0
        total_changes = 0

        for i in tqdm(range(len(df)), desc=f"Processing {ont}"):
            try:
                row = df.iloc[i]

                # 检查是否有注释
                prop_annotations = row['prop_annotations']
                terms_in_ont = [t for t in prop_annotations if t in terms]

                if len(terms_in_ont) == 0:
                    print(f"Skipping protein {i} - no annotations in {ont}")
                    skipped += 1
                    continue

                # 保存旧预测
                old_preds = row['preds'].copy()

                # 转换数据格式
                protein_data = {
                    "proteins": row["proteins"],
                    "genes": row["genes"],
                    "sequences": row["sequences"],
                    "interpros": row["interpros"],
                    "orgs": row["orgs"],
                    "uniprot_text": row.get("uniprot_text", ""),
                    "preds": row["preds"].copy(),
                    "diam_preds": row.get("diam_preds", {})
                }

                # 执行精炼
                updated_predictions = refiner.refine_all_predictions(
                    protein_data,
                    terms,
                    terms_dict,
                    score_threshold=score_threshold
                )

                # 回写到 DataFrame
                df.at[i, 'preds'] = updated_predictions

                # 统计变化
                changes = np.sum(np.abs(updated_predictions - old_preds) > 0.01)
                total_changes += changes

                if changes > 0:
                    print(f"Protein {i}: Updated {changes} predictions")

            except Exception as e:
                print(f"Error processing protein {i}: {e}")
                import traceback
                traceback.print_exc()

        # 保存结果
        processed = len(df) - skipped
        print(f"\n{'-' * 60}")
        print(f"Ontology {ont} Summary:")
        print(f"  Processed: {processed} proteins")
        print(f"  Skipped: {skipped} proteins")
        print(f"  Total changes: {total_changes}")
        print(f"  Avg changes per protein: {total_changes / processed:.2f}")
        print(f"{'-' * 60}\n")

        # 保存
        output_file = f'{data_dir}/{ont}/test_predictions_langgraph_run{run_number}.pkl'
        df.to_pickle(output_file, protocol=4)
        print(f"Saved to: {output_file}")


@ck.command()
@ck.option('--original_file', type=str, required=True, help='Original predictions file')
@ck.option('--langgraph_file', type=str, required=True, help='LangGraph refined predictions file')
@ck.option('--ont', type=str, required=True, help='Ontology (mf/bp/cc)')
def compare_results(original_file, langgraph_file, ont):
    """
    比较原始预测和 LangGraph 精炼后的结果
    """

    print(f"\n{'=' * 60}")
    print(f"Comparing Results for {ont}")
    print(f"{'=' * 60}\n")

    # 加载数据
    df_original = pd.read_pickle(original_file)
    df_langgraph = pd.read_pickle(langgraph_file)

    total_proteins = len(df_original)
    total_predictions = 0
    total_changes = 0
    score_increases = 0
    score_decreases = 0

    change_distribution = []

    for i in range(total_proteins):
        old_preds = df_original.iloc[i]['preds']
        new_preds = df_langgraph.iloc[i]['preds']

        diff = new_preds - old_preds
        changes = np.abs(diff) > 0.01

        total_predictions += len(old_preds)
        total_changes += np.sum(changes)
        score_increases += np.sum(diff > 0.01)
        score_decreases += np.sum(diff < -0.01)

        change_distribution.extend(diff[changes].tolist())

    # 统计
    print(f"Total Proteins: {total_proteins}")
    print(f"Total Predictions: {total_predictions}")
    print(f"Total Changes: {total_changes} ({total_changes / total_predictions * 100:.2f}%)")
    print(f"  Increases: {score_increases} ({score_increases / total_changes * 100:.2f}%)")
    print(f"  Decreases: {score_decreases} ({score_decreases / total_changes * 100:.2f}%)")
    print(f"\nChange Statistics:")
    print(f"  Mean change: {np.mean(change_distribution):+.4f}")
    print(f"  Median change: {np.median(change_distribution):+.4f}")
    print(f"  Std dev: {np.std(change_distribution):.4f}")
    print(f"  Max increase: {np.max(change_distribution):+.4f}")
    print(f"  Max decrease: {np.min(change_distribution):+.4f}")

    # 保存比较报告
    report = {
        "ontology": ont,
        "total_proteins": int(total_proteins),
        "total_predictions": int(total_predictions),
        "total_changes": int(total_changes),
        "change_percentage": float(total_changes / total_predictions * 100),
        "score_increases": int(score_increases),
        "score_decreases": int(score_decreases),
        "mean_change": float(np.mean(change_distribution)),
        "median_change": float(np.median(change_distribution)),
        "std_change": float(np.std(change_distribution))
    }

    output_file = f"comparison_report_{ont}.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {output_file}")


@ck.group()
def cli():
    """LangGraph Protein Refinement CLI"""
    pass


cli.add_command(main_langgraph)
cli.add_command(compare_results)

if __name__ == "__main__":
    cli()