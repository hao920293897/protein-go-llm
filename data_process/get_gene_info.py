import requests
import xml.etree.ElementTree as ET
import json
import os
import pickle
import pandas as pd
import time
import multiprocessing
from multiprocessing import Pool
from typing import Dict, Tuple
from tqdm import tqdm

def collect_all_gene_ids(base_dir: str) -> list:
    """Collect all unique InterPro IDs from train/valid/test pickle files in mf/bp/cc directories"""
    gene_ids = set()
    aspects = ['mf', 'bp', 'cc']

    for aspect in aspects:
        for split in ['train_data.pkl', 'test_data.pkl', 'valid_data.pkl']:
            file_path = os.path.join(base_dir, aspect, split)
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} does not exist, skipping...")
                continue

            try:
                df = pd.read_pickle(file_path)
                if 'genes' in df.columns:
                    for gene_id in df['genes']:
                        gene_ids.add(gene_id)
                else:
                    print(f"Warning: 'interpros' column not found in {file_path}")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    print(f"Collected {len(gene_ids)} unique Gene IDs")
    with open("gene_ids.txt", "w") as f:
        for gene in gene_ids:
            f.write(gene + "\n")
    return list(gene_ids)

def load_gene_ids():
    with open("gene_ids.txt", "r") as f:
        gene_ids = f.read().splitlines()
    return gene_ids

import requests
import xml.etree.ElementTree as ET
import time
import random


def fetch_gene_info_batch(gene_ids):
    """Fetch gene information for a batch of gene IDs from NCBI"""

    # ⚠️ 限制batch大小（NCBI建议）
    gene_ids = gene_ids[:200]

    ids = ",".join(map(str, gene_ids))

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    params = {
        "db": "gene",
        "id": ids,
        "retmode": "xml"
    }

    # 重试机制
    max_retries = 3

    for attempt in range(max_retries):

        try:
            r = requests.get(url, params=params, timeout=30)

            if r.status_code == 200:
                break

            print(f"Retry {attempt+1}: HTTP {r.status_code}")

        except Exception as e:
            print(f"Retry {attempt+1}: {e}")

        # 指数退避 + 随机sleep
        time.sleep(0.5 * (attempt + 1) + random.random() * 0.5)

    else:
        print(f"Failed batch: {ids}")
        return []

    # 解析XML
    try:
        root = ET.fromstring(r.text)
    except Exception as e:
        print(f"Error parsing XML for batch {ids}: {e}")
        return []

    genes = []

    for gene in root.findall(".//Entrezgene"):

        gene_id = gene.findtext(".//Gene-track_geneid")
        symbol = gene.findtext(".//Gene-ref_locus")
        desc = gene.findtext(".//Gene-ref_desc")
        summary = gene.findtext(".//Entrezgene_summary")

        genes.append({
            "gene_id": gene_id,
            "symbol": symbol,
            "description": desc,
            "summary": summary
        })

    # ⚠️ 限速（非常重要）
    time.sleep(0.34 + random.random() * 0.2)

    return genes

def fetch_all_gene_info_parallel(gene_ids: list, num_processes: int = 4, batch_size: int = 10) -> Dict[str, dict]:
    """Fetch all gene information from NCBI in parallel with multiprocessing"""
    gene_list = list(gene_ids)
    # Split into batches
    batches = [gene_list[i:i + batch_size] for i in range(0, len(gene_list), batch_size)]
    all_gene_info = {}

    print(f"Fetching {len(gene_list)} genes in {len(batches)} batches with {num_processes} processes...")
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(fetch_gene_info_batch, batches),
            total=len(batches),
            desc="Fetching gene info"
        ))

    for batch_result in results:
        for gene in batch_result:
            if gene['gene_id'] is not None:
                all_gene_info[gene['gene_id']] = gene

    missing = [gene_id for gene_id in gene_ids if str(gene_id) not in all_gene_info]
    if missing:
        print(f"Warning: {len(missing)} gene IDs could not be fetched: {missing[:10]}...")

    return all_gene_info


def fetch_all_gene_info_serial(gene_ids: list, batch_size: int = 10, delay: float = 0.3) -> Dict[str, dict]:
    """Fetch all gene information serially from NCBI with progress bar"""
    gene_list = list(gene_ids)
    # Split into batches
    batches = [gene_list[i:i + batch_size] for i in range(0, len(gene_list), batch_size)]
    all_gene_info = {}

    for batch in tqdm(batches, desc="Fetching gene info"):
        results = fetch_gene_info_batch(batch)
        for gene in results:
            if gene['gene_id'] is not None:
                all_gene_info[gene['gene_id']] = gene
        time.sleep(delay)

    missing = [gene_id for gene_id in gene_ids if str(gene_id) not in all_gene_info]
    if missing:
        print(f"Warning: {len(missing)} gene IDs could not be fetched: {missing[:10]}...")

    return all_gene_info


def save_gene_info(gene_info: Dict[str, dict], output_path: str):
    """Save gene information to JSON file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(gene_info, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(gene_info)} gene entries to {output_path}")


def load_gene_info(input_path: str) -> Dict[str, dict]:
    """Load gene information from JSON file"""
    with open(input_path, 'r', encoding='utf-8') as f:
        gene_info = json.load(f)
    print(f"Loaded {len(gene_info)} gene entries from {input_path}")
    return gene_info


def main():
    # Configuration
    # base_data_dir = "/Users/lhao/workspace/data/data/"  # Change this to your data directory containing mf/bp/cc
    base_data_dir = "../../deepgozero-main/data/"
    output_file = "gene_info.json"

    # Collect all gene IDs
    all_gene_ids = load_gene_ids()

    if not all_gene_ids:
        print("No gene IDs found. Exiting...")
        return

    # Fetch gene info from NCBI API
    use_parallel = True
    num_processes = 4  # Number of parallel processes
    batch_size = 10    # Number of genes per batch
    if use_parallel:
        gene_info = fetch_all_gene_info_parallel(all_gene_ids, num_processes, batch_size)
    else:
        gene_info = fetch_all_gene_info_serial(all_gene_ids, batch_size)

    # Save result
    save_gene_info(gene_info, output_file)


if __name__ == "__main__":
    main()
# print(fetch_gene_info([16542]))