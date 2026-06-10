import requests
import os
import json
import pandas as pd
import time
from typing import Dict, List, Set
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


STRING_API = "https://string-db.org/api"
OUTPUT_FORMAT = "json"
# SPECIES = 10090  # mouse (10090 for mouse, 9606 for human)
DEFAULT_SCORE_THRESHOLD = 0.4  # Minimum confidence score

def extract_main_uniprot_id(accessions_str: str) -> str:
    """Extract main UniProt ID (first P开头 ID) from accessions string"""
    # Split by semicolon
    accessions = [acc.strip() for acc in accessions_str.split(';') if acc.strip()]
    # Find first accession that starts with 'P'
    # If no P开头, return first one
    if accessions:
        return accessions[0]
    return None

def collect_protein_ids_for_aspect(base_dir: str, aspect: str) -> set:
    """Collect all unique protein IDs from train/valid/test pickle files for a specific aspect"""
    protein_ids = set()

    for split in ['train_data.pkl', 'test_data.pkl', 'valid_data.pkl']:
        file_path = os.path.join(base_dir, aspect, split)
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} does not exist, skipping...")
            continue

        try:
            df = pd.read_pickle(file_path)
            if 'accessions' in df.columns:
                for accessions_str in df['accessions']:
                    if pd.isna(accessions_str):
                        continue
                    prot_id = extract_main_uniprot_id(str(accessions_str))
                    if prot_id:
                        protein_ids.add(prot_id)
            else:
                print(f"Warning: 'accessions' column not found in {file_path}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    print(f"[{aspect}] Collected {len(protein_ids)} unique protein IDs")
    with open(os.path.join(base_dir, aspect, "protein_ids.txt"), "w") as f:
        for protein_id in protein_ids:
            f.write(protein_id + "\n")
    return protein_ids

def load_protein_ids(base_dir, aspect):
    with open(os.path.join(base_dir, aspect, "protein_ids.txt"), "r") as f:
        protein_ids = []
        for line in f:
            protein_ids.append(line.strip())
    return protein_ids

MAX_RETRIES = 5
TIMEOUT = 30
BASE_DELAY = 0.5

def get_single_protein_interactions(uniprot_id: str):

    url = f"{STRING_API}/{OUTPUT_FORMAT}/network"

    params = {
        "identifiers": uniprot_id,
    }

    for attempt in range(MAX_RETRIES):

        try:
            response = requests.get(
                url,
                params=params,
                timeout=TIMEOUT
            )

            # 成功
            if response.status_code == 200:
                return {
                    "uniprot_id": uniprot_id,
                    "interactions": response.json()
                }

            # 404 不重试
            if response.status_code == 404:
                return {
                    "uniprot_id": uniprot_id,
                    "interactions": []
                }

            # 其它 HTTP 错误 → 重试
            print(f"HTTP {response.status_code} for {uniprot_id}, retry...")

        except requests.exceptions.Timeout:
            print(f"Timeout for {uniprot_id}, retry {attempt+1}")

        except requests.exceptions.ConnectionError:
            print(f"Connection error for {uniprot_id}, retry {attempt+1}")

        except Exception as e:
            print(f"Error {uniprot_id}: {e}")

        # 指数退避
        sleep_time = BASE_DELAY * (2 ** attempt)
        time.sleep(sleep_time)

    # 所有重试失败
    return {
        "uniprot_id": uniprot_id,
        "interactions": []
    }


def get_all_interactions_parallel(protein_ids: List[str], score_threshold: float = DEFAULT_SCORE_THRESHOLD,
                                 num_processes: int = 4) -> list:
    """Get interactions for all proteins in parallel with multiprocessing"""
    from multiprocessing import Pool

    print(f"Fetching interactions for {len(protein_ids)} proteins with {num_processes} processes...")

    # Prepare arguments for each process
    tasks = [(prot_id, score_threshold, 0.1) for prot_id in protein_ids]

    all_interactions = []
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.starmap(get_single_protein_interactions, tasks),
            total=len(tasks),
            desc="Fetching proteins"
        ))
    # print("results:", len(results))
    # Flatten results
    # for result in results:
    #     all_interactions.extend(result)
    # print('all_interactions:', len(all_interactions))
    # print('all_interactions:', all_interactions[0])
    return results

def get_all_interactions_parallel_new(protein_ids, max_workers=4):

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        for r in tqdm(
            executor.map(get_single_protein_interactions, protein_ids),
            total=len(protein_ids)
        ):
            results.append(r)

    return results


def collect_all_protein_ids(base_dir: str, num_processes:int) -> Set[str]:
    """Collect all unique protein IDs from all aspects (mf, bp, cc)"""
    aspects = ['mf', 'bp', 'cc']
    # all_proteins = set()

    for aspect in aspects:
        aspect_proteins = collect_protein_ids_for_aspect(base_dir, aspect)
        # all_proteins.update(aspect_proteins)
        if not aspect:
            print("No protein IDs found. Exiting...")
            return

        protein_list = list(aspect_proteins)
        # if max_proteins is not None:
        #     protein_list = protein_list[:max_proteins]

        print(f"\nFetching interactions for {len(protein_list)} proteins with {num_processes} parallel processes...")

        # Fetch all interactions in parallel
        all_interactions = get_all_interactions_parallel_new(protein_list, num_processes)
        print(f"\nCollected {len(all_interactions)} total interactions")

        # Collect all string IDs for mapping to UniProt
        # Build final result
        all_interactions_combined = []
        cnt = 0
        # Process and format interactions
        for uniport_interaction in all_interactions:
            # print("uniport_interaction:", uniport_interaction)
            uniprot_id = uniport_interaction['uniprot_id']
            interactions = uniport_interaction['interactions']
            interactions_combined = {'uniprot_id': uniprot_id, 'interactions': []}
            for interaction in interactions:
                p1 = interaction["preferredName_A"]
                p2 = interaction["preferredName_B"]
                s1 = interaction["stringId_A"]
                s2 = interaction["stringId_B"]
                score = interaction["score"]
                if score>=0.5:
                    interactions_combined['interactions'].append(interaction)
                    cnt += 1
            if len(interactions_combined['interactions']) > 0:
                all_interactions_combined.append(interactions_combined)
        print(f"\nTotal {aspect} interactions: {cnt} for {len(all_interactions_combined)} proteins")

        # Save to JSON
        save_interaction_network(all_interactions_combined, f"{aspect}_protein_interactions.json")

        # print(f"\nTotal collected {aspect} {len(all_proteins)} unique protein IDs across all aspects")

def save_interaction_network(interaction_data: List, output_path: str):
    """Save interaction network to JSON file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for interaction in interaction_data:
            f.write(json.dumps(interaction) + '\n')
        # json.dump(interaction_data, f, indent=2, ensure_ascii=False)
    print(f"Saved interaction network  to {output_path}")


def main():
    # Configuration
    base_data_dir = "../../deepgozero-main/data/"
    output_file = "protein_interactions.json"
    score_threshold = DEFAULT_SCORE_THRESHOLD
    num_processes = 4   # Number of parallel processes for requests
    max_proteins = 50 # Set to None to fetch all, or a number for testing

    # Collect all protein IDs from data
    print("Collecting protein IDs...")
    collect_all_protein_ids(base_data_dir, 4)




if __name__ == "__main__":
    main()