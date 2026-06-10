

import json
import os
import pickle
import pandas as pd
import requests
import time
import multiprocessing
from multiprocessing import Pool
from typing import Dict, Tuple
from tqdm import tqdm


def collect_all_interpro_ids(base_dir: str) -> list:
    """Collect all unique InterPro IDs from train/valid/test pickle files in mf/bp/cc directories"""
    interpro_ids = set()
    aspects = ['mf', 'bp', 'cc']

    for aspect in aspects:
        for split in ['train_data.pkl', 'test_data.pkl', 'valid_data.pkl']:
            file_path = os.path.join(base_dir, aspect, split)
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} does not exist, skipping...")
                continue

            try:
                df = pd.read_pickle(file_path)
                if 'interpros' in df.columns:
                    for interpros_list in df['interpros']:
                        for ipr in interpros_list:
                            interpro_ids.add(ipr)
                else:
                    print(f"Warning: 'interpros' column not found in {file_path}")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    print(f"Collected {len(interpro_ids)} unique InterPro IDs")
    with open('interpro_ids.txt', 'w') as f:
        for ipr_id in interpro_ids:
            f.write(ipr_id + '\n')
    return list(interpro_ids)

def load_interpro_ids():
    with open('interpro_ids.txt', 'r') as f:
        interpro_ids = f.read().splitlines()
    return interpro_ids

def fetch_interpro_description(ipr_id: str) -> tuple:
    """Fetch InterPro description from API"""
    base_url = "https://www.ebi.ac.uk/interpro/api/entry/InterPro/"
    url = f"{base_url}{ipr_id}/"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        data = data['metadata']
        # Combine all description text fields
        name = data['name']['name']
        if 'description' in data and isinstance(data['description'], list):
            all_text = []
            for desc_item in data['description']:
                if 'text' in desc_item:
                    # Remove HTML tags if needed, keep them here for now
                    text = desc_item['text']
                    all_text.append(text)
            combined_text = ' '.join(all_text)
            # Remove HTML tags
            combined_text = combined_text.replace('<p>', '').replace('</p>', ' ')
            combined_text = combined_text.replace('<ul>', '').replace('</ul>', ' ')
            combined_text = combined_text.replace('<li>', '').replace('</li>', ', ')
            combined_text = ' '.join(combined_text.split())
            return ipr_id, combined_text.strip(), name
        return ipr_id, None, name
    except Exception as e:
        return ipr_id, None, None


def fetch_all_interpro_descriptions_parallel(interpro_ids: list, num_processes: int = 8) -> Dict[str, str]:
    """Fetch all InterPro descriptions in parallel with multiprocessing"""
    interpro_list = list(interpro_ids)
    all_descriptions = {}

    print(f"Fetching {len(interpro_list)} InterPro descriptions with {num_processes} processes...")

    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(fetch_interpro_description, interpro_list),
            total=len(interpro_list),
            desc="Fetching InterPro descriptions"
        ))

    for ipr_id, desc, name in results:
        if desc is not None and name is not None:
            all_descriptions[ipr_id] = name +'#@@#'+ desc

    missing = [ipr_id for ipr_id, desc, _ in results if desc is None]
    if missing:
        print(f"Warning: {len(missing)} InterPro IDs could not be fetched: {missing[:10]}...")

    return all_descriptions


def fetch_all_interpro_descriptions_serial(interpro_ids: list, delay: float = 0.5) -> Dict[str, str]:
    """Fetch all InterPro descriptions serially with progress bar"""
    all_descriptions = {}
    missing = []

    for ipr_id in tqdm(interpro_ids, desc="Fetching InterPro descriptions"):
        ipr_id, desc, name = fetch_interpro_description(ipr_id)
        # print(f"interpro {ipr_id} {name}: {desc}")
        if desc is not None and name is not None:
            all_descriptions[ipr_id] = name+'#@@#'+desc
        else:
            missing.append(ipr_id)
        time.sleep(delay)

    if missing:
        print(f"Warning: {len(missing)} InterPro IDs could not be fetched: {missing[:10]}...")

    return all_descriptions


def save_interpro_descriptions(descriptions: Dict[str, str], output_path: str):
    """Save InterPro descriptions to JSON file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(descriptions, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(descriptions)} InterPro descriptions to {output_path}")


def load_interpro_descriptions(input_path: str) -> Dict[str, str]:
    """Load InterPro descriptions from JSON file"""
    with open(input_path, 'r', encoding='utf-8') as f:
        descriptions = json.load(f)
    print(f"Loaded {len(descriptions)} InterPro descriptions from {input_path}")
    return descriptions


def main():
    # Configuration
    # base_data_dir = "/Users/lhao/workspace/data/data/"  # Change this to your data directory containing mf/bp/cc
    base_data_dir = "../../deepgozero-main/data/"
    output_file = "interpro_descriptions.json"

    # Collect all InterPro IDs
    all_interpro_ids = collect_all_interpro_ids(base_data_dir)

    if not all_interpro_ids:
        print("No InterPro IDs found. Exiting...")
        return

    # Fetch descriptions from InterPro API
    use_parallel = True
    num_processes = 4  # Number of parallel processes
    if use_parallel:
        descriptions = fetch_all_interpro_descriptions_parallel(all_interpro_ids, num_processes)
    else:
        descriptions = fetch_all_interpro_descriptions_serial(all_interpro_ids)

    # Save result
    save_interpro_descriptions(descriptions, output_file)


if __name__ == "__main__":
    main()