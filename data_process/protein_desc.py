#将下述代码修改为蛋白质的描述获取，区别：
## 数据中蛋白质的id列为accessions，数据内容：P35918; C5H7S5; Q8VCD0; 取P开头的主ID即可， 接口https://rest.uniprot.org/uniprotkb/search?query=P35918&format=json，
## 最后的数据不要把mf/cc/bp的下的蛋白质数据整合到一起，分开获取并分别存储在对应的目录
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


def extract_main_uniprot_id(accessions_str: str) -> str:
    """Extract main UniProt ID (first P开头 ID) from accessions string"""
    # Split by semicolon
    accessions = [acc.strip() for acc in accessions_str.split(';') if acc.strip()]
    # Find first accession that starts with 'P'
    for acc in accessions:
        if acc.startswith('P'):
            return acc
    # If no P开头, return first one
    if accessions:
        return accessions[0]
    return None


def collect_protein_ids_for_aspect(base_dir: str, aspect: str) -> set:
    """Collect all unique protein IDs from train/valid/test pickle files for a specific aspect"""
    protein_ids = set()

    for split in ['train_data.pkl', 'test_data.pkl', 'valid.pkl']:
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
    return protein_ids


def fetch_protein_description(uniprot_id: str) -> tuple:
    """Fetch protein description from UniProt API"""
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        'query': uniprot_id,
        'format': 'json'
    }

    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        description = None
        if 'results' in data and len(data['results']) > 0:
            result = data['results'][0]
            # Get description from different fields
            desc_parts = []

            # Add protein name
            if 'proteinDescription' in result:
                pd = result['proteinDescription']
                if 'recommendedName' in pd:
                    rec_name = pd['recommendedName']
                    if 'fullName' in rec_name and 'value' in rec_name['fullName']:
                        desc_parts.append(rec_name['fullName']['value'])

                # Alternative names
                if 'alternativeNames' in pd:
                    for alt in pd['alternativeNames']:
                        if 'fullName' in alt and 'value' in alt['fullName']:
                            desc_parts.append(alt['fullName']['value'])

            # Add function description
            if 'comments' in result:
                for comment in result['comments']:
                    if comment.get('commentType') == 'FUNCTION':
                        if 'texts' in comment:
                            for text in comment['texts']:
                                desc_parts.append(text.get('value', ''))

            # Add gene info
            if 'genes' in result:
                for gene in result['genes']:
                    if 'geneName' in gene and 'value' in gene['geneName']:
                        desc_parts.append(f"Gene: {gene['geneName']['value']}")

            # Combine all description
            if desc_parts:
                description = ' '.join(desc_parts)
            else:
                # If no description found, just use ID
                description = uniprot_id

        return uniprot_id, description
    except Exception:
        return uniprot_id, None


def fetch_all_protein_descriptions_parallel(protein_ids: set, num_processes: int = 8) -> Dict[str, str]:
    """Fetch all protein descriptions in parallel with multiprocessing"""
    protein_list = list(protein_ids)
    all_descriptions = {}

    print(f"Fetching {len(protein_list)} protein descriptions with {num_processes} processes...")

    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(fetch_protein_description, protein_list),
            total=len(protein_list),
            desc="Fetching protein descriptions"
        ))

    for uniprot_id, desc in results:
        if desc is not None:
            all_descriptions[uniprot_id] = desc

    missing = [uniprot_id for uniprot_id, desc in results if desc is None]
    if missing:
        print(f"Warning: {len(missing)} proteins could not be fetched: {missing[:10]}...")

    return all_descriptions


def fetch_all_protein_descriptions_serial(protein_ids: set, delay: float = 0.5) -> Dict[str, str]:
    """Fetch all protein descriptions serially with progress bar"""
    all_descriptions = {}
    missing = []

    for uniprot_id in tqdm(protein_ids, desc="Fetching protein descriptions"):
        uniprot_id, desc = fetch_protein_description(uniprot_id)
        if desc is not None:
            all_descriptions[uniprot_id] = desc
        else:
            missing.append(uniprot_id)
        time.sleep(delay)

    if missing:
        print(f"Warning: {len(missing)} proteins could not be fetched: {missing[:10]}...")

    return all_descriptions


def save_protein_descriptions(descriptions: Dict[str, str], output_path: str):
    """Save protein descriptions to JSON file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(descriptions, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(descriptions)} protein descriptions to {output_path}")


def load_protein_descriptions(input_path: str) -> Dict[str, str]:
    """Load protein descriptions from JSON file"""
    with open(input_path, 'r', encoding='utf-8') as f:
        descriptions = json.load(f)
    print(f"Loaded {len(descriptions)} protein descriptions from {input_path}")
    return descriptions


def process_aspect(base_dir: str, aspect: str, output_dir: str, use_parallel: bool = True, num_processes: int = 8):
    """Process a single aspect (mf/bp/cc) and save descriptions separately"""
    print(f"\n{'='*60}")
    print(f"Processing aspect: {aspect}")
    print('='*60)

    # Collect protein IDs
    protein_ids = collect_protein_ids_for_aspect(base_dir, aspect)

    if not protein_ids:
        print(f"No protein IDs found for {aspect}, skipping...")
        return

    # Fetch descriptions
    if use_parallel:
        descriptions = fetch_all_protein_descriptions_parallel(protein_ids, num_processes)
    else:
        descriptions = fetch_all_protein_descriptions_serial(protein_ids)

    # Save to output directory with aspect name
    output_file = os.path.join(output_dir, f"{aspect}_protein_descriptions.json")
    save_protein_descriptions(descriptions, output_file)


def main():
    # Configuration
    base_data_dir = "../../deepgozero-main/data/"  # Base directory containing mf/bp/cc
    output_dir = base_data_dir  # Save to the same directory structure
    aspects = ['mf', 'bp', 'cc']
    use_parallel = True
    num_processes = 8  # Number of parallel processes

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each aspect separately
    for aspect in aspects:
        process_aspect(base_data_dir, aspect, output_dir, use_parallel, num_processes)

    print("\nAll processing completed!")


if __name__ == "__main__":
    main()