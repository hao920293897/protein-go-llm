## 已知mf/bp/cc不同GO术语目录下的train_data.pkl, test_data.pkl, valid.pkl均为pd.DataFrame，
## 读取上述文件内容，针对prop_annotations（数据内容：[GO:0004713, GO:0016310, GO:0002040, GO:0042592, GO:0008283, ...]）通过接口
## 如https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/GO:0004713,GO:0016310,GO:0001568 获取数据中所有GO术语的name及definition text，并保存问字典
import json
import os
import pickle
import pandas as pd
import requests
import time
from typing import Dict, Optional
from tqdm import tqdm


def collect_all_go_terms(base_dir: str) -> set:
    """Collect all unique GO terms from train/valid/test pickle files in mf/bp/cc directories"""
    go_terms = set()
    aspects = ['mf', 'bp', 'cc']

    for aspect in aspects:
        for split in ['train_data.pkl', 'test_data.pkl', 'valid_data.pkl']:
            file_path = os.path.join(base_dir, aspect, split)
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} does not exist, skipping...")
                continue

            try:
                df = pd.read_pickle(file_path)
                if 'prop_annotations' in df.columns:
                    for annotations in df['prop_annotations']:
                        for go_term in annotations:
                            go_terms.add(go_term)
                else:
                    print(f"Warning: 'prop_annotations' column not found in {file_path}")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    print(f"Collected {len(go_terms)} unique GO terms")
    return go_terms


def fetch_go_definitions_batch(go_terms: list) -> Dict[str, Dict]:
    """Fetch GO term definitions and names in batch using QuickGO API"""
    base_url = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/"
    go_str = ",".join(go_terms)
    url = f"{base_url}{go_str}"

    headers = {
        "Accept": "application/json"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        result = {}
        if 'results' in data:
            for term in data['results']:
                go_id = term['id']
                name = term.get('name', '')
                definition = term.get('definition', {}).get('text', '')
                result[go_id] = {
                    'name': name,
                    'definition': definition
                }
        return result
    except Exception as e:
        print(f"Error fetching batch {go_str[:50]}...: {e}")
        return {}


def fetch_all_go_definitions(go_terms: set, batch_size: int = 20, delay: float = 1.0) -> Dict[str, Dict]:
    """Fetch all GO definitions with batching and rate limiting"""
    go_list = list(go_terms)
    all_definitions = {}
    total_batches = (len(go_list) + batch_size - 1) // batch_size

    pbar = tqdm(total=total_batches, desc="Fetching GO definitions")
    for i in range(0, len(go_list), batch_size):
        batch = go_list[i:i + batch_size]
        batch_result = fetch_go_definitions_batch(batch)
        all_definitions.update(batch_result)
        pbar.update(1)

        if i + batch_size < len(go_list):
            time.sleep(delay)
    pbar.close()

    # Check for any missing terms
    missing = [g for g in go_terms if g not in all_definitions]
    if missing:
        print(f"Warning: {len(missing)} terms could not be fetched: {missing[:10]}...")

    return all_definitions


def save_go_definitions(definitions: Dict[str, Dict], output_path: str):
    """Save GO definitions to pickle file"""
    with open(output_path, 'w') as f:
        json.dump(definitions, f, indent=4)
    print(f"Saved {len(definitions)} GO definitions to {output_path}")


def load_go_definitions(input_path: str) -> Dict[str, Dict]:
    """Load GO definitions from pickle file"""
    with open(input_path, 'rb') as f:
        definitions = pickle.load(f)
    print(f"Loaded {len(definitions)} GO definitions from {input_path}")
    return definitions


def download_go_obo(output_path: str = "go.obo") -> None:
    """Download GO ontology from http://purl.obolibrary.org/obo/go.obo"""
    url = "https://purl.obolibrary.org/obo/go.obo"
    print(f"Downloading GO ontology from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(output_path, 'wb') as f:
        total_size = int(response.headers.get('content-length', 0))
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    print(f"GO ontology saved to {output_path}")


def parse_go_obo(obo_path: str = "go.obo") -> Dict[str, Dict]:
    """Parse go.obo file and extract GO term name and definition"""
    go_dict = {}
    current_id = None
    current_name = None
    current_def = None
    in_term = False

    print(f"Parsing {obo_path}...")
    with open(obo_path, 'r') as f:
        for line in tqdm(f, desc="Parsing terms"):
            line = line.strip()
            if line == "[Term]":
                if current_id is not None and current_name is not None:
                    go_dict[current_id] = {
                        'name': current_name,
                        'definition': current_def or ''
                    }
                in_term = True
                current_id = None
                current_name = None
                current_def = None
            elif line == "[Typedef]":
                in_term = False
            elif in_term and line.startswith("id: GO:"):
                current_id = line.split()[0]
            elif in_term and line.startswith("name: "):
                current_name = line[5:].strip()
            elif in_term and line.startswith("def: "):
                # definition is in quotes, extract it
                def_text = line[4:].strip()
                if def_text.startswith('"'):
                    end_quote = def_text.find('"', 1)
                    if end_quote > 0:
                        current_def = def_text[1:end_quote]
            elif in_term and line.startswith("alt_id: "):
                # Handle alternative IDs if needed
                pass

    # Add the last term
    if current_id is not None and current_name is not None:
        go_dict[current_id] = {
            'name': current_name,
            'definition': current_def or ''
        }

    print(f"Parsed {len(go_dict)} GO terms")
    return go_dict


def parse_go_obo_with_goatools(obo_path: str = "go.obo") -> Dict[str, Dict]:
    """Alternative parsing using goatools"""
    try:
        from goatools.obo_parser import GODag
    except ImportError:
        print("goatools not installed, falling back to manual parsing")
        return parse_go_obo(obo_path)

    print(f"Parsing {obo_path} with goatools...")
    go = GODag(obo_path, optional_attrs={'def'})
    go_dict = {}
    for go_id, term in tqdm(go.items(), desc="Extracting terms"):
        go_dict[go_id] = {
            'name': term.name,
            'definition': term.defn if term.defn else '',
            'namespace': term.namespace,
        }
    print(f"Parsed {len(go_dict)} GO terms")
    return go_dict


def main():
    # Option 1: Fetch only terms from dataset using QuickGO API
    # Configuration
    base_data_dir = "/Users/lhao/workspace/data/data/"  # Change this to your data directory containing mf/bp/cc
    output_file = "go_definitions_all.pkl"

    # Collect all GO terms
    all_go_terms = collect_all_go_terms(base_data_dir)

    if not all_go_terms:
        print("No GO terms found. Exiting...")
        return

    # Fetch definitions from QuickGO
    definitions = fetch_all_go_definitions(all_go_terms)

    # Save result
    save_go_definitions(definitions, output_file)


def main_download_parse_obo(output_file: str = "go_definitions_obo.json", use_goatools: bool = True):
    """Main function to download go.obo and parse all GO terms"""
    obo_path = "/Users/lhao/workspace/data/data/go.obo"

    # Download if not exists
    if not os.path.exists(obo_path):
        download_go_obo(obo_path)

    # Parse
    if use_goatools:
        definitions = parse_go_obo_with_goatools(obo_path)
    else:
        definitions = parse_go_obo(obo_path)

    # Save
    save_go_definitions(definitions, output_file)


if __name__ == "__main__":
    # Uncomment the one you need:
    # main()  # Fetch from QuickGO for GO terms in your dataset
    main_download_parse_obo()  # Download and parse full go.obo
    # main()