import pandas as pd
import click as ck
import json
from tqdm import tqdm

def load_interpro_names(entry_file):
    interpro_map = {}
    with open(entry_file) as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            interpro_id = parts[0]
            name = parts[2]
            interpro_map[interpro_id] = name
    print(f"example interpro_map[IPR000126]:{interpro_map['IPR000126']}")
    return interpro_map

def parse_accessions(acc):
    return [x.strip() for x in acc.split(";") if x.strip()]

def parse_string_ids(ids):

    if isinstance(ids, list):
        return ids
    return [ids]

def preprocess_dataframe(df, interpro_map):

    rows = []

    for _, r in tqdm(df.iterrows()):
        ipr_text_list = []
        interpros = r["interpros"]
        for ipr in interpros:
            desc = interpro_map.get(ipr,'')
            ipr_text_list.append(f"{ipr} ({desc})")

        rows.append({
            "protein_name": r["proteins"],
            "sequence": r["sequences"],
            "uniprot_ids": parse_accessions(r["accessions"]),
            "string_ids": parse_string_ids(r["string_ids"]),
            "interpros": interpros,
            "interpro_names": ' '.join(ipr_text_list),
            "go_terms": r["prop_annotations"]
        })

    return pd.DataFrame(rows)

@ck.command()
@ck.option(
    '--ont', '-ont', default='mf',
    help='Prediction model')
def main(ont):
    train = pd.read_pickle(f"../deepgozero-main/data/{ont}/train_data.pkl")
    valid = pd.read_pickle(f"../deepgozero-main/data/{ont}/valid_data.pkl")
    test = pd.read_pickle(f"../deepgozero-main/data/{ont}/test_data.pkl")

    interpro_map = load_interpro_names("../deepgozero-main/data/entry.list")

    train = preprocess_dataframe(train, interpro_map)
    valid = preprocess_dataframe(valid, interpro_map)
    test = preprocess_dataframe(test, interpro_map)
    print(test.head(1))
    convert(train, f"data/{ont}/train_instruct.json")
    convert(valid, f"data/{ont}/valid_instruct.json")
    convert(test, f"data/{ont}/test_instruct.json")

    # train.to_json(f"data/{ont}/train_processed.json", orient="records")
    # valid.to_json(f"data/{ont}/valid_processed.json", orient="records")
    # test.to_json(f"data/{ont}/test_processed.json", orient="records")


def build_prompt(row):
    seq = row["sequence"][:800]
    prompt=f"""Protein name: {row["protein_name"]}. InterPro domains: {row["interpro_names"]}. Return a list of GO terms."""
    # interpros = ", ".join(row["interpro_names"])
#     prompt = f""" You are a bioinformatics expert. Predict Gene Ontology (GO) functional annotations for the given protein.
#             Requirements:
#             - Only output valid GO IDs (format: GO:XXXXXXX)
#             - Multiple labels allowed
#             - Output as a comma-separated list
#             Protein name: {row["protein_name"]}
#             InterPro domains: {row["interpro_names"]}
#             Return a list of GO terms.
#         """
    return prompt

def build_output(row):
    return ", ".join(row["go_terms"])


def convert(df, output_file):
    # df = pd.read_json(input_file)
    records = []
    for _, r in tqdm(df.iterrows()):
        records.append({
            "instruction": build_prompt(r),
            "output": build_output(r)
        })
    with open(output_file, "w") as f:
        json.dump(records, f, indent=2)


if __name__ == "__main__":
    main()