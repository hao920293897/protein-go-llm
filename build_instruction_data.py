import json
import pandas as pd


def build_prompt(row):
    seq = row["sequence"][:800]
    interpros = ", ".join(row["interpro_names"])
    prompt = f"""
            You are a bioinformatics expert.
            Predict Gene Ontology (GO) functional annotations for the given protein.
            Requirements:
            - Only output valid GO IDs (format: GO:XXXXXXX)
            - Multiple labels allowed
            - Output as a comma-separated list
            Protein name:
            {row["protein_name"]}
            InterPro domains:
            {interpros}
            UniProt IDs:
            {",".join(row["uniprot_ids"])}
            Return a list of GO terms.
        """
    return prompt

def build_output(row):
    return ", ".join(row["go_terms"])


def convert(input_file, output_file):
    df = pd.read_json(input_file)
    records = []
    for _, r in df.iterrows():
        records.append({
            "instruction": build_prompt(r),
            "output": build_output(r)
        })
    with open(output_file, "w") as f:
        json.dump(records, f, indent=2)


convert("data/train_processed.json", "data/train_instruct.json")
convert("data/valid_processed.json", "data/valid_instruct.json")
convert("data/test_processed.json", "data/test_instruct.json")