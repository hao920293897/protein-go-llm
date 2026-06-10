import json
import torch
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import click as ck
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

class InstructDataset(Dataset):
    def __init__(self, instructs):
        self.data = instructs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
class DeepGoLLM:

    def __init__(
        self,
        model_name="Qwen/Qwen2-7B-Instruct",
        device="cuda:0",
        max_new_tokens=64,
        few_shot_k=0,
        examples=None
    ):
        self.device = device
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.few_shot_k = few_shot_k
        self.examples = examples[:few_shot_k] if few_shot_k>0 else []

        # ===== load model =====
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(device)
        # ⚠️ 必须设置
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # =========================
    # 🔹 Prompt构造
    # =========================
    def build_prompt(self, prompt):
        few_shots = ''
        if self.few_shot_k > 0:
            for ex in self.examples[:self.few_shot_k]:
                few_shots += f"""
                Example:
                Protein domains: {ex['input']}
                GO terms:{ex['output']}
                """

        # ===== 当前样本 =====
        return few_shots+prompt

    def collate_fn(self, batch):
        prompts = []
        labels = []
        for sample in batch:
            prompt = self.build_prompt(sample['instruction'])
            prompts.append(prompt)
            labels.append(sample['output'])
        return prompts, labels

    # =========================
    # 🔹 生成
    # =========================
    def generate(self, prompts):
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0
            )
        decoded = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )
        return decoded

    # =========================
    # 🔹 提取GO
    # =========================
    def extract_go_terms(self, text):
        return list(set(re.findall(r"GO:\d{7}", text)))

    # =========================
    # 🔹 GO → one-hot
    # =========================
    def go_to_onehot(self, go_terms, terms_dict):

        vec = np.zeros(len(terms_dict), dtype=np.float32)

        for go in go_terms:
            if go in terms_dict:
                vec[terms_dict[go]] = 1

        return vec

    # =========================
    # 🔹 单样本预测
    # =========================
    def predict_one(self, row, terms_dict, ipr_desc_dict):

        ipr_text = self.build_ipr_text(row.interpros, ipr_desc_dict)
        prompt = self.build_prompt(ipr_text)
        output = self.generate([prompt])[0]
        go_terms = self.extract_go_terms(output)
        return self.go_to_onehot(go_terms, terms_dict)

    # =========================
    # 🔹 批量预测（推荐）
    # =========================
    def predict_dataset(
            self,
            test_instructs,
            terms_dict,
            batch_size=4,
            num_workers=0
    ):
        dataset = InstructDataset(test_instructs)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: self.collate_fn(x)
        )
        all_preds = []
        results = []
        # ===== 进度条 =====
        for batch_prompts, lables in tqdm(dataloader, desc="Predicting"):
            outputs = self.generate(batch_prompts)
            for offset, out in enumerate(outputs):
                go_terms = self.extract_go_terms(out)
                vec = self.go_to_onehot(go_terms, terms_dict)
                all_preds.append(vec)
                results.append({"instruction":batch_prompts[offset], "labels":lables[offset], "preds":out})

        return np.array(all_preds), results

def build_labels(df, terms_dict):
    labels = np.zeros((len(df), len(terms_dict)), dtype=np.float32)
    for i, row in enumerate(df.itertuples()):
        for go in row.prop_annotations:
            if go in terms_dict:
                labels[i, terms_dict[go]] = 1
    return labels

LLM_PATH = {
    'llama3': '/data/shared/model/llama-3.1-8b-instruct/',
    'qwen': '/data/shared/model/Qwen2.5-7B-Instruct/'
}
@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Prediction model')
@ck.option(
    '--ont', '-ont', default='mf',
    help='Prediction model')
@ck.option(
    '--model_name', '-model', default='llama3',
    help='LLM name')
@ck.option(
    '--batch-size', '-bs', default=4,
    help='Batch size for training')
@ck.option(
    '--few-shot', '-k', default=3,
    help='few shots for instruct')
@ck.option(
    '--device', '-d', default='cuda:0',
    help='Device')

def main(data_root, ont, model_name, batch_size, few_shot, device):

    # data_root = "data"
    # ont = "mf"

    # ===== load =====
    test_df = pd.read_pickle(f"{data_root}/{ont}/test_data.pkl")
    terms_df = pd.read_pickle(f"{data_root}/{ont}/terms_zero_10.pkl")
    entry_file = f'{data_root}/entry.list'
    train_instruction = json.load(open(f"data/{ont}/train_instruct.json"))
    test_instruction = json.load(open(f"data/{ont}/test_instruct.json"))


    terms = terms_df["gos"].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}


    # ===== InterPro描述 =====
    ipr_desc_dict = load_interpro_names(entry_file)

    # ===== few-shot examples =====
    # examples = train_instruction[:5]

    # ===== 初始化模型 =====
    model = DeepGoLLM(
        model_name=LLM_PATH[model_name],
        few_shot_k=few_shot,  # 🔥 改这里
        examples=train_instruction,
        device=device
    )

    # ===== 预测 =====
    preds = model.predict_dataset(
        test_instruction,
        terms_dict,
        batch_size=batch_size
    )

    # ===== 保存（对接evaluate.py）=====
    test_df["preds"] = list(preds)
    test_df.to_pickle(f"{data_root}/{ont}/predictions_llm.pkl")

    # ===== 评测 =====
    # labels = build_labels(test_df, terms_dict)
    # evaluate_predictions(preds, labels)

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

    return interpro_map

if __name__ == "__main__":
    main()