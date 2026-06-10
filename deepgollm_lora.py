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
from peft import LoraConfig, get_peft_model
import torch.optim as optim
from torch import nn

LLM_PATH = {
    'llama3': '/data/shared/model/llama-3.1-8b-instruct/',
    'qwen': '/data/shared/model/Qwen2.5-7B-Instruct/'
}

TASK_INSTRUCTION = """You are a bioinformatics expert. Predict Gene Ontology (GO) functional annotations for the given protein. Requirements: 
- Only output valid GO IDs (format: GO:XXXXXXX)
- Multiple labels allowed
- Output as a comma-separated list
"""

class DeepGoDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        prompt = self.build_prompt(sample["instruction"])
        # 🔥 关键：拼接 label

        return {
            "prompt": prompt,
            "output": sample["output"]
        }
    def build_prompt(self, sample):
        prompt = TASK_INSTRUCTION+""+sample+"\n GO terms:"
        return prompt

def avg_prompt_length(dataset, tokenizer):
    total_len = 0
    go_num = 0
    for i in range(len(dataset)):
        sample = dataset[i]
        prompt = sample["prompt"]
        output = sample["output"].split(', ')
        go_num += len(output)
        tokens = tokenizer(
            prompt,
            add_special_tokens=False
        )["input_ids"]

        total_len += len(tokens)

    return total_len / len(dataset), go_num/len(dataset)

def unified_collate_fn(
    batch,
    tokenizer,
    max_length=1024,
    is_train=True
):
    input_texts = []
    input_ids_list = []
    labels_list = []          # train用（tensor）
    raw_labels_list = []      # test用（文本）
    for item in batch:
        prompt = item["prompt"]
        if is_train:
            output = item["output"]
            full_text = prompt + output
        else:
            full_text = prompt
            raw_labels_list.append(item["output"])  # ⭐保存原始label

        input_texts.append(full_text)

        # =========================
        # tokenize
        # =========================
        full_ids = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            padding=True
        )["input_ids"]
        input_ids_list.append(full_ids)

        if is_train:
            split_token = "GO terms:"
            split_idx = full_text.find(split_token)
            assert split_idx != -1, "必须包含 GO terms:"
            prompt_text = full_text[:split_idx + len(split_token)]
            prompt_ids = tokenizer(
                prompt_text,
                add_special_tokens=False
            )["input_ids"]
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
            labels = labels[:len(full_ids)]
            labels_list.append(labels)

    # =========================
    # padding
    # =========================
    if is_train:
        batch_enc = tokenizer.pad(
            {
                "input_ids": input_ids_list,
            },
            padding=True,
            return_tensors="pt"
        )
        input_ids = batch_enc["input_ids"]

        # 手动 pad labels
        max_len = input_ids.shape[1]

        padded_labels = []
        for labels in labels_list:
            pad_len = max_len - len(labels)
            padded_labels.append(labels + [-100] * pad_len)

        labels_out = torch.tensor(padded_labels, dtype=torch.long)
    else:
        batch_enc = tokenizer.pad(
            {
                "input_ids": input_ids_list
            },
            padding=True,
            return_tensors="pt"
        )
        labels_out = raw_labels_list   # ⭐这里是文本！

    # =========================
    # 统一输出
    # =========================
    result = {
        "input_text": input_texts,
        "input_ids": batch_enc["input_ids"],
        "attention_mask": batch_enc["attention_mask"],
        "labels": labels_out
    }

    return result

class DeepGoLLM(nn.Module):

    def __init__(
        self,
        model_name="Qwen/Qwen2-7B-Instruct",
        device="cuda:0",
        max_new_tokens=64,
        use_lora=False   # ⭐新增
    ):
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.use_lora = use_lora

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        ).to(device)

        # ===== LoRA =====
        if use_lora:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none"
            )
            self.model = get_peft_model(self.model, lora_config)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, input_ids, attention_mask, labels=None):

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,  # 🔥 HuggingFace自动计算loss
        )

        loss = outputs.loss
        logits = outputs.logits

        return {
            "loss": loss,
            "logits": logits
        }

    def predict_dataset(
            self,
            test_data_loader,
            terms_dict,
    ):
        results = []
        # ===== 进度条 =====
        for batch in tqdm(test_data_loader, desc="Predicting"):
            inputs = {
                "input_ids": batch["input_ids"].to(self.device),
                "attention_mask": batch["attention_mask"].to(self.device),
            }
            labels = inputs['labels']
            outputs = self.generate(inputs)
            for offset, out in enumerate(outputs):
                go_terms = self.extract_go_terms(out)
                vec = self.go_to_onehot(go_terms, terms_dict)
                # all_preds.append(vec)
                labels_onehot = self.go_to_onehot(labels[offset].split(", "), terms_dict)
                results.append({"labels": labels[offset], "preds":', '.join(go_terms), 'preds_onehot': vec, 'labels_onehot': labels_onehot})
        return results

    def generate(self, inputs):
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=0.0
            )
        decoded = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )
        return decoded

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

def save_lora(self, path):
    self.model.save_pretrained(path)

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
    '--device', '-d', default='cuda:0',
    help='Device')
@ck.option(
    '--lr', '-lr', default=2e-5,
    help='learning rate')
@ck.option(
    '--epochs', '-ep', default=4,
    help='epochs')
def main(data_root, ont, model_name, batch_size, device, lr, epochs):

    # data_root = "data"
    # ont = "mf"

    # ===== load =====
    # test_df = pd.read_pickle(f"{data_root}/{ont}/test_data.pkl")
    terms_df = pd.read_pickle(f"../deepgozero-main/data/{ont}/terms_zero_10.pkl")
    train_instruction = json.load(open(f"data/{ont}/train_instruct.json"))
    test_instruction = json.load(open(f"data/{ont}/test_instruct.json"))

    terms = terms_df["gos"].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    train_dataset, test_dataset = DeepGoDataset(train_instruction), DeepGoDataset(test_instruction)
    print("train_data size:", train_dataset.__len__())
    print("test_data size:", test_dataset.__len__())

    model = DeepGoLLM(LLM_PATH[model_name], device=device, max_new_tokens=256, use_lora=True)
    train_avg_len, train_go_num = avg_prompt_length(train_dataset, model.tokenizer)
    test_avg_len, test_go_num = avg_prompt_length(test_dataset, model.tokenizer)
    print(f"训练集平均token长度: {train_avg_len:.2f}, {train_go_num:.2f}")
    print(f"测试集平均token长度：{test_avg_len:.2f}, {test_go_num:.2f}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: unified_collate_fn(x, model.tokenizer, is_train=True)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: unified_collate_fn(x, model.tokenizer, is_train=False)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in tqdm(range(epochs), desc="Training"):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"epoch:{epoch}"):
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device)
            }
            outputs = model(**inputs)

            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            print(f"Loss: {loss:.4f}")
        print(f"Epoch {epoch} Loss: {total_loss / len(train_loader):.4f}")

        results = model.predict_dataset(test_loader, terms_dict)

if __name__ == "__main__":
    main()