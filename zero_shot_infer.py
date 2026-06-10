from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

model_name = "Qwen/Qwen2-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)


def generate(prompt):

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200
    )

    return tokenizer.decode(outputs[0])

data = json.load(open("data/test_instruct.json"))
for sample in data[:10]:
    pred = generate(sample["instruction"])
    print("Prediction")
    print(pred)