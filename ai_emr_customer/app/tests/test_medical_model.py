import sys
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

ziya_model_dir = ""  # ziya模型合并后的路径

model = LlamaForCausalLM.from_pretrained(ziya_model_dir, device_map='auto')
tokenizer = LlamaTokenizer.from_pretrained(ziya_model_dir)
model = PeftModel.from_pretrained(model, "shibing624/ziya-llama-13b-medical-lora")
device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_prompt(instruction):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:{instruction}\n\n### Response: """


sents = ['一岁宝宝发烧能吃啥药', "who are you?"]
for s in sents:
    q = generate_prompt(s)
    inputs = tokenizer(q, return_tensors="pt")
    inputs = inputs.to(device=device)

    generate_ids = model.generate(
        **inputs,
        max_new_tokens=120, 
        do_sample=True, 
        top_p=0.85, 
        temperature=1.0, 
        repetition_penalty=1.0
    )

    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
    print(output)
    print()

