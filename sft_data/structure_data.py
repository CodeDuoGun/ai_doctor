import json
from utils.log import logger

def chunk_text(text, max_length=1500):
    paragraphs = text.split("\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) < max_length:
            current += para + "\n"
        else:
            chunks.append(current.strip())
            current = para + "\n"
    if current:
        chunks.append(current.strip())
    return chunks

def call_gpt_structure(text_chunk):
    system_prompt = (
        "你是一个专业医学教材编辑助手。请从以下教材片段中提取结构化训练样本，"
        "格式如下：{\"instruction\": ..., \"input\": \"\", \"output\": ...}。\n"
        "你可以从医学术语、检查项目、疾病解释、治疗方案等角度提问。\n"
        "一次生成多个样本，每个样本单独输出，生成结果为JSON数组。"
    )

    user_prompt = f"医学教材内容如下：\n{text_chunk}\n\n请生成结构化问答样本："

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
    )
    
    content = response["choices"][0]["message"]["content"]
    try:
        data = json.loads(content)
        return data if isinstance(data, list) else [data]
    except Exception:
        print("⚠️ GPT输出不是有效JSON，原始输出如下：\n", content)
        return []

def process_text_file(text_path, out_path):
    with open(text_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    chunks = chunk_text(full_text)
    all_samples = []

    for idx, chunk in enumerate(chunks):
        print(f"⏳ 正在处理第 {idx+1}/{len(chunks)} 块...")
        samples = call_gpt_structure(chunk)
        all_samples.extend(samples)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 完成：保存到 {out_path}")
