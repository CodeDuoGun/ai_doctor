import spacy

# 加载 spaCy 英文模型（如处理中文请加载相应的中文模型）
nlp = spacy.load("zh_core_web_sm")

def classify_entity(entity_text, sentence_text):
    """
    根据句子中的关键词，对人名进行角色分类：
    - 包含 "Dr.", "Doctor" 或 "医生" 的句子认为该人是医生；
    - 包含 "patient", "患者" 或 "病人" 的句子认为该人是患者；
    - 否则标记为 unknown。
    """
    sent_lower = sentence_text.lower()
    if "dr." in sent_lower or "doctor" in sent_lower or "医生" in sentence_text:
        return "doctor"
    elif "patient" in sent_lower or "患者" in sentence_text or "病人" in sentence_text:
        return "patient"
    else:
        return "unknown"

def process_text(text):
    """
    处理输入文本：
      1. 使用 NER 提取文本中的 PERSON 实体；
      2. 根据所在句子的上下文判断角色；
      3. 返回包含人名和对应角色的列表。
    """
    doc = nlp(text)
    results = []
    for ent in doc.ents:
        # 只对标注为人名的实体进行处理
        if ent.label_ == "PERSON":
            # 获取实体所在的句子
            sentence = ent.sent.text
            role = classify_entity(ent.text, sentence)
            results.append({"name": ent.text, "role": role, "context": sentence})
    return results

if __name__ == "__main__":
    # 示例文本：包含英文和中文提示（注意中文部分 spaCy 英文模型可能无法正确识别）
    text = ("医生李华正在看望患者王明。")
    
    entities = process_text(text)
    
    # 输出结果
    for item in entities:
        print(f"Name: {item['name']}, Role: {item['role']}")
        print(f"Context: {item['context']}")
        print("-" * 50)
