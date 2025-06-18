import torch
from PIL import Image
import clip

def test_image_equal_text():
    # 加载 CLIP 模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # 准备图像和文本
    image = preprocess(Image.open("pexels-aloismoubax-1562983_dog.jpg")).unsqueeze(0).to(device)
    texts = ["一只猫在草地上", "一只狗在沙滩上", "一辆红色的汽车"]
    text_tokens = clip.tokenize(texts).to(device)

    # 计算图像和文本特征
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

    # 归一化特征
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # 计算相似度
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(3)

    # 输出结果
    for value, index in zip(values, indices):
        print(f"{texts[index]}: {value.item():.2f}%")


def test_classify_clip():
    from transformers import CLIPModel, CLIPProcessor
    import torch
    from PIL import Image

    # 加载预训练的 CLIP 模型和处理器
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, processor = clip.load("ViT-B/32", device=device) #['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    # 加载图像
    image_path = 'pexels-aloismoubax-1562983_dog.jpg'
    image = Image.open(image_path).convert("RGB")

    # 提供文本描述
    # texts = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]
    texts = ["猫", "狗", "鸟"]

    # 使用 CLIPProcessor 对图像和文本进行预处理
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    # inputs = {"input_ids": tensor, "attention_mask": tensor, "pixel_values": tensor}
    

    # 将输入传递给模型进行推理
    outputs = model(**inputs)

    # 计算相似度
    logits_per_image = outputs.logits_per_image  # 获取图像-文本匹配的相似度
    probs = logits_per_image.softmax(dim=1)  # 归一化为概率

    # 输出结果
    for i, text in enumerate(texts):
        print(f"{text}: {probs[0, i].item():.2f}")
    
# test_image_equal_text()
test_classify_clip()