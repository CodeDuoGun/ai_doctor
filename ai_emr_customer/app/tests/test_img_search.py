import clip
import torch
from PIL import Image
import os

# 加载模型和预处理方法
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 示例图片路径
image_paths = ["/Users/tangxueduo/Projects/sihui_nlp/app/data/img/pexels-aloismoubax-1562983_dog.jpg", "/Users/tangxueduo/Projects/sihui_nlp/app/data/img/pexels-zoujunlin-29996484_cat.jpg"]

# 对所有图片进行预处理并编码
image_tensors = [preprocess(Image.open(p)).unsqueeze(0).to(device) for p in image_paths]
image_input = torch.cat(image_tensors, dim=0)
with torch.no_grad():
    image_features = model.encode_image(image_input)

# 文本检索 query 示例
# text_descriptions = ["找一个狗的图片", "找一个猫的图片"]
text_descriptions = ["a dog picture", "a cat picture"]
text_tokens = clip.tokenize(text_descriptions).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)

# 归一化特征
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

# 计算相似度 [文本数, 图像数]
similarity = text_features @ image_features.T

# 输出结果
print("文本检索图片（Text -> Image）:")
for i, desc in enumerate(text_descriptions):
    top_image_idx = similarity[i].topk(1).indices[0].item()
    print(f"\"{desc}\" 最匹配的是图片：{image_paths[top_image_idx]}")

print("\n图片检索文本（Image -> Text）:")
for i, path in enumerate(image_paths):
    top_text_idx = similarity[:, i].topk(1).indices[0].item()
    print(f"图片 {path} 最匹配的是文本：\"{text_descriptions[top_text_idx]}\"")

