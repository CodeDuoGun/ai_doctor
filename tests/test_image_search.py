# 图像入库
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from elasticsearch import Elasticsearch
import base64
import os

# 初始化
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
es = Elasticsearch("http://localhost:9200")

INDEX_NAME = "test_image_search"

# 创建索引（含 dense_vector）
def create_index():
    if es.indices.exists(index=INDEX_NAME):
        return
    es.indices.create(
        index=INDEX_NAME,
        body={
            "mappings": {
                "properties": {
                    "image_url": {"type": "keyword"},
                    "image_vector": {
                        "type": "dense_vector",
                        "dims": 512,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "tags": {"type": "text"}
                }
            }
        }
    )

# 图像向量提取
def get_image_vector(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    return outputs[0].cpu().tolist()

# 图像入库
def index_images(folder):
    for fname in os.listdir(folder):
        if fname.endswith(".png") or fname.endswith(".jpg"):
            path = os.path.join(folder, fname)
            vector = get_image_vector(path)
            es.index(index=INDEX_NAME, body={
                "image_url": f"/static/images/{fname}",  # 或 S3/GCS URL
                "image_vector": vector,
                "tags": "组织架构图 公司结构图 部门关系图"
            })

create_index()
index_images("org_chart_images")  # 本地目录

#¥ 对话图像检索
def get_text_vector(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
    return outputs[0].cpu().tolist()

def search_similar_images(query_text, top_k=3):
    vector = get_text_vector(query_text)

    # 构建 ES 向量查询 
    body = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'image_vector') + 1.0",
                    "params": {"query_vector": vector}
                }
            }
        }
    }

    response = es.search(index=INDEX_NAME, body=body)
    hits = response["hits"]["hits"]
    return [hit["_source"]["image_url"] for hit in hits]

# 集成对话系统
# 示例对话历史
dialogue_history = [
    "你们的公司是怎么组织的？",
    "就是结构方面，部门怎么分的？"
]

# 合并对话语义（真实可用LLM生成 query）
query = "公司的组织架构图是什么样的？"

# 搜图
images = search_similar_images(query)
for url in images:
    print(f"🔍 推荐图像: {url}")
