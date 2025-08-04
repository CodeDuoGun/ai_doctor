import requests
from app.utils.log import logger
from concurrent.futures import ThreadPoolExecutor
from app.utils.tool import perf_counter_timer, normalize_vector
def get_embedding(text, model, dims:int=1024):
    """bge 向量检索 文本向量化"""
    if not text:
        return [0.0] * dims
    embedding = model.encode(text).tolist()
    return embedding
def get_bgem3_embedding(bgem3model, sentence):
    embedding = bgem3model.encode([sentence])
    return embedding.tolist()[0]

@perf_counter_timer
def get_bge_code_embedding(query:str, model):
    try:
        sentences = [
            query
        ]
        embeddings = model.encode(sentences, convert_to_tensor=True)
    except Exception as e:
        logger.error(f"embedding {query} failed for {e}")
        return [1.0] * 1536
    return embeddings.tolist()[0]


def get_multi_process_bgem3_embedding(bgem3model, sentence):
# from sentence_transformers import SentenceTransformer
    with ThreadPoolExecutor(8) as executor:
        future = executor.submit(get_bgem3_embedding, bgem3model,sentence)
        result = future.result()
        return result



@perf_counter_timer
def get_doubao_embedding(text, dims:int=0):
    """文本向量化"""
    json_data = {
        # "model": "doubao-embedding-vision-241215",
        "model": "doubao-embedding-large-text-240915", # dims=4096
        "input": [text],
        }

    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer f43df693-455a-4e3e-b987-d76d7b57f4c3'
    }

    response = requests.post('https://ark.cn-beijing.volces.com/api/v3/embeddings', headers=headers, json=json_data)
    
    if response.status_code == 200:
        embedding = response.json()["data"][0]["embedding"]

        # embedding = normalize_vector(embedding)
        return embedding
    else:
        logger.error(f"Error: {response.status_code}, {response.text}")
        return []

# res1 = get_doubao_embedding("四惠南区中医门诊如何退号")
# res2 = get_doubao_embedding("四惠南区中医门诊如何退号")
# print(len(res1), res1[:10])
# print(len(res2), res2[:10])