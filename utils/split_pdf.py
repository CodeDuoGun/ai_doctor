from utils.log import logger
import traceback
from ai_emr_customer.app.rag.split import load_and_split_docx
from ai_emr_customer.app.rag.vdb.es.es_processor import es_handler
from ai_emr_customer.app.embedding.tool import get_bgem3_embedding


def save2csv():

    pass


def insert2es(self):
    pass


def extract_pdf2text(pdf_file):
    documents = load_and_split_docx(pdf_file, chunk_size=1000, chunk_overlap=20)
    # 大模型抽取知识信息
    extract_res = []
    try:
        for doc in documents:
            pass
            
    except Exception:
        logger.error(f"extract doc err {traceback.format_exc()}")
        res = []


import re
import fitz
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. PDF 解析 & 多级目录识别
def parse_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    data = []
    current_pian, current_bingzhong, current_leixing = "", "", ""
    buffer_text, buffer_page = "", None

    def flush_buffer():
        if current_bingzhong and current_leixing and buffer_text.strip():
            data.append({
                "篇": current_pian,
                "病名": current_bingzhong,
                "信息类型": current_leixing,
                "正文": buffer_text.strip(),
                "页码": buffer_page
            })

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        lines = text.split("\n")
        for line in lines:
            if re.match(r"^第[一二三四五六七八九十百]+篇\s+", line):
                current_pian = re.sub(r"^第[一二三四五六七八九十百]+篇\s*", "", line)
                continue
            if re.match(r"^第[一二三四五六七八九十百]+章\s+", line):
                flush_buffer()
                current_bingzhong = re.sub(r"^第[一二三四五六七八九十百]+章\s*", "", line)
                current_leixing = ""
                buffer_text = ""
                continue
            if re.match(r"^第[一二三四五六七八九十百]+节\s+", line):
                flush_buffer()
                current_leixing = re.sub(r"^第[一二三四五六七八九十百]+节\s*", "", line)
                buffer_text = ""
                buffer_page = page_num
                continue
            buffer_text += line + "\n"
    flush_buffer()
    return data

# 2. 分块
def chunk_text(text, max_tokens=500):
    sentences = re.split(r"(。|！|\!|？|\?)", text)
    chunks, current_chunk = [], ""
    for sent in sentences:
        if len(current_chunk) + len(sent) > max_tokens:
            chunks.append(current_chunk)
            current_chunk = sent
        else:
            current_chunk += sent
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# 3. 向量化 & 入库
def build_faiss_index(data):
    model = SentenceTransformer("moka-ai/m3e-base")
    texts, meta = [], []
    for item in data:
        chunks = chunk_text(item["正文"])
        for idx, chunk in enumerate(chunks):
            texts.append(chunk)
            meta.append({
                "病名": item["病名"],
                "信息类型": item["信息类型"],
                "chunk_id": f"{item['病名']}-{item['信息类型']}-{idx}",
                "原文": chunk
            })
    vectors = model.encode(texts, convert_to_numpy=True)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index, meta

INDEX_NAME = "medical_pdf_chunks"
# bgem3_model = 

# 1. 创建 ES 索引（dense_vector）
def create_index():
    if es_handler.indices.exists(index=INDEX_NAME):
        es_handler.indices.delete(index=INDEX_NAME)

    mapping = {
        "mappings": {
            "properties": {
                "病名": {"type": "keyword"},
                "信息类型": {"type": "keyword"},
                "原文": {"type": "text"},
                "chunk_id": {"type": "keyword"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": 768,  # m3e-base 输出维度
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }
    es_handler.indices.create(index=INDEX_NAME, body=mapping)

# 2. 向 ES 写入数据
def build_es_index(data):
    # model = SentenceTransformer("moka-ai/m3e-base")
    docs = []
    for item in data:
        chunks = chunk_text(item["正文"])
        for idx, chunk in enumerate(chunks):
            # vector = model.encode(chunk).tolist()
            vector = get_bgem3_embedding(bgem3model, chunk)
            doc = {
                "病名": item["病名"],
                "信息类型": item["信息类型"],
                "chunk_id": f"{item['病名']}-{item['信息类型']}-{idx}",
                "原文": chunk,
                "embedding": vector
            }
            docs.append(doc)
            # 批量写入（这里直接单条写入，生产环境建议 bulk）
            es_handler.index(index=INDEX_NAME, document=doc)
    print(f"已写入 {len(docs)} 条记录到 ES")

# 3. 向量检索
def search_es(query, top_k=5):
    model = SentenceTransformer("moka-ai/m3e-base")
    query_vec = model.encode(query).tolist()

    res = es_h.knn_search(
        index=INDEX_NAME,
        knn={
            "field": "embedding",
            "query_vector": query_vec,
            "k": top_k,
            "num_candidates": 50
        },
        source=["病名", "信息类型", "原文"]
    )

    hits = res["hits"]["hits"]
    return [{"病名": h["_source"]["病名"],
             "信息类型": h["_source"]["信息类型"],
             "原文": h["_source"]["原文"],
             "score": h["_score"]} for h in hits]

# ========== 测试 ==========

# 先创建索引
create_index()
# 测试
pdf_path = "医学教材.pdf"
structured_data = parse_pdf(pdf_path)
# index, meta = build_faiss_index(structured_data)

# 4. 检索示例
def search(query, index, meta, top_k=5):
    model = SentenceTransformer("moka-ai/m3e-base")
    q_vec = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_vec, top_k)
    return [meta[i] for i in I[0]]

# results = search("肝硬化的病因", index, meta)
# for r in results:
#     print(r["病名"], r["信息类型"], r["原文"][:50])
