from elasticsearch import Elasticsearch
from app.config import config

# 初始化客户端（根据你的地址修改）
es = Elasticsearch(f"http://{config.ES_HOST}:{config.ES_PORT}",  basic_auth=(config.ES_USER, config.ES_AUTH), request_timeout=80)

# 要分析的文本和分词器
analyze_body = {
    "analyzer": "ik_smart", # "ik_max_word" 最大细粒度
    "text": "萎缩性胃炎"
}

# 调用 _analyze 接口
response = es.indices.analyze(body=analyze_body)

# 输出分词结果
print("分词结果：")
for token in response["tokens"]:
    print(token["token"])
