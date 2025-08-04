from openai import OpenAI
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q,Index
from elasticsearch_dsl.query import ScriptScore, Q, MatchAll
# from app.rag.split import load_and_split_docx
import re
import json
from app.utils.log import logger
import math
import traceback
import requests
import argparse
from app.constants import Speeches
from app.model.llm_factory import LLMFactory
from app.model.embedding.tool import get_embedding, get_doubao_embedding,get_bgem3_embedding
from app.config import config
from FlagEmbedding import BGEM3FlagModel

bgem3model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)


client = OpenAI(
    api_key=config.DEEPSEEK_API_KEY,
    base_url=config.DEEPSEEK_API_URL
)
history = []
es = es = Elasticsearch('http://localhost:9200', basic_auth=('elastic', 'gungun'))

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather of an location, the user shoud supply a location first",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"]
            },
        }
    },
]

def handle_consultation(history):
    """
    根据对话历史调用知识库查询相关信息，返回咨询结果。
    这里的实现仅作示例，请替换为实际的知识库查询逻辑。
    """
    # 示例：从知识库中根据用户问题获取答案
    # result = knowledge_base_query(history)
    result = "这是知识库返回的咨询结果。"
    return result

def hangle_registration(history):
    """
    根据对话历史调用挂号功能的接口（或 func tool），返回挂号结果。
    这里的实现仅作示例，请替换为实际的挂号调用逻辑。
    """
    # 示例：调用挂号 API
    # result = func_tool_register(history)
    result = "这是调用挂号接口后返回的结果。"
    return result

def handle_doctor_lookup(doctor_ids: list):
    """Method to handle doctor lookup by ids
    @param: doctor_ids: list, 医生的 ID 列表
    @return: json, 医生信息
    """
    result = []
    try:
        url = "https://api-backend.sihuiyiliao.com/doctor_service/api/doctor/coze/doctors"
        resp = requests.post(url, json={"doctor_ids": doctor_ids})
        # TODO: @txueduo catch error
        # raise resp.raise_for_status()
        result = resp.json()["data"]
    except Exception:
        logger.error(f"request url{url} failed {traceback.format_exc()}")
    return result


def semantic_search(index_name, query_sentence, top_k:int=5):
    """
    根据一句话的语义进行查询，比如 '帮我找擅长看胃病的医生',"王全胜医生的出诊时间是什么时候啊"
    """
    response = []
    try:
        query = {
            "query": {
                "multi_match": {
                    "query": query_sentence,
                    # 针对多个字段进行查询，可根据数据情况增加或调整字段
                    "fields": [
                        "擅长病种（新媒体推病种+擅长）",
                        "治疗特色",
                        "主要成就",
                        "姓名",
                        "出诊时间",
                        "挂号费",
                        "职称/职务"
                    ],
                    "fuzziness": "AUTO"  # 自动模糊匹配，可捕捉一些拼写或语义上的相似性
                }
            }
        }
        response = es.search(index=index_name, body=query)
        # logger.debug(response)['hits']['hits']
    except Exception:
        logger.error(traceback.format_exc())
    return response["hits"]["hits"]

def multi_field_search_v2(index_name, keywords: str, fields: list = ["擅长病种（新媒体推病种+擅长）", "姓名"]):
    s = Search(using=es, index=index_name)

    # 将查询字符串拆分为关键词列表
    # keywords_list = keywords.split()

    # 构建布尔查询
    bool_query = Q("bool")
    if "姓名" in fields:
        bool_query.must.append(Q("term", **{"姓名.keyword": keywords}))

    # 对其他字段进行模糊匹配（多字段匹配）
    other_fields = [field for field in fields if field != "姓名"]
    if other_fields:
        bool_query.should.append(Q("multi_match", query=keywords, fields=other_fields, operator="or"))

    # 设置布尔查询的最小匹配条件
    bool_query.minimum_should_match = 1  # 至少匹配一个 should 子句

    # 执行查询
    s = s.query(bool_query)
    results = s.execute()["hits"]["hits"]

    return results

def multi_field_search(index_name, keywords:str, fields:list=["擅长病种（新媒体推病种+擅长）", "姓名"]):
    s = Search(using=es, index=index_name)
    # 将多个关键词用空格拼接
    # query_str = " ".join(keywords)
    query_str = keywords
    query = Q("multi_match", query=query_str, fields=fields, operator="or", size=5)
    s = s.query(query)
    results = s.execute()["hits"]["hits"]
    return results

def search_by_keyword(index: str, keyword: str,fields: list=["擅长病种（新媒体推病种+擅长）", "姓名"], size: int = 5):
        """
        :param index: 要查询的索引名
        :param fields: 需要匹配的字段列表，如 ["title", "description"]
        :param keyword: 搜索关键词
        :param size: 返回结果条数
        """
        # 构造 multi_match 查询
        query_body = {
            "query": {
                "multi_match": {
                    "query": keyword,
                    "fields": fields,
                    "type": "best_fields",      # 可选: best_fields、most_fields、cross_fields、phrase、phrase_prefix
                    "operator": "or"           
                }
            },
            "size": size
        }

        response = es.search(index=index, body=query_body)
        hits = response.get("hits", {}).get("hits", [])
        return hits

def query_rewrite(query, llm):
    """调用具体工具时，根据对话历史获取完整的问题表达"""
    rewrite_prompt = f"""
    角色：你是一个问题重写助手，擅长根据对话历史，对当前医疗相关问题进行精准补全和合理重写，以提升对话的质量和流畅性。
    技能：
        技能 1: 补全和重写问题
            1. 仔细分析给定的医疗场景对话历史。
            2. 基于对话历史，对当前提出的医疗相关问题进行补全，确保问题信息完整。
            3. 运用恰当的医疗术语和自然语言表达方式，对补全后的问题进行重写，使其逻辑更清晰、表达更准确。
            4. 重写后的问题应符合医疗场景的语言习惯和专业要求。

    限制
        仅处理医疗场景相关的对话内容，其他领域话题，直接返回原问题。
        重写后的问题应简洁明了，避免冗长复杂的表述。 
    参考示例：
        示例1:
            对话历史:
                - 用户：四惠中医院有哪些医生？
                - 助手：四惠中医院有很多知名专家，如肿瘤科、 肿瘤科、微创介入科、中医内科、乳腺外科、中西医妇科、儿科、骨科、康复理疗科、针灸科、皮肤科、超声科、中西医眼科等。
            问题: 支持医保么？
            输出: 四惠中医院支持医保么？
        示例2: 直接返回原问题
            问题：邓紫棋是哪里的人
            输出：邓紫棋是哪里的人
    """
    try:
        messages = [      
            {"role": "system", "content": rewrite_prompt},
            {"role": "user", "content": query}]
        rewrite_res = llm.call_intent_stream(messages, temperature=0.3,max_tokens=100)
        return rewrite_res
    except Exception:
        logger.error(f"rewrite {query} error {traceback.format_exc()}")
        return query

def handle_question(query, history:list, llm, score_threshold:float=0.5, debug:bool=True, model_name: str=""):

    """
    """
    # 拼接成一个整体的查询上下文
    if history:
        combined_query ="\n".join([
                    f"角色 {msg['role']}:\n对话内容: {msg['content']}"
                    for msg in history
                ])
    else:
        combined_query = ""
    if model_name == "deepseek-r1":
        system_prompt = """

"""
    else:
        system_prompt = f"""
## 角色 你是四惠医疗健康智能问答助手，根据对话历史和背景知识，专业、口语化的解答用户的问题
## 技能
1. 医院信息查询（医保、发票、营业时间、设施、资质等）
2. 医疗问题解答
3. 症状导诊，推荐医生
4. 医生/专家介绍
5. 精确理解并确认用户目的或意图

## 意图澄清流程

### 如果用户意图不明确
引导知道明确用户意图。如"为了更好帮助您，能否具体说明您想了解什么内容？“您想咨询哪家医院的医保情况呢？”

### 如果用户提供模糊症状
"为了更好地为您提供导诊，请补充以下信息：
1. 主要症状及持续时间
2. 患者姓名、性别和年龄
3. 是否有在服用药物
(例如：'头痛3天，张三，男35岁，正在服用布洛芬')"

## 分类处理

### QA类问题
输出:qa

### 导诊类问题
[当收集完整症状+患者信息+用药情况后输出:doctor]

### 人物介绍
输出:introduce

### 非医疗内容
输出:un_medical

## 注意事项
1. 医疗建议需注明"仅供参考，具体请咨询专业医生"
2. 对用药指导需特别谨慎
3. 涉及急症时提示立即就医
4. 保护患者隐私，不存储个人信息

## 限制
如果输出属于类别，只输出doctor、introduce、un_medical、qa 中的一个。
引导类直接输出文本
背景知识：{combined_query}
"""
    messages = [      
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}]
    print(messages)
    res = llm.call_intent_stream(messages, chat_model="doubao-1.5-pro-32k-250115", temperature=0.6,max_tokens=400)
    if "qa" in res:
        pass
    elif "introduce" in res:
        pass
    elif "doctor" in res:
        pass
    else:
        pass
    print(res)
    # rewrite_query = query_rewrite(query, llm)

def prompt_gen():
    """
    """
    prompt = """你是一位大模型提示词生成专家，请根据用户的需求编写一个智能助手的提示词，来指导大模型进行内容生成，要求：
    # 医疗问题，如果不能具体明确用户目的或意图，需要引导直至明确用户具体意图，如某医院医保咨询、某医院发票、某医院物流、某医院几点下班？某医院设施咨询、某医院资质查询、某医院介绍等
    # 明确意图后，分类属于qa、还是导诊、还是人物介绍、还是非医疗内容
    # 如果是qa。输出qa 走qa处理逻辑。如果没有高分的q，普通问答
    # 如果是导诊，需要引导提供 病情症状/疾病名称+患者姓名性别年龄+用药情况,用户都提供后，输出 doctor
    # 如果是人物介绍，输出 introduce， 检索文档给出内容 
    # 如果是非医疗内容走 输出 un_medical 普通问答"""
    messages = [      
            {"role": "system", "content": prompt},
            {"role": "user", "content": "生成一个高质量的提示词"}]
    res = llm.call_intent_stream(messages, temperature=0.3,max_tokens=800)
    print(res)
    return res


def search_with_knn(index_name, query, top_k):
    query_vector = get_embedding(query)
    num_candidates = math.ceil(top_k * 1.5)
    knn = {"field": "vector", "query_vector": query_vector, "k": top_k, "num_candidates": num_candidates}

    results = es.search(index=index_name, knn=knn, size=top_k)
    return results["hits"]["hits"] if results else []

def search_hybrid_es(index_name, query, history, top_k=5):
    """
    """
    vector_results = search_by_vector(index_name, query, top_k=top_k)
    keyword_results = search_by_keyword(index_name, query, top_k=top_k)
    # 合并检索结果，简单去重（可根据业务调整排序和融合策略）
    combined = list(dict.fromkeys(vector_results + keyword_results))
    combined = combined[:top_k]
    return "\n\n".join(combined)

def search_by_vector(index_name, query, top_k=3, embedding_model:str="bge-large-zh-v1.5"):
    """"""
    # 获取查询向量
    try:
        if embedding_model=="bge-large-zh-v1.5":
            query_vector = get_embedding(query)
        elif embedding_model == "bge-m3":
            query_vector = get_bgem3_embedding(bgem3model,query)
        else:
            query_vector = get_doubao_embedding(query)
        if index_name==f"{config.env_version}_doctor_info":
            script_query = Q(
            'script_score',
            query=Q('match_all'),
            script={
                'source': "cosineSimilarity(params.query_vector, 'goodvector') + 1.0",
                'params': {'query_vector': query_vector}
                }
            )
        else:
            script_query = Q(
                'script_score',
                query=Q('match_all'),
                script={
                    'source': "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    'params': {'query_vector': query_vector}
                }
            )

        # 构建搜索对象并执行查询
        search = Search(using=es, index=index_name).query(script_query)
        response = search.execute()["hits"]["hits"][:top_k]
    except Exception as e:
        logger.error(traceback.format_exc())
        response = None
    return response


def create_index_with_mapping(index_name, embedding_model:str="doubao"):
    """
    创建包含全文和向量字段的索引映射
    """
    try:
        es.indices.delete(index=index_name)
    except Exception:
        pass
    mapping = {
        "mappings": {
            "properties": {
                "content": {"type": "text"},  # 存储文本内容，用于全文检索
                "metadata": {"type": "object"},  # 存储其他元数据，如页码、来源等
                "embedding": {
                    "type": "dense_vector",  # 向量字段，用于向量检索
                    "dims": 1024 if embedding_model!="doubao" else 4096,            # 维度数，依据你所用的嵌入模型（这里以 OpenAI 为例）
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=mapping)
        logger.debug(f"索引 {index_name} 创建成功")
    else:
        logger.debug(f"索引 {index_name} 已存在")

import uuid
def index_documents(index_name, documents, embedding_model:str="bge-large-zh-v1.5"):
    """
    将切分后的文档块向量化，并存入 Elasticsearch
    """
    for doc in documents:
        # 生成唯一 id
        doc_id = str(uuid.uuid4())
        # 生成向量，调用 embed_query（也可以用 embed_document，效果相似）
        if embedding_model=="bge-large-zh-v1.5": 
            vector = get_embedding(doc.page_content)
        else:
            vector = get_doubao_embedding(doc.page_content)
        # 构建文档结构（包含全文、元数据和向量）
        # logger.debug(doc.metadata)
        # import pdb
        # pdb.set_trace()
        doc_body = {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "embedding": vector
        }
        # 入库
        es.index(index=index_name, id=doc_id, body=doc_body)
    logger.debug("文档入库完成")

def call_model(messages, chat_model:str="deepseek-chat", json_schema=None):
    """
    """
    try:
        response = client.chat.completions.create(
            model=chat_model,
            messages=messages,
            temperature=0,
            max_tokens=600,
            top_p=1,
            response_format=json_schema,
            # functions=functions,
            # function_call="auto",
            # tools=tools
        )
        res = response.choices[0].message.content
        logger.debug(res)
        logger.debug(type(res))
        if json_schema:
            try:
                res = res.strip("```json").strip("```").strip(" ")
                res = json.loads(res)
            except Exception:
                pattern = r"json\n(.*?)\n"
                matches = re.findall(pattern, res, re.DOTALL)
                for match in matches:
                    json_data = json.loads(match.strip())
                    print("Extracted JSON Data:")
                    print(json.dumps(json_data, indent=4))
                    return json_data
            return res
    except Exception:
        logger.error(traceback.format_exc())
    return res

def call_model_stream(messages, chat_model:str="deepseek-chat", json_schema=None):
    """
    """
    stream_res = client.chat.completions.create(
        model=chat_model,
        messages=messages,
        temperature=0,
        max_tokens=300,
        top_p=1,
        response_format=json_schema,
        stream=True,
        # functions=functions,
        # function_call="auto",
        # tools=tools
    )
    res= ""
    for event in stream_res:
        res += event.choices[0].delta.content
    if json_schema:
        try:
            res = res.strip("```json").strip("```").strip(" ")
            res = json.loads(res)
        except Exception:
            logger.error(traceback.format_exc())
    logger.debug(res)
    return res


def process_intent(intent, history):
    """
    根据意图和对话历史调用相应的处理函数。
    """
    logger.debug(f"客服消息: {history}")
    resp = call_model(history)
    return resp

def limit_history_length(history, max_length=6):
    """
    限制对话历史的长度，保留最新的 max_length 条消息。
    """
    while len(history) > max_length:
        history.pop(0)  # 删除最旧的消息
    return history

def main(llm):
    global history
    while True:
        try:
            user_input = input("User> ")
            if user_input.lower() == "exit":
                break
            # 构造意图模型的对话历史
            # intent_history.append({"role": "user", "content": user_input})
            # intent_history = limit_history_length(intent_history)
            # intent = call_model(intent_system_prompt+intent_history).content.strip()
            # logger.debug(f"意图识别结果:{intent}")

            # 第二步：根据意图调用客服模型提供完整服务
            # 这里将用户问题作为对话历史的一部分传递给客服模型
            history = limit_history_length(history)
            # prompt_gen()
            handle_question(user_input, history, llm)
        except Exception as e:
            logger.debug(f"Error: {e}")
            break   

if __name__=="__main__":

    llm_factory = LLMFactory()
    llm = llm_factory.create_llm("coze_deepseek")
    llm.init_client(config.ARK_API_KEY, config.ARK_BASE_URL)
    main(llm)
