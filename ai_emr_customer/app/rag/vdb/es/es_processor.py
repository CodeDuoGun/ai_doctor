import pandas as pd
import time
import uuid
import re
from elasticsearch import Elasticsearch,helpers
from elasticsearch.helpers import bulk, BulkIndexError
from app.model.embedding.tool import get_embedding, get_doubao_embedding, get_bgem3_embedding, get_bge_code_embedding
import uuid
from app.utils.log import logger
import traceback
from elasticsearch_dsl import Search, Q, Index
from app.utils.tool import perf_counter_timer, normalize_vector
from app.config import config
from concurrent.futures import ThreadPoolExecutor, as_completed
import aiohttp
from typing import List


class ElasticsearchHandler:
    def __init__(self):#, model_name="BAAI/bge-large-zh-v1.5"):
        """
        :param index_name: str, 索引名称
        """
    
        self.es = Elasticsearch(f'http://{config.ES_HOST}:{config.ES_PORT}', basic_auth=(config.ES_USER, config.ES_AUTH), request_timeout=80)
        # self.es = Elasticsearch(f'http://localhost:9200', basic_auth=('elastic', 'gungun'))
        self.embedding_model = None
        if config.embedding_model == "bge":
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer("BAAI/bge-large-zh-v1.5")

    def create_index(self, index_name, mappings=None, need_del: bool=False):
        """
        创建索引
        """
        if need_del:
            try:
                self.es.indices.delete(index=index_name)
                # self.es.indices.delete(index="es_doctor_info")
            except Exception:
                pass

        if not self.es.indices.exists(index=index_name):
            self.es.indices.create(index=index_name, body=mappings)
            logger.info(f"索引 {index_name} 创建成功")
        else:
            logger.info(f"索引 {index_name} 已存在")


    def create_index_with_mapping(self, index_name, embedding_model:str="doubao"):
        """
        创建包含全文和向量字段的索引映射
        """
        # print(self.es.ping())
        try:
            self.es.indices.delete(index=index_name)
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
                    },
                    "bge_vector": {
                        "type": "dense_vector",  # 向量字段，用于向量检索
                        "dims": 1024,            # 维度数，依据你所用的嵌入模型（这里以 OpenAI 为例）
                        "index": True,
                        "similarity": "cosine"
                    },
                    "bge_code_vector": {
                        "type": "dense_vector",  # 向量字段，用于向量检索
                        "dims": 1536,            
                        "index": True,
                        "similarity": "cosine"
                    }

                }
            }
        }
        if not self.es.indices.exists(index=index_name):
            self.es.indices.create(index=index_name, body=mapping)
            logger.debug(f"索引 {index_name} 创建成功")
        else:
            logger.debug(f"索引 {index_name} 已存在")

    def index_documents(self, index_name, documents, embedding_model_type:str="doubao", bge_model=None):
        """
        将切分后的文档块向量化，并存入 Elasticsearch
        """
        for doc in documents:
            # 生成唯一 id
            doc_id = str(uuid.uuid4())
            # 生成向量，调用 embed_query（也可以用 embed_document，效果相似）
            # if bge_model:
            #     bgem3_vector = get_bgem3_embedding(bge_model, doc.page_content)
            # else:
            #     bgem3_vector = [0.0] * 1024
            if embedding_model_type == "bge_code":
                bge_code_vector = get_bge_code_embedding(doc.page_content, bge_model)
            else:
                bge_code_vector = [0.0] * 1536 #  TODO: 需要额外处理 都是0会计算报错，

            vector = get_doubao_embedding(doc.page_content)
            
            # 构建文档结构（包含全文、元数据和向量）
            # logger.debug(doc.metadata)
            # import pdb
            # pdb.set_trace()
            doc_body = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "embedding": vector,
                "bge_code_vector":bge_code_vector
            }
            # 入库
            self.es.index(index=index_name, id=doc_id, body=doc_body)
        logger.debug("文档入库完成")
    

    def update_document(self,index_name, doc_id, doc):
        """
        更新索引中的文档
        :param index_name: str, 索引名称
        :param doc_id: str, 文档 ID
        :param doc: dict, 更新的文档内容
        """
        self.es.update(index=index_name, id=doc_id, body={"doc": doc})
        logger.info(f"文档 {doc_id} 更新成功")

    def delete_document(self, index_name, doc_id):
        """
        删除索引中的文档
        :param index_name: str, 索引名称
        :param doc_id: str, 文档 ID
        """
        self.es.delete(index=index_name, id=doc_id)
        logger.info(f"文档 {doc_id} 删除成功")
    
    def search_keyword(self, index_name, query, top_k=5):
        """
        使用关键词检索：通过 match 查询对文档内容进行检索，
        返回与 query 匹配度较高的 top_k 个文档
        """
        es_query = {
            "size": top_k,
            "query": {
                "match": {
                    "content": query
                }
            }
        }
        results = self.es.search(index=index_name, body=es_query)
        # return results
        # TODO： 看下返回结果
        hits = results["hits"]["hits"]
        retrieved_docs = []
        for hit in hits:
            doc = hit["_source"]
            doc_text = f"标题：{doc.get('title', '')}\n内容：{doc.get('content', '')}"
            retrieved_docs.append(doc_text)
        return retrieved_docs
    
    
    @perf_counter_timer 
    def search_by_vector(self, index_name, query, top_k=5,doctor_location:str="", embedding_model_type:str="doubao", embed_model=None):
        """"""
        try:
            if embedding_model_type=="beg_large_zh":
                query_vector = get_embedding(query, embed_model)
            elif embedding_model_type == "bge_code":
                query_vector = get_bge_code_embedding(query, embed_model) 
            else:
                query_vector = get_doubao_embedding(query)
            if index_name==f"{config.env_version}_doctor_info":
                script_query = Q(
                'script_score',
                query=Q('match_all'),
                script={
                    'source': "cosineSimilarity(params.query_vector, 'goodvector') + 1.0",
                    'params': {'query_vector': query_vector}
                    },
                )
            elif index_name==f"{config.env_version}_primary_disease":
                script_query = Q(
                'script_score',
                query=Q('match_all'),
                script={
                    'source': "cosineSimilarity(params.query_vector, 'primary_disease_vector') + 1.0",
                    'params': {'query_vector': query_vector}
                    },
                )
            elif index_name==f"{config.env_version}_doctor":
                # 优先根据医生所属医院检索
                cosine_script = {
                        'source': "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                        'params': {'query_vector': query_vector}
                        }
                if doctor_location:
                    script_query = Q(
                    'script_score',
                    query=Q(
                        'bool',
                        filter=[
                            Q('match', 所在区域=doctor_location)  # 替换为您需要的区域
                        ]
                    ),
                    script=cosine_script,
                    )
                else:
                    script_query = Q(
                    'script_score',
                    query=Q('match_all'),
                    script=cosine_script
                    )
            elif index_name==f"{config.env_version}_qa":
                if embedding_model_type == "doubao":
                    script_query = Q(
                    'script_score',
                    query=Q('match_all'),
                    script={
                        'source': "cosineSimilarity(params.query_vector, 'q_vector') + 1.0",
                        'params': {'query_vector': query_vector}
                        },
                    )
                elif embedding_model_type == "bgem3":
                    script_query = Q(
                    'script_score',
                    query=Q('match_all'),
                    script={
                        'source': "cosineSimilarity(params.query_vector, 'q_bgem3_vector') + 1.0",
                        'params': {'query_vector': query_vector}
                        },
                    )
            elif index_name==f"{config.env_version}_primary_disease":
                if embedding_model_type == "doubao":
                    script_query = Q(
                    'script_score',
                    query=Q('match_all'),
                    script={
                        'source': "cosineSimilarity(params.query_vector, 'q_vector') + 1.0",
                        'params': {'query_vector': query_vector}
                        },
                    )
                elif embedding_model_type == "bgem3":
                    script_query = Q(
                    'script_score',
                    query=Q('match_all'),
                    script={
                        'source': "cosineSimilarity(params.query_vector, 'q_bgem3_vector') + 1.0",
                        'params': {'query_vector': query_vector}
                        },
                    )
                
            else:
                if embedding_model_type=="bge_large_zh":
                    script_query = Q(
                    'script_score',
                    query=Q('match_all'),
                    script={
                        'source': "cosineSimilarity(params.query_vector, 'bge_vector') + 1.0",
                        'params': {'query_vector': query_vector}
                    }
                )
                elif embedding_model_type=="bge_code":
                    script_query = Q(
                        'script_score',
                        query=Q('match_all'),
                        script={
                            'source': "cosineSimilarity(params.query_vector, 'bge_code_vector') + 1.0",
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
            search = Search(using=self.es, index=index_name).query(script_query)
            response = search.execute()["hits"]["hits"]
        except Exception as e:
            logger.error(traceback.format_exc())
            response = None
        return response
    
    @perf_counter_timer 
    def search_by_keyword(self, index: str, 
                          keyword: str,
                          fields: list=["擅长病种（新媒体推病种+擅长）", "姓名","所在区域"], 
                          size: int = 5, 
                          doctor_name:str="", 
                          doctor_location:str="",
                          ):
        """
        :param index: 要查询的索引名
        :param fields: 需要匹配的字段列表，如 ["title", "description"]
        :param keyword: 搜索关键词
        :param size: 返回结果条数
        """
        # 构造 multi_match 查询
        if index == f"{config.env_version}_doctor_info":
            fields = ["擅长病种（新媒体推病种+擅长）"]
        elif index == f"{config.env_version}_doctor":
            fields = ["擅长", "所在区域", "姓名"] 
        elif index == f"{config.env_version}_secondary_disease":
            fields = ["secondary_disease"] 
        elif index == f"{config.env_version}_primary_disease":
            fields = ["primary_disease"] 

        if not doctor_name and not doctor_location:
            if index == f"{config.env_version}_doctor":
                query_body = {
                    "query": {
                        "multi_match": {
                            "query": keyword,
                            # "type": "phrase",
                            "fields": fields,
                            # "fuzziness": "AUTO"
                        }
                    },
                    "size": size
                }
            else:
                query_body = {
                    "query": {
                        "multi_match": {
                            "query": keyword,
                            "type": "phrase",
                            "fields": fields,
                            # "fuzziness": "AUTO"
                        }
                    },
                    "size": size
                }

        elif doctor_name and doctor_location:
            query_body = {
                "query": {
                    "bool": {
                        "must": {
                            "multi_match": {
                                "query": keyword,
                                "fields": fields,
                                "type": "best_fields",  # 可选: best_fields、most_fields、cross_fields、phrase、phrase_prefix
                                "operator": "or"
                            }
                        },
                        "filter": {
                            "bool": {
                                "must": [
                                    {
                                        "match": {
                                            "姓名": doctor_name  # 示例过滤条件，根据实际需求修改
                                        }
                                    },
                                    {
                                        "match": {
                                            "所在区域": doctor_location,  # 示例过滤条件，根据实际需求修改
                                        }
                                    }]
                            }
                        }
                    }
                },
                "size": size
            }
        elif doctor_location:
            print("location search")
            query_body = {
                "query": {
                    "bool": {
                        "must": {
                            "multi_match": {
                                "query": keyword,
                                "fields": fields,
                                "type": "best_fields",  # 可选: best_fields、most_fields、cross_fields、phrase、phrase_prefix
                                "operator": "or"
                            }
                        },
                        "filter": {
                            "match": {
                                "所在区域": doctor_location,  # 示例过滤条件，根据实际需求修改
                            }
                        }
                    }
                },
                "size": size
            }
        elif doctor_name:
            query_body = {
                "query": {
                    "bool": {
                        "should": {
                            "multi_match": {
                                "query": keyword,
                                "fields": fields,
                                "type": "best_fields",  # 可选: best_fields、most_fields、cross_fields、phrase、phrase_prefix
                                "operator": "or"
                            }
                        },
                        "filter": {
                            "match": {
                                "姓名": doctor_name,
                            }
                        }
                    }
                },
                "size": size
            }

        response = self.es.search(index=index, body=query_body)
        hits = response["hits"]["hits"]
        return hits

    def search_hybrid(self,
                      index: str,
                      doctor_name: str,
                      condition: str,
                      k: int = 5,
                      alpha: float = 0.3,
                      name_boost: float = 3.0):
        """
        混合检索：对医生姓名做 BM25，对症状描述做向量检索，
        最终得分 = alpha * 向量相似度 + (1-alpha) * BM25 得分
        """
        q_vec = get_doubao_embedding(condition)

        body = {
            "size": k,
            "query": {
                "script_score": {
                    "query": {
                        "bool": {
                            "should": [
                                {
                                    "match": {
                                        "姓名": {
                                            "query": doctor_name,
                                            "fuzziness": "AUTO",
                                            "boost": name_boost
                                        }
                                    }
                                },
                                {
                                "multi_match": {
                                    "query": condition,
                                    "fields": ["擅长^1.5"],  # 关键词检索
                                    # "fuzziness": "AUTO"
                                }
                                },
                            ]
                        }
                    },
                    "script": {
                        "source": (
                            "params.alpha * cosineSimilarity(params.q, 'goodvector') "
                            "+ (1 - params.alpha) * _score"
                        ),
                        "params": {
                            "q": q_vec,
                            "alpha": alpha
                        }
                    }
                }
            }
        }
        res = self.es.search(index=index, body=body)
        return res["hits"]["hits"]

    def hybrid_search_doc(
        self,
        index: str,
        query:str,
        k: int = 5,
        alpha: float = 0.4,
        bge_model=None,
        name_boost: float = 3.0):
        print(111111)
        if bge_model:
            query_vector = get_embedding(query, bge_model)
        else:
            query_vector = get_doubao_embedding(query)

        if bge_model:
            body = {
                "size": k,
                "query": {
                    "script_score": {
                        "query": {
                            "bool": {
                                "should": [
                                    {
                                        "match": {
                                            "content": {
                                                "query": query,
                                                "boost": 1.0
                                            }
                                        }
                                    }
                                ]
                            }
                        },
                        "script": {
                            "source": (
                                "params.alpha * cosineSimilarity(params.q, 'bge_vector') "
                                "+ (1 - params.alpha) * _score"
                            ),
                            "params": {
                                "q": query_vector,
                                "alpha": alpha
                            }
                        }
                    }
                }
            } 
        else:
            body = {
                "size": k,
                "query": {
                    "script_score": {
                        "query": {
                            "bool": {
                                "should": [
                                    {
                                        "match": {
                                            "content": {
                                                "query": query,
                                                "boost": 1.0
                                            }
                                        }
                                    }
                                ]
                            }
                        },
                        "script": {
                            "source": (
                                "params.alpha * cosineSimilarity(params.q, 'embedding') "
                                "+ (1 - params.alpha) * _score"
                            ),
                            "params": {
                                "q": query_vector,
                                "alpha": alpha
                            }
                        }
                    }
                }
            } 
        res = self.es.search(index=index, body=body)
        logger.debug(f"length hits:{len(res)}")
        return res["hits"]["hits"]

    
    def get_history_from_es(self, conversation_id, deviceId:str="", index_name:str="chat_history"):
        # 构造ES查询，假设文档中存有 uid、source 和 deviceId 字段
        s = Search(using=self.es, index=index_name)
        s = s.filter("term", conversation_id=conversation_id).filter("term", deviceId=deviceId)
        response = s.execute()
        
        # 如果查询到数据，则获取第一个文档的 history 字段，否则返回空列表
        if response.hits.total.value > 0:
            doc = response.hits[0]
            historyx = doc.history if hasattr(doc, "history") else []
        else:
            historyx = []
        
        # 保持历史记录数不超过 5 对（即最多 10 个记录）
        while len(historyx) > 5 * 2:
            historyx.pop(0)
            historyx.pop(0)
        
        # 过滤掉 content 为空的记录
        try:
            historyx = [item for item in historyx if len(item.get("content", "")) > 0]
        except Exception:
            pass
        
        return historyx
        
    
    def add_chat_history(self, conversation_id, history,index_name:str="chat_history"):
        """
        存入一组对话历史到 es 
        @param index_name: ES 索引名称
        @param conversation_id: 对话ID
        @param history: 对话历史，格式为 list，例如:
                        [{"role":"user", "content": "ghg"}, {"role":"assistant", "content": "fhghliG"}]
        @return: ES 响应结果
        """
        # 构造文档数据
        # TODO: @txueduo 每次写入删除旧的history
        document = {
            "conversation_id": conversation_id,
            "history": history,
        }
        # 将文档存入指定索引
        response = self.es.index(index=index_name, document=document)
        return response
    
    def semantic_search(self, index_name, query_sentence, top_k:int=5):
        """
        根据一句话的语义进行查询，比如 '帮我找擅长看胃病的医生',"王全胜医生的出诊时间是什么时候啊"
        """
        query = {
            "query": {
                "multi_match": {
                    "query": query_sentence,
                    # 针对多个字段进行查询，可根据数据情况增加或调整字段
                    "fields": [
                        "擅长病种（新媒体推病种+擅长）",
                        "治疗特色",
                        "所在区域",
                        "姓名",
                        "出诊时间",
                        "挂号费",
                        "职称/职务"
                    ],
                    "fuzziness": "AUTO"  # 自动模糊匹配，可捕捉一些拼写或语义上的相似性
                }
            }
        }
        response = self.es.search(index=index_name, body=query)#[:top_k]
        return response["hits"]["hits"]
    
    # TODO: @txueduo 修改下面保证所有格式兼容
    def bulk_excel_insert(self, xlsx_file, index_name:str="alpha_doctor_info",embed_model:str="doubao"):
        xlsx = pd.ExcelFile(xlsx_file, engine='openpyxl')
        # 获取所有工作表名称
        sheet_names = xlsx.sheet_names
        logger.debug(f"所有工作表:{sheet_names[1:]}")
        for sheet in sheet_names[1:]:
            df = pd.read_excel(xlsx, sheet_name=sheet, skiprows=1, engine='openpyxl')
            # 删除所有列均为 NaN 的行
            df = df.dropna(subset=['姓名'])
            df = df.drop(['出生年份', '序号'], axis=1)
            df = df.fillna("")
            # 重命名 某列
            df = df.rename(columns={'所在\n区域': '所在区域'})
            df["职称/职务"] = df['职称/职务'].str.replace('\n', '、', regex=False)
            df["擅长病种（新媒体推病种+擅长）"] = df['擅长病种（新媒体推病种+擅长）'].str.replace(',', '，', regex=False)
            print(df['职称/职务'])
            # 清洗id列
            def clean_number(s):
                # 使用正则表达式提取所有数字
                numbers = re.findall(r'\d+', str(s))
                # 如果没有找到数字，返回 None 或其他默认值
                if not numbers:
                    return -999
                # 将第一个数字转换为整数
                return int(numbers[0])

            df["ID"] = df["ID"].apply(clean_number)
            actions = []
            if sheet in ("肿瘤"):
                df = df.dropna(axis=1, how='all')
                df = df.drop("Unnamed: 23", axis=1)
            # 遍历每一行数据
            for _, row in df.iterrows():
                # 将每行数据转换为字典
                doc = row.to_dict()
                # 对需要转换为数值类型的字段进行转换，防止出现 NaN
                doc["年龄"] = 0 if doc.get("年龄") == '——' else doc.get("年龄", 0)
                doc["挂号费"] = doc.get("挂号费", "")
                doc["处方单价"] = doc.get("处方单价", "")

                # 假设我们利用“擅长病种（新媒体推病种+擅长）”字段来生成向量，也可以选择其他字段或拼接多个字段
                text_for_embedding = doc['姓名'] + doc.get("擅长病种（新媒体推病种+擅长）", "")# + doc['治疗特色']
                if embed_model == "doubao":
                    # 使用 doubao 模型进行向量化
                    doc["goodvector"] = get_doubao_embedding(text_for_embedding)
                else:
                    # 使用其他模型进行向量化
                    embed_model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
                    doc["goodvector"] = get_embedding(embed_model, text_for_embedding)

                action = {
                    "_index": index_name,
                    "_source": doc
                }
                actions.append(action)

            logger.warning(f"sheet_name: {sheet}, actions: {len(actions)}")
            try: 
                helpers.bulk(self.es, actions)
            except BulkIndexError as e:
                logger.error(f"{sheet} BulkIndexError: {e}")
                for error in e.errors:
                    pass
                    logger.error(f"Failed documents:{error}")
                    
            except Exception:
                logger.error(f"error to bulk {sheet} for {traceback.format_exc()}")
        logger.debug("XLSX 数据导入完成")

    def bulk_insert_by_file(self, file, index_name:str="alpha_doctor", embed_model:str="doubao", sheet_name:str=""):
        if index_name == f"{config.env_version}_doctor":
            df = pd.read_excel(file, engine='openpyxl')
            # 清洗列表
            actions = []
            for _, row in df.iterrows():
                # 将每行数据转换为字典
                doc = row.to_dict()
                new_doc = {}
                new_doc["所在区域"] = doc["地区"]
                new_doc["序号"] = doc["序号"]
                new_doc["姓名"] = doc["姓名"]
                new_doc["简介"] = doc["简介"]
                new_doc["ID"] = doc["ID"]
                new_doc["擅长"] = doc["擅长"]
                new_doc["执业医院"] = doc["执业医院"]
                new_doc["出诊地点"] = doc["出诊地点"]
                text_for_embedding = doc["姓名"]+ "擅长：" + doc["擅长"]
                if embed_model == "doubao":
                    # 使用 doubao 模型进行向量化
                    new_doc["goodvector"] = get_doubao_embedding(text_for_embedding)
                    new_doc["vector"] = get_doubao_embedding(new_doc["出诊地点"]+text_for_embedding)
                else:
                    # 使用其他模型进行向量化
                    embed_model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
                    new_doc["goodvector"] = get_embedding(embed_model, text_for_embedding)
                action = {
                    "_index": index_name,
                    "_source": new_doc
                }
                actions.append(action)
            try: 
                helpers.bulk(self.es, actions)
            except BulkIndexError as e:
                logger.error(f"BulkIndexError: {e}")
                for error in e.errors:
                    logger.error(f"Failed documents:{error}")
                    
            except Exception:
                logger.error(f"error to bulk for {traceback.format_exc()}")
        else:
            xlsx = pd.ExcelFile(file, engine='openpyxl')
            actions = []
            # 获取所有工作表名称
            sheet_names = xlsx.sheet_names
            logger.debug(f"所有工作表:{sheet_names}")
            df = pd.read_excel(xlsx, sheet_name=sheet_name, engine='openpyxl')
            for _, row in df.iterrows():
                doc = row.to_dict()
                if sheet_name == "一级病种": 
                    action = {
                        "_index": index_name,
                        "_source": {"doctors": doc["专家姓名"], "primary_disease": doc["疾病种类"], "primary_disease_vector": get_doubao_embedding(doc["疾病种类"])}
                    }
                else:
                    action = {
                        "_index": index_name,
                        "_source": {"doctors": doc["专家姓名"], "secondary_disease": doc["疾病种类"], "kw_secondary_disease": doc["疾病种类"]}
                    }

                actions.append(action)
            try: 
                helpers.bulk(self.es, actions)
            except BulkIndexError as e:
                logger.error(f"BulkIndexError: {e}")
                for error in e.errors:
                    pass
                    logger.error(f"Failed documents:{error}")
                    
            except Exception:
                logger.error(f"error to bulk for {traceback.format_exc()}")


    def batch_qa(self, qa_data, actions:list, index_name:str="alpha_qa",embed_type:str="doubao", embed_model=None):
        """组成五个一组的qa并行处理"""
        with ThreadPoolExecutor(max_workers=len(qa_data)) as executor:
            # 提交任务到线程池
            if embed_type == "bge_code":
                bge_code_futures = [executor.submit(get_bge_code_embedding, q, embed_model) for q in qa_data.keys()]
            futures = [executor.submit(get_doubao_embedding, q) for q in qa_data.keys()]
            answers = [executor.submit(get_doubao_embedding, qa_data[q]) for q in qa_data.keys()]


            # vector
            results = [future.result() for future in as_completed(futures)]
            answers_results = [future.result() for future in as_completed(answers)]
            if config.USE_BGE_CODE:
                bge_results = [future.result() for future in as_completed(bge_code_futures)]
        for index,q in enumerate(list(qa_data.keys())):
            data_map = {
                "kw_question": q,
                "question": q, 
                "answer": qa_data[q],
                "answer_vector": answers_results[index],
                "q_vector": results[index],
                # "q_bge_code_vector": bge_results[index] if config.USE_BGE_CODE else [0.0] * 1536
            }
            action = {
                "_index": index_name,
                "_source": data_map
            }
            actions.append(action)

        yield actions
            
     

    def bulk_insert_qa(self, qa_data, index_name:str="alpha_qa", embed_type:str="doubao", embed_model=None):
        """
        qa_data: {"q1": a1, "q2": a2...}
        """
        actions = []
        batch_size = 10
        res = {}
        total = len(qa_data)
        try: 
            for q, a in qa_data.items():
                res[q] = a
                if len(res) % batch_size == 0:
                    total  -= len(res) 
                    for actions in self.batch_qa(res, actions, index_name, embed_type, embed_model):
                        logger.debug(f"{total} ready to es_qa")
                        time.sleep(1) # doubao 每分钟请求次数限制导致
                        helpers.bulk(self.es, actions)
                        actions = []
                        res = {}
            if res:
                for q in res.keys():
                    q_vector = get_doubao_embedding(q)

                    if embed_type == "bge_code":
                        q_bge_code_vector = get_bge_code_embedding(q, embed_model)
                        # answer_bge_code_vector = get_bge_code_embedding(qa_data[q], embed_model)

                    data_map = {
                    "kw_question": q,
                    "question": q, 
                    "answer": qa_data[q],
                    "answer_vector": get_doubao_embedding(qa_data[q]),
                    # "answer_bge_code_vector": answer_bge_code_vector,
                    "q_vector": q_vector,
                    # "q_bge_code_vector": q_bge_code_vector if embed_type == "bge_code" else [0.0] * 1536
                    }
                    action = {
                        "_index": index_name,
                        "_source": data_map
                    }
                    actions.append(action)
                helpers.bulk(self.es, actions)
                logger.info(f"last qa actions{len(actions)}")

        except BulkIndexError as e:
            logger.error(f"BulkIndexError: {e}")
            for error in e.errors:
                pass
                logger.error(f"Failed documents:{error}")
        logger.info(f"success insert num of {len(actions)} qa data to es")
    

    def update_qa(self, qa_data, index_name: str = "alpha_qa", embed_type: str = "doubao"):
        for q, a in qa_data.items():
            # 计算向量
            if embed_type == "doubao":
                q_vector = get_doubao_embedding(q)
            
            # 构造文档体
            doc = {
                "kw_question": q,
                "question": q,
                "answer": a,
                "q_vector": q_vector,
                "answer_vector": get_doubao_embedding(a)
            }
            
            # 1. 尝试根据 question 查找已有文档
            query = {
                "query": {
                    "match": {
                        "kw_question": q 
                    }
                }
            }
            resp = self.es.search(index=index_name, body=query, size=1)
            
            if resp["hits"]["total"]["value"] > 0:
                # 已有文档，取出文档 ID 并更新
                doc_id = resp["hits"]["hits"][0]["_id"]
                try:
                    self.es.update(
                        index=index_name,
                        id=doc_id,
                        body={"doc": doc}
                    )
                    logger.info(f"Updated existing QA, index_name:{index_name}, id={doc_id}, question={q!r}")
                except Exception as e:
                    logger.error(f"Failed to update doc id={doc_id}: {e}")
            else:
                # 不存在，则插入新文档
                try:
                    self.es.index(
                        index=index_name,
                        document=doc
                    )
                    logger.info(f"Inserted new QA,  index_name:{index_name}, question={q!r}")
                except Exception as e:
                    logger.error(f"Failed to insert new doc for question={q!r}: {e}")


    def bulk_insert(self, index_name, data):
        """
        批量插入数据到索引
        :param index_name: str, 索引名称
        :param data: list, 数据列表
        """
        actions = []
        for item in data:
            doc = {
                "_index": index_name,
                "_source": {
                    "sheet_name": item['sheet_name'],
                    "text": item['text'],
                    "embedding": item['embedding']
                }
            }
            actions.append(doc)
        if actions:
            bulk(self.es, actions)
            logger.info(f"批量插入 {len(actions)} 条数据成功")

    @perf_counter_timer
    async def search_qa_by_answer_async(self, index_name, answer: str, top_k: int = 5,
                                        embed_model_type: str = "doubao", embed_model=None):
        from aiohttp import BasicAuth
        auth = BasicAuth(config.ES_USER, config.ES_AUTH)
        if embed_model_type == "doubao":
            answer_vector = get_doubao_embedding(answer)  # 注意：必须是同步函数或改成 async 的

            script_score = {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'answer_vector') + 1.0",
                    "params": {"query_vector": answer_vector}
                }
            }

            body = {
                "size": top_k,
                "query": {
                    "script_score": script_score
                }
            }

        url = f"http://{config.ES_HOST}:{config.ES_PORT}/{index_name}/_search"  # 替换为你的 ES 地址

        async with aiohttp.ClientSession(auth=auth) as session:
            async with session.post(url, json=body) as resp:
                if resp.status != 200:
                    raise Exception(f"Search failed with status {resp.status}")
                res = await resp.json()
                return res["hits"]["hits"]


    def search_qa_by_answer(self, index_name, answer:str, top_k:int=5, embed_model_type:str="doubao", embed_model=None):
        """"""
        if embed_model_type == "doubao":
            answer_vector = get_doubao_embedding(answer)
            script_score= {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'answer_vector') + 1.0",
                        "params": {
                            "query_vector": answer_vector
                        }
                    }
                }
        body = {
            "size": top_k,
            "query": {
                "script_score": script_score
            }
        }
        res = self.es.search(index=index_name, body=body)
        logger.info(f"search_qa_by_answer cost: {res.get('took')} took")
        return res["hits"]["hits"]

    def search_qa_question(self, index_name, query:str, top_k:int=5, embed_model_type:str="doubao", embed_model=None):
        """余弦相似度检索最相关的topk个question"""
        if embed_model_type == "doubao":
            query_vector = get_doubao_embedding(query)
            script_score= {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'q_vector') + 1.0",
                        "params": {
                            "query_vector": query_vector
                        }
                    }
                }
        elif embed_model_type == "bgem3":
            query_vector = get_bgem3_embedding(embed_model, query)
            script_score= {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'q_bgem3_vector') + 1.0",
                    "params": {
                        "query_vector": query_vector
                    }
                }
            }
        elif embed_model_type == "bge_code":
            query_vector = get_bge_code_embedding(query, embed_model)
            script_score= {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'q_bge_code_vector') + 1.0",
                    "params": {
                        "query_vector": query_vector
                    }
                }
            }
                    
        else:
            # 使用 BGE zh模型进行向量化
            bgezh_model = None
            query_vector = get_embedding(query, bgezh_model)
            script_score= {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'q_bgezh_vector') + 1.0",
                    "params": {
                        "query_vector": query_vector
                    }
                }
            }
        body = {
            "size": top_k,
            "query": {
                "script_score": script_score
            }
        }
        res = self.es.search(index=index_name, body=body)
        logger.info(f"search_qa_question cost: {res.get('took')} took")
        return res["hits"]["hits"]


    def delete_qa(self,qa_data:dict, index_name:str="alpha_qa"):
        """
        删除指定问题的QA
        :param qa_data: {q1:a1,q2:a2}
        :param index_name: 索引名称，默认为 "alpha_qa"
        """
        try:
            for question, answer in qa_data.items():
                query = {
                    "query": {
                        "match": {
                            "kw_question": question
                        }
                    }
                }
                resp = self.es.search(index=index_name, body=query, size=1)
                assert len(resp['hits']['hits']) == 1
                print(f"resP: {resp['hits']['hits'][0]['_source']['kw_question']}")

                response = self.es.delete_by_query(
                    index=index_name,  # 指定索引名称
                    body={
                        "query": {
                            "match": {
                                "kw_question": question
                            }
                        }
                    }
                )

            logger.info(f"Deleted {response['deleted']} documents.")
        except Exception:
            logger.error(f"delete qa error {traceback.format_exc()}")
    
    def update_doctor(self, doctor_data:dict, index_name:str="alpha_doctor", embed_model:str="doubao"):
        """
        @param: doctor_data, {2: {"所在区域":"", ...}}
        @param: index_name, es index
        """
        actions = []
        try:
            for doctor_id, doc in doctor_data.items():
                text_for_embedding = doc["姓名"]+ "擅长：" + doc["擅长"]
                if embed_model == "doubao":
                    # 使用 doubao 模型进行向量化
                    doc["goodvector"] = [float(v) for v in get_doubao_embedding(text_for_embedding)]
                    doc["vector"] = [float(v) for v in get_doubao_embedding(doc["出诊地点"]+text_for_embedding)]
                else:
                    # 使用其他模型进行向量化
                    embed_model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
                    doc["goodvector"] = get_embedding(embed_model, text_for_embedding)
                self.es.index(index=index_name, document=doc)
            logger.info(f"doctor data {doctor_data.keys()} update to index {index_name} success")
        except Exception:
            logger.error(f"doctor_data {doctor_data.keys()} update to index {index_name} err for {traceback.format_exc()}")
    

    def delete_doctor(self, doctor_ids:List[str], index_name:str="alpha_doctor"):
        """删除指定id的医生，默认医生id唯一标识医生"""
        for id in doctor_ids:
            try:
                query = {
                    "query": {
                        "match": {
                            "ID": id
                        }
                    }
                }
                resp = self.es.search(index=index_name, body=query, size=1)
                assert len(resp['hits']['hits']) == 1
                print(f"resP: {resp['hits']['hits'][0]['_source']['姓名']}")

                response = self.es.delete_by_query(
                    index=index_name,  # 指定索引名称
                    body={
                        "query": {
                            "match": {
                                "ID": id
                            }
                        }
                    }
                )

                logger.info(f"Deleted {response['deleted']} doctor.")
            except Exception:
                logger.error(f"delete doctor error {traceback.format_exc()}")

    def update_disease(self, index_name:str,data:list):
        """增量更新病种医生: disease: doctor_names
        """
        for doc in data:
            if "primary_disease" in index_name: 
                doc = {"doctors": doc["专家姓名"], "primary_disease": doc["疾病种类"], "primary_disease_vector": get_doubao_embedding(doc["疾病种类"])}
            else:
                doc= {"doctors": doc["专家姓名"], "secondary_disease": doc["疾病种类"], "kw_secondary_disease": doc["疾病种类"]}
            self.es.index(index=index_name, document=doc)

es_handler = ElasticsearchHandler()
