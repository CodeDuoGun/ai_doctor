from app.utils.log import logger
from app.rag.vdb.es.es_processor import ElasticsearchHandler
from app.config import config
from app.utils.tool import gen_qa_by_file

def test_vector_search():
    from elasticsearch import Elasticsearch

    from elasticsearch_dsl import Search, Document, Index, connections, DenseVector, Text, Q
    connections.create_connection(hosts=['http://127.0.0.1:9200'], http_auth=("elastic","gungun"), timeout=30)
    es = Elasticsearch(f'http://localhost:9200', basic_auth=('elastic', 'gungun'))
    index_name = "doctor_info"
    search = Search(index=index_name)
    search = search.extra(size=10000)
    s = search.query()
    index = Index(index_name)
    index.refresh()
    response = s.execute()
    for i in range(len(response)):
        logger.debug(response[i])


if __name__=="__main__":
    import argparse
    from app.rag.split import load_and_split_docx
    if config.USE_BGE_CODE:
        from sentence_transformers import SentenceTransformer
    # bge_model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
    # from FlagEmbedding import BGEM3FlagModel
    # bgem3model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        bge_code_model = SentenceTransformer("BAAI/bge-code-v1")
    else:
        bge_code_model = None
    bgem3model = None
    bge_model = None
    parser = argparse.ArgumentParser(description="Elasticsearch操作")
    parser.add_argument("--index_name", type=str, default="base_qa", help="索引名称")
    parser.add_argument("--host", type=str, default="localhost", help="Elasticsearch主机地址")
    parser.add_argument("--port", type=str, default="9200", help="Elasticsearch端口")
    parser.add_argument("--embed_model", type=str, default="doubao", help="embedding 模型")
    parser.add_argument("--alpha", type=bool, default=True, help="是否为本地使用")

    args = parser.parse_args()
    es_handler = ElasticsearchHandler()
    # 写入qa
    print("load bge success")
    if args.index_name == "base_qa" or args.index_name == "test_word":
        file_path = "app/data/AI智能客服知识库5-6.docx" if args.index_name == "test_word" else "app/data/AI智能客服知识库5-6-删减版.docx"
        chunks = load_and_split_docx(file_path, chunk_size=500,chunk_overlap=10)
        # 入库
        es_handler.create_index_with_mapping(f"{config.env_version}_{args.index_name}")
        es_handler.index_documents(f"{config.env_version}_{args.index_name}", chunks)
    elif args.index_name == "test_qa":
        qa_mappings = {
            "mappings": {
                "properties": {
                    "kw_question": {"type": "keyword"},  # 关键词问题
                    "question": {"type": "text"},
                    "q_vector": {
                        "type": "dense_vector",  # 向量字段，用于向量检索
                        "dims": 1024 if args.embed_model!="doubao" else 4096,            # 维度数，依据你所用的嵌入模型（这里以 OpenAI 为例）
                        "index": True,
                        "similarity": "cosine"
                    },
                    "answer": {"type": "text"},
                    "answer_vector": {
                        "type": "dense_vector",  # 向量字段，用于向量检索
                        "dims": 1024 if args.embed_model!="doubao" else 4096,            # 维度数，依据你所用的嵌入模型（这里以 OpenAI 为例）
                        "index": True,
                        "similarity": "cosine"
                    },
                    # "q_bge_code_vector": {
                    #     "type": "dense_vector",  # 向量字段，用于向量检索
                    #     "dims": 1536,            # 维度数，依据你所用的嵌入模型（这里以 OpenAI 为例）
                    #     "index": True,
                    #     "similarity": "cosine"
                    # }
                }
            }
        }
        index_name = f"{config.env_version}_{args.index_name}"
        es_handler.create_index(index_name, mappings=qa_mappings, need_del=True)
        qa_res = gen_qa_by_file("app/data/qa_out_final_0526.csv")
        # print(qa_res)
        es_handler.bulk_insert_qa(qa_res, index_name,embed_type="bge_code", embed_model=bge_code_model)

    elif args.index_name == "qa":
        qa_mappings = {
        "mappings": {
            "properties": {
                "kw_question": {"type": "keyword"},  # 关键词问题
                "question": {"type": "text"},
                "q_vector": {
                    "type": "dense_vector",  # 向量字段，用于向量检索
                    "dims": 1024 if args.embed_model!="doubao" else 4096,            # 维度数，依据你所用的嵌入模型（这里以 OpenAI 为例）
                    "index": True,
                    "similarity": "cosine"
                },
                "answer": {"type": "text"},
                "answer_vector": {
                    "type": "dense_vector",  # 向量字段，用于向量检索
                    "dims": 1024 if args.embed_model!="doubao" else 4096,            # 维度数，依据你所用的嵌入模型（这里以 OpenAI 为例）
                    "index": True,
                    "similarity": "cosine"
                },
                # "bge_vector": {
                #     "type": "dense_vector",  # 向量字段，用于向量检索
                #     "dims": 1024,            # 维度数，依据你所用的嵌入模型（这里以 OpenAI 为例）
                #     "index": True,
                #     "similarity": "cosine"
                # }
            }
        }
    }
        index_name = f"{config.env_version}_qa"
        es_handler.create_index(index_name, mappings=qa_mappings, need_del=True)
        qa_res = gen_qa_by_file("app/data/qa_out_final_0611.csv")
        # print(qa_res)
        es_handler.bulk_insert_qa(qa_res, index_name)

    elif args.index_name == "chat_history":
        pass
    elif args.index_name == "doctor":
        print("准备写入医生信息")
        mapping = {
            "mappings": {
                "properties": {
                    "序号": {"type": "integer"},
                    # "姓名": {"type":  "text", "analyzer": "standard"},
                    "姓名": {"type":  "keyword"},
                    "ID": {"type": "keyword"},
                    "所在区域": {"type": "text", "analyzer": "standard"},
                    "特殊称号/职称": {"type": "keyword"},
                    "执业医院": {"type": "text", "analyzer": "standard"},
                    "goodvector": {"type": "dense_vector", "dims": 1024 if args.embed_model != "doubao" else 4096},
                    "简介": {"type": "text"},
                    "擅长": {"type": "text", "analyzer": "ik_max_word"},
                    "出诊地点": {"type": "text", "analyzer": "standard"},
                    "vector": {"type": "dense_vector", "dims": 1024 if args.embed_model != "doubao" else 4096}
                }
            }
        }
        es_handler.create_index(f"{config.env_version}_{args.index_name}", mappings=mapping, need_del=True)
        es_handler.bulk_insert_by_file("app/data/医师id 20250613.xlsx", f"{config.env_version}_{args.index_name}")

    # 写入 doctor
    elif args.index_name == "doctor_info":
        # TODO: @txueduo 修改字段为en
        print("准备写入医生信息")
        mapping = {
            "mappings": {
                "properties": {
                    "序号": {"type": "integer"},
                    # "姓名": {"type":  "text", "analyzer": "standard"},
                    "姓名": {"type":  "keyword"},
                    "ID": {"type": "keyword"},
                    "性别": {"type": "keyword"},
                    "出生年份": {"type": "text"},
                    "年龄": {"type": "integer"},
                    "所在区域": {"type": "text", "analyzer": "standard"},
                    "职称/职务": {"type": "keyword"},
                    "职位": {"type": "text", "analyzer": "standard"},
                    "擅长病种（新媒体推病种+擅长）": {"type": "text", "analyzer": "standard"},
                    "goodvector": {"type": "dense_vector", "dims": 1024 if args.embed_model != "doubao" else 4096},
                    "荣誉头衔": {"type": "text"},
                    "主要成就": {"type": "text"},
                    "教育背景": {"type": "text"},
                    "学历": {"type": "keyword"},
                    "抖音名称": {"type": "text"},
                    "快手名称": {"type": "text"},
                    "社会职务（3个）": {"type": "text"},
                    "挂号费": {"type": "text"},
                    "出诊时间": {"type": "text"},
                    "处方单价": {"type": "text"},
                    "治疗特色": {"type": "text", "analyzer": "standard"},
                    "治疗案例1": {"type": "text"},
                    "治疗案例2": {"type": "text"}
                }
            }
        }
        es_handler.create_index(f"{config.env_version}_{args.index_name}", mappings=mapping, need_del=True)
        es_handler.bulk_excel_insert("app/data/doctor_info_0409.xlsx", f"{config.env_version}_{args.index_name}")
    elif args.index_name == "primary_disease" or args.index_name == "secondary_disease":
        print(f"准备写{args.index_name}数据")
        if args.index_name == "primary_disease":
            mapping = {
                "mappings": {
                    "properties": {
                        "primary_disease": {"type": "text"},
                        # "姓名": {"type":  "text", "analyzer": "standard"},
                        "doctors": {"type":  "text"},
                        "primary_disease_vector": {"type": "dense_vector", "dims": 1024 if args.embed_model != "doubao" else 4096}
                    }
                }
            }
        else:
            mapping = {
                "mappings": {
                    "properties": {
                        "secondary_disease": {"type": "text"},
                        # "姓名": {"type":  "text", "analyzer": "standard"},
                        "doctors": {"type":  "text"},
                        "kw_secondary_disease": {"type": "keyword"}
                    }
                }
            }
        es_handler.create_index(f"{config.env_version}_{args.index_name}", mappings=mapping, need_del=True)
        sheet_name = "二级病种" if args.index_name == "secondary_disease" else "一级病种"
        es_handler.bulk_insert_by_file("app/data/四惠医疗互联网医院疾病库-20250508-V3.xlsx", f"{config.env_version}_{args.index_name}", sheet_name=sheet_name)

    # test_vector_search()
   