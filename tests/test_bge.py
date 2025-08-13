import time
import torch
from utils.log import logger
from ai_emr_customer.app.rag.vdb.es.es_processor import es_handler
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-code-v1")
print(f"load bge_code success")

from FlagEmbedding import BGEM3FlagModel
bgem3_model = BGEM3FlagModel('BAAI/bge-m3',  use_fp16=True) 

def test_bgem3_saerch(query, top_k:int=10):
    search_results = es_handler.search_by_vector(f"alpha_test_qa", query, top_k=top_k, embedding_model_type="bge_code", embed_model=model)
    context = "\n".join([f"内容: {hit['_source']['content']}" for hit in search_results])
    print(context)

def test_bge_search(query, is_qa:bool=False, top_k:int=10):
    if is_qa:
        qa_results = es_handler.search_qa_question(f"alpha_test_qa", query, top_k=top_k,embed_model_type="bge_code", embed_model=model)
        # qa_questions = [{"question": hit['_source']['question'], "answer": hit['_source']['answer'], "score": hit["score"]} for hit in qa_results]
        qa_questions = [{"question": hit['_source']['question'], "score": hit["_score"]} for hit in qa_results]
        print(f"bge search qa res: {qa_questions}")
        return qa_results
    else:   
        search_results = es_handler.search_by_vector(f"alpha_test_qa", query, top_k=top_k, embedding_model_type="bge_code", embed_model=model)
        context = "\n".join([f"内容: {hit['_source']['content']}" for hit in search_results])
        print(context)

def test_doubao_search(query, index_name="prod_qa", top_k:int=10):
    qa_results = es_handler.search_qa_question(index_name, query, top_k=top_k)
    # qa_questions = [{"question": hit['_source']['question'], "answer": hit['_source']['answer'], "score": hit["score"]} for hit in qa_results]
    qa_questions = [{"question": hit['_source']['question'], "score": hit["_score"]} for hit in qa_results]
    print(f"doubao search qa res: {qa_questions}")
    return qa_results

def test_doubao_answer_search(query, index_name="alpha_test_qa",top_k:int=10):
    qa_results = es_handler.search_qa_by_answer(index_name, query, top_k=top_k)
    # qa_questions = [{"question": hit['_source']['question'], "answer": hit['_source']['answer'], "score": hit["score"]} for hit in qa_results]
    qa_questions = [{"question": hit['_source']['question'], "answer": hit['_source']['answer'], "score": hit["_score"]} for hit in qa_results]
    print(f"doubao search qa res BY answer: {qa_questions}")
    return qa_results


def main():
    t0 =time.time()
    query = "四惠西区医院可以用医保么？"
    print(f"question: {query}")
    res1 = test_bge_search(query,is_qa=True)
    res2 = test_doubao_search(query)
    res3 = test_doubao_answer_search(query)
    
    rerank_data = [[query, hit["_source"]['question']]for hit in res1] + [[query, hit["_source"]['question']] for hit in res2] + [[query, hit["_source"]['question']] for hit in res3]
    # rerank_answer_data = [[query, hit["_source"]['question']]for hit in res1] + [[query, hit["_source"]['question']] for hit in res2] + [[query, hit["_source"]['question']] for hit in res3]
    logger.debug(f"rerank data: {rerank_data}")

    from FlagEmbedding import FlagReranker
    reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)  # Setting use_fp16 to True speeds up computation with a slight performance degradation
    scores = reranker.compute_score(rerank_data)
    logger.debug(f"rerank scores: {scores}")
    reranked_results = [{"text": result[1], "score": score} for result, score in zip(rerank_data, scores)]
    reranked_results.sort(key=lambda x: x["score"], reverse=True)

    # 输出重排后的结果
    for result in reranked_results:
        print(result)
        print("\n")
        
    print(f"search cost: {time.time() - t0}")
    
main()