from app.schema.chat import ChatRequest
from app.utils.log import logger
from app.model.llm_factory import LLMFactory
from app.rag.vdb.es.es_processor import es_handler
import traceback
from app.service.conversation import get_conversation, update_conversation
import requests
from app.constants import Speeches,MessageEventStatus, MODLE_MAPPING
import json
from app.rag.vdb.func_tools import extract_entities
from app.rag.vdb.doctor_retrieve import disease_doctors, get_doctor_ids, multi_search_doctor
import os
from app.config import config
from concurrent.futures import ThreadPoolExecutor,as_completed
import uuid
from app.prompts.customer_prompt import customer_system_prompt
from app.prompts.condition_summary_prompt import condition_summary_prompt
from app.prompts.think_prompt import think_system_prompt
from app.prompts.rewrite_question import rewrite_prompt
from app.utils.tool import perf_counter_timer 
from app.schema.chat import ChatRequest
import concurrent.futures
import time
import torch

interval_time = 0.04
# 模拟调用意图分类模型的接口
def limit_history_length(history, max_length=config.chat_history_num):
    """
    限制对话历史的长度，保留最新的 max_length 条消息。
    """
    # TODO: 对话历史暂时保留，其实不需要维护很多
    return history[-max_length*2:]
    while len(history) > max_length:
        history.pop(0)  # 删除最旧的消息
    return history


@perf_counter_timer
def judge_intent(query: str, convesation_id: str, chat_model:str="coze_deepseek"):
    """单轮意图识别，主要处理转人工需求"""
    prompt = f"""# 角色: 你是一个精准的意图识别专家，判断用户是否需要寻找人工客服
## 技能
- 精准识别文本内容是否包含"转人工", "人工客服" "人工"关键词。如果包含，输出0，否则，输出1
- 输出仅限两个数字 0 或 1,0表示需要寻找人工，1表示其他意图

## 限制
1. 严格限制返回结果为int 0 或 1。
2. 不附加解释说明，仅返回数字结果。
3. 严格依据"转人工", "人工客服" "人工"这三个关键词。
    """
    llm_factory = LLMFactory()
    llm = llm_factory.create_llm(chat_model)
    llm.init_client(config.ARK_API_KEY, config.ARK_BASE_URL)
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": query}]
    resp = llm.call_intent_stream(messages)
    logger.info(f"trace_id: {convesation_id}, query: {query} 意图结果：{resp}")
    return resp if isinstance(resp, int) else int(resp)
    

@perf_counter_timer
def handle_doctor_lookup(doctor_ids: list):
    """Method to handle doctor lookup by ids
    @param: doctor_ids: list, 医生的 ID 列表
    @return: json, 医生信息
    """
    result = []
    try:
        url = config.DOCTOR_SEARCH_URL
        resp = requests.post(url, json={"doctor_ids": doctor_ids})
        if resp.status_code != 200:
            logger.error(f"request url {url} failed {resp.status_code}")
            return result
        result = resp.json()["data"]
    except Exception:
        logger.error(f"request url {url} failed {traceback.format_exc()}")
    return result

def search_multi_doctor_by_name(trace_id, name):
    hits = []
    for doctor in name.split(","):
        logger.debug(name.split(","))
        res = es_handler.search_by_keyword( f"{config.env_version}_doctor", doctor,doctor_name=doctor)
        if res:
            hits.extend(res)
    logger.info(f"{trace_id} hits {len(hits)}")
    return hits

def retrieve_doctors(llm, user_query: str, history: list, trace_id:str, rewrite_query:str="",chat_model="coze_deepseek", must_name:bool=False):    
    ents = extract_entities(llm, trace_id, user_query, rewrite_query or user_query, must_name=must_name, history=history)  # 同前，使用大模型做 NER 提取
    name, doctor_location, cond, doctor_hospital = ents["doctor_name"], ents["doctor_location"], ents["condition"], ents["doctor_hospital"]
    logger.info(f"{trace_id}, ents:{ents}")
    # TODO: 确认是否符合要求
    if not name and not cond:
        cond = user_query
    by_name = False
    if name or must_name:
        by_name = True
    if must_name:
        # 如果根据人名，没有找到医生，应该返回未找到医生
        hits = search_multi_doctor_by_name(trace_id, name)
        if not hits:
            return hits, by_name

        # 根据id去重逻辑
        res = {}
        for hit in hits:
            res[hit['_source']["ID"]] = hit
        hits = list(res.values())
        logger.info(f"{trace_id}, doctor hits_res: {len(hits)}")
        return hits, by_name

    # 如果有人名
    if name:
        # 仅姓名 → 关键词检索
        hits = search_multi_doctor_by_name(trace_id, name)
    else:
        # 仅症状 → 混合检索
        kw_hits, vector_res = multi_search_doctor(cond, doctor_hospital, doctor_location)
        hits = list(kw_hits) + list(vector_res)

    # 根据id去重逻辑
    #TODO: 暂时注释
#     if hits:
#         doctors = "\n".join([f"医生姓名：{hit['_source']['姓名']}, 所属医院: {hit['_source']['出诊地点']}, 擅长疾病: {hit['_source']['擅长']}" for hit in hits])
#         system_prompt = f"""
# ## 角色：你是一个精准的医生推荐专家
# ## 工作流
#     1. 根据医生信息,为用户选择最适合用户的医生.
#     2. 严格以list的形式返回所有符合的医生姓名

# ## 限制
#     1. 只返回医生姓名列表，禁止输出其他内容。

# ## 输出示例
# 示例1: 有符合条件的医生
#     输出： ["张三", "李四", "王五"]
# 示例2: 没有符合条件的医生
#     输出： []
# """
#         messages = [{"role": "system", "content": system_prompt},
#                     {"role": "user", "content": rewrite_query},
#                     {"role": "user", "content": f"请根据以下医生信息，选择最适合用户的医生：\n{doctors}"}]
#         response = llm.call_intent_stream(messages, chat_model="doubao-1.5-pro-32k-250115",temperature=0.1, json_schema=None, max_tokens=200)
#         # response = llm.call_intent_stream(messages, json_schema=None, max_tokens=800)
#         logger.info(f"{trace_id}, LLM filterd doctor result: {response}, resp type: {type(response)}")
#     # TODO: json_schema 需要定义

#     if response:
#         if isinstance(response, str):
#             try:
#                 # 尝试将字符串解析为 JSON
#                 response = json.loads(response)
#             except Exception:
#                 logger.error(f"{trace_id}, LLM filterd doctor result is not json: {response}")
#                 response = "" 
#         if isinstance(response, list):
#             # 如果有医生姓名，过滤医生
#             hits = [hit for hit in hits if hit['_source']['姓名'] in response]
#             logger.info(f"{trace_id}, LLM filterd doctor hits: {len(hits)}")
#             return hits[:8], by_name
            
    if hits:
        res = {}
        for hit in hits:
            res[hit['_source']["ID"]] = hit
        hits = list(res.values())
        logger.info(f"{trace_id}, doctor hits_res: {len(hits)}")
    return hits[:8], by_name

def introduce_doctors(llm, query, rewrited_query, sse_data, history, conversation_id, trace_id):
    """
    """
    hits_res, by_name = retrieve_doctors(llm, query, history, trace_id, rewrited_query, must_name=True)
    if not hits_res:
        text_chunks = [Speeches.NoDoctorSpeech[i:i+config.CHUNK_SIZE] for i in range(0, len(Speeches.NoDoctorSpeech), config.CHUNK_SIZE)]
        for text in text_chunks:
            sse_data["content"] = text
            yield f"event: conversation\n\n"
            yield f'data: {json.dumps(sse_data, ensure_ascii=False)}\n\n'  
            time.sleep(interval_time)

        # history.append({"role": "tool", "content": []})
        sse_data["event"] = MessageEventStatus.COMPLETED
        sse_data["content"] = "\n\n" + Speeches.NoDoctorSpeech
        logger.info(f"{trace_id},msg_id: {sse_data['msg_id']}, content: {Speeches.NoDoctorSpeech}")
        history.append({"role": "assistant", "content":sse_data["content"]})
        update_conversation(conversation_id, history, trace_id)
        yield f"event: conversation\n\n"
        yield f'data: {json.dumps(sse_data, ensure_ascii=False)}\n\n'
        yield f"event: [DONE]\n\n"
        yield f"data: [DONE]\n\n"
    else:
        # TODO: 理论上只返回一个医生，返回文本类型的医生介绍
        content = ""
        for hit in hits_res[:1]:
            doctor_info = ""
            good_at_info = ""
            doctor_info = f"{hit['_source']['姓名']}"
            if hit['_source']['简介']:
                doctor_info+=f"\n简介：{hit['_source']['简介'].strip('、')}"
            good_at_info = f"\n擅长：{hit['_source']['擅长'].strip('、')}"
            if hit['_source']['出诊地点']:
                doctor_info += f"\n出诊地点：{hit['_source']['出诊地点'].strip('、')}"
            logger.debug(f"doctor_info:{doctor_info}")
            logger.debug(f"good_at_info: {good_at_info}")
            
            # 让大模型来介绍
            prompt = f"""你是一个专业的医疗助手，擅长介绍医生信息。根据你现有的知识，请根据以下医生资料:\n{doctor_info}\n{good_at_info}\n，生成一段口语化的介绍.\n
            限制：
                1.禁止给出医生资料之外的信息，如联系方式、出诊费用、出诊时间。
                2.只输出医生的介绍，包括：姓名、简介、出诊地点、擅长。严格禁止输出其他内容。
                3.最后请用自然、亲切的语气结束，例如：“如果您有需要，我可以帮您推荐合适的医生进行挂号～还有其他想了解的吗？”
                4.对于资料中没有的信息，请明确指出“暂时没有相关信息”或“资料中未提及”，并提示：您也可以直接描述您的症状，小惠会尽力为您推荐合适的医生哟。
            """
            messages = [{"role": "system", "content": prompt}] + history
            responses =  llm.chat_coze_stream(sse_data, messages, history, conversation_id, trace_id, query, chat_model="coze_deepseek", need_stream_content=True, need_update_history=True)
            for chunk in responses:
                yield chunk
            content += sse_data["content"]

        logger.info(f"{trace_id}, Got doctor info of {len(hits_res)} results: {content}")
        logger.info(f"{trace_id}, success send data :{sse_data}")
        yield f"event: [DONE]\n\n"
        yield f"data: [DONE]\n\n"
    

def search_qa(index_name, query, top_k):
    return es_handler.search_qa_question(index_name, query, top_k=top_k)

def search_qa_answer(index_name, query, top_k):
    return es_handler.search_qa_by_answer(index_name, query, top_k=top_k)

@perf_counter_timer
def perform_searches(rewrite_query, query):
    index_name = f"{config.env_version}_qa"
    qa_size = config.qa_size
    qa_answer_size = config.qa_answer_size

    # 定义两个查询任务
    tasks = [
        (search_qa, index_name, rewrite_query, qa_size),
        (search_qa_answer, index_name, query, qa_answer_size)
    ]

    # 使用 ThreadPoolExecutor 并发执行任务
    with concurrent.futures.ThreadPoolExecutor(3) as executor:
        # 提交任务到线程池
        futures = [executor.submit(func, *args) for func, *args in tasks]

        # 收集结果
        results = []
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error during search: {e}")

    return results


def intent_rewrite_agent(llm, history, query,trace_id):
    """
    """

    try:
        messages = [      
            {"role": "system", "content": rewrite_prompt}] + history
        
        new_query = llm.call_intent_stream(messages, chat_model="doubao-1.5-pro-32k-250115", temperature=0.8)
        logger.info(f"trace_id: {trace_id}  rewrite {query} TO {new_query}") 
    except Exception:
        logger.error(f"trace_id:{trace_id} rewrite {query} error {traceback.format_exc()}")
    return new_query or query

def doubao_search_task(trace_id, query, top_k:int=7):
    # search_results = es_handler.search_by_vector(f"{config.env_version}_base_qa", query, top_k=top_k, embedding_model="doubao")
    search_results = es_handler.hybrid_search_doc(f"{config.env_version}_base_qa",query, top_k=top_k)

    # logger.debug(search_results)
    scores = [hit['_score'] for hit in search_results]
    logger.info(f"{trace_id}, Got doc scores {scores}")
    res = [] 
    for hit in search_results:
        tmp = {}
        tmp["score"] =hit['_score']
        tmp["content"] = hit['_source']['content']
        res.append(tmp)
    logger.debug(f"****doubao doc: {res}")
    # search_results = [f"内容: {hit['_source']['content']}" if hit['_score'] > config.score_threshold else "" for hit in search_results]
    search_results = [f"内容: {hit['_source']['content']}" for hit in search_results]
    if search_results:
        context = "\n".join(search_results)
    else:context=""
    return search_results

def bge_search_task(trace_id, query, top_k:int=7, bge_model=None):
    search_results = es_handler.hybrid_search_doc(f"{config.env_version}_base_qa",query, top_k=top_k,bge_model=bge_model)
    # search_results = es_handler.search_by_vector(f"{config.env_version}_base_qa", query, top_k=top_k, bge_model=bge_model)
    # logger.debug(search_results)
    scores = [hit['_score'] for hit in search_results]
    logger.info(f"{trace_id}, Got doc scores {scores}")
    res = [] 
    for hit in search_results:
        tmp = {}
        tmp["score"] =hit['_score']
        tmp["content"] = hit['_source']['content']
        res.append(tmp)
    # search_results = [f"内容: {hit['_source']['content']}" if hit['_score'] > config.score_threshold else "" for hit in search_results]
    logger.debug(f"****bge doc: {res}")
    search_results = [f"内容: {hit['_source']['content']}" for hit in search_results]
    if search_results:
        context = "\n".join(search_results)
    else:context=""
    return search_results
    
def multi_search_task(trace_id, llm, rewrite_query, qa_results, questions, qa_questions,history, qa_results_by_answer, tokenizer, reranker):
    with ThreadPoolExecutor(3)  as executor:
        # reranked_results = gen_reranked_results(trace_id, rewrite_query, qa_results, qa_results_by_answer, tokenizer, reranker)
        # most_similar_answer, final_qa_idx = most_similar_task(trace_id, llm, rewrite_query, qa_results, questions, qa_questions,history)
        task1 = executor.submit(gen_reranked_results, trace_id, rewrite_query, qa_results, qa_results_by_answer, tokenizer, reranker)
        task2 = executor.submit(most_similar_task, trace_id, llm, rewrite_query, qa_results, questions, qa_questions,history)
        
        res1 = task1.result()
        res2 = task2.result()
    return [res1, res2]

def most_similar_task(trace_id, llm, rewrite_query, qa_results, questions, qa_questions,history, top_k:int=7):
    answers = [hit['_source']['answer'] for hit in qa_results]
    qa_prompt = f"""你将获得一个问题, 不要回答问题，根据对话历史找出最符合用户想要咨询的问题。并输出对应的序号。\n
## 要求：\n"""
    for q_idx, q in enumerate(questions):
        qa_prompt += f"如果问题:{q} 最符合用户咨询的问题,就输出{q_idx}\n"
    qa_prompt += f"""\n用户咨询的问题：{rewrite_query}\n
## 判断标准强化说明：
1.识别医院/平台：
用户问题中如果提及“互联网医院”、“线上医院”或未指明但语境明显为线上服务，则归类为“互联网医院”。
准确区分不同院区（见下方医院别名表），避免混淆四惠、郁证中心等其他机构的问题。
2.理解用户意图：
关注关键词：退挂号费、退款、退药、物流、加号、预约、复诊、客服、看诊、视频问题等。


## 限制：
1. 只输出数字，不要输出其他内容

## 技能
1. 根据对话历史找出最符合用户想要咨询的问题。并输出对应的序号
2. 辨别用户问题中，医院的不同叫法，可能的医院叫法如下：
北京四惠中医医院：四惠总院、总院、四惠中医、四惠中医院、北京四惠中医院
北京四惠西区医院：四惠西区、西区医院、西区、四惠西区医院、北京四惠西区
北京四惠南区门诊部：四惠南区、南区医院、南区门诊、南区门诊部、北京四惠南区
上海圣保堂中医门诊：上海四惠、上海四惠医院、上海圣保堂、圣保堂、圣保堂门诊、上海四惠医院
南宁桂派中医门诊：南宁四惠医院、南宁四惠、南宁桂派、桂派、桂派中医、桂派中医门诊
杭州四惠医院：杭州四惠、杭州四惠医院、杭州医院
郑州四惠中医门诊部：郑州医馆、郑州门诊/医院
北京四惠中西医结合肿瘤会诊中心：会诊中心、会诊、中西医结合会诊、中西医结合会诊中心、四惠会诊中心
北京中医郁证临床学科示范基地：郁证中心、郁证示范基地、郁证基地
互联网医院：互联网医院、线上医院

## 示例：
示例1: 正确输出
问题列表：
如果问题：“如何预约参加市图书馆的阅读活动？” 最符合用户咨询的问题，就输出0
如果问题：“市图书馆有哪些儿童阅读活动？” 最符合用户咨询的问题，就输出1
如果问题：“市图书馆有哪些著名作家的讲座？” 最符合用户咨询的问题，就输出2
如果问题：“市图书馆的开放时间和服务特色是什么？” 最符合用户咨询的问题，就输出3
如果问题：“市图书馆有哪些推荐的阅读书目？” 最符合用户咨询的问题，就输出4
如果问题：“如何预约市图书馆的讲座？” 最符合用户咨询的问题，就输出5
问题：
“怎么预约你们的讲座？”
输出：5
示例2: 错误输出, 未精确区分问题中的主体
问题列表：
如果问题：“四惠西区医院可以做CT检查吗” 最符合用户咨询的问题，就输出0
如果问题：“四惠西区医院提供哪些影像检查服务？” 最符合用户咨询的问题，就输出1
如果问题：“四惠中医医院可以做CT检查吗” 最符合用户咨询的问题，就输出2
如果问题：“四惠南区门诊的具体位置在哪里？” 最符合用户咨询的问题，就输出3
如果问题：“四惠西区医院是否设有住院部？” 最符合用户咨询的问题，就输出4
如果问题：“四惠南区门诊的特色是什么？” 最符合用户咨询的问题，就输出5
问题：四惠南区医院可以做核磁吗？
错误输出：1
正确输出：5
    """

    logger.debug(f"qa_promt: {qa_prompt}")
    system_prompt = [{"role": "system", "content": qa_prompt}]
    final_qa_idx, retry_num = 0, 0
    while retry_num < 4:
        final_qa_idx = llm.call_intent_stream(system_prompt+history,json_schema=None)
        try:
            final_qa_idx = int(final_qa_idx)
            break
        except Exception:
            retry_num +=1
            logger.info(f"{trace_id} retried {retry_num}")
            final_qa_idx = 0
    # 根据最佳内容找到最佳答案。并返回
    logger.info(f"score: {qa_questions[final_qa_idx]['score']}, {trace_id} most similar question is {final_qa_idx}:{questions[final_qa_idx]}:{answers[final_qa_idx]}")
    return answers[final_qa_idx], final_qa_idx


def gen_reranked_results(trace_id, query, qa_results, qa_results_by_answer, tokenizer, reranker):
    rerank_data = [[query, hit["_source"]['question'],hit["_source"]['answer']]for hit in qa_results] + [[query, hit["_source"]['question'],hit["_source"]['answer']] for hit in qa_results_by_answer]
    rerank_score_data = [[query, hit["_source"]['question']] for hit in qa_results] + [[query, hit["_source"]['question']] for hit in qa_results_by_answer] + [[query, hit["_source"]['question']] for hit in qa_results[:2]]
    # 加入rank
    with torch.no_grad():
        inputs = tokenizer(rerank_score_data, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
    # scores = reranker.compute_score(rerank_data)
    logger.debug(f"rerank scores: {scores}")
    reranked_results = [{"question": result[1], "score": score, "answer": result[2]} for result, score in zip(rerank_data, scores)]
    reranked_results.sort(key=lambda x: x["score"], reverse=True)
    
    # answer 去重, 如果answer相同，删除重复的
    # reranked_results = sorted(reranked_results, key=lambda x: x["score"], reverse=True)
    reranked_results = [result for i, result in enumerate(reranked_results) if result["answer"] not in [r["answer"] for r in reranked_results[:i]]]
    logger.info(f"{trace_id} reranked results: {reranked_results[:10]}")
    return reranked_results


def chat_service(request:ChatRequest, backend_id, bge_embedding_model=None, reranker=None, tokenizer=None):
    """
    聊天服务service,该服务负责处理聊天请求，并返回相应的结果。
    """
    trace_id = f"{backend_id}_{request.conversation_id}"
    try:
        history = get_conversation(request.conversation_id)
        # TODO: history 目前存储了10轮，不删除
        if history:
            logger.info(f"{trace_id}, get chat history: {history}")
            history = limit_history_length(history)
        else: history=[]
        llm_factory = LLMFactory()
        # TODO: @txueduo 获取 model_config
        llm = llm_factory.create_llm(request.model_name)
        if request.model_name == "deepseek":
            llm.init_client(config.DEEPSEEK_API_KEY, config.DEEPSEEK_API_URL)
        else:
            llm.init_client(config.ARK_API_KEY, config.ARK_BASE_URL)
        query = request.query.strip()
        sse_data = {"conversation_id":request.conversation_id,"msg_id": str(uuid.uuid4()), "event": MessageEventStatus.DELTA, "role":"assistant","content":"","content_type":"text", "reasoning_content": "", "finish_reasoning": False, "tools": []}
        history.append({"role": "user", "content":query})

        # 改用下model, 开启深度思考时使用r1， 未开启使用豆包pro
        if request.model_name == "coze_deepseek-r1":
            messages = [{"role": "system", "content": think_system_prompt}] + history
            resp = llm.chat_coze_stream(sse_data, messages, history, request.conversation_id, trace_id, query, chat_model=request.model_name, max_tokens=500, temperature=0.8)
            for chunk in resp:
                if "DONE" in chunk:
                    # response = chunk[5:]
                    break
                yield chunk
        # refer_doctors, by_name = retrieve_doctors(llm, query, history, trace_id, query, must_name=True)
        # refer_doctor_info = [f'医生：{hit["_source"]["姓名"]}，出诊医院：{hit["_source"]["出诊地点"]}' if hit else "" for hit in refer_doctors]
        messages = [{"role": "system", "content": customer_system_prompt}] + history
        response = llm.call_intent_stream(messages, chat_model="doubao-1.5-pro-32k-250115",temperature=0.1, json_schema=None, max_tokens=800)
        logger.info(f"{trace_id}, {query}, intent result: {response}")
        intent = response
        # 人名检索
        if "disease_search" in intent:
            rewrite_query = intent_rewrite_agent(llm, history, query,trace_id)
            # 如果按照疾病种类未找到，走doctor检索
            hits_res = disease_doctors(llm, trace_id,rewrite_query)
            if not hits_res:
                hits_res, by_name = retrieve_doctors(llm, query, history, trace_id, rewrite_query)
            
            if hits_res:
                doctor_ids = get_doctor_ids(trace_id, f"{config.env_version}_doctor", hits_res)
                doctor_info = handle_doctor_lookup(doctor_ids)
            if not hits_res or not doctor_info:
                text_chunks = [Speeches.NoDoctorSpeech[i:i+config.CHUNK_SIZE] for i in range(0, len(Speeches.NoDoctorSpeech), config.CHUNK_SIZE)]
                for text in text_chunks:
                    sse_data["content"] = text
                    yield f"event: conversation\n\n"
                    yield f'data: {json.dumps(sse_data, ensure_ascii=False)}\n\n'
                    time.sleep(interval_time)

                sse_data["event"] = MessageEventStatus.COMPLETED
                sse_data["content"] = Speeches.NoDoctorSpeech
                history.append({"role": "assistant", "content":Speeches.NoDoctorSpeech})
                update_conversation(request.conversation_id, history,trace_id)
                yield f"event: conversation\n\n"
                yield f'data: {json.dumps(sse_data, ensure_ascii=False)}\n\n'  
                logger.info(f"{trace_id} send content: {Speeches.NoDoctorSpeech}")
                yield f"event: [DONE]\n\n"
                yield f"data: [DONE]\n\n"
            else:
                # 1. 先返回推荐医生固定话术
                sse_data["event"] = MessageEventStatus.DELTA
                chunk_lists = [Speeches.RecommendDoctorSpeech[i:i+config.chunk_size] for i in range(0, len(Speeches.RecommendDoctorSpeech), config.chunk_size)]
                for chunk in chunk_lists:
                    sse_data["content"] = chunk
                    # logger.info(f"Sending {trace_id}, query:{query}, content: {chunk}")
                    yield f"event: conversation\n\n"
                    yield f"data: {json.dumps(sse_data, ensure_ascii=False)}\n\n"
                    time.sleep(interval_time)
                sse_data["content"] = "\n" + Speeches.RecommendDoctorSpeech
                sse_data["event"] = MessageEventStatus.COMPLETED
                yield f"event: conversation\n\n"
                yield f"data: {json.dumps(sse_data, ensure_ascii=False)}\n\n"
                # 2. 再返回医生列表
                sse_data["content_type"] = "object"
                sse_data["content"] = {"doctor_data": doctor_info}
                yield f"event: conversation\n\n"
                yield f"data: {json.dumps(sse_data, ensure_ascii=False)}\n\n"
                history.append({"role": "assistant", "content":Speeches.RecommendDoctorSpeech})
                history.append({"role": "tool", "name": "doctor_search", "tool_call_id": "0","content": json.dumps(doctor_info, ensure_ascii=False)})
                # history.append({"role": "assistant", "content":Speeches.RecommendDoctorSpeech})
                update_conversation(request.conversation_id, history,trace_id)
                logger.info(f"{trace_id}, success send data :{sse_data}")
                yield f"event: [DONE]\n\n"
                yield f"data: [DONE]\n\n"

        elif "doctor" in intent:
            rewrite_query = intent_rewrite_agent(llm, history, query,trace_id)
            # 病情小结+健康科普
            condition_messages = [{"role": "system", "content": condition_summary_prompt}] + history
            condition_summary = llm.chat_coze_stream(sse_data, condition_messages, history, request.conversation_id, trace_id, query, chat_model="coze_deepseek", need_stream_content=True)
            for chunk in condition_summary:
                yield chunk
            # 根据对话内容生成对话小结
            # condition_content = sse_data['content']
            # hits_res, by_name = retrieve_doctors(llm, query, history, trace_id, condition_content) # TODO: 用这个提取不到医院信息,后面优化
            hits_res, by_name = retrieve_doctors(llm, query, history, trace_id,  rewrite_query)
            doctor_ids = []
            if hits_res:
                doctor_ids = []
                for hit in hits_res:
                    doctor_id = hit['_source']['ID']
                    if isinstance(doctor_id, str):
                        ids = doctor_id.split("、")
                        doctor_ids.extend(ids)
                    else: 
                        ids = doctor_id
                        doctor_ids.append(ids)
            elif not by_name:
                doctor_ids = config.DEFAULT_DOCTOR_IDS
            logger.info(f"{trace_id} got doctor ids {doctor_ids}")
            doctor_info = handle_doctor_lookup(doctor_ids)
            # doctor_material = []
            # for doc in doctor_info:
            #     # 姓名 医院 擅长
            #     doctor_material.append(doc["doctor_name"], )
            if (not hits_res and by_name) or not doctor_info:
                text_chunks = [Speeches.NoDoctorSpeech[i:i+config.CHUNK_SIZE] for i in range(0, len(Speeches.NoDoctorSpeech), config.CHUNK_SIZE)]
                for text in text_chunks:
                    sse_data["content"] = text
                    yield f"event: conversation\n\n"
                    yield f'data: {json.dumps(sse_data, ensure_ascii=False)}\n\n'
                    time.sleep(interval_time)

                sse_data["event"] = MessageEventStatus.COMPLETED
                sse_data["content"] = Speeches.NoDoctorSpeech
                history.append({"role": "assistant", "content":Speeches.NoDoctorSpeech})
                update_conversation(request.conversation_id, history,trace_id)
                yield f"event: conversation\n\n"
                yield f'data: {json.dumps(sse_data, ensure_ascii=False)}\n\n'  
                logger.info(f"{trace_id} send content: {Speeches.NoDoctorSpeech}")
                yield f"event: [DONE]\n\n"
                yield f"data: [DONE]\n\n"
                return
            

            # 2. 先返回推荐医生固定话术
            sse_data["event"] = MessageEventStatus.DELTA
            chunk_lists = [Speeches.RecommendDoctorSpeech[i:i+config.chunk_size] for i in range(0, len(Speeches.RecommendDoctorSpeech), config.chunk_size)]
            sse_data["content"] = "\n\n"
            yield f"event: conversation\n\n"
            yield f'data: {json.dumps(sse_data, ensure_ascii=False)}\n\n'  
            for chunk in chunk_lists:
                sse_data["content"] = chunk
                # logger.info(f"Sending {trace_id}, query:{query}, content: {chunk}")
                yield f"event: conversation\n\n"
                yield f"data: {json.dumps(sse_data, ensure_ascii=False)}\n\n"
                time.sleep(interval_time)
            sse_data["content"] = "\n" + Speeches.RecommendDoctorSpeech
            sse_data["event"] = MessageEventStatus.COMPLETED
            yield f"event: conversation\n\n"
            yield f"data: {json.dumps(sse_data, ensure_ascii=False)}\n\n"
            # 3. 再返回医生列表
            sse_data["content_type"] = "object"
            sse_data["content"] = {"doctor_data": doctor_info}
            yield f"event: conversation\n\n"
            yield f"data: {json.dumps(sse_data, ensure_ascii=False)}\n\n"
            history.append({"role": "assistant", "content":Speeches.RecommendDoctorSpeech})
            history.append({"role": "tool", "name": "doctor_search", "tool_call_id": "0","content": json.dumps(doctor_info, ensure_ascii=False)})
            # history.append({"role": "assistant", "content":Speeches.RecommendDoctorSpeech})
            update_conversation(request.conversation_id, history,trace_id)
            logger.info(f"{trace_id}, success send data :{sse_data}")
            yield f"event: [DONE]\n\n"
            yield f"data: [DONE]\n\n"
        elif "introduce" in intent:# or doctor_hits:
            rewrite_query = intent_rewrite_agent(llm, history, query,trace_id)
            response = introduce_doctors(llm, query, rewrite_query, sse_data, history, request.conversation_id, trace_id)
            for chunk in response:
                time.sleep(interval_time)
                yield chunk
        elif "qa" in intent:
            rewrite_query = intent_rewrite_agent(llm, history, query,trace_id)
            # agent 召回语义最相关的10个问题，让llm给出最符合用户问题的q。根据q对应的a，给出权威严谨的回复。
            rewrite_query = rewrite_query.replace("总院", "中医医院").replace("北京四惠医疗互联网医院","互联网医院")
            results1, results2 = perform_searches(rewrite_query, query)
            qa_results, qa_results_by_answer = (results1, results2) if len(results1) > len(results2) else (results2, results1)
            
            qa_questions = [{"question": hit['_source']['question'], "score": hit["_score"]} for hit in qa_results]
            qa_answer_questions = [{"question": hit['_source']['question'], "score": hit["_score"]} for hit in qa_results_by_answer]
            questions = [hit['_source']['question'] for hit in qa_results]
            logger.info(f"{trace_id} recalled {len(qa_questions)} QA: {qa_questions}\n QA_ANSWER: {qa_answer_questions}")
            if reranker:
                reranked_results, similar_results = multi_search_task(trace_id, llm, rewrite_query, qa_results, questions, qa_questions,history, qa_results_by_answer, tokenizer, reranker)
                most_similar_answer = similar_results[0]
                # reranked_results = gen_reranked_results(trace_id, rewrite_query, qa_results, qa_results_by_answer, tokenizer, reranker)
                # most_similar_answer = ""

            else:
                most_similar_answer, final_qa_idx = most_similar_task(trace_id, llm, rewrite_query, qa_results, questions, qa_questions,history)
    
            # 根据问题作为参考答案
            if reranker:
                final_refer_answers = "\n\n".join([hit['answer'] for hit in reranked_results[:7]])
                final_refer_answers = "\n参考资料:\n" + final_refer_answers + most_similar_answer
            else:
                refer_answers_results = [{"question": hit['_source']['question'], "answer": hit['_source']['answer'], "score": hit["_score"]} for hit in qa_results_by_answer] 
                logger.info(f"{trace_id} refer QA answers: {refer_answers_results}")
                refer_answers = "\n\n".join([hit['_source']['answer'] for hit in qa_results_by_answer])
                final_refer_answers = "\n参考资料:\n" + refer_answers + most_similar_answer
            logger.info(f"{trace_id} final refer answers:\n {final_refer_answers}")
            # if True:
            #     answer_prompt = f"根据你现有的知识，辅助以搜索到的文件资料：\n{final_refer_answers}\n 回答问题：\n{rewrite_query}\n 尽可能多的覆盖到文件资料"
            # else:
            answer_prompt = f"""根据对话历史，针对用户问题 {rewrite_query} 从答案{final_refer_answers}中精确提取最相关的内容， 简洁清晰有礼貌的给出答案。禁止输出其他无关内容\n
            技能：
                1. 精准提取答案中的关键信息，紧扣用户问题进行回应，保证回答内容与问题匹配，不能答非所问。
                2. 若答案中没有能准确回应用户问题的信息（即使“部分相关”但不是用户“真正关注点”），参考回复如下：
                    您好！这个问题小惠还在学习中，您先联系人工看看，或者拨打客服热线400−689−6699吧。我会加速学习，来更好的为您服务！
                3. 精准区分用户问题是针对以下哪个医疗机构，并给出相应的回答。
                    机构名称：
                    北京四惠中医医院
                    北京四惠西区医院
                    北京四惠南区门诊部
                    上海圣保堂中医门诊
                    南宁桂派中医门诊
                    杭州四惠医院
                    郑州四惠中医门诊部：郑州医馆、郑州门诊/医院
                    北京四惠中西医结合肿瘤会诊中心
                    郁证临床学科示范基地：郁证中心、郁证示范基地、郁证基地
                    互联网医院：互联网医院、线上医院

                4. 如果询问“退挂号费” “退号/取消预约/取消挂号”时，从以下内容中提取相关信息：
                    您好～为了不耽误您的退号办理，小惠将各家医院的退号退款方式整理如下啦👇
                        1. 线下医院（如北京四惠中医医院、西区、南区、杭州、上海、南宁、郑州）：
                        - 请在就诊当天，携带身份证或医保卡前往挂号窗口或自助机办理退号
                        - 拨打各医院客服电话协助您退款退号: 1.北京四惠中医医院：010-67289999\n2.北京四惠西区医院：010-88849999\n3.北京四惠南区医院：010-67289966\n4.中西医结合肿瘤会诊中心：010-67289999\n5.杭州四惠医院：0571-8868 5312\n6.南宁桂派中医门诊：0771-5556788\n7.上海圣保堂中医门诊： 021-66261616\n8.郁证中心：010-67289999\n9.郑州四惠中医门诊：0371-65012822

                        2. 互联网医院：
                        - 可联系您的医助协助退款，或拨打客服热线 400-689-6699
                        费用一般 3-7 个工作日到账，如有问题可随时联系小惠哈～

                    您好!线下医院、四惠中医医院、四惠西区医院、四惠南区中医门诊部、杭州四惠医院、圣保堂中医门诊部、南宁桂派中医门诊部、中西医结合肿瘤会诊中心、郑州门诊退号方法及流程如下：\n1.携带身份证或医保卡\n2.前往挂号窗口或自助机办理退号\n3.退号需在就诊当日完成\n4.联系人工客服或拨打各医院的客服电话协助您退号退款，给您带来的不便，敬请谅解！
                    四惠医疗互联网医院、郁症示范基地的退号方法如下：\n1、联系服务您的医助，医助会及时为您办理\n2、联系人工客服或者拨打400-689-6699客服电话\n给您带来的不便，敬请谅解！
    
                5. 如果询问“药品退货类”，“复诊/加号类“， “退款类（包含药品/检查项目/挂号等费用退还）“,"物流/快递等问题"时，从以下内容中提取相关信息：
                    关于线下医院、四惠中医医院、四惠西区医院、四惠南区中医门诊部、杭州四惠医院、圣保堂中医门诊部、南宁桂派中医门诊部、中西医结合肿瘤会诊中心的退款/退药问题、复诊或者加号问题、物流或快递问题、药品配送问题的解决办法和流程如下:\n1、联系服务您的医助，医助会及时为您办理\n2、联系人工客服或拨打各医院的客服电话，给您带来的不便，敬请谅解! 

                    关于互联网医院复诊流程小惠为您说明:\n1.互联网医院复诊建议提前5天预约挂号\n2.若即将断药，可通过互联网医院提前复诊
                    关于互联网医院的退款/退药问题，解决办法和流程如下：\n1.联系服务您的医助，医助会及时为您办理\n2.联系人工客服或者拨打400-689-6699客服电话\n给您带来的不便，敬请谅解！

                    退费到账时间一般3～7个工作日，如果没有收到相应退款，请联系客服人员及时为您处理！
        
                6. 如果询问“视频看诊可信度”时，或“视频看诊链接获取”问题时，从以下内容中提取相关信息：
                    您可以放心，互联网视频看诊在中医领域已经得到了国家的认可和推广。古代中医没有检查设备，只能靠望闻问切诊断疾病，随着医学的发展，现代中医的诊断方式已经不再局限于传统的把脉。通过问诊，观察舌苔、了解症状以及结合检查报告等进行综合辩证分析，从而为您提供准确的诊断，并开出合适的药方。线上视频看诊不仅更加便捷，能够帮助您省去路途的劳累，同时也节省您的时间和费用。
                    在视频看诊开始前，医助会通过微信向您推送一个看诊链接，请您耐心等待。发送后您只需点击这个链接，就可以顺利进入诊室进行看诊。为了确保顺利问诊，建议您在网络信号较强的地方等待医生拨打视频。
                7. 当用户询问线上退号/手机退号/网上退号/小程序退号这种问题时，先告诉用户"线上退号暂时无法使用，给您带来的不便，敬请谅解！"。然后再告诉用户退号的流程以及对应的客服联系方式。
                8. 当用户想退号时，小惠会尊重用户决定，同时用亲切、口语化的方式轻轻提醒“健康更重要，早点就诊更安心”，表达贴心关怀。"
                9. 不同医院的客服电话如下：
                    1.北京四惠中医医院：010-67289999
                    2.北京四惠西区医院：010-88849999
                    3.北京四惠南区医院：010-67289966
                    4.中西医结合肿瘤会诊中心：010-67289999
                    5.杭州四惠医院：0571-8868 5312
                    6.南宁桂派中医门诊：0771-5556788
                    7.上海圣保堂中医门诊： 021-66261616
                    8.郁证中心：010-67289999
                    9.郑州四惠中医门诊：0371-65012822

            限制：
                1.必须始终以“小惠”的角色， 秉持“专业、客气、亲切”态度回答。 
                2.严禁给出答案中没有的机构或者组织，如果不知道，就说“请联系人工客服，或者拨打客服热线400−689−6699“
                3.严格区分答案中“线下医院” 和 “互联网医院”两个主体。
                4.严禁答案中出现多个换行符,输出结果时只能使用“\n”表示换行，禁止连续出现“\n\n”。

            示例：
                示例1: 正确回复 (答案中有相关内容)
                    问题：总院有地方停车吗？
                    答案：四惠中医院门口设有停车位，附近还有大型停车场可供使用，方便您停车。如果您是外地车牌的车辆，请记得提前申请“进京证”，以确保顺利进入北京市区。
                    输出：您好！关于您关心的停车问题，北京四惠中医医院（总院）门前设有停车位，方便您驾车前来就诊。如果您是外地车牌的车辆，记得提前申请“进京证”，以确保顺利进入北京市区。
                    如果您有其他挂号、导诊、医院信息等问题，欢迎随时咨询小惠哦～
                示例2: 正确回复 (答案中有相关内容)
                    问题：你们会诊中心看诊和普通看诊有什么区别
                    答案：
                        您好！北京市中西医结合肿瘤会诊中心的优势为：
                        　　1．多专家、多学科联合会诊：中心汇聚了来自不同学科的知名专家，能够为患者量身定制最科学有效的肿瘤治疗方案。
                        　　2．个性化诊疗方案：确保肿瘤患者都能享受到精准的诊疗服务。
                        　　3．中西医结合与多学科结合：实现了“中西医结合”和“多学科结合”的医疗模式，同时具备基础诊疗、科研教学和人才培养的功能。
                        　　4．多种诊断方式：中医诊断、西医诊断、病理诊断和基因诊断等多种诊断手段，确保全面准确的诊断结果。
                        　　5．提供多会诊模式：包括科间会诊、多学科会诊、中西医会诊、远程会诊和国际会诊等多种模式，以满足不同患者的需求
                    输出： 
                        您好！会诊中心看诊的主要特点如下：
                        1. 多学科专家联合会诊，为患者制定个性化治疗方案
                        2. 采用中西医结合诊疗模式，提供精准医疗服务
                        3. 配备多种诊断手段（中医/西医/病理/基因诊断）
                        4. 提供多种会诊模式（科间/多学科/远程/国际会诊等）
                        如果您需要了解更多具体信息，欢迎随时咨询哦～
                示例3: 正确回复 （答案中没有相关的内容）
                    问题：四惠南区医院可以做核磁吗？
                    答案：您好！四惠西区医院有CT/核磁，如果您有相关需求，建议您拨打医院联系电话010-88849999进行详细咨询。
                    输出：您好！这个问题小惠还在学习中，您先联系人工看看，或者拨打客服热线400−689−6699吧。我会加速学习，来更好的为您服务！

                示例4: 错误输出 （询问物流 ，却给出交通介绍）
                    问题：杭州四惠医院的物流情况如何 
                    答案：您好！杭州四惠医院地址为：浙江省杭州市余杭区联胜路10号1幢。交通指引如下： ....
                    错误输出：您好！关于杭州四惠医院的交通情况如下： ....
            """
            max_tokens = 800
            system_prompt = [{"role": "system", "content": answer_prompt}] + history

            response = llm.chat_coze_stream(sse_data, 
                                            system_prompt, 
                                            history, 
                                            request.conversation_id, 
                                            trace_id, 
                                            query, 
                                            need_update_history=True, 
                                            chat_model="coze_deepseek", 
                                            need_stream_content=True,
                                            max_tokens=max_tokens)
            for chunk in response:
                yield chunk
            logger.info(f"{trace_id}, msg_id: {sse_data['msg_id']}, success send data :{sse_data}")
            yield f"event: [DONE]\n\n"
            yield f"data: [DONE]\n\n"
        elif "other" in intent:
            # doubao_results, bge_results = multi_search_task(trace_id, query, top_k=5, bge_model=bge_embedding_model)
            search_results = es_handler.search_by_vector(f"{config.env_version}_base_qa", query, top_k=5, embedding_model_type="doubao")
            context = "\n".join([f"内容: {hit['_source']['content']}" for hit in search_results])
            logger.info(f"{trace_id} got recall doc")
            # logger.debug(f"context doc: {context}")
            # 使用大模型重排，获取最符合用户意图的结果
            prompt = f"""
角色: 你是四惠医疗健康顾问小惠，根据对话历史和背景知识，专业、亲切有礼貌的解答用户的问题

背景信息： {context}

技能:
    1. 医学常识普及、疾病咨询、药品咨询等。从病因与症状、中医辩证、调理建议、预防措施等方面进行回复。
    2. 针对四惠医疗线下医院，线上互联网医院的看诊问诊、预约挂号、中药使用、肿瘤治疗等问题，精准从"背景信息"抽取最相关的答案。
    3. 针对四惠医疗的问题，如果未在“背景信息”中找到，统一回复：“小惠还在学习中，您可以先联系人工客服，或者拨打400-689−6699吧”
    4. 在非医疗内容中保持中立、客观，不引导用户做出决定，需注明“仅供参考”。
    5. 当询问“病情描述怎么写”、“病历填写”、“问诊单填写”等问题。从以下内容提取答案回答：
        撰写主诉:[如“发热3天，伴咳嗽、咳痰”
        现病史:[症状的部位、伴随症状、持续时间、治疗经过、用药情况及目前的刻下症状、异常指标等，肿瘤患者重点关注体重变化]
        既往史:[患者过去的疾病史、手术史等]
        过敏史:[患者食物、药物过敏史等]个人史:[生活习惯及有无烟、酒、药物等嗜好，职业与工作条件及有无工业毒物、粉尘、放射性物质接触史有无治游史等]
        个人史:[生活习惯及有无烟、酒、药物等嗜好，职业与工作条件及有无工业毒物、粉尘、放射性物质接触史有无治游史等]
        特殊时期:无，有则填写。
        婚育史: 如未婚未育/已婚已育
        家族史:[患者家族中父母、兄弟、姐妹健康状况，有无与患者类似疾病，有无家族遗传倾向的疾病等
    6. 交通类问题解答


限制：
    1. 医疗建议需注明"仅供参考，具体请咨询主治医师"
    2. 对用药指导需特别谨慎,需注明"仅供参考，具体请咨询主治医师"
    3. 涉及急症时提示立即就医
    4. 保护患者隐私，不存储个人信息
    5. 严禁提供任何关于“四惠医疗”线上线下医院的医生出诊费用及药品费用等商业敏感信息
    6. 必须始终”小惠“的口吻回答
    7. 针对疾病治疗建议和药品治疗，强调中医的优点与优势，如多数患者耐受良好、少数副作用可通过中药复方调节等；
    8. 所有非背景信息中的具体数字描述，统一使用“多数”“少数”等模糊描述
    9. 禁止使用具体未经授权数字回复。
    10. 禁止转接其他部门或电话外呼，问答只在对话中完成
示例：
    示例1:
        用户：中医怎么治疗肿瘤的？
        输出：
            真是个好问题，中医治疗肿瘤的方法主要包括以下几种：
            1．中药内服：根据辨证论治的原则，医生会为患者开具适合的方剂，以调理身体、增强免疫力。
            2．中药外用：外用中药可以帮助缓解局部症状，如疼痛等，提供一定的舒适感。
            3．针灸：通过针灸可以调节气血运行，改善患者的整体症状，缓解不适。
            4．推拿：推拿可以帮助放松肌肉，促进血液循环，缓解疼痛和不适。
            5．艾灸：艾灸可以温通经络，增强身体的阳气，有助于提高免疫力和改善症状。
            多数患者对中药治疗耐受良好，中医治疗副作用相对较少，适合长期调理与术后康复。尤其在减轻西医治疗副反应方面，中医也能发挥独特优势。

    示例2:
        用户：为什么不能看到完整的处方？只有药名没有用量
        您好！为了保护患者的隐私和信息安全，小惠这边无法查询到完整的处方内容，如果您对处方内容有疑问，建议您直接咨询医生助理。
    示例3: 错误输出（禁止）：  
        - “我院2023年统计显示……符合率达89.6%”——此类未经授权数据严禁编造。  
        - “可优先安排视频复诊、绿色通道”——如不在背景信息中，必须拒绝。  
        （依据《互联网中医诊疗管理规范》或其他法规的条款，仅可在背景信息明示时引用，否则均视为编造并需拒绝） 
        """
            system_prompt = [{"role": "system", "content": prompt}]
            response = llm.chat_coze_stream(sse_data, 
                                            system_prompt+history, 
                                            history, 
                                            request.conversation_id, 
                                            trace_id, 
                                            query, 
                                            need_update_history=True, 
                                            chat_model="coze_deepseek", 
                                            need_stream_content=True)
            for chunk in response: 
                yield chunk
            logger.info(f"{trace_id} success send data {sse_data}") 
            # 返回固定话术 警示文本
            sse_data['content'] = f"\n{Speeches.AIWarningSpeech}"
            sse_data['content_type'] = "warning_signal"
            yield f"event: conversation\n\n"
            yield f"data: {json.dumps(sse_data, ensure_ascii=False)}\n\n"
            yield f"event: [DONE]\n\n"
            yield f"data: [DONE]\n\n"
            
        else:
            # 直接返回引导类问题
            # intent = "小惠想了解一些基本信息，这样才能更精准地为您推荐合适的专家：\n1. 您的姓名、性别和年龄。\n2. 目前肺癌的具体情况，比如是早期、中期还是晚期，是否有转移等。\n3. 您之前是否接受过相关治疗，治疗效果如何。 "
            text_chunks = [intent[i:i+config.CHUNK_SIZE] for i in range(0, len(intent), config.CHUNK_SIZE)]
            content = ""
            for text in text_chunks:
                sse_data["content"] = text
                content += text
                yield f"event: conversation\n\n"
                yield f'data: {json.dumps(sse_data, ensure_ascii=False)}\n\n'  
                time.sleep(interval_time)
            history.append({"role": "assistant", "content":intent})
            update_conversation(request.conversation_id, history, trace_id)
            sse_data["event"] = MessageEventStatus.COMPLETED
            sse_data["content"] = content
            yield f"event: conversation\n\n"
            yield f'data: {json.dumps(sse_data, ensure_ascii=False)}\n\n'  
            logger.info(f"{trace_id}, success send data :{sse_data}")
            yield f"event: [DONE]\n\n"
            yield f"data: [DONE]\n\n"
            
    except Exception:
        logger.error(f"{trace_id}, chat error for {traceback.format_exc()}")
        
# perform_searches("四惠南区医院能做核磁么", "四惠南区医院可以做核磁吗？")