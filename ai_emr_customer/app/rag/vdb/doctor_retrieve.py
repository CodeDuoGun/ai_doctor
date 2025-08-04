from app.utils.log import logger
from app.config import config
from app.rag.vdb.es.es_processor import es_handler
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.rag.vdb.func_tools import extract_disease

def disease_doctors(llm, trace_id:str, rewrite_query:str="",chat_model="coze_deepseek", top_k:int=7):    
    """"""
    # return []
    res = []
    ents = extract_disease(llm, trace_id, rewrite_query)  # 同前，使用大模型做 NER 提取
    cond = ents["condition"]
    logger.info(f"{trace_id},ents:{ents}")
    with ThreadPoolExecutor(3) as executor:
        task1 = executor.submit(es_handler.search_by_keyword,f"{config.env_version}_secondary_disease", cond, size=top_k) #关键词检索二级表
        # task2 = executor.submit(es_handler.search_by_keyword, f"{config.env_version}_primary_disease", cond, size=top_k) # 语义检索一级表
        task2 = executor.submit(es_handler.search_by_vector, f"{config.env_version}_primary_disease", cond, top_k=5, doctor_location="") # 语义检索一级表
        # task3 = executor.submit(es_handler.search_by_keyword, f"{config.env_version}_doctor", cond, size=top_k) # 语义检索
        
        for future in as_completed([task1, task2]):
            res.append(future.result())
    # print(res[1])
    res[1] = [hit for hit in res[1] if hit["_score"]>1.9]
    hits = res[0] or res[1]# or res[2]
    # 过滤低分医生
    # hits = [hit for hit in hits if hit['_score'] >= 2.5]
    return hits[:8]


def multi_search_doctor(cond, doctor_hospital="", doctor_location="", top_k:int=5):
    with ThreadPoolExecutor(3)  as executor:
        # kw_hits = es_handler.search_by_keyword(f"{config.env_version}_doctor", cond, size=5, doctor_location="", doctor_name="")
        # hits = es_handler.search_by_vector(f"{config.env_version}_doctor", doctor_hospital+cond, top_k=5, doctor_location=doctor_location)
        task1 = executor.submit(es_handler.search_by_keyword, f"{config.env_version}_doctor", cond, size=top_k, doctor_location="", doctor_name="")
        task2 = executor.submit(es_handler.search_by_vector, f"{config.env_version}_doctor", doctor_hospital+","+cond, top_k=top_k, doctor_location=doctor_location)
        
        res1 = task1.result()
        res2 = task2.result()
    return [res1, res2]
    
def get_id_by_name(index_name:str, doctor_name:str):
    doctor_name = doctor_name.strip()
    logger.info(f"ready to search doctor: {doctor_name}")
    res = es_handler.search_by_keyword(index_name, "", size=5, doctor_location="", doctor_name=doctor_name)
    if res:
        logger.info(f"get_id_by_name {doctor_name} res {res[0]['_source']['ID']}")
        return res[0]["_source"]["ID"]
    else:
        return ""


def get_doctor_ids(trace_id:str, es_index_name, hits_name_res):
    doctor_ids = []
    for hit in hits_name_res:
        if "doctors" in hit['_source']:# 二级病种表
            doctor_names = hit['_source']['doctors'].split("、")
            for doctor_name in doctor_names:
                doctor_id = get_id_by_name(es_index_name, doctor_name)
            
                if not doctor_id:
                    continue
                if isinstance(doctor_id, str):
                    ids = doctor_id.split("、")
                    doctor_ids.extend(ids)
                else: 
                    ids = doctor_id
                    doctor_ids.append(ids)
        else:
            doctor_id = hit['_source']["ID"]
            if isinstance(doctor_id, str):
                ids = doctor_id.split("、")
                doctor_ids.extend(ids)
            else: 
                ids = doctor_id
                doctor_ids.append(ids)
    logger.info(f"{trace_id} got doctor ids {doctor_ids}")
    return list(set(doctor_ids)) 

