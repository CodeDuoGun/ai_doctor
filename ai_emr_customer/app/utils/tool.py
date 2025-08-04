import time
import base64
import pandas as pd
from app.utils.log import logger
import uuid
import re
import traceback

def generate_message_id():
    return str(uuid.uuid4())


def perf_counter_timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行目标函数
        end = time.perf_counter()  # 记录结束时间
        logger.debug(f"{func.__name__} cost time: {end - start:.6f} s")
        return result
    return wrapper


def remove_special_punctuation(text, special_punctuation='**'):
    try:
        # 使用正则表达式替换这些特殊标点符号为空字符串
        cleaned_text = re.sub(r'\s*\*\*\s*', '', text)
        # cleaned_text = re.sub(special_punctuation, '', text)
    except Exception:
        logger.error(f"normalize failed for {text} cause {traceback.format_exc()}")
        return text
    return cleaned_text


def split_multi_q(questions, qa_data:dict, answer:str=""):
    """把多个q按照换行符分割成一个个q"""
    res = questions.strip("\n").split('\n') if questions else []
    for q in res:
        q = q.strip(" ")
        if not q:
            continue
        qa_data[q] = answer
    # 逐行放入list中
    return qa_data

def gen_qa_by_file(file_path:str="app/data/qa_out.csv"):
    df = pd.read_excel(file_path, engine='openpyxl')
    qa_res = {}
    # i = 0
    for _, row in df.iterrows():
        # i+=1
        doc = row.to_dict()
        # if i > 250:
        #     break
        qa_res = split_multi_q(doc["new_question"], qa_res, doc["答案"])
    # 临时入库脚本
    logger.debug(f"q num: {len(qa_res)}")
    return qa_res


def read_prompt_from_file(file_path):
    """从文件中读取提示词"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            prompt = file.read()
        return prompt
    except FileNotFoundError:
        logger.error(f"文件 {file_path} 未找到。")
        return
    except Exception as e:
        logger.error(f"读取文件时发生错误: {e}")
        return


import numpy as np

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def convert_image_file_to_base64(image_path):
    """将图片文件转换为 Base64 编码字符串"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
