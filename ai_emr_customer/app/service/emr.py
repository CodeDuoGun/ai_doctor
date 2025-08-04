# -*- coding: utf-8 -*-     
from app.config import config
from app.utils.log import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import traceback
from app.utils.backend_service import request_backend_emr, get_backend_chat_his

PATIENT_CH_MAP = {
    "name": "姓名",
    "age": "年龄",
    "sex": "性别",
    "disease_description": "病情描述",
    "present_disease": "现病史",
    "family_history": "家族史",
    "allergy_history": "过敏史",
    "personal_history": "个人史",
    "specialdetial": "特殊时期",  # TODO: 这个字段的含义是什么？
    "tongue_face_results": "舌照面照信息",
    "check_report_results": "检查报告结果信息",
    "questionnaire_results": "问卷异常回复分析",
    "chat_records": "医生患者问诊聊天记录",
    "medical_history": "历史病历信息",
    "no_data": "无"
}


"""
totolist:
1. 获取现有emr 和问诊单信息
    1.1 基础信息
    1.2 病情描述
    1.3 现病史
    1.4 既往史
    1.5 家族史
    1.6 过敏史
    1.7 个人史 # 什么意思？

获取检查报告，若无异常指标，给出一个总的描述
获取舌照面照，给出标准化描述
问诊单问卷内容
历史病历，取最近一次的
聊天记录（音视频、文本、图片）后端提供，
"""

# TODO: 能否让后端直接给出
def prepare_emr_data(id):
    """
    """
    # 问诊单资料
    logger.info(f"准备病历数据，id: {id}")
    emr_data = request_backend_emr(id)
    if not emr_data:
        return
    # format 获取基本信息包括（性别、年龄、身高、体重、婚育史、病情描述、现病史、既往史、家族史、过敏史）、历史病历信息、问卷问题、检查报告、舌照面照、聊天记录、
    patient_info = {}
    patient_info["basic_info"] = {
        "name": emr_data.get("username", ""),
        "age": emr_data.get("age", ""), 
        "sex": "男" if emr_data.get("sex", 0) == 1 else "女", 
        "disease_description": emr_data.get("description", ""), 
        "present_disease": emr_data.get("present_history", ""), 
        "family_history": emr_data.get("familydetial", ""), 
        "allergy_history": emr_data.get("allergydetial", ""), 
        "personal_history": emr_data.get("personal_history", ""), 
        "specialdetial": emr_data.get("specialdetial", ""),
        }
    
    # images
    patient_info["check_report_images"] = emr_data.get("other_img", "")
    patient_info["tongue_face_images"] = emr_data.get("tongue", "")
    
    # [{q1:a1}, {q2:a2}...]
    # patient_info["questionnaire"] = emr_data.get("question_answer", {}).get("data", [])
    # TODO: 涉及业务过多，先忽略
    patient_info["questionnaire"] = []
    return patient_info
        

def get_chat_msgs(id=None):
    chat_msgs = get_backend_chat_his()
    return chat_msgs

def tongue_face_diagnose_task(tonge_face_images:list):
    # 先获取舌照/面照分类结果

    # 再分别进行检测
    tongue_results = ""
    face_results = ""
    return tongue_results, face_results


def report_diagnose_task(report_imgs):
    """根据病情描述和检查报告中的异常结果"""
    if not report_imgs:
        return 

def extract_disease_history():
    """
    医学信息抽取助手，给出现病史
    | 字段名           | 说明                   | 示例                     |
| ------------- | -------------------- | ---------------------- |
| **主诉症状**（主症）  | 当前最主要的表现、症状及持续时间     | 咳嗽3月、胸痛1周              |
| **起病时间与方式**   | 起病时间及是否突发、缓慢         | 三月前起病，起初干咳，渐有痰         |
| **症状演变过程**    | 症状如何变化（加重、缓解、迁延）     | 逐渐加重，后转为咳痰             |
| **伴随症状**      | 除主诉以外伴随的其他临床表现       | 食欲下降、体重减轻、低热           |
| **既往用药情况**    | 曾用药物与治疗方法及其效果        | 使用过阿司匹林、甲硝唑，效果不佳       |
| **相关检查与结果**   | 当前相关的实验室检查、影像学检查等    | 血小板136ml/mg，血脂184ml/ug |
| **疾病背景与基础病史** | 与现病密切相关的长期疾病或病史      | 肝癌5年，现为第二次复诊           |
| **功能状态/影响**   | 疾病对生活质量、进食、精神状态等影响   | 食量骤减，精神差，乏力            |
| **就诊动因/复诊原因** | 患者为何来就诊、复诊（主动 or 被动） | 因症状未缓解来院复诊             |
| **治疗反应或变化**   | 已尝试治疗后的病情变化          | 未见明显改善                 |
| **并发症或异常发现**  | 当前并发的严重症状或并发病        | 积液5ml，疑胸水形成            |
舌诊面诊结果  精神尚可、面色红润；舌质红，苔薄黄
    """
    pass

def gen_chief_claim():
    pass

def gen_ai_emr(id=None):
    """
    由于模型未必掌握细致辨证逻辑，可以引入医学知识库辅助推理：将中医辨证标准（如《中医临床诊疗指南》）转为“结构化规则”（如舌苔黄 + 咳黄痰 + 脉浮 → 风热犯肺）。
    使用规则引导 LLM，例如在 Prompt 中嵌入：“参考《中医辨证分型对照表》：风热犯肺证的典型表现为：咳嗽黄痰，咽喉红肿，舌红苔黄……”

    百川Baichuan2-13B（中医知识可调优）	中文理解好，可微调或量化部署，适合本地模型。
    ChatGLM3-6B / 6B-Med	医疗微调版，已在中医与病历抽取方向有工作。

    """
    try:
        content = ""
        # TODO 并行获取问诊单和聊天记录结果
        patient_info = prepare_emr_data(id)
        if not patient_info:
            logger.info("未获取到患者问诊单数据")
            return
        basic_info = ",".join(f"{PATIENT_CH_MAP[k]}:{v}" for k, v in patient_info["basic_info"].items() if v)  # 基础信息
        content = f"患者基本信息: {basic_info}\n"
        logger.debug(f"patient_basic_info: {basic_info}")
        import pdb
        pdb.set_trace()
        patient_info['chat_records'] = get_chat_msgs(id)
        result = {}
        
        tonge_face_res = tongue_face_diagnose_task()
        report_diagnose_res = report_diagnose_task()

        if tonge_face_res:
            content = content + f"舌照面照信息: {tonge_face_res}"
        if report_diagnose_res:
            content = content + f"舌照面照信息: {report_diagnose_res}"


        # 基于基本信息、舌照面照分析结果、异常问卷问题分析结果、检查报告异常指标分析结果、历史病历提取结果
        with ThreadPoolExecutor(max_workers=3) as executor:
            # TODO: 多线程完成
            tasks = {
                "description": executor.submit(prepare_emr_data, patient_info.get("id")),
                "tongue_face_results": executor.submit(tongue_face_diagnose_task, patient_info.get("id")),
                "chief_complaint": executor.submit(task(), patient_info.get("id")),
                "report_diagnose_results": executor.submit(report_diagnose_task, patient_info.get("id")),
            }
            for task_name, future in tasks.items():
                result[task_name] = future.result()
        
        disease_text = f""
        
        
        # 根据现病史内容，调用中西医辩证模型，智能诊断和开方
        pass


    except Exception:
        logger.error(f"生成AI病历失败: {traceback.format_exc()}")
    else:
        logger.info("AI病历生成成功")
    return result


if __name__ == "__main__":
    gen_ai_emr(id=617494)