from app.utils.log import logger
import traceback


tools =[
    {
        "type": "function",
        "function": {
            "name": "extract_entities",
            "description": """抽取文本中某个具体名字的医生姓名、医生所在城市、医生所属医院、疾病、症状、病种，根据给定文本返回相应的医生姓名、医生所在位置、医生所属医院、医学专有名词,四种提取结果。
            特别注意：禁止将患者名字或其他非医生身份的姓名识别为医生。\n
            支持多位医生及其对应城市和医院，结果以逗号分隔的字符串形式返回。\n 例子1：query=广西和北京有擅长肺癌的医生么， 输出{"doctor_name":"","condition":"肺癌", "doctor_location": "广西,北京", "doctor_hospital":""}""",
            "parameters": {
                "properties": {
                    "doctor_name": {
                        "type": "string",
                        "description": "医生姓名列表，多个姓名用逗号分隔，若无则返回空字符串"
                    },
                    "condition": {
                        "type": "string",
                        "description": "疾病或症状描述，若无则返回空字符串"
                    },
                    "doctor_location": {
                        "type": "string",
                        "description": "医生所在城市列表，多个城市用逗号分隔，与 doctor_name 一一对应，若无则返回空字符串"
                    },
                    "doctor_hospital": {
                        "type": "string",
                        "description": "医生所属医院列表，多个医院用逗号分隔，与 doctor_name 一一对应，若无则返回空字符串"
                    }
                },
                "required": [],
                "type": "object",
            },
        },
    },
]

disease_tools = [
    {       
        "type": "function",
        "function": {
            "name": "extract_disease",
            "description": """
            抽取某段文本中的某个疾病、病种时使用该plugin，根据给定文本返回相应的病种提取结果。\n 
示例：
    示例1：
    query=39岁女性昨天手上刚出现很多有点痒的水泡，可能是汗疱疹、手癣、接触性皮炎等疾病，应该挂哪个号
    输出:{"condition":"汗疱疹、手癣、接触性皮炎"}
    示例2：
    query=北京四惠中医医院谁能治疗肺癌晚期
    输出:{"condition":"肺癌"}
""",
            "parameters": {
                "properties": {
                    "condition": {
                        "type": "string",
                        "description": "疾病/病种名称，若无则返回空字符串"
                    }
                },
                "required": [],
                "type": "object",
            },
        },
    },
]
def extract_disease(llm, trace_id, rewrite_query:str="") -> dict:
    """
    返回 {'condition': str}
    """
    system_prompt = """
    ## 角色：你是一位信息抽取专家，每次必须调用工具extract_disease，并从文本中识别并提取病症关键词。

    ## 技能
    1. 准确识别并提取文本中的病症关键词。
    2. 将抽取结果整理为标准 JSON 格式，包含以下字段：
        condition：病症关键词
    3. 提取的病症关键词应涵盖疾病名称、病种等医学术语。

    ## 限制
    1.严格按照 JSON 格式返回结果。
    2.对于无法识别的字段，使用空字符串 "" 表示。

    ## 示例:
    示例一：
        问题：北京有哪些擅长治疗糖尿病的医生
        输出：{"condition": "糖尿病"}
    
    示例二：
        问题：北京四惠中医医院谁能治疗肺癌晚期
        输出：肺癌
        
    示例三：症状不是疾病
        问题：找治疗肚子疼的医生
        输出：{"condition": ""}
    
    示例四：
        问题： 推荐一个皮肤病专家
        输出： 皮肤病
    
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": rewrite_query}
    ]
    logger.info(f"{trace_id} extract_disease query: {rewrite_query}")
    data = llm.coze_ds_with_func(trace_id, messages, tools=disease_tools, temperature=0.3, tool_choice="required")
    if not data:
        logger.error(f"{trace_id} extract info of {data} error {traceback.format_exc()}")
        return {"condition": rewrite_query}
    else:
        # 确保返回字段，过滤“晚期”，“中期”文本
        if data.get("condition"):
            data["condition"] = data["condition"].replace("晚期", "").replace("中期", "").replace("早期","").strip()
            return {"condition": data["condition"]}
        # 确保返回字段
        return {
            "condition": data.get("condition", "").strip(),
        }
    
def extract_entities(llm, trace_id, query: str, rewrite_query:str="", history:list=[], must_name:bool=False) -> dict:
    """
    返回 {'doctor_name': str, 'condition': str}
    """
    system_prompt = """
## 角色：你是一位信息抽取专家，必须调用工具extract_entities，并从文本中识别并提取医生姓名、医生所在城市、医生所属医院、以及病症关键词四类信息。


## 技能
    1. 准确识别并提取文本中的医生姓名、所在城市、所属医院以及病症关键词。
    2. 将抽取结果整理为标准 JSON 格式，包含以下字段：
        doctor_name：医生姓名
        doctor_location：医生所在城市
        doctor_hospital：医生所属医院
        condition：病症关键词
    3. 病症关键词的提取要求如下：
        首位关键词必须为你结合文本判断出的患者可能所患疾病（如“筋膜炎”、“肌肉劳损性结节”等），基于上下文症状进行合理医学推断；
        后续关键词包括：
            明确提到的疾病名称；
            典型症状（如“发热”“咳嗽”“结节”）；
            医学术语描述的病种、体征、生理异常；
        不包含生活方式、情绪、饮食、睡眠习惯等非医学性描述。
    4. 精准区分医生姓名与患者姓名，避免将患者姓名误识为医生姓名。
    5. 病症关键词应限定为与疾病、症状、体征、生理异常相关的医学术语，排除饮食偏好、作息习惯、情绪状态等生活方式描述。

## 限制
    1.严格按照 JSON 格式返回结果。
    2.对于无法识别的字段，使用空字符串 "" 表示。

## 示例:
    示例一:
        用户问题：预约山野村务主任（血液科专家）的门诊号
        输出：{"doctor_name": "山野村务", "doctor_location": "","doctor_hospital": "", "condition": ""}
    示例二：
        用户问题：请问是否有擅长治疗头痛伴随耳鸣和恶心症状的中西医结合专家推荐
        输出：{"doctor_name": "", "doctor_location": "", "doctor_hospital": "", "condition": "头痛 耳鸣 恶心"}

    示例三：
        用户问题：北京有哪些擅长治疗糖尿病的医生
        输出：{"doctor_name": "", "doctor_location": "北京", "doctor_hospital": "", "condition": "糖尿病"}

    示例四： 
        问题：王素梅医生在治疗儿童多发性抽动症、多动症、自闭症、学习困难综合征、急/慢性咳嗽、反复呼吸道感染、厌食症、难治性腹泻、过敏性紫癜、鼻炎、蛋白尿、遗尿症、儿童湿疹、腺样体肥大等儿科常见病及疑难病方面的疗效介绍有哪些
        输出：{"doctor_name": "王素梅", "condition": "儿童多发性抽动症、多动症、自闭症、学习困难综合征、急/慢性咳嗽、反复呼吸道感染、厌食症、难治性腹泻、过敏性紫癜、鼻炎、蛋白尿、遗尿症、儿童湿疹、腺样体肥大", "doctor_location": "", "doctor_hospital": ""}

    示例五：推断可能患有的疾病
        问题：25岁男性大腿肌肉结节出现三天左右，骑车等肌肉收缩时疼、走路略微不适，曾患同腿髂胫束摩擦综合征，应挂什么科？
        输出：{"doctor_name": "", "doctor_location": "", "doctor_hospital": "","condition": "筋膜炎、 肌肉劳损、纤维化、髂胫束摩擦综合征、肌肉收缩时疼"}
    
    示例六：（错误示例，将患者姓名误识为医生姓名）
        用户问题：推荐北京四惠中医医院有号的肺癌专家，患者是76岁女性郑思思
        错误输出：{"doctor_name": "郑思思", "doctor_location": "北京", "doctor_hospital": "北京四惠中医医院","condition": "肺癌"}
        正确输出：{"doctor_name": "", "doctor_location": "北京", "doctor_hospital": "北京四惠中医医院","condition": "肺癌"}
    示例七： 张三视频/张三图文
        问题：张三视频
        输出：{"doctor_name": "张三", "doctor_location": "", "doctor_hospital": "","condition": ""}
    
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": rewrite_query}
    ]
    # messages += history


    data = llm.coze_ds_with_func(trace_id, messages, tools=tools, temperature=0.3, tool_choice="required")
    if not data:
        logger.error(f"{trace_id} extract info of {data} error {traceback.format_exc()}")
        if must_name:
            return {"doctor_name": query, "condition": "", "doctor_location": "", "doctor_hospital":""}
        return {"doctor_name": '', "condition": rewrite_query, "doctor_location": "", "doctor_hospital":""}
    else:
        # 确保返回字段
        return {
            "doctor_name": data.get("doctor_name", "").strip(),
            "condition": data.get("condition", "").strip(),
            "doctor_location": data.get("doctor_location", "").strip(),
            "doctor_hospital": data.get("doctor_hospital", "").strip(),
        }
