import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor, as_completed



s = """您好！南宁桂派中医门诊部位于广西南宁市青秀区，环境优雅，交通便利，建筑面积1700平方米。
门诊部拥有国家级名老中医2人，桂派中医大师4人，国家二级教授7人，广西名中医9人。汇聚了广西中医药大学原副校长罗伟生、广西中医药大学第一附属医院原院长黄贵华、广西中医药大学附属瑞康医院主任医师黄适等众多知名专家团队。此外，在广西名中医、国家瑶医药学科带头人李彤主任的带领下，门诊部组建了特色瑶医肿瘤治疗团队。
门诊部始终秉承集团“聚全国名医，为人民服务”的使命，为患者提供贴心、周到的服务。从挂号、问诊到治疗、康复，每一个环节都力求做到最好。多年来，门诊部凭借其优质的医疗服务、精湛的医疗技术和良好的医德医风，赢得了广大患者的信赖和好评。"""


from openai import OpenAI

client = OpenAI(
            api_key="f43df693-455a-4e3e-b987-d76d7b57f4c3",
            base_url="https://ark.cn-beijing.volces.com/api/v3"
        )

def get_completion(messages, chat_model:str="deepseek-v3-250324", json_schema=None, temperature:int=0.5, functions=None,stream=True):
    retry_num = 0
    while retry_num < 3:
        try:
            stream_res = client.chat.completions.create(
                    model=chat_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=400,
                    top_p=1,
                    response_format=json_schema,
                    stream=stream,
                    functions=functions,
                    function_call="auto",
                    # tools=tools
                )

            res= ""
            for event in stream_res:
                if event.choices:
                    res += event.choices[0].delta.content
            print(res)
            return res
        except Exception:
            retry_num += 1
            print(f"retry_num: {retry_num}")
            continue
    return res

def read_csv(csv_file:str="app/data/智能客服知识库5-6.xlsx"):
    df = pd.read_excel(csv_file)
    print(df["答案"])
    new_df = df.filter(['问题', '答案'])
    df_map =new_df.to_dict()
    return df_map

def main():
    df_map = read_csv()
    questions = []
    i = 0
    for index, answer in df_map["答案"].items():
        i+=1
        if i<=250:
            continue
        print(f'question: {df_map["问题"][index]}')
        print(f"answer: {answer}")
        prompt = f"""你是一个精准的问题生成器。
请根据以下文本内容，生成尽可能15种不同表述的问题，确保问题的多样性。每个问题应考虑用户可能的多种问法，包括但不限于以下几种类型：

每个问题应该覆盖文本中的关键信息、主要概念和细节，确保不遗漏重要内容。

文本: {answer}
输出格式：每个问题单独一行，问题之间用换行符分隔。请确保问题的多样性和覆盖面，避免重复和相似的问题。

示例：
    输入："糖尿病的常见治疗方法包括饮食控制、规律运动以及药物治疗。对于2型糖尿病患者，初期治疗通常通过饮食和运动来控制血糖。如果这些方法无效，医生可能会推荐口服降糖药物。如果症状更严重，可能需要注射胰岛素来调节血糖水平。"
    输出：
    糖尿病的常见治疗方法有哪些？
    糖尿病的治疗方法包括哪些方面？
    是否可以说糖尿病的治疗主要依赖饮食控制和运动？
    如果2型糖尿病患者的饮食和运动方法不起效，接下来会采用什么治疗？
    如果一个糖尿病患者需要注射胰岛素，这说明了什么？
    为什么医生会推荐口服降糖药物作为2型糖尿病的治疗手段？
    请问在治疗糖尿病时，胰岛素和口服降糖药物的作用有什么不同？
    糖尿病患者应该如何判断何时需要胰岛素治疗？
输出限制：
    不要输出问题序号
"""
        messages = [{"role": "user", "content": answer}, 
                    {"role": "system", "content": prompt}]
        res = get_completion(messages)
        questions.append(res)
        # print(f"questions: {questions}")
        # break
    try:
        df_map["new_question"] = pd.Series(questions)
        new_df = pd.DataFrame.from_dict(df_map)
        new_df.to_csv('app/data/qa_out_end.csv', index=False)  
    except Exception as e:
        with open("./data.json", "w") as fp:
            fp.write(json.dumps(df_map, ensure_ascii=False))
        

def run_multi_task():
    res = []
    with ThreadPoolExecutor(2)  as executor:
        task1 = executor.submit(doubao_search_task)
        task2 = executor.submit(other_search_task)
        
        for future in as_completed([task1, task2]):
            res.append(future.result)
    return res


def doubao_search_task(es):
    questions_doubao = es.search_by_vector()
    pass

def other_search_task(es):
    questions_other = es.search_by_vector()
    pass

def rerank_results():
    """重排结果 qa时可以不需要"""
    pass


def qa2es():
    """
    """


def chat():
    # 两路召回7个最相关的问题q_vector, a_vector, q2_vector, v2_vector
    while True:
        query = input()        


        system_promt = f"""
根据给出的多个问题，选择一个和当前问题最相似的问题。
当前问题{query}
示例：
    当前问题：
    多个问题：
    输出：

    """
    
main()