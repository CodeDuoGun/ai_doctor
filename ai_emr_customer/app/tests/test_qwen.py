import os
from openai import OpenAI

def test_qwen():
    client = OpenAI(
        api_key="sk-4ef42187cc2e47999d0c835c10fc5a78",  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
    )

    completion = client.embeddings.create(
        model="text-embedding-v3",
        input='你好',
        dimensions=1024,
        encoding_format="float"
    )

    print(completion.model_dump_json())


def test_ds():
    from openai import OpenAI

    # for backward compatibility, you can still use `https://api.deepseek.com/v1` as `base_url`.
    client = OpenAI(api_key="sk-a17c861e74ce49c6878efcae740baeed", base_url="https://api.deepseek.com/v1")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "今天周几"},
    ],
        max_tokens=1024,
        temperature=0.3,
        stream=True
    )
    for chunk in response:
        print(2)
        print(chunk)

import requests
tools =[
    {
        "type": "function",
        "function": {
            "name": "search_doctor",
            "description": """医生检索Plugin，当用户需要查询某个具体名字的医生、或擅长某种疾病、症状、病种的医生时使用该plugin，根据给定文本返回相应的人名和医学专有名词两种提取结果。\n 例子1：query=有擅长肺癌的医生么， 输出{"doctor_name":"","condition":"肺癌"}""",
            "parameters": {
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户提到的医生姓名，若无则返回空字符串"
                    },
                    "condition": {
                        "type": "string",
                        "description": "疾病或症状描述，若无则返回空字符串"
                    }
                },
                "required": [],
                "type": "object",
            },
        },
    },
]
def test_bot():
    client = OpenAI(
            api_key="f43df693-455a-4e3e-b987-d76d7b57f4c3",
            base_url="https://ark.cn-beijing.volces.com/api/v3"
        )
    messages = [
        {"role": "system", "content": "你是一个智能医疗助手，当用户提到人物时，调用工具 search_doctor"},
        {"role": "user", "content": "张大宁是谁"}
    ]
    response = client.chat.completions.create(
        model="deepseek-v3-250324",
        messages=messages,
        temperature=0.3,
        max_tokens=500,
        top_p=1,
        tools=tools,
        tool_choice="",
        response_format=None,
        stream=True,
    )
    for chunk in response:
        print(chunk)
    

def main():
    # test_ds()
    # test_qwen()
    test_bot()

main()

