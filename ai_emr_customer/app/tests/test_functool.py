import json
from app.rag.vdb.es.es_processor import es_handler
search_tool = {
    "type": "function",
    "function": {
        "name": "search_es_by_description",
        "description": "根据用户提供的自然语言描述，在Elasticsearch中搜索相关内容",
        "parameters": {
            "type": "object",
            "properties": {
                "query_description": {
                    "type": "string",
                    "description": "描述用户要查找的内容，比如疾病名称、物流信息、医生介绍等"
                }
            },
            "required": ["query_description"]
        }
    }
}

import openai

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "四惠中医医院的物流怎么查？"}
    ],
    tools=[search_tool],
    tool_choice="auto"
)


tool_call = response["choices"][0]["message"]["tool_call"]
arguments = json.loads(tool_call["arguments"])
query_text = arguments["query_description"]

# 

tool_result = es_handler.search_qa_question(query_text)

follow_up = openai.ChatCompletion.create(
    model="",
    messages=[
        {"role": "user", "content": "四惠中医医院的物流怎么查？"},
        response["choices"][0]["message"],
        {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": tool_result
        }
    ]
)

print(follow_up["choices"][0]["message"]["content"])
