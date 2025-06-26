import openai
import json
from openai import OpenAI
from log import logger


def get_gpt35_data(data,question):
    client = OpenAI(api_key="",base_url="http://openai.4ukids.cn/v1",)
    final_prompt=[{"role": "system","content": data},{"role": "user", "content": question}]
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=final_prompt,
    temperature=0,
    top_p=1
    )
    # response=response.model_dump_json()
    # response=response.replace('true','True').replace('false','False').replace('null','None').replace('Null','None')
    # response=eval(response)
    # logger.info(f"chat gpt response:{response}") 
    # return response["choices"][0]["message"]["content"]
    return response.choices[0].message.content

class ChatGPT():
    def __init__(self, model_path = 'gpt-3.5-turbo', api_key = None):
        openai.api_key = api_key
        self.model_path = model_path

    def chat(self, message):

        return get_gpt35_data(message, message)
        # response = openai.ChatCompletion.create(
        #     model=self.model_path,
        #     messages=[
        #         {"role": "user", "content": message}
        #     ]
        # )
        # return response['choices'][0]['message']['content']