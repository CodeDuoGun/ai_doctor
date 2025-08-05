import requests
import json

from typing import Generator, Optional
from llm_v2.llm.base import BaseLLMClient
import time
from utils.tool import convert_image_file_to_base64
import os


class  DoubaoAIClient(BaseLLMClient):
    def chat(self, messages: list[dict], stream: bool = False, llm_model: str="doubao-seed-1-6-250615") -> str:
        # url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": llm_model,
            "messages": messages,
            "temperature": 0.7,
            "thinking":{"type": "disabled"},
            "max_tokens": 1000,
            "stream": stream
        }
        t0 = time.time()
        response = requests.post(self.base_url, headers=headers, json=payload, stream=stream)
        if not response.ok:
            raise Exception(f"OpenAI error: {response.text}")

        if stream:
            return self._parse_stream_response(response)
        else:
            print(f"Request took {time.time() - t0} seconds")
            return response.json()["choices"][0]["message"]["content"]
    
    def validate_img():
        """
        当图片分辨率高，使用模型上下文窗口限制为 32k，每个图片转为 1312 个 token ，单轮可传入的图片数量为 32000 ÷ 1312 = 24 张。
        当图片分辨率小，使用模型上下文窗口限制为 32k，每个图片转为 256 个 token，单轮可传入的数量为 32000 ÷ 256 = 125 张。
        单张图片小于 10 MB。
        使用 base64 编码，请求中请求体大小不可超过 64 MB。
        token 计算：min(图片宽 * 图片高÷784, 单图 token 限制)
        """
        pass
    
    def chat_with_img(self, question:str, files:list=[]):
        """"""
        image_contents =[]        
        for file_path in files:
            file_path = "app/data/test_img.jpg"
            # TODO: 如果不是http
            if not file_path.startswith("https"):
                img_content = f"data:image/{file_type};base64,{convert_image_file_to_base64(file_path)}"
            else:
                img_content = file_path
            
            file_type = os.path.basename(file_path).split('.')[-1]  # 获取文件类型
            image_contents.append({
                        "type": "image_url",
                        "image_url": {"url": img_content}
                    })
            
        # TODO: history
        messages = [
                {"role": "system", "content": reportor_system_prompt},
                {"role": "user", "content": [
                    {
                        "type": "text",
                        "text": question
                    },
                ]}
            ]
        messages.extend(image_contents)
        
        response = client.chat.completions.create(
            model="doubao-1-5-vision-pro-32k-250115",
            messages=messages,
            stream=True,
            # thinking={"type": "disabled"}
        )

        # 打印响应内容
        content = ""
        for chunk in response:
            if chunk:
                print(chunk)
                content += chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
        print("Final Response Content:", content)


    def _parse_stream_response(self, response: requests.Response) -> Generator[str, None, None]:
        for line in response.iter_lines():
            # print(f"line: {line}")
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    line_str = line_str[6:].strip()
                    # print(f"line_str: {line_str}")
                    if line_str == "[DONE]":
                        break
                    try:
                        content = json.loads(line_str)["choices"][0]["delta"].get("content", "")
                        yield content
                    except Exception as e:
                        print(f"Error parsing error: {e}")
                        continue

if __name__ == "__main__":
    # Example usage
    from default_config import config
    client = DoubaoAIClient(base_url=config.ARK_API_URL, api_key=config.ARK_API_KEY)
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    
    # Streaming response
    for chunk in client.chat(messages, stream=True):
        print(chunk)
    
    # Non-streaming response
    response = client.chat(messages, stream=False)
    print(response)