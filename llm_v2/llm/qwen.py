import requests
import json
from typing import Generator, Optional
from llm_v2.llm.base import BaseLLMClient
from utils.tool import convert_image_file_to_base64
import os

class QwenAIClient(BaseLLMClient):
    def chat(self, messages: list[dict], stream: bool = False, llm_model: str = "qwen-max") -> str | Generator[str, None, None]:
        url = f"{self.base_url}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": llm_model,
            "messages": messages,
            "temperature": 0.7,
            "stream": stream,
            "max_tokens": 1000,
        }

        response = requests.post(url, headers=headers, json=payload, stream=stream)

        if not response.ok:
            raise Exception(f"OpenAI error: {response.text}")

        if stream:
            return self._parse_stream_response(response)
        else:
            return response.json()["choices"][0]["message"]["content"]
        
    def chat_with_img(self,question, files:list):
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
            model="qwen-vl-max-2025-04-08",
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
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    line_str = line_str[6:].strip()
                    if line_str == "[DONE]":
                        break
                    try:
                        content = json.loads(line_str)["choices"][0]["delta"].get("content", "")
                        # content = eval(line_str)["choices"][0]["delta"].get("content", "")
                        yield content
                    except Exception:
                        continue


if __name__ == "__main__":
    # Example usage
    from default_config import config
    client = QwenAIClient(base_url=config.QWEN_API_URL, api_key=config.DASHSCOPE_API_KEY)
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    
    # Streaming response
    for chunk in client.chat(messages, stream=True):
        print(chunk)
    
    # Non-streaming response
    response = client.chat(messages, stream=False)
    print(response)