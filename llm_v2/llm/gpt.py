import requests
import json
from typing import Generator, Optional
from app.model.llm.base import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    def chat(self, messages: list[dict], stream: bool = False, llm_model: str = "gpt-4o") -> str | Generator[str, None, None]:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": llm_model,
            "messages": messages,
            "temperature": 0.7,
            "stream": stream
        }

        response = requests.post(url, headers=headers, json=payload, stream=stream)

        if not response.ok:
            raise Exception(f"OpenAI error: {response.text}")

        if stream:
            return self._parse_stream_response(response)
        else:
            return response.json()["choices"][0]["message"]["content"]
    
    def chat_with_img(self):
        pass

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

# if __name__ == "__main__":
#     # Example usage
#     from default_config import config
#     client = OpenAIClient(base_url=config.OPENIAI_BASE_URL, api_key=config.OPENIAI_API_KEY)
#     messages = [{"role": "user", "content": "Hello, how are you?"}]
    
#     # Streaming response
#     for chunk in client.chat(messages, stream=True):
#         print(chunk)
    
#     # Non-streaming response
#     response = client.chat(messages, stream=False)
#     print(response)