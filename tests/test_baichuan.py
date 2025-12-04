import requests
import json

def do_request():
    url = "https://api.baichuan-ai.com/v1/chat/completions"
    api_key = "your_api_key"

    data = {
        "model": "Baichuan-M2-Plus",
        "messages": [
            {
                "role": "user",
                "content": "25岁健康女性种植牙，刚做完植入种植体，请问手术后是否需要服用抗生素"
            }
        ],
        "stream": True
    }

    json_data = json.dumps(data)

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + "sk-c1acdf2e09f54ed0c3cb8f12621f9e0c"
    }

    response = requests.post(url, data=json_data, headers=headers, timeout=60, stream=True)

    if response.status_code == 200:
        print("请求成功！")
        print("请求成功，X-BC-Request-Id:", response.headers.get("X-BC-Request-Id"))
        for line in response.iter_lines():
            if line:
                print(line.decode('utf-8'))
    else:
        print("请求失败，状态码:", response.status_code)
        print("请求失败，body:", response.text)
        print("请求失败，X-BC-Request-Id:", response.headers.get("X-BC-Request-Id"))

if __name__ == "__main__":
    do_request()



              
