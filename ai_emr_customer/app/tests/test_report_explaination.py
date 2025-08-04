from openai import OpenAI
import base64
import os 
from app.prompts.report import system_prompt
def convert_image_file_to_base64(image_path):
    """将图片文件转换为 Base64 编码字符串"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string


client = OpenAI(api_key="f43df693-455a-4e3e-b987-d76d7b57f4c3", base_url="https://ark.cn-beijing.volces.com/api/v3")  # 替换为你的 API Key

def request_llm():
    user_prompt = "请根据以下医学报告内容，提供详细的解读和建议。"
    file_path = "app/data/test_img.jpg"
    file_type = os.path.basename(file_path).split('.')[-1]  # 获取文件类型
    response = client.chat.completions.create(
        model="doubao-1-5-vision-pro-32k-250115",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {
                    "type": "text",
                    "text": user_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{file_type};base64,{convert_image_file_to_base64(file_path)}"
                    }
                }
            ]}
        ],
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

def explain_report(report: str) -> str:
    """
    根据用户上传的医学报告，提供详细的解读和建议。
    """
    # 这里可以调用模型进行解读
    # 模拟返回结果
    return f"解读结果：{report} 的解读和建议。"


def load_data() -> str:
    """
    加载测试数据。
    """
    # 这里可以从文件或数据库加载测试数据
    return "这是一个医学报告的示例文本。"

def huatuo_report_explaination():
    """
    医学报告解读测试函数。
    """
    query = 'What does the picture show?'
    image_paths = ['image_path1']
    # Load model directly
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("FreedomIntelligence/HuatuoGPT-Vision-7B")

    from HuatuoGPT_Vision.cli import HuatuoChatbot
    bot = HuatuoChatbot(huatuogpt_vision_model_path) # loads the model 
    output = bot.inference(query, image_paths) # generates
    print(output)  # Prints the model output

def test_baichuan():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    # 1. Load pre-trained model and tokenizer
    model_name = "baichuan-inc/Baichuan-M1-14B-Instruct"  
    tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True,torch_dtype = torch.bfloat16).cuda()
    # 2. Input prompt text
    prompt = "May I ask you some questions about medical knowledge?"

    # 3. Encode the input text for the model
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    ) 
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 4. Generate text
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # 5. Decode the generated text
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


    # 6. Output the result
    print("Generated text:")
    print(response) 


def test_qwen_vl():
    """"""
    import os
    from openai import OpenAI

    client = OpenAI(
        api_key="sk-4ef42187cc2e47999d0c835c10fc5a78",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    """
    "https://images.sihuiyiliao.com/xlys/patient/case/2025-06-02/a92b6eae-a833-b0aa-6b87-30f54b17b9ff.jpg",
    "https://images.sihuiyiliao.com/xlys/patient/case/2025-06-02/cb20a934-c47a-fecb-2d74-31a954c3deb3.jpg",
    "https://images.sihuiyiliao.com/xlys/patient/case/2025-06-02/b228ccb6-7427-c22f-4f95-260c767320d4.jpg",
    "https://images.sihuiyiliao.com/xlys/patient/case/2025-06-02/c14fea31-935c-081b-9133-38b48e2ed271.jpg",
    "https://images.sihuiyiliao.com/xlys/patient/case/2025-06-02/f999c8b0-72aa-72b4-f82a-cd4620524beb.jpg",
    "https://images.sihuiyiliao.com/xlys/patient/case/2025-06-02/8261dfd1-12b1-2590-9a29-05bcaab19a4d.jpg",
    "https://images.sihuiyiliao.com/xlys/patient/case/2025-06-02/c6892c76-48d4-4dc6-8c50-57af3a6fc8ca.jpg",
    "https://images.sihuiyiliao.com/xlys/patient/case/2025-06-02/733655a9-05a9-f5f6-5d0e-5565b7489167.jpg"
    """
    completion = client.chat.completions.create(
        model="qwen-vl-max-2025-04-08", # 此处以qwen-vl-max-latest为例，可按需更换模型名称。模型列表：https://help.aliyun.com/model-studio/getting-started/models
        messages=[
        {"role":"system","content":[{"type": "text", "text": system_prompt}]},
        {"role": "user","content": [
            # 第一张图像url，如果传入本地文件，请将url的值替换为图像的Base64编码格式


            # {"type": "image_url","image_url": {"url": "https://images.sihuiyiliao.com/xlys/patient/case/2025-06-02/a92b6eae-a833-b0aa-6b87-30f54b17b9ff.jpg"},},
            # # 第二张图像url，如果传入本地文件，请将url的值替换为图像的Base64编码格式
            # {"type": "image_url","image_url": {"url": "https://images.sihuiyiliao.com/xlys/patient/case/2025-06-02/cb20a934-c47a-fecb-2d74-31a954c3deb3.jpg"},},
            # {"type": "image_url","image_url": {"url": "https://images.sihuiyiliao.com/xlys/patient/case/2025-06-02/b228ccb6-7427-c22f-4f95-260c767320d4.jpg"},},
            # {"type": "image_url","image_url": {"url": "https://images.sihuiyiliao.com/xlys/patient/case/2025-06-02/c14fea31-935c-081b-9133-38b48e2ed271.jpg"},},
            # {"type": "image_url","image_url": {"url": "https://images.sihuiyiliao.com/xlys/patient/case/2025-06-02/f999c8b0-72aa-72b4-f82a-cd4620524beb.jpg"},},
            # {"type": "image_url","image_url": {"url": "https://images.sihuiyiliao.com/xlys/patient/case/2025-06-02/8261dfd1-12b1-2590-9a29-05bcaab19a4d.jpg"},},
            # {"type": "image_url","image_url": {"url": "https://images.sihuiyiliao.com/xlys/patient/case/2025-06-02/c6892c76-48d4-4dc6-8c50-57af3a6fc8ca.jpg"},},
            # {"type": "image_url","image_url": {"url": "https://images.sihuiyiliao.com/xlys/patient/case/2025-06-02/733655a9-05a9-f5f6-5d0e-5565b7489167.jpg"},},
            {"type": "image_url","image_url": {"url": f'data:image/jpg;base64,{convert_image_file_to_base64("app/data/test_img.jpg")}'}},
            {"type": "text", "text": "分析一下"},
                ],
            }
        ],
        stream=True,
    )
    

    content = ""
    for chunk in completion:
        if chunk:
            print(chunk)
            content += chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
    print(f"content: {content}")

if __name__ == "__main__":
    # huatuo_report_explaination()
    # test_baichuan()
#     request_llm()
    test_qwen_vl()