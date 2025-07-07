from openai import OpenAI
import base64
import os

client = OpenAI(api_key="f43df693-455a-4e3e-b987-d76d7b57f4c3", base_url="https://ark.cn-beijing.volces.com/api/v3")  # 替换为你的 API Key
def convert_image_file_to_base64(image_path):
    """将图片文件转换为 Base64 编码字符串"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string

def extract_image_info_by_model():
    user_prompt = "提取图片中的文本，基于提取的文字，给出一个摘要或者总结。要求精准无误、逻辑清晰。内容包括患者所有信息、检查报告信息、指标信息。最后给出病人的病情分析和治疗建议"
    system_prompt = """
## Role 你是一个医学专家，擅长分析医学影像和报告。请根据用户提供的图片内容进行详细分析，并给出专业的建议。\n
## Skills
- 图像文字识别（OCR）能力
- 医学术语理解与归纳能力
- 检查报告与指标的逻辑整合能力
- 提供病情判断与治疗建议的专业能力

## Goals:
- 输出结构清晰的分析报告，四个部分：
  1. 患者信息：姓名、年龄、性别、就诊日期、就诊医院、科室、接诊医生
  2. 检查报告内容和指标解读：提取原始报告文字并逐项分析指标意义
  3. 病情分析：根据检查内容判断可能疾病或病变状态
  4. 治疗建议：基于医学常规提出初步建议（如进一步检查、药物治疗、随访等）

## Rules
1. 所有信息必须基于图像中提取的文字内容。
2. 输出语言为中文，要求医学术语准确、逻辑清晰。
3. 不进行无根据的推断，仅基于已有信息提供合理解释。
4. 输出结构保持一致，四段分明，简洁但不遗漏关键信息。

## Workflows
1. 执行OCR提取图像中文字。
2. 解析患者资料和检查信息。
3. 识别并解读指标、检查内容。
4. 结合医学常识与经验，生成病情分析与治疗建议。

示例：
    输出示例1：
        1.患者信息
        张三，35岁女性，2023年6月3日就诊于罗定市人民医院妇科门诊。接待医生：王丽。

        2.检查报告内容和指标解读
        B超检查显示子宫大小为5.8×4.2×3.5cm，形态规则，子宫内膜厚度12mm，回声均匀，属正常范围。左侧卵巢发现大小为3.0×2.5cm的卵巢，内部见一低回声囊性区，约2.1×1.8cm，边界清晰，内容透声良好，提示为典型卵巢囊肿表现。右侧卵巢形态、大小正常，无异常发现。

        3.病情分析
        患者左侧卵巢囊性病变符合单纯性卵巢囊肿的声像图特征，多为功能性囊肿或良性囊肿。无明显恶性特征，暂无明显急性并发症表现。

        4.治疗建议
        建议定期随访复查（建议1-3个月内复查一次B超），观察囊肿大小变化。期间注意月经周期规律及腹痛等异常症状。如囊肿持续增大或合并症状，可考虑进一步MRI检查或手术处理。无需急于干预，注意健康管理。

"""
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


if __name__ == "__main__":
    extract_image_info_by_model()