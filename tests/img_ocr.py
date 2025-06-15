from openai import OpenAI
import json
import base64

client = OpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", api_key="sk-325f79c2085d481c9a8a3e625c9a698b")

def load_img():
    pass

system_prompt = """提取图片内容,并严格按照下面的json返回。
输出示例：
{
  "likes": 0,
  "follows": 0,
  "followers": 0,
  "remarks": "无明显规避内容",
  "total_videos": 0,
  "storefront": true,
  "ai_check": false
}
# 字段说明
1.likes 表示获赞数，没有为0
2.follows 表示关注数，没有为0
3.followers 表示粉丝数，没有为0
4.remarks 表示备注内容，没有为空字符串
5.total_videos 表示作品数量，没有为0
6.storefront 表示是否开通，有则true，否则false
7.ai_check 表示截图类型和用户传入类型“video_type”是否匹配，如果匹配返回true，否则false

限制：
1.必须严格按照json返回
2.禁止用```包裹json,直接输出
"""

def img2base64(img_path):
    with open(img_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def main(img1_path:str, img2_path, video_type: str):
    """输入两张图片"""
    messages = [
        {
            "role": "system",
            "content": [
                    {
                        "type": "text", "text": system_prompt
                    }
                ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img2base64(img1_path)
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img2base64(img2_path)
                        },
                },
                {"type": "text", "text": f"解析图片内容并返回json结果， video_type:{video_type}"},
            ],
        },
    ]
    # messages = [{"role":"system", "constant": system_prompt}, {"role": "user", "constant": ""}]
    resp = client.chat.completions.create(messages=messages, model="qwen-vl-max-latest", stream=True)
    content = ""
    for chunk in resp:
        if chunk.choices[0]:
            chunk = chunk.choices[0].delta.content
            content += chunk
    print(content)
    res = json.loads(content)
    print(res)
    # 规则匹配
    if 0 < res.get("likes", 0) < 1:
        pass
    if 0 < res.get("followers", 0) < 1: # 粉丝数
        pass
    if res.get("remarks", "") in ("laji"): # 备注
        return False
    if res.get("storefront", False): # 橱窗
        return False
    if 0< res.get("total_videos", 1) < 1: # 作品数
        return False
    if not res.get("ai_check", False):
        return False

    # 判断规则
    return True

if __name__== "__main__":
    main("data/cache/test0.jpg", "data/cache/test0.jpg", video_type="唱歌")