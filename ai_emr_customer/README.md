# ai_emr_customer
医疗智能体服务，不限于智能病历、智能客服

# 数据清洗
1、来自阿里天池数据集
2、医学教材构建问答对
3、
任务类型	推荐训练量	验证集比例
简单问答（知识类）	3万 - 20万对	10%-15%
问诊模拟	≥ 5000 条	1000 条左右
病历摘要/结构化	5000 - 20000 条	1000 - 3000 条
示例如下：
{
  "instruction": "请简要总结流感的主要症状",
  "input": "",
  "output": "发热、头痛、乏力、咽痛、肌肉酸痛和鼻塞。"
}


# 模型选择
Qwen1.5, ChatGLM3, Baichuan2, Mistral（中文任务建议中文模型）

# 问诊


# 报告解读
微调



# 病历生成
python3 -m vllm.entrypoints.openai.api_server --model app/model/qwen3_lora_merged --tokenizer app/model/qwen3_lora_merged --trust-remote-code  --dtype float16

https://voives.sihuiyiliao.com/202508/xlys_hx1754126556959.mp3 # 电话问诊


# 诊疗意见


# 处方建议
