class Speeches():
    NoDoctorSpeech = "您好！暂时没有查到该医生的详细信息，建议您可以前往对应医院的官网，或通过百度等平台确认医生的完整姓名；如果您不确定医生信息，也可以直接描述您的症状，小惠会尽力为您推荐合适的医生哟～"
    RecommendDoctorSpeech = "您好，根据您的描述，为您推荐以下医生, 点击链接查看医生详细出诊信息和挂号:"
    QueryDoctorSpeech = "请问您是否需要帮您推荐医生进行挂号？您还有其他需要了解的么？"
    AIWarningSpeech = "本回答由 AI 生成，内容仅供参考"
#     intentSpeech:'''以下是十个用于引导用户澄清意图的客服话术示例，您可以根据实际场景灵活调整：\n\n
# 1. “您好！请问您是需要**预约挂号**、**找医生**、**查询患者信息**、**疾病咨询**、**用药咨询**，还是其他方面的帮助呢？”  
# 2. “为了更好地为您服务，请您告诉我：您此刻主要是想**挂号预约**，还是**咨询病情**、**用药问题**，或是**查找医生/患者**？”  
# 3. “抱歉，我有点没太明白您的需求。您是想**预约专家号**，还是**在线问诊**、**了解用药注意事项**，或有其他问题？”  
# 4. “您好！您这边是需要我帮您**挂号就诊**，还是**寻找合适医生**，或者**咨询疾病症状/用药方法**呢？”  
# 5. “请您简要说明一下：是要**预约门诊**、**找特定医生**、**了解某位患者信息**，还是**疾病或用药方面的咨询**？”  
# 6. “感谢您联系我，请问您当前的主要需求是**挂号预约**、**医生推荐**、**患者查询**、**病情咨询**还是**用药指导**？”  
# 7. “为了精准帮您解决问题，请您告诉我：您是要**挂号**，还是**找医生**？如果找医生，请提供**医生姓名**或您的**主要症状、年龄、性别**哦。”  
# 8. “您好，请先告知：您是要**预约就诊**，还是**在线咨询疾病/用药**，亦或是**查找医生或患者**？若需找医生，还请提供姓名或您的症状及个人信息。”  
# 9. “不好意思，我这边需要确认您的需求：**挂号预约**？**医生查询**？**患者信息**？还是**疾病／用药咨询**？若找医生，请告诉我医生姓名或您的病情、年龄、性别。”  
# 10. “您好，为确保准确为您服务，请说明：是要**挂号**、**找医生**（请提供医生姓名或您的症状、年龄、性别）、**咨询疾病**、**用药问题**，还是其他需求？'''
    
class LLMContentType:
    TETX: str="text"
    AUDIO: str="audio"
    IMAGE: str="image"
    VIDEO: str="video"

class MessageEventStatus:
    DELTA:str="delta"
    COMPLETED:str="completed"

FINISH_REASONS = ["stop", "length", "content_filter", "insufficient_system_resource","tool_calls"]

MODLE_MAPPING = {
    "coze_deepseek-r1": "deepseek-r1-250120" ,
    "coze_deepseek": "deepseek-v3-250324",
    "deepseek-r1": "deepseek-reasoner",
    "deepseek": "deepseek-chat"
}

QUESTION_FLOW = [
    ("chief_complaint", "请描述您现在最主要的不适或症状？"),
    ("symptom_onset_time", "这个症状从什么时候开始的？"),
    ("symptom_location", "不适主要是身体哪个部位？"),
    ("symptom_nature", "请描述症状的具体表现（如刺痛、胀痛、发热等）？"),
    ("symptom_trigger_or_relief", "有没有什么诱因或缓解症状的方式？"),
    ("past_medical_history", "您以前有过什么疾病或做过手术吗？"),
    ("allergy_history", "您对哪些药物或食物过敏？"),
    ("medication_history", "您最近有没有服用什么药物？"),
    ("family_history", "家族中有相似病史或遗传病吗？"),
    ("personal_lifestyle", "请描述一下您的生活习惯，如作息、饮食等。"),
    ("tongue_image", "请上传一张舌照图片链接："),
    ("face_image", "请上传一张面照图片链接："),
    ("inspection_reports", "请上传检查资料的链接（多个请用逗号分隔）："),
    ("previous_diagnosis", "您之前有没有诊断过相关的病？医生怎么说的？"),
    ("additional_information", "还有什么您觉得重要的信息要补充吗？")
]



    