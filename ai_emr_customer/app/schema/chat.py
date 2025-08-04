from pydantic import BaseModel

class ChatRequest(BaseModel):
    conversation_id: str
    query: str
    model_name: str= "coze_deepseek"

class CreateChatItem(BaseModel):
    uid: str
    device_id:str="shyl"

class IntentItem(BaseModel):
    conversation_id:str
    query: str

class UpdateQAItem(BaseModel):
    question: str
    answer: str=""
    new_question: str
    new_answer: str