import json
import re
from app.utils.redis_db import redis_tool as r
from app.utils.log import logger

# TODO: 确认是否需要其他字段构建唯一key
def create_conversation(chat_id, user_id: str, device_id:str="shyl", history:list=[]):
    """
    会话id采取时间戳形式
    """
    conversation_id = f"{chat_id}_{user_id}"
    key = f"history_{conversation_id}_{device_id}"
    if r.exist(key):
        logger.info(f"{key} exists, do not create again")
        return conversation_id
    r.set_expire(key, json.dumps(history))
    return conversation_id


def get_conversation(conversation_id, device_id:str="shyl"):
    key = f"history_{conversation_id}_{device_id}"
    if not r.exist(key):
        r.set_expire(key, json.dumps([]))
        return []
    data = r.get(key)
    if data:
        return json.loads(data)
    return []


def update_conversation(conversation_id, new_history, trace_id: str="", device_id:str="shyl"):
    key = f"history_{conversation_id}_{device_id}"
    if not r.exist(key):
        logger.error(f"{trace_id} Not found key of {conversation_id}")
    # for his in new_history:
    #     if isinstance(his, str):
    #         his["content"] = remove_special_punctuation(his["content"])
    r.set_expire(key, json.dumps(new_history))
    logger.info(f"{trace_id} 成功更新对话")

def delete_conversation(conversation_id, device_id:str="shyl"):
    key = f"history_{conversation_id}_{device_id}"
    r.delete(key)
    logger.debug(f"删除对话：{key}")