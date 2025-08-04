import requests 
from app.utils.log import logger
import traceback

def request_backend_emr(id):
    try:
        #TODO: url 从配置获取
        case_detail_url = f"https://api-backend.sihuiyiliao.com/orders_service/api/registerOrder/caseDetail?register_order_id=617494&id={id}"
        # case_detail_url = f"https://api-backend.sihuiyiliao.com/orders_service/api/prescriptionOrder/caseDetail?id={id}"
        headers = {
                "Accept": "application/json",
    "Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJodHRwOi8vMTcyLjE2LjAuNDA6OTUwNi91c2Vycy91c2VyTG9naW4iLCJpYXQiOjE3NTA5ODk4ODQsImV4cCI6MTc2NTM4OTg4NCwibmJmIjoxNzUwOTg5ODg0LCJqdGkiOiJwa29ZMmZOMmpJM1ZSRWRYIiwic3ViIjoiMTA4OTMiLCJwcnYiOiJiYzY2MDUxZWFkMzE3NTYxNTRjY2I3ZjU4MGRhNTMxZTcxZWE4OTkyIn0.WcNXrz9oXCjFsavnci1F7_2yWXbvJoSZfLrW2pjA3NerDebGXos6BpcHCSOv6eSWJZlY-hlWmS0INAtaOor_-A",
        }

        data = requests.get(case_detail_url, headers=headers)
        print(data.status_code)
        if data.status_code != 200:
            logger.error(f"调用接口{case_detail_url}获取病历数据失败")
            return

        data = data.json()
        if data.get("code") != 200:
            logger.error(f"获取病历数据失败，错误信息: {data['code']}")
            return
        return data["data"]
    except Exception:
        logger.error(f"获取病历数据失败: {traceback.format_exc()}")
        return

def content_handler(content, msg_type):
    if msg_type == 1:# audio
        # asr
        text = ""

    elif msg_type == 2: # video
        # 1、直接调用视频理解接口
        pass
        # 2、提取 音频，再做asr

    elif msg_type == 4: # img
        pass
        # 暂时不使用，使用问诊单中的图片
    

def get_backend_chat_his():
    url1 = 'https://api-backend.sihuiyiliao.com/users_service/api/message/list?user_info_id=755545&patient_id=583407&doctor_id=432&time=1751090054982'
    url2 = 'https://api-backend.sihuiyiliao.com/users_service/api/message/list?user_info_id=755545&patient_id=583407&doctor_id=432&time=1751334929577'
    chat_url = [url1, url2]
    headers = {
                "Accept": "application/json",
    "Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJodHRwOi8vMTcyLjE2LjAuNDA6OTUwNi91c2Vycy91c2VyTG9naW4iLCJpYXQiOjE3NTA5ODk4ODQsImV4cCI6MTc2NTM4OTg4NCwibmJmIjoxNzUwOTg5ODg0LCJqdGkiOiJwa29ZMmZOMmpJM1ZSRWRYIiwic3ViIjoiMTA4OTMiLCJwcnYiOiJiYzY2MDUxZWFkMzE3NTYxNTRjY2I3ZjU4MGRhNTMxZTcxZWE4OTkyIn0.WcNXrz9oXCjFsavnci1F7_2yWXbvJoSZfLrW2pjA3NerDebGXos6BpcHCSOv6eSWJZlY-hlWmS0INAtaOor_-A",
        }
    chat_messages = []
    for url in chat_url:
        resp = requests.get(url, headers=headers)
        print(resp.status_code)
        if resp.status_code != 200:
            logger.error(f"调用接口{url}获取病历数据失败")
            return

        resp = resp.json()
        if resp["code"] != 200:
            logger.error(f"获取病历数据失败，错误信息: {resp['code']}")
            return
        for chat_msg in resp["data"]:
            role = "doctor" if "doctor" in chat_msg["from"] else "patient"
            if chat_msg["type"] in (1,2,4):
                content = ""
            else:
                content = chat_msg[content]
            content