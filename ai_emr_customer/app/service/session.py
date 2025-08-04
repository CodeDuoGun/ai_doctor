from app.constants import QUESTION_FLOW

class SessionProcessor():
    def __init__(self):
        self.user_sessions = {}  # key: user_id or session_id

    def init_session(self, user_id):
        self.user_sessions[user_id] = {
            "current_index": 0,
            "data": {}
        }

    async def handle_user_message(self, user_id, user_input):
        session = self.user_sessions.get(user_id)
        if not session:
            self.init_session(user_id)
            session = self.user_sessions[user_id]

        current_index = session["current_index"]
        current_field, _ = QUESTION_FLOW[current_index]

        # 保存用户输入
        session["data"][current_field] = user_input.strip()

        # 更新到下一个问题
        session["current_index"] += 1

        if session["current_index"] >= len(QUESTION_FLOW):
            # 问卷完成
            collected_data = session["data"]
            return "预问诊信息已收集完成，感谢您的配合！", collected_data

        # 提下一个问题
        next_question = QUESTION_FLOW[session["current_index"]][-1]
        return next_question, None
