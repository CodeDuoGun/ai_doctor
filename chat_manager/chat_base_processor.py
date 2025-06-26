from chat_manager.chat_instance import ChatInstance


class ChatManagerBase:
    def __init__(self):
        self.parent = None

    def init(self, parent: ChatInstance):
        self.parent = parent

    async def run(self):
        pass
    