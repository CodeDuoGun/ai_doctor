from chat_manager.chat_instance import ChatInstance


class ChatManagerBase:
    def __init__(self, websocket, nefreal):
        self.parent = None
        self.websocket = websocket
        self.nefreal = nefreal

    def init(self, parent: ChatInstance):
        self.parent = parent

    async def run(self):
        pass
    