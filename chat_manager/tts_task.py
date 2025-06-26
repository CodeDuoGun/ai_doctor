from chat_manager.chat_base_processor import ChatManagerBase
from utils.log import logger
import traceback
from ttsreal_v2 import CustomTTS

tick_time = 0.01
class ChatTTS(ChatManagerBase):
    def __init__(self):
        super().__init__()


    def init_data(self):
        self.is_running = False
        self.tts = CustomTTS()

    async def run(self):
        self.init_data()
        while self.parent.is_valid:
            nlp_info = await self.parent.runtime_data.nlp_answer_queue.get()
            await self.gen_tts(nlp_info)
            pass