from chat_manager.chat_base_processor import ChatManagerBase
from utils.log import logger
import traceback
from ttsreal_v2 import CustomTTS
import asyncio

tick_time = 0.01
class ChatTTS(ChatManagerBase):
    def __init__(self, websocket, nefreal):
        super().__init__(websocket, nefreal)


    def init_data(self):
        self.is_running = False
        self.tts = CustomTTS(self.nefreal)

    async def run(self):
        self.init_data()
        while self.parent.is_valid:
            try:
                logger.debug(f"TTS start")
                if not self.parent.is_valid():
                    self.is_running = False
                    break
                nlp_info = await self.parent.runtime_data.nlp_answer_queue.get()
                if nlp_info:
                    self.tts.txt_to_audio(nlp_info["text"])
                else:
                    await asyncio.sleep(tick_time)
                    continue
                
                await asyncio.sleep(tick_time)
            except Exception:
                logger.error(f"TTS exist {traceback.format_exc()}")
                self.is_running = False
                break