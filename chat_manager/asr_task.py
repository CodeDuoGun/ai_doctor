from chat_manager.chat_base_processor import ChatManagerBase
from utils.log import logger
from asr.FunASR import AliFunASR
import asyncio
import traceback

tick_time = 0.01

class ChatASR(ChatManagerBase):
    def __init__(self):
        super().__init__()
    
    def init_data(self):
        self.is_running = False
        self.asr = AliFunASR()

    async def run(self):
        while self.parent.is_valid:
            self.init_data()
            logger.debug(f"ASR running")
            if not self.parent.is_valid():
                self.is_running = False
                break
            asr_msg = await self.parent.runtime_data.asr_msg_queue.get()
            if asr_msg:
                await self.asr.recognize(asr_msg)
            else:
                await asyncio.sleep(tick_time)
                continue
            
            await asyncio.sleep(tick_time)
    
            