from chat_manager.chat_base_processor import ChatManagerBase
from utils.log import logger
from asr.FunASR import AliFunASR
import asyncio
import traceback
import json
import time

tick_time = 0.01

class ChatASR(ChatManagerBase):
    def __init__(self, websocket, nefreal):
        super().__init__(websocket, nefreal)
    
    def init_data(self):
        self.is_running = False
        self.asr = AliFunASR()

    async def run(self):
        try:
            while self.parent.is_valid:
                self.init_data()
                logger.debug(f"ASR running")
                if not self.parent.is_valid():
                    self.is_running = False
                    break
                vad_msg = await self.parent.runtime_data.queue_vad.get()
                if vad_msg:
                    t0 = time.perf_counter()
                    asr_result =  await self.asr.recognize(vad_msg)
                    logger.info(f"asr cost time {time.perf_counter() - t0}")
                    await self.websocket.send(json.dumps({"asr": asr_result}, ensure_ascii=False))
                    await self.parent.runtime_data.asr_msg_queue.put(asr_result)

                else:
                    await asyncio.sleep(tick_time)
                    continue
                
                await asyncio.sleep(tick_time)
        except Exception:
            logger.error(f"asr error {traceback.format_exc()}")
        logger.info(f"exit asr task")
        
            