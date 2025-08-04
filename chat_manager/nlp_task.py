from chat_manager.chat_base_processor import ChatManagerBase
from utils.log import logger
from llm.Qwen import Qwen
import asyncio
import traceback
import json

tick_time = 0.01

class ChatNLP(ChatManagerBase):
    def __init__(self, websocket, nefreal):
        super().__init__(websocket, nefreal)
    
    def init_data(self):
        self.is_running = False
        self.llm = Qwen()
        self.history = []

    
    async def get_nlp_resp(self, msg):
        if not self.history:
            self.history= [
                {"role": "system", "content": ""}
            ]
        self.history.append({"role": "user", "content": msg})
        nlp_res = await self.llm.chat_stream(msg, messages=self.history)
        # TODO: nlp 切分,要保证连续
        answer_id = "nlp_answer"
        self.history.append({"role": "assistant", "content": nlp_res})
        await self.websocket.send(json.dumps({"nlp": nlp_res}, ensure_ascii=False))
        await self.put_nlp2tts(answer_id, nlp_res)

    async def put_nlp2tts(self, answer_id: int, nlpmsg_chunk: str):
        """
        如果发生打断，清空后，这里也不应该继续放入上次的结果
        """
        if nlpmsg_chunk:
            data = {
                "sentence_id": answer_id,
                "text": nlpmsg_chunk,
            }
            logger.info(f"{data['sentence_id']} nlpmsg_chunk:{nlpmsg_chunk}")
            self.parent.runtime_data.nlp_answer_queue.put_nowait(data)
    
    
    async def run(self):
        try:
            self.init_data()
            while self.parent.is_valid():
                # 获取
                logger.debug(f"NLP start")
                if not self.parent.is_valid():
                    self.is_running = False
                    break
                asr_msg = await self.parent.runtime_data.asr_msg_queue.get()
                if asr_msg:
                    await self.get_nlp_resp(asr_msg)
                else:
                    await asyncio.sleep(tick_time)
                    continue
               
                await asyncio.sleep(tick_time)
        except Exception:
            logger.error(f"nlp err :{traceback.format_exc()}")

        logger.info(f"stopping_room   exit nlp logic")