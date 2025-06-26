from chat_manager.chat_base_processor import ChatManagerBase
from utils.log import logger
from llm.Qwen import Qwen
import asyncio
import traceback

tick_time = 0.01

class ChatNLP(ChatManagerBase):
    def __init__(self):
        super().__init__()
    
    def init_data(self):
        self.is_running = False
        self.llm = Qwen()

    
    async def get_nlp_resp(self, msg):
        nlp_res = await self.llm.chat_stream(msg)
        # TODO: nlp 切分,要保证连续
        answer_id = "nlp_answer"
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
                logger.debug("NLP start")
                if self.default_msg_falg:
                    logger.info(f"nlp_队列收到消息 default")
                    data = {
                        "room_id": self.parent.init_data.live_id,
                        "msg_id": "default",
                        "sentence_id": "default",
                        "asr_text": "default",
                        "last_msg_id": "default"
                    }
                    await self.get_nlp_resp(data)
                    self.default_msg_falg = False
                    continue

                if not self.parent.is_valid():
                    self.is_running = False
                    break
                await self.connect()
                
                await asyncio.sleep(tick_time)
        except Exception:
            logger.error(f"nlp err :{traceback.format_exc()}")
        finally:
            await self.close()

        logger.info(f"stopping_room {self.parent.init_data.live_id}  exit nlp logic")