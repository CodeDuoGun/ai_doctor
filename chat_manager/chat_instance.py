import asyncio
import traceback
from utils.log import logger

from chat_manager.chat_data import ChatRuntimeData

class ChatInstance:
    def __init__(self):
        self._thread = None
        self.runtime_data = ChatRuntimeData()
        self.is_running = False
        self.tasks = []
        self._businesses = []
    
    def is_valid(self):
        return self.is_running
    
    async def thread_function(self):
        try:
            tasks = []
            # TODO: 暂时用不到Wav2Lip
            # from chat_manager.wav2lip_v1 import ChatWav2Lip
            # self._businesses.append(ChatWav2Lip())

            # TODO：目前做一句话识别，不做vad
            # from room_business.business_vad import BusinessVad
            # self._businesses.append(BusinessVad())

            from chat_manager.nlp_task import ChatNLP
            nlp_processor = ChatNLP()
            self._businesses.append(nlp_processor)
            #用来处理打断，需要停止nlp
            # self.runtime_data.nlp_interrupt_handler = nlp
            from chat_manager.tts_task import ChatTTS
            tts_processor = ChatTTS()
            self._businesses.append(tts_processor)
    

            # TODO: 暂时让 ASR 在这里吧
            from chat_manager.asr_task import ChatASR
            asr_processor = ChatASR()
            self._businesses.append(asr_processor)

            # logger.debug(f"{self.init_data.live_id} 任务队列就绪，准备启动")
            for business in self._businesses:
                business.parent = self
                self.tasks.append(self.loop.create_task(business.run()))
            # if self.tasks:
            #     await asyncio.gather(*self.tasks)  # 等待所有任务完成,是否有必要这么写?
            # if len(tasks) > 0:
            #     self.loop.run_until_complete(asyncio.gather(*tasks))
            logger.info(f'启动：{self.init_data.live_id} 房间启动的全部准备工作都做好了')
        except:
            logger.error(f'{self.init_data.live_id} 启动任务队列出错 {traceback.format_exc()}')
    def start_tasks_in_main_thread(self):
        self.loop = asyncio.get_event_loop()  # 获取主线程的事件循环
        # await self.thread_function()  
        asyncio.create_task(self.thread_function())# 调度任务，不阻塞主线程

    def close_async_queue(self,queuue : asyncio.Queue):
        try:
            if queuue:
                queuue.put_nowait(None)
                queuue = None
        except:
            pass

    async def start_chat(self):
        if self.is_running:
            return
        if self._thread is not None: # TODO要改
            error_msg = (
                "self._thread exist 没有调用停止直播或这重复开始直播，或者线程还没死"
            )
            return error_msg
        pass
        self.start_tasks_in_main_thread()


    def reset(self):
        self._thread = None
        self.is_running = False
        self._businesses.clear()
        # 清理队列
        self.runtime_data.shutup()
        # self.runtime_data = ChatRuntimeData()

    async def stop_chat(self):
        self.is_running = False
        try:
            if self._thread is not None:
                self._thread.join(timeout=20)

            self.close_async_queue(self.runtime_data.asr_msg_queue)
            self.close_async_queue(self.runtime_data.nlp_answer_queue)
            self.close_async_queue(self.runtime_data.queue_25d_audio)
            logger.info(f"停播结束")
            return 200
        except Exception:
            logger.error(
                f"stop live of  error {traceback.format_exc()}"
            )
            return 500