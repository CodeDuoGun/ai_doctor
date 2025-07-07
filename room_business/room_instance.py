import asyncio
import enum
import json
import multiprocessing
import threading
import time
import traceback

import numpy as np

from config import config
from constants import InferAction, Status, WarmupEvent
from log import logger
from room_business.room_data import RoomInitData, RoomRuntimeData
from wav2lip.high_perf.buffer.infer_buffer import Infer_Cache
from wav2lip.high_perf.buffer.share_memory_variable import ShareMemoryInt
from wav2lip.high_perf.buffer.video_buff import Video_Cache
from wav2lip.infer.wav2lip_processor import process_wav2lip_predict
from wav2lip.room_resource.model_people_local import change_background, download


# 可能的状态，启动中（starting），运行中（RTCReadying），stopping，stopped
class RoomState(enum.Enum):
    starting = 1
    ready = 2
    stopping = 3
    stopped = 4
    dead = 5


class RoomInstance:
    def __init__(self):
        self._thread = None
        self.init_data = RoomInitData()
        self.runtime_data = RoomRuntimeData()
        self.is_running = False
        self._businesses = []
        self.tasks = []

        self.wav2lip_process: multiprocessing.Process = None
        self.wav2lip_request_queue: multiprocessing.Queue = None
        self.wav2lip_result_queue: multiprocessing.Queue = None
        self.wav2lip_stop_event  = None

        self.wav2lip_process_started = False
        self.wav2lip_process_state = RoomState.stopped
        self.read_index_share_memory = None
        self.write_index_share_memory = None

        self.video_cache = None
        self.data_cache = None

        self.share_memory_cache_threshold = 2 * int(config.g_cache_unplayed_duration * 25)
        self.rtc_ready = False

        self.loop = None
        self.last_batch_speak = False #  记录上一音频是否还在说话
        self.next_batch_speak = False # 记录下一段音频是否还在说话, 默认说话停止
        self.asr_call_back_task = None
        self.nlp_call_back_task = None

    def reset(self):
        self._thread = None
        self.is_running = False
        self._businesses.clear()
        # 清理队列
        self.runtime_data.shutup()
        self.runtime_data = RoomRuntimeData()


    # 判断直播间是否有效
    def is_valid(self):
        return self.is_running and self.wav2lip_process_state == RoomState.ready

    # 线程创建函数，内部会创建多个逻辑协程
    async def thread_function(self):
        try:
            logger.debug(f"{self.init_data.live_id} wav2lip 已经启动成功,现在到任务队列了")

            tasks = []

            # 交互
            from room_business.business_interaction import BusinessInteraction

            self._businesses.append(BusinessInteraction())

            # 缓存
            from room_business.business_cache_v1 import BusinessCache

            self._businesses.append(BusinessCache())

            # Wav2Lip
            from room_business.business_wav2lip_v1 import BusinessWav2Lip

            self._businesses.append(BusinessWav2Lip())

            from room_business.business_vad import BusinessVad
            self._businesses.append(BusinessVad())

            from room_business.business_nlp_v1 import BusinessNLP
            nlp = BusinessNLP()
            self._businesses.append(nlp)
            #用来处理打断，需要停止nlp
            self.runtime_data.nlp_interrupt_handler = nlp

            logger.debug(f"{self.init_data.live_id} 任务队列就绪，准备启动")
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

    async def wait_room_ready(self, infer_ready_event,timeout=100):
        try:
            await asyncio.wait_for(infer_ready_event.wait(), timeout=timeout)
            return self.wav2lip_process_state == RoomState.ready
        except asyncio.TimeoutError:
            logger.error(f"Timeout! {self.init_data.live_id} can not get infer proc state.")
            return False
        

    async def prepare_model_people(self,model_speaker_id,person_config,bg_url):
        await download(model_speaker_id, person_config)
        await change_background(model_speaker_id,bg_url)  

    def create_wav2lip_process(self,infer_ready_event):
        # 延时，先让flask返回

        if self.wav2lip_process is None:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self.prepare_model_people(self.init_data.pkl_name,self.init_data.pkl_config,self.init_data.background))
            # asyncio.run_coroutine_threadsafe()
            
            self.wav2lip_stop_event = multiprocessing.Event()
            self.wav2lip_request_queue = multiprocessing.Queue()
            self.wav2lip_result_queue = multiprocessing.Queue()

            self.wav2lip_process = multiprocessing.Process(
                target=process_wav2lip_predict,
                args=(
                    self.init_data.live_id,
                    self.wav2lip_request_queue,
                    self.wav2lip_stop_event,
                    self.wav2lip_result_queue,
                ),
            )
            self.wav2lip_process.start()
            
            def warm_up_infer_process():
                # 遍历字典并使用 enumerate 获取当前索引
                total_warmup_model_num = len(self.init_data.video_model_info_map)
                logger.info(f" will send warm up request for  {self.init_data.live_id} {total_warmup_model_num}")
                for index, (actor, modelinfo) in enumerate(
                    self.init_data.video_model_info_map.items()
                ):
                    params = [
                        InferAction.WARMUP,
                        actor,
                        modelinfo.modeling_url,
                        index,
                        total_warmup_model_num,
                    ]
                    self.wav2lip_request_queue.put(params)
                result = "nok"
                try:
                    result = self.wav2lip_result_queue.get(
                        timeout=5 * total_warmup_model_num
                    )
                    
                    logger.info(
                        f"{self.init_data.live_id} infer process warm up got result {result}"
                    )
                except Exception:
                    logger.error(
                        f"{self.init_data.live_id} infer process failed warm up  {traceback.format_exc()}"
                    )
                    pass
                finally:
                    if result == WarmupEvent.WARMUP_DONE:
                        self.wav2lip_process_state = RoomState.ready
                    else:
                        self.wav2lip_process_state = RoomState.dead
                        self.wav2lip_process.terminate()

                    infer_ready_event.set()
                if result == WarmupEvent.WARMUP_DONE:
                    return True
                return False

            if warm_up_infer_process():
                self.wav2lip_process_state = RoomState.ready
                self.wav2lip_process_started = True
            else:
                self.wav2lip_process_state = RoomState.dead
            logger.info(
                f" room has finished to start {self.init_data.live_id} and have result  {self.wav2lip_process_state}"
            )

    def is_read_fast(self):
        # 共享内存实际读取的内存索引与写入的内存索引，如果差距太小就继续生成wav2lip
        read_i = self.read_index_share_memory.read()  # 视频帧数转换为音频实际读取地址
        write_i = self.write_index_share_memory.read()
        is_read_fast = write_i <= read_i + self.share_memory_cache_threshold

        if is_read_fast:
            logger.debug(
                f"{self.init_data.live_id} business_wav2lip 生成速度不够读取速度  读：{read_i} 写：{write_i} read gap {read_i-write_i}"
            )
        # logger.debug(f'性能监控：生成速度，读取速度  读：{read_i} 写：{write_i} {read_threshold}')
        return is_read_fast

    def is_write_fast(self):
        read_i = self.read_index_share_memory.read()  # 视频帧数转换为音频实际读取地址
        write_i = self.write_index_share_memory.read()
        is_write_fast = (
            write_i - read_i > self.share_memory_cache_threshold
        )  # TODO写太快也不行，但是这个会导致send_to_unity的话术太快(有很大概率时读指针初始化仍然不正确)
        if is_write_fast:
            logger.debug(
                f"{self.init_data.live_id} business_wav2lip 生成速度很快 {is_write_fast}，可以做点别的  读：{read_i} 写：{write_i} {self.share_memory_cache_threshold} write gap {write_i - read_i } "
            )
        # logger.debug(f'生成速度很快 {is_write_fast}，可以做点别的  读：{read_i} 写：{write_i} {self.share_memory_cache_threshold} gap {write_i - read_i } ')
        return is_write_fast

    # 开始直播
    def start_tasks_in_main_thread(self):
        self.loop = asyncio.get_event_loop()  # 获取主线程的事件循环
        # await self.thread_function()  
        asyncio.create_task(self.thread_function())# 调度任务，不阻塞主线程

    async def start_live(self, init_data: RoomInitData,room_monitor):

        if self.is_running:
            return
        mel_block_size = (config.wav2lip_batch_size, 80, 16)
        audio_block_size = (np.prod(mel_block_size).tolist(), 1)
        self.init_data = init_data
        self.runtime_data.room_monitor = room_monitor
        self.runtime_data.init_ai_person(self.init_data.pkl_name,self.init_data.background,self.init_data.ai_person,self.init_data.pkl_config)
        self.data_cache = Infer_Cache(
            "infer_cache",
            init_data.live_id,
            create=True,
            mel_block_shape=mel_block_size,
            audio_block_shape=audio_block_size,
        )
        #TODO: 获取channel
        self.video_cache = Video_Cache(
            "video_cache",
            "video_audio_cache",
            init_data.live_id,
            "speech_id_cache",
            resolution=config.video_resolution,
            audio_block_size=int(audio_block_size[0]/5),
            create=True,
            channel=3,
        )
        # 指针的初始化顺序都提前到推理进程启动之前
        self.read_index_share_memory = ShareMemoryInt(
            f"bota_video_read_frame_{self.init_data.live_id}"
        )
        self.read_index_share_memory.write(0)
        logger.info(f"read_index:{self.read_index_share_memory.read()}")

        # read_index_share_memory.write(0)#读指针在渲染端控制，理论上这边忽略重置读写指针,这边重置可能导致问题，正常情况来说，这边一定先启动。
        self.write_index_share_memory = ShareMemoryInt(
            f"full_write_video_frame_{self.init_data.live_id}"
        )
        self.write_index_share_memory.write(0)  # 重置写指针
        logger.info(f"write_index:{self.write_index_share_memory.read()}")

        infer_ready_event = asyncio.Event() 

        wav2lip_start_thread = threading.Thread(target=self.create_wav2lip_process,args=(infer_ready_event,))
        wav2lip_start_thread.start()

        error_msg = ""
        self.init_data = init_data

        if self._thread is not None: # TODO要改
            error_msg = (
                "self._thread exist 没有调用停止直播或这重复开始直播，或者线程还没死"
            )
            return error_msg
        
        self.is_running = await self.wait_room_ready(infer_ready_event)
        
        self.start_tasks_in_main_thread()

        logger.info(f"当前room {self.init_data.live_id} running state is {self.is_running} {self.wav2lip_process_state}")

        return error_msg
    async def asrcallback(self, websocket):
        try:
            logger.debug(
                f"asrcallback task start :{self.init_data.live_id}"
            )
            while (
                # not self.room_render.runtime_data.asr_callback_queue.empty()
                # and 
                self.is_valid()
            ):
                logger.debug(f"{self.init_data.live_id}  asrcallback_run_time_cal begin")
                asr_msg = await self.runtime_data.asr_callback_queue.get()
                logger.debug(f"{self.init_data.live_id} recv_asr_message {asr_msg}.")
                if asr_msg is not None:
                    await websocket.send(json.dumps(asr_msg))
                else:
                    logger.info(f"{self.init_data.live_id} none asr message,must be someone ask me quit.")
                    break
                logger.debug(f"{self.init_data.live_id}  asrcallback_run_time_cal end")
            self.asr_call_back_task = None
            logger.info(f"asr task is done {self.init_data.live_id}")
            logger.info(f"stopping_room {self.init_data.live_id}  exit business_asrcallback logic")
        except Exception:
            logger.error(f"asr callback error:{traceback.format_exc()}")
        except BaseException as e :
            logger.error(f' receved error {e} {traceback.format_exc()}')

    async def nlpcallback(self, websocket):
        """Return nlp result for js"""
        try:
            logger.debug(
                f"business_nlpjs start:{self.init_data.live_id}"
            )
            while (
                # not self.room_render.runtime_data.nlp_callback_queue.empty()
                # and 
                self.is_valid()
            ):
                logger.debug(f"{self.init_data.live_id}  nlpcallback_run_time_cal begin")
                nlp_msg = await self.runtime_data.nlp_callback_queue.get()
                logger.debug(f"{self.init_data.live_id} recv_nlp_answer:{nlp_msg}")
                if nlp_msg is not None:
                    # await websocket.send(f"nlp_{nlp_msg}")
                    await websocket.send(json.dumps(nlp_msg))
                else:
                    logger.info(f"{self.init_data.live_id} none nlp message,must be someone ask me quit.")
                    break
                logger.debug(f"{self.init_data.live_id}  nlpcallback_run_time_cal end")
            logger.info(f"{self.init_data.live_id} business_nlpcallback task is done ")
            logger.info(f"stopping_room {self.init_data.live_id}  exit business_nlpcallback logic")
            self.nlp_call_back_task = None
        except Exception:
            logger.error(f"nlpmsg error:{traceback.format_exc()}")
        except BaseException as e :
            logger.error(f' receved error {e} {traceback.format_exc()}')
            
    def create_asr_call_back_task(self,asr_call_back_websock):
        if not self.asr_call_back_task:
            self.asr_call_back_task = self.loop.create_task(self.asrcallback(asr_call_back_websock))
            self.tasks.append(self.asr_call_back_task)

    def create_nlp_call_back_task(self,websocket):
        logger.info("create_nlp_call_back_task")
        if not self.nlp_call_back_task:
            logger.info("create_nlp_call_back_task append task begin.")
            self.nlp_call_back_task = self.loop.create_task(self.nlpcallback(websocket))
            # self.nlp_call_back_task = self.loop.create_task(self.nlpcallback(websocket))
            logger.info("create_nlp_call_back_task append task end.")
            self.tasks.append(self.nlp_call_back_task)  

    @DeprecationWarning
    def start_thread(self):
        '''
        异步模式下不再需要这个函数
        '''
        if self._thread is not None:
            self._thread.start()

    # 结束直播
    def close_async_queue(self,queuue : asyncio.Queue):
        try:
            if queuue:
                queuue.put_nowait(None)
                queuue = None
        except:
            pass
    async def stop_live(self):
        try:
            self.is_running = False
            self.wav2lip_process_started = False

            """
            停止顺序：
            1.先停止主进程的任务
            2.再停止推理进程
            3.最后清除各个队列
            """
            if self._thread is not None:
                self._thread.join(timeout=20)
                # self.reset()
            logger.info(f"stopping_room {self.init_data.live_id} is stopping,firstly clear callback queue")
            self.close_async_queue(self.runtime_data.remote_audio_queue)
            self.close_async_queue(self.runtime_data.queue_vad)
            self.close_async_queue(self.runtime_data.asr_audio_queue)
            self.close_async_queue(self.runtime_data.asr_callback_queue)
            self.close_async_queue(self.runtime_data.nlp_callback_queue)
            self.close_async_queue(self.runtime_data.asr_msg_queue)
            self.close_async_queue(self.runtime_data.queue_25d_audio)
            self.close_async_queue(self.runtime_data.send_command_queue)
            self.close_async_queue(self.runtime_data.queue_interaction)
            # self.close_async_queue(self.runtime_data.remote_audio_queue)
            self.close_async_queue(self.wav2lip_result_queue)
            self.close_async_queue(self.runtime_data.audio_queue)
            self.close_async_queue(self.runtime_data.video_queue)
            logger.info(f"room_id:{self.init_data.live_id} 清除消息队列")
            await asyncio.sleep(0.5)#让出CPU以便于其他task停止。
            logger.info(f"stopping_room {self.init_data.live_id} is stopping,clear tasks")

            # 停止推理进程
            if self.wav2lip_process is not None:
                self.wav2lip_stop_event.set()
                
                # 通知推理进程停止
                logger.info(f"stopping_room {self.init_data.live_id} wav2lip_stop_event 已经发送")
                self.wav2lip_request_queue.put(None, block=False)
                logger.info(f"stopping_room {self.init_data.live_id} wav2lip_request_queue 队列发送空，通知对方退出")
                if self.wav2lip_process: #可能提前为空了。
                    # logger.info(f"stopping_room {self.init_data.live_id} join")
                    # self.wav2lip_process.join()
                    logger.info(f"stopping_room {self.init_data.live_id} terminate")
                    self.wav2lip_process.terminate()
                    logger.info(f"stopping_room {self.init_data.live_id} wav2lip_process设为none")
                    self.wav2lip_process = None
                await asyncio.sleep(1)
                self.wav2lip_result_queue = None
                self.wav2lip_request_queue = None
            self.reset()


            # 清理共享内存---
            if self.data_cache is not None:
                logger.info(f"stopping_room 清除推理cache")
                self.data_cache.destroy()
                self.data_cache = None
            if self.video_cache is not None:
                logger.info(f"stopping_room 清除 video_cache")
                self.video_cache.destory()
                self.video_cache = None
            if self.read_index_share_memory is not None:
                logger.info(f"stopping_room 准备清除读索引{self.read_index_share_memory.buffer_name}")
                self.read_index_share_memory.destroy()
                self.read_index_share_memory = None
            if self.write_index_share_memory is not None:
                logger.info(
                    f"stopping_room 准备清除写索引{self.write_index_share_memory.buffer_name}"
                )
                self.write_index_share_memory.destroy()
                self.write_index_share_memory = None
            logger.info(f"room_id:{self.init_data.live_id} 准备清除任务队列")
            await asyncio.sleep(0.5) #让出CPU以便于其他task正常停止
            if self.loop is not None:
                for task in self.tasks:
                    task.cancel()
            logger.info(f"room_id:{self.init_data.live_id} 停播结束")
        except Exception:
            logger.error(
                f"stop live of {self.init_data.live_id} error {traceback.format_exc()}"
            )
            return Status.ERROR
        return Status.SUCCESS
