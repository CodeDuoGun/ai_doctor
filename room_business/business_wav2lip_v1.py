import asyncio
import time
import traceback


import wav2lip.utils.performance_tools as performance_tools
from log import logger
from room_business.business_base import BusinessBase
from room_business.room_data import *

tick_time = 0.01


class BusinessWav2Lip(BusinessBase):
    def __init__(self):
        super().__init__()
        self.can_run = False
        # 当前场景id
        self.current_scene_id = ""
        # 当前视频素材帧
        self.current_video_section_frame = 1
        # 当前视频素材播放方向：True为正，False为反
        self.current_video_play_direction = True
        self.temp_result = []
        self.speech_id = ""
        self.node_id = ""

    async def wait_wav2lip_result(self, params):

        time_diff = performance_tools.TimeDiff()
        start_cnt = time.perf_counter()
        output_resutl = None
        try:
            
            await self.parent.loop.run_in_executor(None, self.parent.wav2lip_request_queue.put, params)
            output_resutl = await self.parent.loop.run_in_executor(None, self.parent.wav2lip_result_queue.get)

        except Exception:
            logger.error(f"wait_wav2lip_result:{traceback.format_exc()}")
        i_cost = time.perf_counter() - start_cnt
        flag = "good"
        if i_cost > 0.17:
            flag = "bad"
        audio_type = "静默" if not str(params[2]).find("silent_") >=0 else "说话"
        # audio_type = "静默"
        # if not str(params[2]).find("silent_") >=0:
        # # if not params[2].startswith("silent_"):
        #     audio_type = "说话"
        logger.info(
            f'{self.parent.init_data.live_id}  wav2lp_time :{audio_type}_{flag}_{params[1]} {time_diff.diff_last("从推送到接收数据耗时")}   {time.perf_counter() - start_cnt}'
        )
        return output_resutl

    async def gen_video(self, speech: Speech,cur_node: Node):
        # logger.info(
        #     f"{self.parent.init_data.live_id}  [{speech.speech_id}] gen_video ing"
        # )
        time_diff = performance_tools.TimeDiff()
        speech.is_wav2lip = True
        try:
            duration = 0
            #下面是判断说话的剩余时长。
            last_node = self.parent.runtime_data.speech_linked_list.get_head()
            while last_node is not None:
                if last_node.data.priority > 0:
                    # priority_node = last_node
                    duration += last_node.data.duration
                else:
                    break
                last_node = last_node.tail
            # TODO: 第一个参数没用，去掉            
            # 记录下是首次开始讲话,  如果接着两句都是在讲话，认为一直在讲话,first_speak=False
            # 只要上句在讲话，就认为继续讲话
            # TODO，这个逻辑可能有bug需要改一下。
            if self.parent.last_batch_speak:
                first_speak = False
            else:
                first_speak = False if not speech.first_speak else True
            
            # 如果下句是静默。一定是讲话结束
            if cur_node.tail:
                logger.debug(f"cur_node.tail.data.speech_id：{cur_node.tail.data.speech_id}")
            if "silent" not in cur_node.data.speech_id and cur_node.tail and "silent" in cur_node.tail.data.speech_id:
                last_speak = True
            else:
                last_speak = False
                
            if self.speech_id != cur_node.data.speech_id:
                if "silent" not in self.speech_id and "silent" in cur_node.data.speech_id:
                    last_speak = True
                self.speech_id = cur_node.data.speech_id
            # TODO: 第二个参数没用，去掉
            params = [
                int(speech.priority == 0),
                speech.speech_id,
                speech.video_model_info.pkl_url,
                self.current_video_section_frame,
                self.current_video_play_direction,
                duration,
                False,
                first_speak,
                last_speak,
            ]

            # ['infer', 1, 'room_f1297e8e-32cd-4b70-94f6-66e8936e8f55_silent_0_0', '2iKrw1uZ01eQBygN', 1, True, 0, False]

            if last_speak and self.parent.runtime_data.send_command:
                await self.parent.runtime_data.send_command_queue.put("play")
            if self.parent.data_cache:
                self.parent.data_cache.put_data(
                    speech.push_unit_data[0], speech.push_unit_data[1], speech.speech_id
                )
                diff = time.time()
                result = await self.wait_wav2lip_result(params)
                if result is None or not self.parent.is_valid():
                    logger.warning(
                        f"{self.parent.init_data.live_id}  wait_wav2lip_result is {result}"
                    )
                    return
                # logger.info(
                #     f"{self.parent.init_data.live_id}  wait_wav2lip_result:{time.time() - diff} "
                # )
                self.current_video_section_frame = result[1]
                self.current_video_play_direction = result[2]
                logger.debug(
                f"{self.parent.init_data.live_id} {speech.speech_id} ready to put into webrtc queue  {self.parent.runtime_data.audio_queue.qsize()}  {self.parent.runtime_data.video_queue.qsize()}"
                    )
                for i in range(5):
                    audio_speech_index,audio_cache = self.parent.video_cache.get_audio()
                    half = len(audio_cache) // 2
                    try:
                        #TODO 可能在这里阻塞,这样会不会导致资源消耗更低？这样基本上就可以做到由推流侧控制速率了,快慢指针似乎不需要了。
                        #TODO 改为put_nowait 如果队列满了，捕获QueueFull异常？？？
                        # asyncio.run_coroutine_threadsafe(
                        #     self.parent.runtime_data.audio_queue.put(audio_cache[:half]),
                        #     self.parent.loop,
                        # )
                        # asyncio.run_coroutine_threadsafe(
                        #     self.parent.runtime_data.audio_queue.put(audio_cache[half:]),
                        #     self.parent.loop,
                        # )
                        # asyncio.run_coroutine_threadsafe(
                        #     self.parent.runtime_data.video_queue.put(
                        #         self.parent.video_cache.get()
                        #     ),
                        #     self.parent.loop,
                        # )
                        video_index ,video_frame = self.parent.video_cache.get()
                        self.parent.runtime_data.audio_queue.put_nowait((audio_speech_index,audio_cache[:half]))
                        self.parent.runtime_data.audio_queue.put_nowait((audio_speech_index,audio_cache[half:]))
                        self.parent.runtime_data.video_queue.put_nowait((video_index,video_frame))
                        if speech.speech_id.startswith("lrid_"):
                            self.parent.runtime_data.room_monitor.speaking(0.04)
                        
                        logger.debug(
                            f"{self.parent.init_data.live_id} {speech.speech_id} rtc video_index {video_index} audio_index {audio_speech_index} :{self.parent.runtime_data.audio_queue.qsize()}"
                        )
                        # logger.debug(
                        #     f"{self.parent.init_data.live_id} {speech.speech_id} rtc  video size:{self.parent.runtime_data.video_queue.qsize()}"
                        # )
                    except :
                        logger.warning(
                            f"{self.parent.init_data.live_id} {speech.speech_id} [{audio_speech_index}] failed to put into web rtc"
                        )
                        # self.full_write_frame_buf.write(self.full_write_frame_buf.read() + 10)
                self.parent.write_index_share_memory.write(self.parent.write_index_share_memory.read() +10)
                def build_client_tts_txt(speech):
                    data = {

                            "room_id":self.parent.init_data.live_id,
                            "msg_id":speech.msg_id,
                            "sentence_id":speech.sentence_id,
                            "msg_type":"tts",
                            "text":speech.tts_text,
                            "image_url": speech.image_url,
                            "video_url": speech.send_video_url,
                            "status": speech.status
                    }
                    return data
                    
                logger.debug(
                    f"{self.parent.init_data.live_id} {speech.speech_id} 数据已经给到webrtc {speech.tts_text} read_idx: {self.parent.read_index_share_memory.read()} write_idx: {self.parent.write_index_share_memory.read()} video_size {self.parent.runtime_data.video_queue.qsize()} audio_dize {self.parent.runtime_data.audio_queue.qsize()}")
                
                if self.node_id and last_speak:
                    await self.parent.runtime_data.bota_server_callback.put(self.node_id)

                if self.node_id and self.node_id != speech.node_id:
                    await self.parent.runtime_data.bota_server_callback.put(self.node_id)
                self.node_id = speech.node_id
                if speech.first_tts and speech.tts_text and len(speech.tts_text.strip()) > 0 :
                    await self.parent.runtime_data.nlp_callback_queue.put(build_client_tts_txt(speech))
                    logger.debug(
                    f"{self.parent.init_data.live_id} {speech.speech_id} {speech.msg_id} 把文本已经通知到客户端 [{speech.tts_text}] into remote queue"
                    )            
                #下面的消息已经不需要所以注释掉
                # if speech.last_tts:#
                #     await self.parent.runtime_data.nlp_callback_queue.put(build_client_tts_txt("tts_end"))
                    # logger.debug(
                    # f"{self.parent.init_data.live_id} {speech.speech_id} 最后的一段文本发送文本结束标记 {speech.tts_text} into remote queue"
                    # )
                
                # await api.send_history(speech.speech_id, payload)
                #TODO 似乎不需要了。
                if self.parent.runtime_data.first_push_time == 0:
                    self.parent.runtime_data.first_push_time = time.time()
                    # logger.debug(
                    #     f"{self.parent.init_data.live_id} [{speech.speech_id}] gen_video succeed:{len(speech.push_unit_data[1])} {time_diff.diff_last('从推送到接收数据耗时')}"
                    # )
            logger.debug(
                    f"{self.parent.init_data.live_id} {speech.speech_id} {speech.tts_text} read_idx: {self.parent.read_index_share_memory.read()} read_idx: {self.parent.write_index_share_memory.read()} video_size {self.parent.runtime_data.video_queue.qsize()} audio_dize {self.parent.runtime_data.audio_queue.qsize()}")
        except Exception:
            logger.error(
                f"[{speech.speech_id}] gen_video error:{traceback.format_exc()}"
            )
        finally :
            #本段音频即使失败也不再重试
            speech.video_url = "success"
            speech.is_wav2lip = False
            speech.had_pushed = True #放push标记

    async def first_run(self):
        while self.parent.is_valid():
            if self.parent.runtime_data.speech_linked_list.head is not None:
                return True
            await asyncio.sleep(tick_time)
        return False

    async def run(self):
        try:
            # if not await self.first_run():
            #     logger.warning("exit wav2lip logic at begin")
            #     return
            logger.info("start wav2lip looping")

            while self.parent.is_valid():
                logger.debug(f"{self.parent.init_data.live_id}  run_time_cal begin")
                begin = time.time()

                
                last_node = None
                is_write_fast = self.parent.is_write_fast()
                is_read_fast  = self.parent.is_read_fast()
                if (
                        not is_write_fast
                        or is_read_fast
                    ):
                    last_node = await self.parent.runtime_data.speech_linked_list.remove_head()
                    rest_in_list = self.parent.runtime_data.speech_linked_list.count()
                    speech_id = "no_node_in_cache_yet"
                    if last_node:
                        speech_id = last_node.data.speech_id
                    logger.debug(f"{self.parent.init_data.live_id} ，生成速度不太快准备推理 {speech_id} 剩余{rest_in_list}")
                else:
                    logger.debug(f"{self.parent.init_data.live_id} ，生成速度快 {is_write_fast} {is_read_fast} 剩余{rest_in_list}")
                    if rest_in_list == 0 :
                        print(f'不应该发生！！！！！！！')
                while last_node is not None:
                    if not self.parent.is_valid():
                        logger.warning('exit wav2lip logic')
                        return

                    if not self.parent.wav2lip_process_started:
                        logger.warning("wav2lip_process_started is ongoing.")
                        break

                    # 如果共享内存实际读取与写入的缓冲不足
                    # 或者 liveroom计算的剩余缓冲不足
                    # 或者 开播缓冲不足, 这个逻辑目前去掉了
                    # 会重新准备音频数据
                    if (
                        not is_write_fast
                        or is_read_fast
                        #这个判断大概率不需要了。

                    ):
                        # 如果当前节点未推送 并且 未生成wav2lip 并且 已准备好音频数据
                        if (
                            (not last_node.data.had_pushed)
                            and last_node.data.video_url == ''
                            and last_node.data.audio_url != ''
                        ):
                            await self.gen_video(last_node.data, last_node)
                            #TODO 这一行还需要么？
                            # self.parent.runtime_data.current_priority = (
                            #     self.parent.runtime_data.current_node.data.priority
                            # )
                            break
                        logger.debug(
                            f"{self.parent.init_data.live_id} try_next_node {last_node.data.speech_id} {last_node.data.had_pushed} {last_node.data.video_url} {last_node.data.audio_url} "
                        )
                        #一次一个节点
                        # last_node = last_node.tail
                        
                    else:
                        if last_node:
                            logger.debug(
                            f"{self.parent.init_data.live_id} wait_no_need_gen_video {last_node.data.speech_id} {last_node.data.had_pushed} {last_node.data.video_url} {last_node.data.audio_url} "
                            )
                        else:
                            logger.debug(
                            f"{self.parent.init_data.live_id} wait_no_need_gen_video  "
                            )
                        break
                logger.debug(
                    f"{self.parent.init_data.live_id} run_time_cal end:{time.time() - begin}"
                )
                await asyncio.sleep(tick_time)

            logger.info(f"stopping_room {self.parent.init_data.live_id}  exit wav2lip logic")
        except Exception:
            logger.error(f"wav2lip err:{traceback.format_exc()}")