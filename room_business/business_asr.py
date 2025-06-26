import asyncio
import os
import shutil
import time
import traceback
import wave

import aiohttp

from config import config
from log import logger
from room_business.business_base import BusinessBase
from utils.tool import extract_audio_data, match_target_amplitude

tick_time = 0.005


class BusinessASR(BusinessBase):
    def __init__(self):
        super().__init__()


    async def run(self):
        buffer_audio = b""
        frames_asr_online = []
        is_speak_final = True
        speech_start_i = -1
        online_check_num= 0
        last_asr_end = False
        frame_buffer = b''
        frame_len = 1920 #目前就是这个样子的
        room_id = self.parent.init_data.live_id
        while self.parent.is_valid():
            # 获取
            logger.debug(f"{room_id} asr start run_time_cal begin")
            # while not self.parent.runtime_data.nlp_answer_queue.empty():
            audio_frame = await self.parent.runtime_data.remote_audio_queue.get()
            if not audio_frame:
                continue
            logger.debug(f"{self.parent.init_data.live_id} asr recved voice")
            frame_buffer = frame_buffer + audio_frame
            while len(frame_buffer) >= frame_len:
            
                frames_asr_online.append(frame_buffer[:frame_len])
                frame_buffer = frame_buffer[frame_len:]
            #-----------------------------TODO 需要重构，这段代码是复制并兼容websocket的

                #下面的代码应该与原来的代码完全一致
                if (
                    len(frames_asr_online) % config.VAD_CHUNK_INTERVAL != 0
                    and is_speak_final
                ):
                    continue

                logger.info(
                    f"{room_id} ASR 缓存音频数量： {len(frames_asr_online)}, id:{self.parent.runtime_data.asr.message_id} online_check_num：{online_check_num} self.parent.runtime_data.asr.online_asr: {self.parent.runtime_data.asr.online_asr}, asrchunks:{len(self.parent.runtime_data.asr.asr_chunks)},last_asr_end:{last_asr_end},speech_start_i:{speech_start_i}"
                )
                await self.parent.runtime_data.info_sender.send("asr_running")
                audio_in = b"".join(frames_asr_online)

                # 检测到人声后，关闭vad，一直向后请求asr,直到asr那识别不到结果。继续开启vad
                if self.parent.runtime_data.asr.vad_flag:
                    speech_start_i, is_speak_final = await self.parent.runtime_data.asr.vad_detect(
                        self.parent.init_data.live_id, audio_in, is_speak_final
                    )
                if speech_start_i == -1:
                    frames_asr_online = []
                    # await self.parent.runtime_data.info_sender.send("asr_end")
                    continue
                # 收到打断，切断上一次asr链，重新收音
                # 如果是持续收音状态, 如果上次没有识别到人声，也不应该打断。不应该进行打断，打断只有vad认为是打断,此时打断nlp之后的所有结果
                # TODO: @txueduo 考虑下一次是静默的情况，再讲话, nlp_status_map 需要删除
                if last_asr_end:
                    logger.info(
                        f"{room_id} 人声打断:{self.parent.runtime_data.asr.message_id}, nlp_status_map:{self.parent.runtime_data.nlp_status_map}"
                    )
    
                    await self.parent.runtime_data.clear_current_session(
                        self.parent.runtime_data.asr.message_id,
                        self.parent.runtime_data.asr.last_message_id,
                    )
                    last_asr_end = False
                    self.parent.runtime_data.asr.last_message_id = self.parent.runtime_data.asr.message_id
                    # 清理后，继续收这次的音
                if (
                    self.parent.runtime_data.asr.speak_end # or online_check_num > config.asr_gap_size
                ):  # 33 为 int(32000/960）
                    logger.info(
                        f"{room_id} {self.parent.runtime_data.asr.message_id} offline audio length:{len(self.parent.runtime_data.asr.asr_chunks)} online_check_num：{online_check_num}"
                    )
                    # 声音结束，asr检测
                    self.parent.runtime_data.asr.sentence_asr = True
                    self.parent.runtime_data.asr.online_asr = False
                    # TODO: @txueduo 落盘测试下效果, 后期删除
                    # save_raw_wav(self.parent.runtime_data.asr.message_id, self.parent.runtime_data.asr.asr_chunks)
                    
                    if config.ASR_MODE != "aliyun":
                        self.parent.runtime_data.asr.close_microsoft_asr()
                    else:
                        await self.parent.runtime_data.asr.close_asr()

                    # flush status
                    is_speak_final = True
                    if self.parent.runtime_data.asr.speak_end:
                        last_asr_end = True
                    speech_start_i = -1
                    online_check_num = 0
                    frames_asr_online = []
                    # self.parent.runtime_data.asr.flush_status()

                    # await self.parent.runtime_data.info_sender.send("asr_end")#通过websocket把文本发出来。
                    logger.info(f"{room_id} {self.parent.runtime_data.asr.message_id} offline_asr end sent ")

                elif self.parent.runtime_data.asr.online_asr:
                    logger.info(
                        f"{room_id} {self.parent.runtime_data.asr.message_id},online_check_num:{online_check_num}, online audio length:{len(audio_in)}, buffer_audio:{len(buffer_audio)}"
                    )
                    # 为了避免长文本检测失败。增大bytes长度, 一个时，audio_in 1920

                    if len(audio_in) > 1920:
                        if config.ASR_MODE != "aliyun":
                            # 音量增强
                            self.parent.runtime_data.asr.stream.write(match_target_amplitude(audio_in))
                            await self.parent.runtime_data.asr.async_microsoft_asr(audio_in)
                        else:
                            await self.parent.runtime_data.asr.async_asr(match_target_amplitude(audio_in))
                        online_check_num += 1
                    elif len(audio_in) == 1920:
                        buffer_audio += audio_in
                    if len(buffer_audio) == 1920*config.ASR_BUFFER_AUDIO:
                        if config.ASR_MODE != "aliyun":
                            self.parent.runtime_data.asr.stream.write(match_target_amplitude(buffer_audio))
                            await self.parent.runtime_data.asr.async_microsoft_asr(buffer_audio)
                        else:
                            await self.parent.runtime_data.asr.async_asr(match_target_amplitude(buffer_audio))
                        buffer_audio = b"" 
                        online_check_num += 1

                frames_asr_online = []
            logger.debug(f"{room_id} {self.parent.init_data.live_id} asr end run_time_cal end")
            await asyncio.sleep(tick_time)
        await self.parent.runtime_data.asr.close_asr()
        logger.info(f"stopping_room {self.parent.init_data.live_id}  exit asr logic")
