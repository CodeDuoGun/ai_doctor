import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, unique
import io
import math
import os
import shutil
import subprocess
import time
import traceback
from typing import Literal
import wave

import aiohttp
import numpy as np

from config import config

from log import logger

from room_business.business_base import BusinessBase
from room_business.room_data import  VADEvent, VADEventType
from room_business.utils.audio_frame import AudioFrame
from room_business.utils.exp_filter import ExpFilter
from room_business.utils.utils import combine_audio_frames, gracefully_cancel
from room_business.utils import onnx_model
from utils.tool import apply_gain, extract_audio_data, match_target_amplitude

tick_time = 0.005
SLOW_INFERENCE_THRESHOLD = 0.2  # late by 200ms
class AudioSaver:
    def __init__(self, filename="output.wav", max_size=1_073_741_824):
        """
        初始化音频保存器，设置最大文件大小为 1GB（默认）
        
        :param filename: 保存的音频文件名
        :param max_size: 文件最大大小，单位字节，默认为 1GB
        """
        self.filename = filename
        self.max_size = max_size
        self.wav_file = None
        if config.enable_vad_audio_save:
            self._init_wav_file()

    def _init_wav_file(self):
        """初始化WAV文件，设置文件头"""
        self.wav_file = wave.open(self.filename, 'wb')
        self.wav_file.setnchannels(1)  # 单声道
        self.wav_file.setsampwidth(2)  # 16-bit 每帧
        self.wav_file.setframerate(16000)  # 16kHz 采样率

    def write_audio_chunk(self, audio_data):
        """
        向 WAV 文件写入音频数据并检查文件大小
        
        :param audio_data: 要写入的音频数据，应该是 bytes 类型
        """
        if config.enable_vad_audio_save:
            if self._get_file_size() < self.max_size:
                self.wav_file.writeframes(audio_data)
            else:
                print(f"文件大小已达到限制 ({self.max_size / 1e9}GB)，停止保存音频")
                self.close()  # 达到大小限制后关闭文件

    def _get_file_size(self):
        """获取当前文件大小"""
        return os.path.getsize(self.filename)

    def close(self):
        """关闭文件"""
        if config.enable_vad_audio_save:
            if self.wav_file:
                self.wav_file.close()
            # print(f"音频已保存到 {self.filename}")
        
class FixedLengthBuffer:
    def __init__(self,maxlength = 64000):
        self.buffer = bytearray()
        self.maxlength = maxlength

    def append(self, data):
        self.buffer.extend(data)
        if len(self.buffer) > self.maxlength:
            self.buffer = self.buffer[-self.maxlength:]

    def to_bytes(self, num):
        if num > len(self.buffer):
            raise ValueError("指定的数量超过了缓冲区中的字节数量")
        result = self.buffer[-num:]
        self.buffer.clear()
        return result
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()
@dataclass
class _VADOptions:
    min_speech_duration: float
    min_silence_duration: float
    prefix_padding_duration: float
    max_buffered_speech: float
    activation_threshold: float
    sample_rate: int
@unique
class AgentState(str, Enum):
    Agent_VAD = "lookat" # VAD已经检测到人声
    Agent_Listening = "listening" # 确定是有人说话，已经打断当前说话
    Agent_Normal = "waiting"    #没有人声
    
class BusinessVad(BusinessBase):
    def __init__(self,min_speech_duration: float = 0.05,
        min_silence_duration: float = 0.25,
        prefix_padding_duration: float = 0.1,
        max_buffered_speech: float = 60.0,
        activation_threshold: float = 0.5,
        sample_rate: Literal[8000, 16000] = 16000,
        force_cpu: bool = True,):
        super().__init__()
        self._opts = _VADOptions(
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
            prefix_padding_duration=prefix_padding_duration,
            max_buffered_speech=max_buffered_speech,
            activation_threshold=activation_threshold,
            sample_rate=sample_rate,
        )
        self._onnx_session = onnx_model.new_inference_session(force_cpu)
        self._model =  onnx_model.OnnxModel(
                onnx_session=self._onnx_session, sample_rate=self._opts.sample_rate
            )
        self._loop = asyncio.get_event_loop()

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._exp_filter = ExpFilter(alpha=0.35)

        self._extra_inference_time = 0.0
        self._recognize_atask = None
        self.pub_speaking = False
        self.room_state = AgentState.Agent_Normal
        self.interrupt_speech_duration: float = 0.5

        self.MIN_VAD_SENCOND = int(32000 * config.MIN_VAD_SENCOND)


    def _recognize(self):
        logger.debug(f' will running _recognize ')
        self._recognize_atask = asyncio.create_task(
            self._recognize_human_speech(old_task = self._recognize_atask)
        )

    async def _recognize_human_speech(self,old_task:asyncio.Task):

        if old_task is not None:
            await gracefully_cancel(old_task)
        running = True
        if config.USING_ALIASR:
            from room_business.asr.ali_asr import AliASR
            asr = AliASR(name = "aliasr",room_instance = self.parent)
        else:
            from room_business.asr.azure_asr import AzureASR 
            #TODO 这个还有bug ！！！！！！！
            asr = AzureASR(name = "azureasr",room_instance = self.parent)
        logger.debug(f' will running _recognize with  asr.')

        # 0.5ms的音频 且能识别两个字
        to_be_sent_buffer = FixedLengthBuffer()
        try:
            while self.parent.is_valid():
                evt = await self.parent.runtime_data.queue_vad.get()
                if evt is None:
                    break
                # input_frames = utils.combine_frames(evt.frames)
                # to_be_sent_buffer = to_be_sent_buffer + input_frames
                logger.debug(f" 收到VAD结果：room_state {self.room_state} 当前文本[{asr.current_text}] {asr.has_valid_speech()} {evt.type} {evt.speech_duration} {len(to_be_sent_buffer)}")
                if evt.type == VADEventType.NORMAL_AUDIO:
                    for frame in evt.frames:
                        to_be_sent_buffer.append(frame.data) 
                    if self.room_state == AgentState.Agent_VAD and asr.has_valid_speech():
                        
                        self.room_state = AgentState.Agent_Listening
                        self.parent.runtime_data.clear_current_session(
                                        asr.message_id,
                                        asr.last_message_id,
                                    )
                        logger.debug(f' {asr.current_text} : 打断结束,并进入收音状态，同时也将asr结果向前端发送 当前room状态{self.room_state}')
                    if self.room_state in [AgentState.Agent_VAD,AgentState.Agent_Listening]:
                        if not asr.speak_end:
                            if len(to_be_sent_buffer) < 3200:
                                # 数据长度不够，累计数据
                                continue 
                            # ASR 进行中,继续发送音频
                            if not self.parent.is_valid():
                                break
                            await asr.async_asr(apply_gain(to_be_sent_buffer.to_bytes(3200),config.APPLY_GAIN_IN_DB))
                            to_be_sent_buffer.clear()
                        else:
                            # ASR结束了。
                            self.room_state = AgentState.Agent_Normal
                            logger.debug(f' 即将关闭ASR {self.room_state} current_text:{asr.current_text} 当前room状态{self.room_state}')
                            await asr.close_asr()
                elif evt.type == VADEventType.INFERENCE_DONE :
                    if self.room_state == AgentState.Agent_Normal and evt.speech_duration > self.interrupt_speech_duration:
                        if len(to_be_sent_buffer) > self.MIN_VAD_SENCOND:
                            
                            self.room_state = AgentState.Agent_VAD
                            logger.debug(f' 即将启动ASR {self.room_state} current_text:{asr.current_text} 当前room状态{self.room_state}')
                            if not self.parent.is_valid():
                                break

                            await asr.async_asr(apply_gain(to_be_sent_buffer.to_bytes(self.MIN_VAD_SENCOND),config.APPLY_GAIN_IN_DB))
                            # self.room_state = AgentState.Agent_Listening
                            # #需要先清除 NLP，然后清除其他队列。
                            # self.parent.runtime_data.clear_current_session(
                            #                 asr.message_id,
                            #                 asr.last_message_id,
                            #             )
                            # logger.debug(f' {asr.current_text} : 打断结束 {asr.last_message_id},并进入收音状态，同时也将asr结果向前端发送 当前room状态{self.room_state} ')
                            to_be_sent_buffer.clear()
                
                if (self.room_state == AgentState.Agent_VAD or self.room_state == AgentState.Agent_Listening ) and VADEventType.END_OF_SPEECH:
                    #这个需要强制停掉 ASR??
                    pass

        except Exception:
            logger.error(f" {self.parent.init_data.live_id} {traceback.format_exc()}")
        finally:
            await asr.close_asr()
        logger.debug(f"{self.parent.init_data.live_id} quit asr logic.")
        self.room_state = AgentState.Agent_Speaking
        self._recognize_atask = None
        # self.parent.runtime_data.queue_vad = None

    async def run(self):
        #可以打断
        #
        try:
            room_id = self.parent.init_data.live_id
            saved_audio_file = f'vaddata/vad_{self.parent.init_data.live_id}.wav'
            saver = AudioSaver(filename=saved_audio_file)
            os.makedirs("vaddata",exist_ok=True)
            logger.debug(f'准备写文件{saved_audio_file}')
            self._recognize()
            inference_f32_data = np.empty(self._model.window_size_samples, dtype=np.float32)

            # a copy is exposed to the user in END_OF_SPEECH
            speech_buffer: np.ndarray | None = None
            speech_buffer_max_reached = False
            speech_buffer_index: int = 0

            # "pub_" means public, these values are exposed to the users through events
            self.pub_speaking = False
            pub_speech_duration = 0.0
            pub_silence_duration = 0.0
            pub_current_sample = 0
            pub_timestamp = 0.0

            pub_sample_rate = 0
            pub_prefix_padding_samples = 0  # size in samples of padding data

            speech_threshold_duration = 0.0
            silence_threshold_duration = 0.0

            input_frames = []
            inference_frames = []
            # resampler: rtc.AudioResampler | None = None

            # used to avoid drift when the sample_rate ratio is not an integer
            input_copy_remaining_fract = 0.0
            

            # self.room_state = AgentState.Agent_Listening
            
            asr_buffer = []
            recv_audio_len = 0 
            while self.parent.is_valid():
                # 获取
                # logger.debug(f"{self.parent.init_data.live_id} asr start run_time_cal begin {recv_audio_len}")
                # while not self.parent.runtime_data.nlp_answer_queue.empty():
            #     # 1.vad 发现人声，发出打断的通知
            #     # 2.ASR结束，再次启动 vad
                input_frame:AudioFrame = await self.parent.runtime_data.remote_audio_queue.get()
                if input_frame is None:
                    break
                saver.write_audio_chunk(input_frame.data)
                recv_audio_len = recv_audio_len + len(input_frame.data)
                # if len(asr_buffer) > 5:
                #     asr_buffer.remove()
                # asr_buffer.append(input_frame)
                # if room_state == AgentState.Agent_Listening:
                #     #收音中,音频只发给ASR，不再做vad检测
                #     pass
                # if room_state == AgentState.Agent_Speaking:
                #     #检测是否被打断。
                #     pass
                if not pub_sample_rate or speech_buffer is None:
                    pub_sample_rate = input_frame.sample_rate

                    # alloc the buffers now that we know the input sample rate
                    pub_prefix_padding_samples = math.ceil(
                        self._opts.prefix_padding_duration * pub_sample_rate
                    )

                    speech_buffer = np.empty(
                        int(self._opts.max_buffered_speech * pub_sample_rate)
                        + int(self._opts.prefix_padding_duration * pub_sample_rate),
                        dtype=np.int16,
                    )

                elif pub_sample_rate != input_frame.sample_rate:
                    logger.error("a frame with another sample rate was already pushed")
                    continue

                input_frames.append(input_frame)
                # if resampler is not None:
                #     # the resampler may have a bit of latency, but it is OK to ignore since it should be
                #     # negligible
                #     inference_frames.extend(resampler.push(input_frame))
                # else:
                inference_frames.append(input_frame)

                # if self._recognize_atask:
                #     logger.debug(f"{self.parent.init_data.live_id} send data to asr.")
                self.parent.runtime_data.queue_vad.put_nowait(
                    VADEvent(
                        type=VADEventType.NORMAL_AUDIO,
                        samples_index=pub_current_sample,
                        timestamp=pub_timestamp,
                        silence_duration=pub_silence_duration,
                        speech_duration=pub_speech_duration,
                        frames=[input_frame],
                        speaking=True,
                    )
                )
                #     continue
                # self.room_state = AgentState.Agent_Listening
                while True:
                    start_time = time.perf_counter()

                    available_inference_samples = sum(
                        [frame.samples_per_channel for frame in inference_frames]
                    )
                    if available_inference_samples < self._model.window_size_samples:
                        logger.debug(f'vad finished one round inference.')
                        break  # not enough samples to run inference

                    input_frame = combine_audio_frames(input_frames)
                    inference_frame = combine_audio_frames(inference_frames)

                    # convert data to f32
                    np.divide(
                        inference_frame.data[: self._model.window_size_samples],
                        np.iinfo(np.int16).max,
                        out=inference_f32_data,
                        dtype=np.float32,
                    )

                    # run the inference
                    # if True or self.room_state == AgentState.Agent_Listening:
                    # logger.debug(f'vad inference begin.')
                    p = await self._loop.run_in_executor(
                        self._executor, self._model, inference_f32_data
                    )
                    # logger.debug(f'vad inference end.')
                    p = self._exp_filter.apply(exp=1.0, sample=p)

                    window_duration = (
                        self._model.window_size_samples / self._opts.sample_rate
                    )

                    pub_current_sample += self._model.window_size_samples
                    pub_timestamp += window_duration

                    resampling_ratio = pub_sample_rate / self._model.sample_rate
                    to_copy = (
                        self._model.window_size_samples * resampling_ratio
                        + input_copy_remaining_fract
                    )
                    to_copy_int = int(to_copy)
                    input_copy_remaining_fract = to_copy - to_copy_int

                    # copy the inference window to the speech buffer
                    available_space = len(speech_buffer) - speech_buffer_index
                    to_copy_buffer = min(self._model.window_size_samples, available_space)
                    if to_copy_buffer > 0:
                        speech_buffer[
                            speech_buffer_index : speech_buffer_index + to_copy_buffer
                        ] = input_frame.data[:to_copy_buffer]
                        speech_buffer_index += to_copy_buffer
                    elif not speech_buffer_max_reached:
                        # reached self._opts.max_buffered_speech (padding is included)
                        speech_buffer_max_reached = True
                        logger.warning(
                            "max_buffered_speech reached, ignoring further data for the current speech input"
                        )

                    inference_duration = time.perf_counter() - start_time
                    self._extra_inference_time = max(
                        0.0,
                        self._extra_inference_time + inference_duration - window_duration,
                    )
                    if inference_duration > SLOW_INFERENCE_THRESHOLD:
                        logger.warning(
                            "inference is slower than realtime",
                            extra={"delay": self._extra_inference_time},
                        )

                    def _reset_write_cursor():
                        nonlocal speech_buffer_index, speech_buffer_max_reached
                        assert speech_buffer is not None

                        if speech_buffer_index <= pub_prefix_padding_samples:
                            return

                        padding_data = speech_buffer[
                            speech_buffer_index
                            - pub_prefix_padding_samples : speech_buffer_index
                        ]

                        speech_buffer[:pub_prefix_padding_samples] = padding_data
                        speech_buffer_index = pub_prefix_padding_samples
                        speech_buffer_max_reached = False

                    def _copy_speech_buffer() -> AudioFrame:
                        # copy the data from speech_buffer
                        assert speech_buffer is not None
                        speech_data = speech_buffer[:speech_buffer_index].tobytes()

                        return AudioFrame(
                            sample_rate=pub_sample_rate,
                            num_channels=1,
                            samples_per_channel=speech_buffer_index,
                            data=speech_data,
                        )

                    if self.pub_speaking:
                        pub_speech_duration += window_duration
                        # 说话的情况，发音频过去
                    else:
                        pub_silence_duration += window_duration

                    if self.parent.runtime_data.queue_vad:
                        logger.debug(f'INFERENCE_DONE： to_copy_int:{to_copy_int} pub_speech_duration:{pub_speech_duration} len(input_frame.data):{len(input_frame.data)} inference_duration:{inference_duration} self.pub_speaking:{self.pub_speaking}')
                        self.parent.runtime_data.queue_vad.put_nowait(
                            VADEvent(
                                type=VADEventType.INFERENCE_DONE,
                                samples_index=pub_current_sample,
                                timestamp=pub_timestamp,
                                silence_duration=pub_silence_duration,
                                speech_duration=pub_speech_duration,
                                probability=p,
                                inference_duration=inference_duration,
                                frames=[
                                    AudioFrame(
                                        data=input_frame.data[:to_copy_int].tobytes(),
                                        sample_rate=pub_sample_rate,
                                        num_channels=1,
                                        samples_per_channel=to_copy_int,
                                    )
                                ],
                                speaking=self.pub_speaking,
                            )
                        )
                    # 是说话
                    if p >= self._opts.activation_threshold:
                        speech_threshold_duration += window_duration
                        silence_threshold_duration = 0.0
                        # 第一次检测到说话
                        if not self.pub_speaking:
                            # 第一次检测到说话，且够长度
                            if speech_threshold_duration >= self._opts.min_speech_duration:
                                self.pub_speaking = True

                                pub_silence_duration = 0.0
                                pub_speech_duration = speech_threshold_duration
                                #可以开始ASR
                                # if self.parent.runtime_data.queue_vad is not None:
                                #     self.parent.runtime_data.queue_vad.put_nowait(None)
                                # self.parent.runtime_data.queue_vad = asyncio.Queue()
                                # self._recognize(self.parent.runtime_data.queue_vad)
                                logger.debug(f'START_OF_SPEECH :  speech_buffer_index:{speech_buffer_index} pub_speech_duration:{pub_speech_duration}  speech_buffer:{len(speech_buffer.data)} self.pub_speaking:{self.pub_speaking}|True')
                                self.parent.runtime_data.queue_vad.put_nowait(
                                    VADEvent(
                                        type=VADEventType.START_OF_SPEECH,
                                        samples_index=pub_current_sample,
                                        timestamp=pub_timestamp,
                                        silence_duration=pub_silence_duration,
                                        speech_duration=pub_speech_duration,
                                        frames=[_copy_speech_buffer()],
                                        speaking=True,
                                    )
                                )
                                # self.parent.runtime_data.asr_audio_queue.put_nowait(
                                #     VADEvent(
                                #         type=VADEventType.START_OF_SPEECH,
                                #         samples_index=pub_current_sample,
                                #         timestamp=pub_timestamp,
                                #         silence_duration=pub_silence_duration,
                                #         speech_duration=pub_speech_duration,
                                #         frames=[_copy_speech_buffer()],
                                #         speaking=True,
                                #     )
                                # )

                    else:
                        # 不是说话
                        silence_threshold_duration += window_duration
                        speech_threshold_duration = 0.0
                        # 当前没有说话，重置索引
                        if not self.pub_speaking:
                            _reset_write_cursor()

                        if (
                            self.pub_speaking
                            and silence_threshold_duration
                            >= self._opts.min_silence_duration
                        ):
                            self.pub_speaking = False
                            pub_speech_duration = 0.0
                            pub_silence_duration = silence_threshold_duration
                            if self.parent.runtime_data.queue_vad:
                                logger.debug(f'END_OF_SPEECH : speech_buffer_index:{speech_buffer_index} pub_speech_duration:{pub_speech_duration}  speech_buffer:{len(speech_buffer.data)} self.pub_speaking:{self.pub_speaking}|False')
                                self.parent.runtime_data.queue_vad.put_nowait(
                                    VADEvent(
                                        type=VADEventType.END_OF_SPEECH,
                                        samples_index=pub_current_sample,
                                        timestamp=pub_timestamp,
                                        silence_duration=pub_silence_duration,
                                        speech_duration=pub_speech_duration,
                                        frames=[_copy_speech_buffer()],
                                        speaking=False,
                                    )
                                )

                            _reset_write_cursor()

                    # remove the frames that were used for inference from the input and inference frames
                    input_frames = []
                    inference_frames = []

                    # add the remaining data
                    if len(input_frame.data) - to_copy_int > 0:
                        data = input_frame.data[to_copy_int:].tobytes()
                        input_frames.append(
                            AudioFrame(
                                data=data,
                                sample_rate=pub_sample_rate,
                                num_channels=1,
                                samples_per_channel=len(data) // 2,
                            )
                        )

                    if len(inference_frame.data) - self._model.window_size_samples > 0:
                        data = inference_frame.data[
                            self._model.window_size_samples :
                        ].tobytes()
                        inference_frames.append(
                            AudioFrame(
                                data=data,
                                sample_rate=self._opts.sample_rate,
                                num_channels=1,
                                samples_per_channel=len(data) // 2,
                            )
                        )
                # logger.debug(f"{self.parent.init_data.live_id} asr start run_time_cal end")
        except:
            logger.error(f' something wrong {traceback.format_exc()}')
        finally:
            logger.info(f"{self.parent.init_data.live_id} quit vad logic.")
            if self.parent.runtime_data.queue_vad is not None:
                logger.info(f"{self.parent.init_data.live_id} 准备关闭ASR发送者.")
                self.parent.runtime_data.queue_vad.put_nowait(None)
            await asyncio.sleep(0.01)
            await gracefully_cancel(self._recognize_atask)
            if saver:
                saver.close()

