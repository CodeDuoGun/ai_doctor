import asyncio
import webrtcvad
import os
import json
import time
import nls
import traceback
import torch
from log import logger
import numpy as np
from config import config
from room_business.asr.utils import str_count
from room_business.business_base import BusinessBase
from room_business.room_data import VADEventType
from room_business.room_instance import RoomInstance
from utils.tool import get_db, spectral_subtraction, save_raw_wav
import queue
import io
import librosa
import soundfile as sf
import azure.cognitiveservices.speech as speechsdk
import constants

nls.enableTrace(False)
URL="wss://nls-gateway-cn-shanghai.aliyuncs.com/ws/v1"
TOKEN=""  #参考https://help.aliyun.com/document_detail/450255.html获取token
APPKEY=""    #获取Appkey请前往控制台：https://nls-portal.console.aliyun.com/applist


# TODO: total size 参数
class AzureASR():
    """
    queue 收data
    while 开启循环get chunk
    create vad , if 人声，asr, else, 静默帧
    create asr , if first send and [chunk, chunk], send asr
    if asr done, close asr, create vad
    """
    def __init__(self, name:str="azure_asr",room_instance:RoomInstance = None) -> None:

        #TODO: 根据不同id 控制返回和销毁
        self.__id = name
        self.room_instance:RoomInstance = room_instance
        self.vad_flag = True
        # 一次声音检测的标记
        self.message_id = ""
        self.last_message_id = "first_asr_message" # 用于处理是否正在nlp

        # asr_chunks 用于存一句话
        self.asr_chunks = b"" # 缓存人声音，目前2s, 640*5*2 6400

        # 区分在线检测和完整性检测
        self.online_asr =False
        logger.info(f"sentence_asr init")
        self.sentence_asr = False

        # 确认是否声音结束标记
        #TODO: 删除sentence_start_num
        self.sentence_start_num = 0
        self.sentence_end_num = 0

        # jiangzhenghao增加以下两个字段
        self.sentence_id = "asr_silent"
        self.session_id = 0
        
        self.sr = None # asr 实例 
        self.speak_end =False # 判断是否结束说话
        # self.vad = webrtcvad.Vad(config.VAD_MODE)
        self.frame_duration = 20
        self.sample_rate = 16000
        self.quantify_value = int(self.sample_rate * 2 * self.frame_duration / 1000.0) # 量化大小
        self.ring_buffer = []

        self.stream = None
        self.audio_config = None
        self.speech_config = speechsdk.SpeechConfig(subscription=constants.Microsoft_Config.key, region=constants.Microsoft_Config.region, speech_recognition_language=constants.Language.zh)
        self.speech_recognizer = None
        self.keep_send_audio = False
        self.asr_start = False

        self.room_id = self.room_instance.init_data.live_id
        self.current_text = ""
        self.interrupted = False
        self.is_speech = False
        self.monitor_task :asyncio.Task = None

        self.last_text_time = None  # 用于存储上次接收到文字的时间戳
        self.check_interval = 0.2   # 检查时间间隔 (秒)
        self.timeout = 1.0          # 超时时间 (秒)

    def has_valid_speech(self):
        if self.is_speech:
            return self.is_speech
        total = str_count(self.current_text)
        if not total is None:
            s_len, count_zh, count_en, count_sp, count_dg, count_pu = total
            self.is_speech = (count_zh + count_en + count_dg) > config.MIN_WORDS
        return False
    
    def send_asr_to_client(self,res,isFinal = False):
        data = {
            "room_id":self.room_id,
            "msg_id":self.message_id,
            "msg_type":"asr",
            "sentence_id":self.sentence_id,
            "text":res,
        }
        if self.has_valid_speech() or isFinal:
            logger.debug(f'{self.room_instance.init_data.live_id} 准备向客户端发ASR结果： {res} ')
            self.room_instance.runtime_data.asr_callback_queue.put_nowait(data)
        # self.room_instance.runtime_data.asr_callback_queue.put_nowait(data)

    async def monitor_speech_timeout(self):
        """
        监控 ASR 返回的超时任务
        """
        while self.asr_start:
            await asyncio.sleep(self.check_interval)
            if self.last_text_time and (time.time() - self.last_text_time) > self.timeout:
                # 如果超过指定时间没有收到文字返回，关闭 ASR
                logger.info(f"No text received for {self.timeout} seconds. Terminating ASR.")
                break  # 退出监控任务
        await self.close_asr()

    def __recognizing_cb(self, evt: speechsdk.SpeechRecognitionEventArgs):
        """callback for recognizing event"""
        logger.debug(f'{self.room_instance.init_data.live_id} __recognizing_cb： {evt} ')
        if evt.result.reason == speechsdk.ResultReason.RecognizingKeyword:
            logger.info(f'{self.room_id} RECOGNIZING KEYWORD: {evt}')
        elif evt.result.reason == speechsdk.ResultReason.RecognizingSpeech:
            res = evt.result.text
            logger.info(f'{self.room_id} {self.message_id} azure.sentence_asr:{self.sentence_asr}, test_on_chg:{res}')
            if res:
                self.last_text_time = time.time()
                self.current_text = res
                self.send_asr_to_client(res)


    def __recognized_cb(self, evt: speechsdk.SpeechRecognitionEventArgs):
        """callback for recognized event"""
        logger.debug(f'{self.room_instance.init_data.live_id} __recognized_cb {evt} ')
        if evt.result.reason == speechsdk.ResultReason.RecognizedKeyword:
            logger.info(f'{self.room_id} RECOGNIZED KEYWORD: {evt}')
        elif evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            res = evt.result.text
            if not res:
                logger.info(f"{self.room_id} {self.message_id} asr 识别结果为空：{res}")

                self.speak_end = True
                return
            
            self.sentence_id = evt.result.result_id
            logger.info(f'{self.room_instance.init_data.live_id} 识别结果 by {evt} {self.sentence_id} -- {res}')
            self.current_text = res
            self.last_text_time = time.time() - 10000000 #结束了，可以让监控任务自行结束
            self.put_audio_chunk(res)
            self.send_asr_to_client(res,isFinal=True)

            self.speak_end = True
            logger.info(f"{self.room_id} lastmsg11:{self.last_message_id}, cur_msg:{self.message_id} test_on_sentence_end:{res}")
        elif evt.result.reason == speechsdk.ResultReason.NoMatch:
            logger.warning(f'{self.room_id} NOMATCH: {evt}')
        self.close_asr()
        self.speak_end = True
    
    def __cancel_cb(self, evt: speechsdk.SpeechRecognitionCanceledEventArgs):
        logger.debug(f'{self.room_instance.init_data.live_id} __cancel_cb {evt} ')
        if evt.result.reason == speechsdk.ResultReason.Canceled:
            logger.info(f'{self.room_id} {self.message_id} CANCELED {evt}')

    def __start_cb(self, evt: speechsdk.SessionEventArgs):
        logger.info(f'{self.room_id} {self.message_id} SESSION STARTED: {evt}')

    def __session_stop_cb(self, evt: speechsdk.SessionEventArgs):
        logger.info(f'{self.room_id} {self.message_id} SESSION STOPED: {evt}')
    
    def __speech_end_cb(self, evt: speechsdk.RecognitionEventArgs):
        logger.info(f"{self.room_id} {self.message_id} speech end {evt}")
        self.close_asr()

    async def async_asr(self, chunk):
        """
        """
        logger.debug(f"{self.room_id} {self.message_id} 开始online {self.asr_start} asr")
        if not self.asr_start and not self.speech_recognizer:
            self.speak_end = False
            self.current_text = ""
            self.session_id = self.session_id + 1
            self.message_id = f"{self.room_id}_{self.session_id}"
            def stop_cb(evt: speechsdk.SessionEventArgs):
                """callback that signals to stop continuous recognition upon receiving an event `evt`"""
                logger.info(f'{self.room_id} {self.message_id} CLOSING on {evt}')
                self.close_asr()
                if self.monitor_task:
                    self.last_text_time = None
                    self.monitor_task.cancel()
                    self.monitor_task = None
            if self.monitor_task:
                    self.last_text_time = None
                    self.monitor_task.cancel()
                    self.monitor_task = None
            self.last_text_time = time.time()
            self.monitor_task = asyncio.create_task(self.monitor_speech_timeout())    
            logger.info(f"{self.room_id} {self.message_id} async_asr 启动")
            self.stream = speechsdk.audio.PushAudioInputStream(stream_format=speechsdk.audio.AudioStreamFormat(
                samples_per_second=16000,
                bits_per_sample=16,
                channels=1,
            ))
            self.audio_config = speechsdk.audio.AudioConfig(stream=self.stream)
            self.speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=self.audio_config)
            self.speech_recognizer.recognizing.connect(self.__recognizing_cb)
            self.speech_recognizer.recognized.connect(self.__recognized_cb)
            self.speech_recognizer.session_started.connect(self.__start_cb)
            # self.speech_recognizer.session_stopped.connect(self.session_stop_cb)
            # self.speech_recognizer.canceled.connect(self.cancel_cb)
            self.speech_recognizer.speech_end_detected.connect(self.__speech_end_cb)
            # Stop continuous recognition on either session stopped or canceled events
            self.speech_recognizer.session_stopped.connect(stop_cb)
            self.speech_recognizer.canceled.connect(stop_cb)

            self.speech_recognizer.start_continuous_recognition_async().get()
            self.asr_start = True
        # self.asr_chunks += chunk
        # self.stream.
        if self.speech_recognizer:
            logger.debug(f"{self.room_id} 发音频出去 开始online {self.asr_start} asr")
            self.stream.write(chunk)

    async def close_asr(self):
        """
        """
        #Close asr
        if self.speech_recognizer:
            self.speech_recognizer.stop_continuous_recognition_async().get()
            self.speech_recognizer = None
            # flush status
            self.flush_status()
            self.stream.close()
            self.stream = None
            self.audio_config = None
        if self.monitor_task:
            self.last_text_time = None
            self.monitor_task.cancel()
            self.monitor_task = None
        logger.info(f"{self.room_id} {self.message_id} asr stoped")
        self.speak_end = False
        self.asr_start = False

    def flush_status(self):
        """
            一次完整收音结束后，更新vad online_asr,offline_asr 相关状态
        """
        
        logger.info(f"{self.room_id} {self.message_id} sentence_asr flush")
        if self.last_message_id != self.message_id:
            self.last_message_id = self.message_id 
        self.vad_flag =True
        self.sentence_asr = False
        self.asr_chunks = b""
        self.online_asr = False
        self.sentence_end_num = 0
        self.sentence_start_num = 0
        
        self.is_speech = False
        self.message_id = ""
        self.sentence_id = ""
        self.current_text = ""
        self.interrupted = True

    def put_audio_chunk(self, audio_txt):
        data = {
            "room_id":self.room_id,
            "msg_id":self.message_id,
            "sentence_id":self.sentence_id,
            "asr_text":audio_txt,
            "last_msg_id": self.last_message_id
        }
        logger.debug(f'{self.room_instance.init_data.live_id} 准备向LLM提问： {audio_txt} ')
        self.room_instance.runtime_data.asr_msg_queue.put_nowait(data)

