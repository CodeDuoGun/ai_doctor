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
from utils.tool import get_db, spectral_subtraction, save_raw_wav
import queue
import io
import librosa
import soundfile as sf
import azure.cognitiveservices.speech as speechsdk
import constants

nls.enableTrace(False)
URL="wss://nls-gateway-cn-shanghai.aliyuncs.com/ws/v1"
TOKEN="9695e79d6ed24a4a9dcdbc38bb0c9bd8"  #参考https://help.aliyun.com/document_detail/450255.html获取token
APPKEY="L7844wm4aavtAPsr"    #获取Appkey请前往控制台：https://nls-portal.console.aliyun.com/applist


# TODO: total size 参数
class CustomASR():
    """
    queue 收data
    while 开启循环get chunk
    create vad , if 人声，asr, else, 静默帧
    create asr , if first send and [chunk, chunk], send asr
    if asr done, close asr, create vad
    """
    def __init__(self,asr_msg_queue, asr_callback_queue: asyncio.Queue, name:str="custom_asr") -> None:
        self.asr_msg_queue = asr_msg_queue
        self.asr_callback_queue = asr_callback_queue
        self.room_instance  = None
        #TODO: 根据不同id 控制返回和销毁
        self.__id = name
        
        self.vad_flag = True
        # 一次声音检测的标记
        self.message_id = ""
        self.last_message_id = "" # 用于处理是否正在nlp

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
        self.vad = webrtcvad.Vad(config.VAD_MODE)
        self.frame_duration = 20
        self.sample_rate = 16000
        self.quantify_value = int(self.sample_rate * 2 * self.frame_duration / 1000.0) # 量化大小
        self.ring_buffer = []

        # 新增microsoft asr , refer to: https://github.com/Azure-Samples/cognitive-services-speech-sdk/
        if config.ASR_MODE != "aliyun":
            self.speech_config = speechsdk.SpeechConfig(subscription=constants.Microsoft_Config.key, region=constants.Microsoft_Config.region, speech_recognition_language=constants.Language.zh)
            self.stream = speechsdk.audio.PushAudioInputStream()
            self.audio_config = speechsdk.audio.AudioConfig(stream=self.stream)
            self.speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=self.audio_config)
            self.keep_send_audio = False
            self.asr_start = False


    @staticmethod
    def get_token():
        #获取token 
        cur_time = time.time()
        if os.path.exists("./token.txt"):
            token_time, cur_token = read_expire_time()
            if int(token_time) - cur_time>=10:
                logger.info(f"不需要更新asr token")
                return cur_token
        logger.info(f"asr token 更新了")
        new_token = get_asr_token()
        return new_token

    
    async def vad_detect(self, room_id: str, data: bytes, is_speak_final: bool=False, sample_rate:int=16000, buffer_size: int=10):
        """
        Args:
            room_id: roomw唯一标识
            audio_chunk: 人声检测用的音频流
            is_speak_final: 判断是否说话结束，

        """ 
        self.session_id+=1
        self.message_id = f"{room_id}_{self.session_id}"
        # save_raw_wav(self.message_id, data, tag="VAD")
        is_speak_start = -1
        frame_len = int(self.quantify_value / 2)
        logger.debug(f"valid_rate_and_frame_length:{webrtcvad.valid_rate_and_frame_length(sample_rate, frame_len)}")
        # 这里进行谱减法音量增强后，再音高过滤，再进行人声检测
        signal, fs = np.frombuffer(data, dtype=np.int16), 16000
        enhanced_signal = spectral_subtraction(signal, fs, self.message_id)
        chunks = list(enhanced_signal[pos:pos + self.quantify_value] for pos in range(0, len(enhanced_signal), self.quantify_value))
        if len(chunks[-1]) != self.quantify_value:
            chunks = chunks[:-1]
        for chunk in chunks:
            voiced = self.vad.is_speech(chunk, sample_rate)
            self.ring_buffer.append("1" if voiced else "0")
            #f = Frame() speech = is_speech 得到有人声音的Frame的个数
            num_voiced = len([1 for speech in self.ring_buffer if speech=="1"])
            # 当说话段数量大于缓冲区的90%时认为人声开始，所以进入if时前10段有人声恰好在ring_buffer里
        logger.info(f"{self.message_id} num_voiced:{num_voiced}, buffer:{len(self.ring_buffer)}")
        if num_voiced > config.VOICED_PERCENT * len(self.ring_buffer):
            logger.info(f"{self.message_id} 检测到人声：num_voiced:{self.ring_buffer}")
            cur_db = get_db(enhanced_signal, self.message_id)
            if cur_db > config.DB_VALUE-10:
                is_speak_start = 1 
                self.online_asr = True
                self.vad_flag = False
                is_speak_final = False
                self.speak_end = False
                self.ring_buffer = []
                if self.last_message_id=="":
                    self.last_message_id = self.message_id 
                logger.info(f"音量{cur_db} 超过阈值 {config.DB_VALUE} {self.message_id} 检测到人声,停止vad lastmsg:{self.last_message_id}")
                return is_speak_start, is_speak_final
        # 一旦 vad判断有人声了，那就无需再做vad的检查了，直到asr说没有人声了，此时继续需要vad
        logger.info(f"{self.message_id} vad静默")
        self.ring_buffer = []
        is_speak_final = True
        return is_speak_start, is_speak_final

    def send_asr_to_client(self,res):
        data = {
            "room_id":self.__id,
            "msg_id":self.message_id,
            "msg_type":"asr",
            "sentence_id":self.sentence_id,
            "text":res,
        }
        self.asr_callback_queue.put_nowait(data)

    def recognizing_cb(self, evt: speechsdk.SpeechRecognitionEventArgs):
        """callback for recognizing event"""
        if evt.result.reason == speechsdk.ResultReason.RecognizingKeyword:
            logger.info(f'RECOGNIZING KEYWORD: {evt}')
        elif evt.result.reason == speechsdk.ResultReason.RecognizingSpeech:
            res = evt.result.text
            logger.info(f'{self.message_id} self.sentence_asr:{self.sentence_asr}, test_on_chg:{res}')
            if res:
                self.send_asr_to_client(res)


    def recognized_cb(self, evt: speechsdk.SpeechRecognitionEventArgs):
        """callback for recognized event"""
        if evt.result.reason == speechsdk.ResultReason.RecognizedKeyword:
            logger.info(f'RECOGNIZED KEYWORD: {evt}')
        elif evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            res = evt.result.text
            if not res:
                logger.warning(f"{self.message_id} asr 识别结果为空：{res}")

                self.speak_end = True
                return
            
            # 关闭asr
            
            self.sentence_id = evt.result.result_id
            logger.info(f'{self.room_instance.init_data.live_id} 识别结果 by {evt} {self.sentence_id} -- {res}')
            self.put_audio_chunk.put(res)
            self.send_asr_to_client(res)

            self.speak_end = True
            logger.info(f"lastmsg11:{self.last_message_id}, cur_msg:{self.message_id} test_on_sentence_end:{res}")
            self.close_microsoft_asr()

        elif evt.result.reason == speechsdk.ResultReason.NoMatch:
            logger.warning(f'NOMATCH: {evt}')
    
    def cancel_cb(self, evt: speechsdk.SpeechRecognitionCanceledEventArgs):
        if evt.result.reason == speechsdk.ResultReason.Canceled:
            logger.info(f'{self.message_id} CANCELED {evt}')

    def start_cb(self, evt: speechsdk.SessionEventArgs):
        logger.info(f'{self.message_id} SESSION STARTED: {evt}')

    def session_stop_cb(self, evt: speechsdk.SessionEventArgs):
        logger.info(f'{self.message_id} SESSION STOPED: {evt}')
    
    def speech_end_cb(self, evt: speechsdk.RecognitionEventArgs):
        logger.info(f"{self.message_id} speech end {evt}")
        self.close_microsoft_asr()

    async def async_microsoft_asr(self, chunk):
        """
        """
        logger.info(f"{self.message_id} 开始online {self.online_asr} asr")
        if not self.asr_start:
            def stop_cb(evt: speechsdk.SessionEventArgs):
                """callback that signals to stop continuous recognition upon receiving an event `evt`"""
                logger.info(f'{self.message_id} CLOSING on {evt}')
                self.close_microsoft_asr()
                

            self.speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=self.audio_config)
            self.speech_recognizer.recognizing.connect(self.recognizing_cb)
            self.speech_recognizer.recognized.connect(self.recognized_cb)
            self.speech_recognizer.session_started.connect(self.start_cb)
            # self.speech_recognizer.session_stopped.connect(self.session_stop_cb)
            # self.speech_recognizer.canceled.connect(self.cancel_cb)
            self.speech_recognizer.speech_end_detected.connect(self.speech_end_cb)
            # Stop continuous recognition on either session stopped or canceled events
            self.speech_recognizer.session_stopped.connect(stop_cb)
            self.speech_recognizer.canceled.connect(stop_cb)

            self.speech_recognizer.start_continuous_recognition_async().get()
            self.asr_start = True
        self.asr_chunks += chunk

    def close_microsoft_asr(self):
        """
        """
        #Close asr
        if self.speech_recognizer:
            self.speech_recognizer.stop_continuous_recognition_async().get()
            self.speech_recognizer = None
            # flush status
            self.flush_status()
            self.stream.close()
        logger.info(f"{self.message_id} asr stoped")



    async def async_asr(self, asr_chunks):
        """
        Args: 
            asr_chunks: 用于asr识别， bytes
        """
        
        logger.info(f"{self.message_id} 开始online {self.online_asr} asr")
        if self.last_message_id=="":
            self.last_message_id = self.message_id 

        if not self.sr:
            logger.info(f"{self.message_id} 重新实例化sr")
            self.sr = nls.NlsSpeechTranscriber(
                        url=URL,
                        token=self.token,
                        appkey=APPKEY,
                        on_sentence_begin=self.on_sentence_begin,
                        on_sentence_end=self.on_sentence_end,
                        on_start=self.on_start,
                        on_result_changed=self.on_result_chg,
                        on_completed=self.on_completed,
                        on_error=self.on_error,
                        on_close=self.on_close,
                        callback_args=[self.__id, self.sentence_asr]
                    )
            logger.info(f"session start:{self.message_id}")
            r = self.sr.start(
                aformat="pcm", # pcm
                enable_intermediate_result=True,
                enable_punctuation_prediction=True,
                enable_inverse_text_normalization=True,
                timeout=5,
                ping_interval=0,
            )
        # self.__slices = zip(*(iter(asr_chunks),) * 640)
        # for i in self.__slices:
        #     self.sr.send_audio(bytes(i))
        self.sr.send_audio(asr_chunks)
        self.asr_chunks += asr_chunks
        # self.sr.ctrl(ex={"test":"tttt"})
        #sleep不能丢
        await asyncio.sleep(0.1)

    async def close_asr(self):
        if self.sr:
            r = self.sr.shutdown()
            await asyncio.sleep(0.01)
            self.sr = None
        else: 
            r = None
        # flush status
        self.flush_status()
        logger.info(f"{self.message_id}: sr stopped:{r}")

    def flush_status(self):
        """
            一次完整收音结束后，更新vad online_asr,offline_asr 相关状态
        """
        
        logger.info(f"{self.message_id} sentence_asr flush")
        self.vad_flag =True
        self.sentence_asr = False
        self.asr_chunks = b""
        self.online_asr = False
        self.sentence_end_num = 0
        self.sentence_start_num = 0
        self.asr_start = False

    async def get_audio_chunk(self):
        res = await self.asr_msg_queue.get()

    def put_audio_chunk(self, audio_txt):
        data = {
            "room_id":self.__id,
            "msg_id":self.message_id,
            "sentence_id":self.sentence_id,
            "asr_text":audio_txt,
            "last_msg_id": self.last_message_id
        }
        self.asr_msg_queue.put_nowait(data)

    async def on_sentence_end_v1(self, message, *args):
        """"""
        # put识别出来的文本内容
        payload = eval(message)
        res = payload["payload"]["result"]
        #TODO: 看起来接口还没有返回，就已经拿到了状态
        logger.info(f"self.sentence_asr:{self.sentence_asr}, self.online_asr:{self.online_asr}")
        if not self.sentence_asr:
            logger.info(f"{self.message_id} test_on_sentence_end:{res}")
        if not res:
            logger.warning(f"asr 识别结果为空：{res}")
            return
        if self.sentence_asr:
            logger.info(f"{self.message_id} Final test_on_sentence_end:{res}")
            self.sentence_end_num +=1
            self.sentence_id = payload["header"]["message_id"]
            self.put_audio_chunk(res)
        
    def on_sentence_end(self, message, *args):
        """ 如果识别人声结束，这个会有返回
        此时关闭asr，更新状态，重新进行收声
        """
        payload = eval(message)
        res = payload["payload"]["result"]
        logger.info(f"self.sentence_asr:{self.sentence_asr}, self.online_asr:{self.online_asr}")
        if not res:
            logger.warning(f"{self.message_id} asr 识别结果为空：{res}")

            self.speak_end = True
            return
        
        # 关闭asr
        self.sentence_id = payload["header"]["message_id"]
        logger.info(f'{self.room_instance.init_data.live_id} 识别结果 by on_sentence_end {self.sentence_id} -- {res}')
        self.put_audio_chunk(res)
        self.send_asr_to_client(res)

        self.speak_end = True
        logger.info(f"lastmsg:{self.last_message_id}, cur_msg:{self.message_id} test_on_sentence_end:{res}")

    def on_error(self, message, *args):
        logger.error(f"ASR on_error message:{message}, args=>{args}")

    def on_close(self, *args):
        pass
        # logger.warning(f"on_close: args=>{args}")

    def on_start(self, message, *args):
        self.sentence_start_num +=1
        logger.info(f"{self.message_id} test_on_start:{message}")
    
    def on_result_chg(self, message, *args):
        payload = eval(message)
        res = payload["payload"]["result"]
        logger.info(f'{self.message_id} self.sentence_asr:{self.sentence_asr}, test_on_chg:{res} {payload}')
        if res:
            self.send_asr_to_client(res)


    def on_completed(self, message, *args):
        """表示识别结束

        """
        logger.info(f"{self.message_id} on_completed:args=>{args} message=>{message}")


    def on_sentence_begin(self, message, *args):
        logger.info(f"test_on_sentence_begin:{message}")


def validate(model,inputs: torch.Tensor):
    with torch.no_grad():
        outs = model(inputs)
    return outs

# Provided by Alexander Veysov
def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/32768
    sound = sound.squeeze()  # depends on the use case
    return sound 

TOKEN_PATH = "./token.txt"
def read_expire_time():
    # 如果从来没有这个文件
    with open(TOKEN_PATH, "r") as fp:
        lines = fp.readlines()
    logger.debug(lines)
    # token ,time
    return lines[0].strip(), lines[1].strip()


def get_asr_token():
    from aliyunsdkcore.client import AcsClient
    from aliyunsdkcore.request import CommonRequest

    # 创建AcsClient实例
    client = AcsClient(
        os.getenv('ALIYUN_AK_ID', ""),
        os.getenv('ALIYUN_AK_SECRET', ""),
        "cn-shanghai"
        )

    # 创建request，并设置参数。
    request = CommonRequest()
    request.set_method('POST')
    request.set_domain('nls-meta.cn-shanghai.aliyuncs.com')
    request.set_version('2019-02-28')
    request.set_action_name('CreateToken')

    try : 
        response = client.do_action_with_exception(request)
        print(response)

        jss = json.loads(response)
        if 'Token' in jss and 'Id' in jss['Token']:
            token = jss['Token']['Id']
            expireTime = jss['Token']['ExpireTime']
            print("token = " + token)
            print("expireTime = " + str(expireTime))
            # 把这个时间落盘
            with open(TOKEN_PATH, "w") as fp:
                fp.write(f"{expireTime}\n") 
                fp.write(f"{token}\n") 
    except Exception:
        logger.error(traceback.format_exc())
    return token 
