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
URL=""
TOKEN=""  #参考https://help.aliyun.com/document_detail/450255.html获取token
APPKEY=""    #获取Appkey请前往控制台：https://nls-portal.console.aliyun.com/applist


# TODO: total size 参数
class AliASR():
    """
    queue 收data
    while 开启循环get chunk
    create vad , if 人声，asr, else, 静默帧
    create asr , if first send and [chunk, chunk], send asr
    if asr done, close asr, create vad
    """
    def __init__(self, name:str="ali_asr",room_instance:RoomInstance = None) -> None:

        #TODO: 根据不同id 控制返回和销毁
        self.__id = name
        self.room_instance = room_instance
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
        self.is_speech = False
        self.quantify_value = int(self.sample_rate * 2 * self.frame_duration / 1000.0) # 量化大小
        self.ring_buffer = []
        self.token = self.get_token()
        self.current_text = ""
        self.room_id = self.room_instance.init_data.live_id
        self.interrupted = True

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

    def has_valid_speech(self):
        if self.is_speech:
            return self.is_speech
        total = str_count(self.current_text)
        if not total is None:
            s_len, count_zh, count_en, count_sp, count_dg, count_pu = total
            self.is_speech = (count_zh + count_en + count_dg) > config.MIN_WORDS
        return False
    async def async_asr(self, asr_chunks):
        """
        Args: 
            asr_chunks: 用于asr识别， bytes
        """
        
        logger.info(f"{self.last_message_id} 开始online {self.online_asr} asr")
        # if self.last_message_id=="":
        # if self.last_message_id != self.message_id:
        #     self.last_message_id = self.message_id 

        if not self.sr:
            self.speak_end = False
            self.session_id = self.session_id + 1
            self.message_id = f"{self.room_id}_{self.session_id}"
            logger.info(f"{self.message_id} 重新实例化sr")
            self.current_text = ""
            self.sr = nls.NlsSpeechTranscriber(
                        url=URL,
                        token=self.token,
                        appkey=APPKEY,
                        on_sentence_begin=self.__on_sentence_begin,
                        on_sentence_end=self.__on_sentence_end,
                        on_start=self.__on_start,
                        on_result_changed=self.__on_result_chg,
                        on_completed=self.__on_completed,
                        on_error=self.__on_error,
                        on_close=self.__on_close,
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

        self.sr.send_audio(asr_chunks)
        self.asr_chunks += asr_chunks
        #sleep不能丢
        await asyncio.sleep(0.01)

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
        self.speak_end = False

    def flush_status(self):
        """
            一次完整收音结束后，更新vad online_asr,offline_asr 相关状态
        """
        
        logger.info(f"{self.message_id} sentence_asr flush")
        if self.last_message_id != self.message_id:
            self.last_message_id = self.message_id 
        self.vad_flag =True
        self.sentence_asr = False
        self.asr_chunks = b""
        self.online_asr = False
        self.sentence_end_num = 0
        self.sentence_start_num = 0
        self.asr_start = False
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

        
    def __on_sentence_end(self, message, *args):
        """ 如果识别人声结束，这个会有返回
        此时关闭asr，更新状态，重新进行收声
        """
        payload = eval(message)
        res = payload["payload"]["result"]
        logger.debug(f"self.sentence_asr:{self.sentence_asr}, self.online_asr:{self.online_asr}")
        if not res:
            logger.warning(f"{self.message_id} asr 识别结果为空：{res}")
            return
        self.current_text = res

        # 关闭asr
        self.sentence_id = payload["header"]["message_id"]
        logger.debug(f' 识别结果 by on_sentence_end {self.sentence_id} -- {res} {self.has_valid_speech()} {self.speak_end}')
        # if self.has_valid_speech():
        self.put_audio_chunk(res)
        self.send_asr_to_client(res,isFinal=True)
        self.speak_end = True
        logger.debug(f"lastmsg:{self.last_message_id}, cur_msg:{self.message_id} test_on_sentence_end:{res} {self.speak_end}")

    def __on_error(self, message, *args):
        logger.error(f"ASR on_error message:{message}, args=>{args}")

    def __on_close(self, *args):
        pass
        # logger.warning(f"on_close: args=>{args}")

    def __on_start(self, message, *args):
        self.sentence_start_num +=1
        logger.info(f"{self.message_id} test_on_start:{message}")
    
    def __on_result_chg(self, message, *args):
        payload = eval(message)
        res = payload["payload"]["result"]
        logger.info(f'{self.message_id} self.sentence_asr:{self.sentence_asr}, test_on_chg:{res} {payload}')
        if res:
            self.current_text = res
            self.send_asr_to_client(res)


    def __on_completed(self, message, *args):
        """表示识别结束

        """
        logger.info(f"{self.message_id} on_completed:args=>{args} message=>{message}")


    def __on_sentence_begin(self, message, *args):
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
