import time
import numpy as np 
import resampy
import dashscope
import os
from utils.log import logger
import base64
from utils.tool import pcm_to_wav
import traceback
import queue
import soundfile as sf
from queue import Queue
from io import BytesIO
from dashscope.audio.tts_v2 import *
from default_config import config
dashscope.api_key = config.DASH_API_KEY
from enum import Enum



class State(Enum):
    RUNNING=0
    PAUSE=1

class CustomTTS():
    def __init__(self, nefreal, fps:int=25):
        self.nerfreal = nefreal
        self.fps = fps # 20 ms per frame
        self.sample_rate = 16000
        self.chunk = self.sample_rate // self.fps # 320 samples per chunk (20ms * 16000 / 1000)
        self.input_stream = BytesIO()

        self.msgqueue = Queue()
        self.state = State.RUNNING
        
    def txt_to_audio(self,msg): 
        self.stream_tts(
            self.gen_cosyvoice(msg)
        )
    
    def stream_tts(self, audio_stream):
        logger.info(f"audio stream {type(audio_stream)}")
        streamlen = len(audio_stream)
        idx = 0
        while streamlen >= self.chunk and self.state==State.RUNNING:
            self.nerfreal.put_audio_frame(audio_stream[idx:idx+self.chunk])
            streamlen -= self.chunk
            idx += self.chunk

    def request_qwen_stream_tts(self, text):
        # text = "那我来给大家推荐一款T恤，这款呢真的是超级好看，这个颜色呢很显气质，而且呢也是搭配的绝佳单品，大家可以闭眼入，真的是非常好看，对身材的包容性也很好，不管啥身材的宝宝呢，穿上去都是很好看的。推荐宝宝们下单哦。"
        response = dashscope.audio.qwen_tts.SpeechSynthesizer.call(
            model="qwen-tts",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            text=text,
            voice="Cherry",
            stream=True
        )
        for chunk in response:
            audio_string = chunk["output"]["audio"]["data"]
            owav_bytes = base64.b64decode(audio_string)
            yield owav_bytes
    
    def gen_cosyvoice(self,text):
        # 模型
        start = time.perf_counter()
        model = "cosyvoice-v2"
        # 音色
        voice = "longxiaochun_v2"

        # 实例化SpeechSynthesizer，并在构造方法中传入模型（model）、音色（voice）等请求参数
        synthesizer = SpeechSynthesizer(model=model, voice=voice, format=AudioFormat.PCM_16000HZ_MONO_16BIT) # pcm bytes
        # synthesizer = SpeechSynthesizer(model=model, voice=voice, format=AudioFormat.WAV_16000HZ_MONO_16BIT) # np
        # 发送待合成文本，获取二进制音频
        res = synthesizer.call(text)
        print('[Metric] requestId为：{}，首包延迟为：{}毫秒'.format(
            synthesizer.get_last_request_id(),
            synthesizer.get_first_package_delay())) 
        end = time.perf_counter()
        logger.info(f"tts result: cost{end-start}")
        audio_np = np.frombuffer(res, dtype=np.int16)
        return audio_np
        # first = True
        # id = 0
        # chunk = res[id:id+32000]
        # while chunk: 
        #     chunk = res[id:id+32000] # 1280 32K*20ms*2
        #     if first:
        #         end = time.perf_counter()
        #         logger.info(f"gpt_sovits Time to first chunk: {end-start}s")
        #         first = False
        #     id += 1
        #     if chunk and self.state==State.RUNNING:
        #         yield chunk
