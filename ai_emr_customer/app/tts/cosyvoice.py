import time
import asyncio
import threading
import numpy as np 
# import resampy
import dashscope
import os
from app.utils.log import logger
import base64
# from utils.tool import pcm_to_wav
import traceback
import queue
from queue import Queue
from io import BytesIO
from dashscope.audio.tts_v2 import *
from app.config import config
dashscope.api_key = config.DASHSCOPE_API_KEY
from enum import Enum
from datetime import datetime
import pyaudio


def get_timestamp():
    now = datetime.now()
    formatted_timestamp = now.strftime("[%Y-%m-%d %H:%M:%S.%f]")
    return formatted_timestamp



class State(Enum):
    RUNNING=0
    PAUSE=1

class CustomTTS():
    def __init__(self, fps:int=25):
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
        streamlen = len(audio_stream)
        idx = 0
        while streamlen >= self.chunk and self.state==State.RUNNING:
            # self.nerfreal.put_audio_frame(audio_stream[idx:idx+self.chunk])
            streamlen -= self.chunk
            idx += self.chunk
            yield audio_stream[idx:idx+self.chunk]

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
        logger.debug(f"tts result: cost{end-start}")
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
    
    def stream_gen_cosyvoice(self, text, play_queue, loop):
        self.audio_chunk = b""

        # 模型
        model = "cosyvoice-v2"
        # 音色
        voice = "longxiaochun_v2"
        # 定义回调接口
        class Callback(ResultCallback):

            def on_open(self):
                print("连接建立：" + get_timestamp())

            def on_complete(self):
                print("语音合成完成，所有合成结果已被接收：" + get_timestamp())

            def on_error(self, message: str):
                print(f"语音合成出现异常：{message}")

            def on_close(self):
                print("连接关闭：" + get_timestamp())
                # 停止播放器

            def on_event(self, message):
                pass

            def on_data(self, data: bytes) -> None:
                # print(get_timestamp() + " 二进制音频长度为：" + str(len(data)),play_queue.qsize())
                # asyncio.run_coroutine_threadsafe(play_queue.put(data), loop)
                play_queue.put(data)
                # play_queue.put(data)


        callback = Callback()
        synthesizer = SpeechSynthesizer(
            model=model,
            voice=voice,
            # format=AudioFormat.PCM_16000HZ_MONO_16BIT,  
            format=AudioFormat.WAV_16000HZ_MONO_16BIT,
            callback=callback,
        )
        def _synth():
            synthesizer.streaming_call(text) 
            # 结束流式语音合成
            synthesizer.streaming_complete()
            # play_queue.put("silent")
        
        from threading import Thread
        thread = Thread(target=_synth)
        thread.start()