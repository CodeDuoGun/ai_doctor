import dashscope
from dashscope.audio.tts_v2 import *
import threading
import os
import base64
import numpy as np
import pyaudio
import time


dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "sk-325f79c2085d481c9a8a3e625c9a698b")
#
def synthesis_text_to_speech_and_play_by_streaming_mode(text):
    '''
    Synthesize speech with given text by streaming mode, async call and play the synthesized audio in real-time.
    for more information, please refer to https://help.aliyun.com/document_detail/2712523.html
    '''
    # Define a callback to handle the result
    complete_event = threading.Event()

    class Callback(ResultCallback):
        def on_open(self):
            self.file = open('result.mp3', 'wb')
            self.audio_stream = b''
            print('websocket is open.')

        def on_complete(self):
            print('speech synthesis task complete successfully.')
            complete_event.set()
            self.file.close()

        def on_error(self, message: str):
            print(f'speech synthesis task failed, {message}')

        def on_close(self):
            print('websocket is closed.')
            self.file.close()

        def on_event(self, message):
            # print(f'recv speech synthsis message {message}')
            pass

        def on_data(self, data: bytes) -> None:
            # send to player
            self.audio_stream += data
            # save audio to file
            self.file.write(data)

    # Call the speech synthesizer callback
    synthesizer_callback = Callback()

    # Initialize the speech synthesizer
    # you can customize the synthesis parameters, like voice, format, sample_rate or other parameters
    speech_synthesizer = SpeechSynthesizer(model='cosyvoice-v1',
                                           voice='loongstella',
                                           callback=synthesizer_callback)
    # 非流式
    speech_synthesizer.call(text)
    # 流式
    # speech_synthesizer.streaming_call(text)
    # speech_synthesizer.streaming_complete()
    print('Synthesized text: {}'.format(text))
    complete_event.wait() 
    print('[Metric] requestId: {}, first package delay ms: {}'.format(
        speech_synthesizer.get_last_request_id(),
        speech_synthesizer.get_first_package_delay()))
    return synthesizer_callback.audio_stream

import os
import requests
import dashscope

def request_qwen_tts():
    text = "那我来给大家推荐一款T恤，这款呢真的是超级好看，这个颜色呢很显气质，而且呢也是搭配的绝佳单品，大家可以闭眼入，真的是非常好看，对身材的包容性也很好，不管啥身材的宝宝呢，穿上去都是很好看的。推荐宝宝们下单哦。"
    response = dashscope.audio.qwen_tts.SpeechSynthesizer.call(
        model="qwen-tts",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        text=text,
        voice="Cherry",
        # stream=True
    )
    audio_url = response.output.audio["url"]
    save_path = "downloaded_audio.wav"  # 自定义保存路径

    try:
        response = requests.get(audio_url)
        response.raise_for_status()  # 检查请求是否成功
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"音频文件已保存至：{save_path}")
    except Exception as e:
        print(f"下载失败：{str(e)}")

def request_qwen_stream_tts():
    p = pyaudio.PyAudio()
    # 创建音频流
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=24000,
                    output=True)

    text = "那我来给大家推荐一款T恤，这款呢真的是超级好看，这个颜色呢很显气质，而且呢也是搭配的绝佳单品，大家可以闭眼入，真的是非常好看，对身材的包容性也很好，不管啥身材的宝宝呢，穿上去都是很好看的。推荐宝宝们下单哦。"
    response = dashscope.audio.qwen_tts.SpeechSynthesizer.call(
        model="qwen-tts",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        text=text,
        voice="Cherry",
        stream=True
    )
    for chunk in response:
        audio_string = chunk["output"]["audio"]["data"]
        wav_bytes = base64.b64decode(audio_string)
        audio_np = np.frombuffer(wav_bytes, dtype=np.int16)
        # 直接播放音频数据
        stream.write(audio_np.tobytes())

    time.sleep(0.8)
    # 清理资源
    stream.stop_stream()
    stream.close()
    p.terminate() 

if __name__ == "__main__":
    request_qwen_tts()
# synthesis_text_to_speech_and_play_by_streaming_mode("你是谁发的")