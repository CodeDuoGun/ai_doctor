import time
import numpy as np
import soundfile as sf
import resampy
import asyncio
import edge_tts

from typing import Iterator
import requests

import queue
from queue import Queue
from io import BytesIO
from threading import Thread, Event
import dashscope
from dashscope.audio.tts_v2 import *
import threading
dashscope.api_key = "sk-325f79c2085d481c9a8a3e625c9a698b"
from enum import Enum

class State(Enum):
    RUNNING=0
    PAUSE=1

class BaseTTS:
    def __init__(self, fps, parent):
        self.parent = parent

        self.fps = fps # 20 ms per frame
        self.sample_rate = 16000
        self.chunk = self.sample_rate // self.fps # 320 samples per chunk (20ms * 16000 / 1000)
        self.input_stream = BytesIO()

        self.msgqueue = Queue()
        self.state = State.RUNNING

    def pause_talk(self):
        self.msgqueue.queue.clear()
        self.state = State.PAUSE

    def put_msg_txt(self,msg): 
        self.msgqueue.put(msg)

    def render(self,quit_event):
        process_thread = Thread(target=self.process_tts, args=(quit_event,))
        process_thread.start()
    
    def process_tts(self,quit_event):        
        while not quit_event.is_set():
            try:
                msg = self.msgqueue.get(block=True, timeout=1)
                self.state=State.RUNNING
            except queue.Empty:
                continue
            self.txt_to_audio(msg)
        print('ttsreal thread stop')
    
    def txt_to_audio(self,msg):
        pass
    

###########################################################################################
class EdgeTTS(BaseTTS):
    def txt_to_audio(self,msg):
        voicename = "zh-CN-YunxiaNeural"
        text = msg
        t = time.time()
        asyncio.new_event_loop().run_until_complete(self.__main(voicename,text))
        print(f'-------edge tts time:{time.time()-t:.4f}s')

        self.input_stream.seek(0)
        stream = self.__create_bytes_stream(self.input_stream)
        streamlen = stream.shape[0]
        idx=0
        while streamlen >= self.chunk and self.state==State.RUNNING:
            self.parent.put_audio_frame(stream[idx:idx+self.chunk])
            streamlen -= self.chunk
            idx += self.chunk
        #if streamlen>0:  #skip last frame(not 20ms)
        #    self.queue.put(stream[idx:])
        self.input_stream.seek(0)
        self.input_stream.truncate() 

    def __create_bytes_stream(self,byte_stream):
        #byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        print(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            print(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            print(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream
    
    async def __main(self,voicename: str, text: str):
        communicate = edge_tts.Communicate(text, voicename)

        #with open(OUTPUT_FILE, "wb") as file:
        first = True
        async for chunk in communicate.stream():
            if first:
                first = False
            if chunk["type"] == "audio" and self.state==State.RUNNING:
                #self.push_audio(chunk["data"])
                self.input_stream.write(chunk["data"])
                #file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                pass

###########################################################################################
class VoitsTTS(BaseTTS):
    def txt_to_audio(self,msg): 
        self.stream_tts(
            self.gpt_sovits(
                msg,
                self.opt.REF_FILE,  
                self.opt.REF_TEXT,
                "zh", #en args.language,
                self.opt.TTS_SERVER, #"http://127.0.0.1:5000", #args.server_url,
            )
        )

    def gpt_sovits(self, text, reffile, reftext,language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        req={
            'text':text,
            'text_lang':language,
            'ref_audio_path':reffile,
            'prompt_text':reftext,
            'prompt_lang':language,
            'media_type':'raw',
            'streaming_mode':True
        }
        # req["text"] = text
        # req["text_language"] = language
        # req["character"] = character
        # req["emotion"] = emotion
        # #req["stream_chunk_size"] = stream_chunk_size  # you can reduce it to get faster response, but degrade quality
        # req["streaming_mode"] = True
        res = requests.post(
            f"{server_url}/tts",
            json=req,
            stream=True,
        )
        end = time.perf_counter()
        print(f"gpt_sovits Time to make POST: {end-start}s")

        if res.status_code != 200:
            print("Error:", res.text)
            return
            
        first = True
        for chunk in res.iter_content(chunk_size=32000): # 1280 32K*20ms*2
            if first:
                end = time.perf_counter()
                print(f"gpt_sovits Time to first chunk: {end-start}s")
                first = False
            if chunk and self.state==State.RUNNING:
                yield chunk

        print("gpt_sovits response.elapsed:", res.elapsed)

    def stream_tts(self,audio_stream):

        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=32000, sr_new=self.sample_rate)
                #byte_stream=BytesIO(buffer)
                #stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk])
                    streamlen -= self.chunk
                    idx += self.chunk 

###########################################################################################
class XTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt,parent)
        self.speaker = self.get_speaker(opt.REF_FILE, opt.TTS_SERVER)

    def txt_to_audio(self,msg): 
        self.stream_tts(
            self.xtts(
                msg,
                self.speaker,
                "zh-cn", #en args.language,
                self.opt.TTS_SERVER, #"http://localhost:9000", #args.server_url,
                "20" #args.stream_chunk_size
            )
        )

    def get_speaker(self,ref_audio,server_url):
        files = {"wav_file": ("reference.wav", open(ref_audio, "rb"))}
        response = requests.post(f"{server_url}/clone_speaker", files=files)
        return response.json()

    def xtts(self,text, speaker, language, server_url, stream_chunk_size) -> Iterator[bytes]:
        start = time.perf_counter()
        speaker["text"] = text
        speaker["language"] = language
        speaker["stream_chunk_size"] = stream_chunk_size  # you can reduce it to get faster response, but degrade quality
        res = requests.post(
            f"{server_url}/tts_stream",
            json=speaker,
            stream=True,
        )
        end = time.perf_counter()
        print(f"xtts Time to make POST: {end-start}s")

        if res.status_code != 200:
            print("Error:", res.text)
            return

        first = True
        for chunk in res.iter_content(chunk_size=960): #24K*20ms*2
            if first:
                end = time.perf_counter()
                print(f"xtts Time to first chunk: {end-start}s")
                first = False
            if chunk:
                yield chunk

        print("xtts response.elapsed:", res.elapsed)
    
    def stream_tts(self,audio_stream):
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=24000, sr_new=self.sample_rate)
                #byte_stream=BytesIO(buffer)
                #stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk])
                    streamlen -= self.chunk
                    idx += self.chunk 

class CustomTTS(BaseTTS):
    def txt_to_audio(self,msg): 
        self.stream_tts(
            self.synthesis_text_to_speech_and_play_by_streaming_mode(
                msg,
            )
        )
    
    def stream_tts(self, audio_stream):
        streamlen = len(audio_stream)
        idx = 0
        while streamlen >= self.chunk and self.state==State.RUNNING:
            self.parent.put_audio_frame(audio_stream[idx:idx+self.chunk])
            streamlen -= self.chunk
            idx += self.chunk
        # for chunk in audio_stream:
        #     if chunk is not None and len(chunk)>0:          
        #         stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
        #         stream = resampy.resample(x=stream, sr_orig=32000, sr_new=self.sample_rate)
        #         #byte_stream=BytesIO(buffer)
        #         #stream = self.__create_bytes_stream(byte_stream)
        #         streamlen = stream.shape[0]
        #         idx=0
        #         while streamlen >= self.chunk:
        #             self.parent.put_audio_frame(stream[idx:idx+self.chunk])
        #             streamlen -= self.chunk
        #             idx += self.chunk 

    def synthesis_text_to_speech_and_play_by_streaming_mode(self, text, ref_voice:str="longyuan"):
        '''
        Synthesize speech with given text by streaming mode, async call and play the synthesized audio in real-time.
        for more information, please refer to https://help.aliyun.com/document_detail/2712523.html
        '''
        # Define a callback to handle the result
        complete_event = threading.Event()

        class Callback(ResultCallback):
            def on_open(self):
                # self.file = open('result.mp3', 'wb')
                self.audio_stream = b''
                print('websocket is open.')

            def on_complete(self):
                print('speech synthesis task complete successfully.')
                complete_event.set()
                # self.file.close()

            def on_error(self, message: str):
                print(f'speech synthesis task failed, {message}')

            def on_close(self):
                print('websocket is closed.')
                # self.file.close()

            def on_event(self, message):
                # print(f'recv speech synthsis message {message}')
                pass

            def on_data(self, data: bytes) -> None:
                # send to player
                self.audio_stream += data
                # save audio to file
                # self.file.write(data)

        # Call the speech synthesizer callback
        synthesizer_callback = Callback()

        # Initialize the speech synthesizer
        # you can customize the synthesis parameters, like voice, format, sample_rate or other parameters
        speech_synthesizer = SpeechSynthesizer(model='cosyvoice-v1',
                                            voice=ref_voice,
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
        