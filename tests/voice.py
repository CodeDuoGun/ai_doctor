import dashscope
from dashscope.audio.tts_v2 import *
import threading
dashscope.api_key = "sk-325f79c2085d481c9a8a3e625c9a698b"

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
    
    
synthesis_text_to_speech_and_play_by_streaming_mode("你是谁发的")