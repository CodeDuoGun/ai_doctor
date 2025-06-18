# For prerequisites running the following sample, visit https://help.aliyun.com/document_detail/xxxxx.html

import os
import requests
from http import HTTPStatus

import dashscope
from dashscope.audio.asr import *

# 若没有将API Key配置到环境变量中，需将your-api-key替换为自己的API Key
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "sk-325f79c2085d481c9a8a3e625c9a698b")


# class QwenASR(BaseAsr):
class QwenASR():
    def __init__(self):
        pass
    
    def recognize(self, audio_bytes):

        class Callback(TranslationRecognizerCallback):
            def on_open(self) -> None:
                print("TranslationRecognizerCallback open.")

            def on_close(self) -> None:
                print("TranslationRecognizerCallback close.")

            def on_event(
                    self,
                    request_id,
                    transcription_result: TranscriptionResult,
                    translation_result: TranslationResult,
                    usage,
            ) -> None:
                print("request id: ", request_id)
                print("usage: ", usage)
                if translation_result is not None:
                    print(
                        "translation_languages: ",
                        translation_result.get_language_list(),
                    )
                    english_translation = translation_result.get_translation("en")
                    print("sentence id: ", english_translation.sentence_id)
                    print("translate to english: ", english_translation.text)
                if transcription_result is not None:
                    print("sentence id: ", transcription_result.sentence_id)
                    print("transcription: ", transcription_result.text)

            def on_error(self, message) -> None:
                print('error: {}'.format(message))

            def on_complete(self) -> None:
                print('TranslationRecognizerCallback complete')


        callback = Callback()

        translator = TranslationRecognizerChat(
            model="gummy-chat-v1",
            format="wav",
            sample_rate=16000,
            callback=callback,
        )

        translator.start()

        try:
            audio_data: bytes = None
            translator.send_audio_frame(audio_bytes)
            # idx = 0
            # while True:
            #     audio_data = audio_bytes[idx:idx+128000]
            #     if not audio_data:
            #         break
            #     else:
            #         if translator.send_audio_frame(audio_data):
            #             print("send audio frame success")
            #         else:
            #             print("sentence end, stop sending")
            #             break
            #     idx+=1
        except Exception as e:
            raise e

        translator.stop()
        return  

# with open("downloaded_audio.wav", "rb") as f:
#     audio_bytes = f.read()

#     print(audio_bytes[:100])

#     trans = QwenASR()
#     trans.recognize(audio_bytes)