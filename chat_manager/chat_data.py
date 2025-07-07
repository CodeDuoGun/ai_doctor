import asyncio

class ChatRuntimeData:
    def __init__(self):
        self.asr_audio_queue = asyncio.Queue()
        
        # 音频队列2.5D
        # self.queue_vad = asyncio.Queue()
        self.queue_25d_audio = asyncio.Queue() # tts
        self.asr_msg_queue = asyncio.Queue() # asr
        self.nlp_answer_queue = asyncio.Queue() # nlp
    