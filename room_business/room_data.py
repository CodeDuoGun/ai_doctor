import asyncio
from dataclasses import dataclass, field
from enum import Enum, unique
import queue
from typing import List

from config import config
from customasr import CustomASR
# from livekit import rtc
from log import logger
from room_business.utils.audio_frame import AudioFrame
from utils.linked_list_v3 import PriorityLinkedList as LinkedList , Node

# Wav2Lip需要的参数数据
class VideoModelInfo:
    def __init__(self):
        self.modeling_url = ""
        self.wav2lip_video_url = ""
        self.mask_url = ""
        self.pkl_url = ""
        self.img_size = ""
        self.frame_config = ""


# TTS需要的参数数据，暂时不需要
class AudioModelInfo:
    pass

def clear(queue):
    queue._queue.clear()

class VideoSection:
    def __init__(self):
        self.can_reverse = False
        # 包含
        self.begin = 0
        # 包含
        self.end = 0

    def count(self):
        return self.end - self.begin + 1


# 话术结构体，现在仅有话术分段
class Speech:
    def __init__(self,priority = 0):
        # 以感叹号!开头表示仅在话术分段使用

        # 用户id
        self.user_id = ""
        self.msg_id = ""
        self.sentence_id = ""
        # 话术
        self.text = ""
        # !已改写话术
        self.gpt_text = ""
        # !父话术id
        self.parent_speech_id = 0
        # 话术id
        self.speech_id = 0
        # !弹幕回复对应话术的id
        self.danmu_response_speech_id = 0
        # 话术优先级,目前仅区分预设话术和交互话术,0为预设话术,1为交互话术
        self.priority = priority
        # !tts原始音频url
        self.audio_url = ""
        # !当前对话属于什么模式
        self.status = 0
        # !视频url
        self.video_url = ""
        # !消息id
        self.node_id = ""
        # 图片url
        self.image_url = ""
        # 视频url
        self.send_video_url = ""
        # 消息id
        self.msg_inedx = 0
        # !用于推流的音频url
        self.push_audio_url = ""
        # !视频用于请求wav2lip的url
        self.wav2lip_video_url = ""
        # !用于推流的图片序列
        self.images = ""
        # !视频mask_url
        self.mask_url = ""
        # !视频pkl_url
        self.pkl_url = ""
        # !视频mask_url
        self.img_size = ""
        # !视频mask_url
        self.modeling_url = ""
        # !视频模型参数
        self.video_model_info: VideoModelInfo = VideoModelInfo()
        # !nlp文本是否写入完毕
        self.is_nlp_over = True  # 本地实时版本NLP不在这里做
        # !是否正在wav2lip
        self.is_wav2lip = False

        # !时长
        self.duration = 0.0
        # !已播放时长
        self.elapse = 0.0
        # !是否已经推送
        self.had_pushed = False
        # !推送时的时间戳
        self.push_time = 0.0
        # !用于推送的最小单元（5帧的数据）
        self.push_unit_data = []
        self.end_with_silent = False

        # 商品id
        self.commodity_id = ""
        # 商品名称
        self.commodity_name = ""
        # 商品价格
        self.commodity_price = 0
        # 佣金
        self.commission_count = 0
        # 商品信息
        self.commodity_info = ""
        # 商品信息
        self.use_rule_info = ""
        # 商品信息
        self.explainText = ""
        # 商品信息
        self.actual_amount = ""

        # 场景id
        self.scene_id = ""
        # 场景名称
        self.scene_name = ""
        # 场景是否显示
        self.scene_show = False

        # -----Unity需要的纯转发数据----begin
        # 背景json
        self.background_json = {}
        # 装饰json
        self.picture_json = []
        # 数字人位置json
        self.person_json = {}
        # -----Unity需要的纯转发数据----end

        # TTS音色ID
        self.voice_speaker_name = ""
        # TTS语速
        self.voice_length = 0

        # 虚拟主播id
        self.video_speaker_id = ""
        # 虚拟主播名字
        self.speaker_name = ""
        # 虚拟主播url
        self.speaker_url = ""

        # UE传来的推理ID
        self.inference_id = ""
        # ue传来的是否是一句话的结尾
        self.is_end = False

        # tts 文本信息
        self.tts_text = ""
        # 标记一段tts开始
        self.first_tts = False
        # 标记一段tts结束
        self.last_tts = False
        
        # 记录首次开始讲话
        self.first_speak = False
        # 记录讲话结束时刻
        self.last_speak = False



    # 打断此话术，即重置部分状态，调整链表中的位置，然后重新改写和生成
    def break_speech(self):
        self.audio_url = ""
        self.video_url = ""
        self.duration = 0
        self.had_pushed = False
        self.elapse = 0.0

    def check_nlp_over(self):
        return True
        # self.is_nlp_over = len(self.gpt_text) >= config.g_tts_min_words_count
        # return self.is_nlp_over


# 直播间初始化结构体
class RoomInitData:
    def __init__(self):
        # 接收到的原始json数据，包含code等
        self.row_json = ""
        # 原始数据解析后的python数据，不含code
        self.data_json_obj = {}
        # 直播间数据
        self.live_data_obj = {}
        # 直播间场景数据
        self.scenes_data_obj = []
        # 直播间预设话术
        self.preset_speech: list[Speech] = []

        # 直播间id
        self.live_id = ""
        # 直播间交互id
        self.interactive_id = ""
        # 直播间名字
        self.live_name = ""

        # Wav2Lip需要的参数数据键值对 Key:speaker_id Value:VideoModelInfo
        self.video_model_info_map = {}

        self.pkl_name = "" #模特id
        self.background = "" # 背景url
        self.ai_person = "" #uid
        self.pkl_config = None
        self.asr_id = None
@unique
class VADEventType(str, Enum):
    START_OF_SPEECH = "start_of_speech"
    INFERENCE_DONE = "inference_done"
    END_OF_SPEECH = "end_of_speech"
    NORMAL_AUDIO = "audio"


@dataclass
class VADEvent:
    """
    Represents an event detected by the Voice Activity Detector (VAD).
    """

    type: VADEventType
    """Type of the VAD event (e.g., start of speech, end of speech, inference done)."""

    samples_index: int
    """Index of the audio sample where the event occurred, relative to the inference sample rate."""

    timestamp: float
    """Timestamp (in seconds) when the event was fired."""

    speech_duration: float
    """Duration of the detected speech segment in seconds."""

    silence_duration: float
    """Duration of the silence segment preceding or following the speech, in seconds."""

    frames: List[AudioFrame] = field(default_factory=list)
    """
    List of audio frames associated with the speech.

    - For `start_of_speech` events, this contains the audio chunks that triggered the detection.
    - For `inference_done` events, this contains the audio chunks that were processed.
    - For `end_of_speech` events, this contains the complete user speech.
    """

    probability: float = 0.0
    """Probability that speech is present (only for `INFERENCE_DONE` events)."""

    inference_duration: float = 0.0
    """Time taken to perform the inference, in seconds (only for `INFERENCE_DONE` events)."""

    speaking: bool = False
    """Indicates whether speech was detected in the frames."""
class RoomRuntimeData:
    def __init__(self,):
        # 直播间首次TTS是否缓存好
        self.first_tts_ok = False
        # 直播间首次Wav2Lip是否缓存好
        self.first_wav2lip_ok = False
        # 当前预设话术id
        self.current_cache_preset_speech_id = ""
        # 直播话术
        self.queue_live: asyncio.Queue[Speech] = asyncio.Queue()
        # 接收到将要缓存的交互话术
        self.queue_interaction: asyncio.Queue[Speech] = asyncio.Queue()
        # 已缓存或正在缓存的话术分段
        # 结构：|已推送或将要推送的话术（5秒）|已缓存或正在缓存的话术（10秒）|
        self.speech_linked_list: LinkedList = LinkedList()
        # 预设话术索引
        self.speech_normal_index = 0
        # 首次推送（播放）的时间戳
        self.first_push_time = 0
        # 历史播放的话术文本
        self.history_speech_text = []
        # 等待发送至服务器的历史话术文本
        self.history_to_send = asyncio.Queue()
        # 当前播放的节点
        # self.current_node: Node = None
        # 当前商品ID
        # self.current_commodity: str = ""
        # # 当前商品信息
        # self.current_commodity_info = ""
        # # 当前商品信息
        # self.current_priority = 0

        # 当前播放到那块共享内存，目前只会在播放时设定初始值
        self.video_play = False
        self.send_command = False
        self.send_command_queue = asyncio.Queue()
        self.bota_server_callback = asyncio.Queue()
        # ------添加 asr task------
        self.info_sender = None
        self.asr : CustomASR = None
        self.remote_audio_queue = asyncio.Queue()

        self.asr_audio_queue = asyncio.Queue()
        
        # 音频队列2.5D
        self.queue_vad = asyncio.Queue()
        self.queue_25d_audio = asyncio.Queue()
        self.asr_msg_queue = asyncio.Queue()
        self.nlp_answer_queue = asyncio.Queue()
        # nlp 前端callback
        self.nlp_callback_queue = asyncio.Queue()
        self.asr_callback_queue = asyncio.Queue()  # asr callback
        # webrtc track quue
        self.shm_gap_size = int(
            config.g_cache_unplayed_duration * 25
        )  # 两倍的空间缓冲区？
        # self.audio_queue: queue.Queue = queue.Queue(maxsize=self.shm_gap_size*2)
        # self.video_queue: queue.Queue = queue.Queue(maxsize=self.shm_gap_size)
        # TODO: @txueduo 这个队列需要单独线程中才能有效，暂时废弃
        self.audio_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.shm_gap_size * 2
        )  # queue.Queue = queue.Queue(maxsize=self.shm_gap_size*2)
        self.video_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.shm_gap_size
        )  # queue.Queue = queue.Queue(maxsize=self.shm_gap_size)

        # 打断后，终止上一次nlp session
        self.nlp_status_map = {}

        # debug
        self.debug_last_status_str = ""

        self.pkl_name = None #模特id
        self.background = None # 背景url
        self.ai_person = None #uid
        self.pkl_config = None

        self.room_monitor = None

        self.nlp_interrupt_handler = None


    def init_ai_person(self,pkl_name,background,ai_person,pkl_config):
        self.pkl_name = pkl_name #模特id
        self.background = background # 背景url
        self.ai_person = ai_person #uid
        self.pkl_config = pkl_config
        
    def shutup(self):  # TODO 清除相关队列 静音，清除全部数据，大概率与on_break类似
        # self.clear_current_session()
        self.dismiss_task_queue()


    def dismiss_task_queue(self):
        if self.nlp_answer_queue:
            # self.nlp_answer_queue.clear()
            clear(self.nlp_answer_queue)
            self.nlp_answer_queue.put_nowait(None)
        if self.queue_25d_audio:
            clear(self.queue_25d_audio)
            # self.queue_25d_audio.clear()
            self.queue_25d_audio.put_nowait(None)
        if self.asr_msg_queue:
            clear(self.asr_msg_queue)
            # self.asr_msg_queue.clear()
            self.asr_msg_queue.put_nowait(None)
        if self.nlp_callback_queue:
            clear(self.nlp_callback_queue)
            # self.nlp_callback_queue.clear()
            self.nlp_callback_queue.put_nowait(None)
        if self.send_command_queue:
            clear(self.send_command_queue)
        # if self.audio_queue:
        #     self.audio_queue.join()
        #     self.audio_queue.put_nowait(None)
        # if self.video_queue:

        #     self.video_queue.join()
        #     self.video_queue.put_nowait(None)

    def push_break_audio(self):
        data = {
            "audio_url": "data/silent_mute.wav",
            "pkl_name": self.pkl_name,
            "inference_id": "silent_200ms",
            "priority": 2,
        }
        logger.debug(f"打断音频放入队列！")
        self.queue_25d_audio.put_nowait(
            (
                data["audio_url"],
                data["pkl_name"],
                data["inference_id"],
                data["priority"],
                False,
                "break",
                "break",
                "",
                "",
                "",
                "",
                "",
                0
            )
        )
        self.room_monitor.listening_to() #收到用户说话
        logger.debug(f"打断消息发出放入队列！")
        
    def clear_current_session(self, msg_id: str = "on_clear", last_msg_id: str = ""):
        """
        清理当前会话，以及上一次的nlp, tts
        """
        # 清除tts，清除nlp，清除asr，不同的状况下，清除东西是否有差异？？？
        # TODO: 优化代码, 清理逻辑保证自上而下
        logger.info(f"clearing queue of {msg_id} last_msg_id：{last_msg_id} {self.nlp_interrupt_handler}")
        if self.nlp_interrupt_handler is not None:
            self.nlp_interrupt_handler.interrupting(msg_id)
            
            #要加一个sleep吧
            
        clear(self.nlp_answer_queue)
        clear(self.asr_msg_queue)
        clear(self.queue_25d_audio)
        clear(self.nlp_callback_queue)
        clear(self.asr_callback_queue)
        self.push_break_audio()
        # if self.nlp_interrupt_handler is not None:
        #     self.nlp_interrupt_handler.say_directly(config.BREAK_SENTENCE)
        logger.info(f"cleared queue of {msg_id} {self.nlp_interrupt_handler}")
        # tts的音频清理按照旧的priority 来处理



    # 获得下一个未缓存的话术分段
    def get_next_uncached(self):
        find_index = 0
        node = self.speech_linked_list.head
        while node is not None:
            if node.data.audio_url == "":
                # logger.info(find_index)
                return node
            node = node.tail
            find_index += 1
        return None

    # 获得下一个未推送的话术分段
    def get_next_unpushed(self):
        node = self.speech_linked_list.head
        while node is not None:
            if not node.data.had_pushed:
                return node
            node = node.tail
        return None

    # 获取已缓存的时长
    def get_unpushed_duration(self):
        duration = 0
        node = self.speech_linked_list.head
        while node is not None:
            if not node.data.had_pushed:
                if node.data.duration > 0:
                    duration += node.data.duration
                else:
                    break
            node = node.tail
        return duration

    def get_pushed_duration(self):
        duration = 0
        node = self.speech_linked_list.head
        while node is not None:
            if node.data.had_pushed:
                duration += node.data.duration
            else:
                break
            node = node.tail
        return duration

    def get_played_duration(self):
        duration = 0
        node = self.speech_linked_list.head
        while node is not None:
            if node.data.elapse > 0:
                duration += node.data.elapse
            else:
                break
            node = node.tail
        return duration

    def get_unplayed_duration(self):
        played_duration = 0
        pushed_duration = 0
        node = self.speech_linked_list.head
        while node is not None:
            if node.data.had_pushed:
                pushed_duration += node.data.duration
                if node.data.elapse > 0:
                    played_duration += node.data.elapse
            else:
                break
            node = node.tail
        return pushed_duration - played_duration

    # 当前是否有高优先级话术
    def has_high_priority_speech(self):
        node = self.speech_linked_list.head
        while node is not None:
            if node.data.priority > 0:
                return True
            node = node.tail
        return False

    # 获得历史已经说过的话术用于发送至gpt
    def get_history_speech_text(self):
        if len(self.history_speech_text) > 5:
            self.history_speech_text = self.history_speech_text[-5:]
        return self.history_speech_text

    # 添加历史说过的话术文本
    def append_history_speech_text(self, text: str):
        self.history_speech_text.append(text)

    def debug_speech_status(self, title="", need_debug=False):
        if not need_debug:
            return need_debug
        status_str = "status    :"
        priority_str = "priority:"
        play_str = "play    :"
        id_str = "id      :"
        node = self.speech_linked_list.head
        while node is not None:
            id_str += node.data.speech_id
            id_str += " "
            if node.data.had_pushed:
                status_str += "pushed "
            else:
                if node.data.video_url != "":
                    status_str += "wav2lipOk "
                else:
                    if node.data.audio_url != "":
                        status_str += "ttsOk "
                    else:
                        if node.data.is_nlp_over:
                            status_str += "nlpOk "
                        else:
                            status_str += "isNlping "
            priority_str += str(node.data.priority)
            if node.data.duration > 0:
                play_str += f"{node.data.elapse}/{node.data.duration} "
            else:
                play_str += str("0/0")

            node = node.tail
        need_log = True
        # if only_change_log:
        #     need_log = False
        #     logs = status_str + priority_str + play_str + id_str
        #     if self.debug_last_status_str != logs:
        #         self.debug_last_status_str = logs
        #         need_log = True

        if need_log:
            logger.debug(f"---{title}---")
            logger.debug(status_str)
            logger.debug(priority_str)
            logger.debug(play_str)
            logger.debug(id_str)

    # 直播间实例


# Wav2lip_2.5D 参数数据
class VideoUEInfo:
    def __init__(self):
        # 音频地址 -- 本机交互
        self.audio_url = ""
        # 使用模特名称
        self.pkl_name = ""
        # id
        self.inference_id = ""
