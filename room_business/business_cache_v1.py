import asyncio
import os
import traceback

from config import config
from room_business.business_base import BusinessBase
from room_business.room_data import *
from utils.audio_split import convert_audio2mel

tick_time = 0.005
MAX_SILENT_COUNT = 1000

#本类主要做两件事
#1.从queue_interaction队列中拿数据放到list中
#2.把静默放入队列，加入静默，不超过1s的数据就行。
#有个问题，高优先级的数据来了，就要清除后面的数据
#   1.队列只有一个打断，
#   2.队列有多个打断，---不可能发生，这意味着，极短的时间（几十毫秒之内）来了两次打断
#   3.队列有说话来了。
class BusinessCache(BusinessBase):
    def __init__(self):
        super().__init__()
        self._is_running_nlp = False
        self.next_speech = Speech()
        self.silent_audio = []
        self.silent_mel = []
        self.silent_ids = []
        self.silent_id_index = 0

    def gen_silent_mel(self):
        self.silent_audio, self.silent_mel = convert_audio2mel(
            config.silent_audio_url, config.split_duration
        )

        logger.debug("gen_silent_mel")
        logger.debug(len(self.silent_audio))
        logger.debug(len(self.silent_mel))

    def gen_silent_ids(self):
        for i in range(MAX_SILENT_COUNT):
            self.silent_ids.append(f"silent_{i}")

    def _build_conversation_nodes(self,interaction_speechs):
        logger.info(
            f"{self.parent.init_data.live_id} 2  交互话术准备进入队列:[{interaction_speechs[0].speech_id}] "
        )
        # new_node = None
        #只接受list
        new_nodes = []
        for speech_data in interaction_speechs:
            new_node = Node(speech_data)
            new_nodes.append(new_node)
        # for speech_data in interaction_speechs:
        #     if new_node is None:
        #         new_node = Node(speech_data)
        #         end_node = new_node
        # #         logger.info(
        # #             f"{self.parent.init_data.live_id} 3  交互话术准备进入队列:[{speech_data.speech_id}]"
        # # )
        #     else:
        #         end_node.insert_node(Node(speech_data))
        #         end_node = end_node.tail
        #         logger.info(
        #             f"{self.parent.init_data.live_id} 3  交互话术准备进入队列:[{speech_data.speech_id}]"
        # )
        logger.info(
            f"{self.parent.init_data.live_id} 新互动话术已经转为队列节点"
        )
        return new_nodes

    async def _find_first_unplayed_node(self,new_speech_nodes):
        if len(new_speech_nodes) == 0 :
            logger.warning(f"{self.parent.init_data.live_id} 收到空打断！！！")
            return
        await self.parent.runtime_data.speech_linked_list.insert_nodes(new_speech_nodes)
            
    async def run(self):
        self.gen_silent_mel()
        self.gen_silent_ids()
        while self.parent.is_valid():
            try:
                # ----begin----交互话术打断处理
                # 如果没有在生成普通话术，且接收到交互话术分段，则进行打断处理   话术分段直接在交互逻辑处理好
                logger.debug(f"{self.parent.init_data.live_id}  run_time_cal begin")
                if not self.parent.runtime_data.queue_interaction.empty():
                    #有新对话的回答，准备插入
                    logger.debug(f"{self.parent.init_data.live_id}  收到打断，不知道是静默打断还是文本内容 当前队列 {self.parent.runtime_data.speech_linked_list.count()}")
                    interaction_speechs = await self.parent.runtime_data.queue_interaction.get()
                    if (
                        interaction_speechs is None or not self.parent.is_valid()
                    ):  # 其实可以不要？？？？
                        break
                    #把音频转为node
                    logger.debug(f"{self.parent.init_data.live_id}  打断数据一共{len(interaction_speechs)}")
                    new_speech_nodes = self._build_conversation_nodes(interaction_speechs)
                    #找到位置准备插入
                    found_node = await self._find_first_unplayed_node(new_speech_nodes)
                    # if new_speech_nodes.data.priority == 2:
                    #     # 真打断，后续直接丢弃
                    #     find_node.tail = None
                    # found_node.insert_node(new_speech_nodes)

                    self.parent.runtime_data.debug_speech_status(
                        f"{self.parent.init_data.live_id} cache: insert interaction"
                    )
                elif (
                    not self.parent.runtime_data.speech_linked_list.more_than_expected_count(2) 
                ):
                    #没有新对话的回答，准备插入静默
                    logger.info(f"{self.parent.init_data.live_id}  无话可说，增加静默")
                    # 同步添加静默节点
                    await self.gen_silent()
                    # ----end----（该缓存的都缓存完了，没有新的话术进来，从预设话术取出一条并缓存放入链表）
                else:
                    
                    logger.debug(f"{self.parent.init_data.live_id} 还有很多话没说完 {self.parent.runtime_data.speech_linked_list.count()} {self.parent.runtime_data.speech_linked_list.more_than_expected_count(2)}")
                    self.parent.runtime_data.debug_speech_status(
                        f"{self.parent.init_data.live_id} "
                    )
                logger.debug(f"{self.parent.init_data.live_id}  run_time_cal end")
                await asyncio.sleep(tick_time)
            except:
                logger.info(f"stopping_room {self.parent.init_data.live_id}  {traceback.format_exc()}")
        logger.info(f"stopping_room {self.parent.init_data.live_id}  exit cache playing logic")

    async def gen_silent(self):

        total_id = self.silent_ids[self.silent_id_index]
        self.silent_id_index += 1
        if self.silent_id_index >= MAX_SILENT_COUNT:
            self.silent_id_index = 0
        # new_node = None
        new_codes = []
        logger.debug(f"{self.parent.init_data.live_id} insert 准备插静默队列长度 {self.parent.runtime_data.speech_linked_list.count()} {self.parent.runtime_data.speech_linked_list.more_than_expected_count(2)}")
        for i in range(int(1 / config.split_duration)):
            new_speech = Speech()
            new_speech.speech_id = f"{self.parent.init_data.live_id}_{total_id}_{i}"
            if i != 0:
                new_speech.is_wav2lip = True

            new_speech.audio_url = os.path.abspath(config.silent_audio_url)
            new_speech.push_unit_data = [self.silent_audio[i], self.silent_mel[i]]
            # new_speech.duration = config.split_duration
            new_speech.duration = (
                len(self.silent_mel[i]) / config.fps
            )  # 一个mel是一张图片
            new_speech.video_model_info.pkl_url = self.parent.init_data.pkl_name

            speech_node = Node(new_speech)
            new_codes.append(speech_node)
            logger.info(f"{self.parent.init_data.live_id} 新 [{new_speech.speech_id}]")
            # if new_node:
            #     new_node.insert_node(speech_node)
            # else :
            #     new_node = speech_node
        await self.parent.runtime_data.speech_linked_list.insert_nodes(new_codes)
        logger.debug(f"{self.parent.init_data.live_id} has insert 静默 新队列长度 {self.parent.runtime_data.speech_linked_list.count()}")

