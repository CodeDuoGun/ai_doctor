import asyncio
import time
import uuid
import os 
import traceback
from utils.audio_split import convert_audio2mel
from config import config
from room_business.business_base import BusinessBase
from log import logger
from room_business.room_data import Speech

tick_time = 0.005

class BusinessInteraction(BusinessBase):
    def __init__(self):
        super().__init__()
        self.send_video = False
    async def check_command(self, tts_text):
        if (tts_text == "subdone" or tts_text == "done") and self.send_video:
            self.parent.runtime_data.send_command = True
            while True:
                if self.parent.runtime_data.send_command_queue.qsize() > 0 and self.parent.runtime_data.send_command:
                    await self.parent.runtime_data.send_command_queue.get()
                    self.parent.runtime_data.send_command = False
                    await self.parent.runtime_data.nlp_callback_queue.put({"command": "play"})
                await asyncio.sleep(0.1)
                if self.parent.runtime_data.video_play :
                    self.send_video = False
                    self.parent.runtime_data.video_play = False
                    self.parent.runtime_data.send_command = False
                    break

    async def run(self):
        global time_clock
        time_clock = 0
        logger.debug('reset time_clock')
        try:
            while self.parent.is_valid():
                # 需要发送到app的消息处理
                # begin = time.time()
                logger.debug(f"{self.parent.init_data.live_id}  run_time_cal begin")
                # 接收到弹幕消息处理
                # while not self.parent.runtime_data.queue_25d_audio.empty(): #如果再来一段怎么办？？？
                if not self.parent.is_valid():
                    break
                #TODO @txueduo ,已知bug: 打断后，还会有上一次的tts音频再次放入队列，
                audio_info = await self.parent.runtime_data.queue_25d_audio.get()
                if not audio_info:
                    logger.debug(f'room {self.parent.init_data.live_id} 收到空音频退出。')
                    break
                logger.debug(f'room {self.parent.init_data.live_id} 收到对话数据：{audio_info}')
                ori_audio_url, recv_pkl_name, inference_id, recv_priority, end,msg_id,sentence_id, tts_text, node_id, image_url, video_url, msg_inedx, status = audio_info

                if video_url != "":
                    self.send_video = True
                
                await self.check_command(tts_text)
                
                # ori_audio_url c/4uBota25d 
                # TODO:  ori_audio_url没有换背景,这个地方直接传绝对路径过来，理论不做任何处理，qt中去掉这个
                if ori_audio_url != '' and os.path.exists(ori_audio_url):
                    audio_chunks, mels = convert_audio2mel(ori_audio_url, config.split_duration)
                    # 改为可配置的！

                    def clean_old_files(current_index):
                        # 确保 current_index 是一个整数
                        if not current_index.isdigit():
                            return
                        current_index = int(current_index)
                        cache_dir = f"{config.AUDIO_CACHE}/{self.parent.init_data.live_id}"
                        # 列出缓存目录中的所有文件
                        for filename in os.listdir(cache_dir):
                            # 检查文件名格式是否符合 "{file_index}.wav"
                            if filename.endswith(".wav"):
                                try:
                                    # 提取文件名中的 file_index 并转换为整数
                                    file_index = int(filename.split(".")[0])
                                    
                                    # 比较并删除小于当前 index 的文件
                                    if file_index < current_index:
                                        file_path = os.path.join(cache_dir, filename)
                                        os.remove(file_path)
                                        # print(f"Deleted old file: {file_path}")
                                except ValueError:
                                    # 如果文件名不符合数字格式，跳过该文件
                                    print(f"Skipping non-standard file: {filename}")
                    cur_index = ori_audio_url.split("/")[-1].replace(".wav","")
                    clean_old_files(cur_index)
                    if 'silent_mute' not in ori_audio_url:
                        os.remove(ori_audio_url)
                    total_id = str(uuid.uuid1())
                    total_id = ''
                    speeches = []
                    logger.debug(f'audio length:{len(audio_chunks)}, mel length: {len(mels)}')
                    min_count = min(len(audio_chunks),len(mels))
                    audio_id = os.path.basename(ori_audio_url).replace(".wav","")
                    for i in range(min_count):
                        new_speech = Speech()
                        if i == 0 and "silent" not in inference_id:
                            new_speech.first_speak = True 
                        new_speech.speech_id = f'lrid_{audio_id}_{recv_priority}_{i}'
                        new_speech.msg_id = msg_id
                        new_speech.sentence_id = sentence_id
                        new_speech.parent_speech_id = 0
                        new_speech.danmu_response_speech_id = new_speech.speech_id
                        new_speech.video_model_info.pkl_url = recv_pkl_name

                        new_speech.node_id = node_id
                        new_speech.image_url = image_url
                        new_speech.send_video_url = video_url
                        new_speech.msg_inedx = msg_inedx
                        new_speech.status = status
                        # new_speech.duration = config.split_duration
                        
                        new_speech.inference_id = inference_id
                        new_speech.audio_url = ori_audio_url
                        new_speech.is_end = end
                        new_speech.tts_text = tts_text
                        ## 槽位0:原始音频，槽位1:mel 供wav2lip用，槽位3:留着给wav2lip的帧使用，
                        # audio_segments_np = np.frombuffer(audio_chunks[i], dtype=np.int8)
                        # logger.error(new_speech.speech_id)
                        # logger.error(audio_segments_np)
                        new_speech.push_unit_data = [audio_chunks[i], mels[i]]
                        new_speech.duration = len(mels[i])/config.fps
                        if i == 0:
                            new_speech.priority = recv_priority
                            new_speech.first_tts  = True
                        else:
                            new_speech.priority = 1
                        
                        if i == min_count-1:
                            new_speech.last_tts  = True
                        logger.debug(f'插入话术:{new_speech.priority} {new_speech.speech_id} {new_speech.tts_text} {new_speech.msg_id} first_speak:{new_speech.first_speak}')
                        speeches.append(new_speech)
                        logger.debug(f'将弹幕话术推送到任务队列[{new_speech.speech_id} {new_speech.inference_id}], first_speak:{new_speech.first_speak}')
                    
                    if len(speeches) > 0 and self.parent.runtime_data.queue_interaction:
                        logger.info(f"{self.parent.init_data.live_id} put speeches:{len(speeches)}, min_count:{min_count}, audio_id:{audio_id}")
                        await self.parent.runtime_data.queue_interaction.put(speeches)
                    # await asyncio.sleep(tick_time)
                elif os.path.exists(ori_audio_url):
                    logger.warning(f' room {self.parent.init_data.live_id} {ori_audio_url} 音频不存在，此时收到的是bota_server的指令 ')
                else:
                    #TODO 下面的代码似乎不需要，我们应该不允许这样的case发生。
                    new_speech = Speech()
                    new_speech.speech_id = str(uuid.uuid1())
                    if self.parent.runtime_data.queue_interaction:
                        logger.info(f"{self.parent.init_data.live_id} put one speech into queue_interaction ")
                        await self.parent.runtime_data.queue_interaction.put(new_speech)

                logger.debug(f"{self.parent.init_data.live_id}  run_time_cal end")
                await asyncio.sleep(tick_time)
        except Exception:
            logger.error(f"interaction err:{traceback.format_exc()}")
        logger.info(f"stopping_room {self.parent.init_data.live_id}  exit interaction logic")
