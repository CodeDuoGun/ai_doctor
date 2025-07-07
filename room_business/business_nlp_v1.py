import asyncio
import json
import string
import time
import traceback
from datetime import datetime
import aiohttp
import uuid
import shutil
import os
import wave
from config import config
from log import logger
from room_business.business_base import BusinessBase
from proto_data.message_pb2 import BotaRequest, BotaMessageType, BotaResponse
import websockets

tick_time = 0.01

class BusinessNLP(BusinessBase):
    def __init__(self):
        super().__init__()
        self.default_msg_falg = True
        
    def init(self):
        self.websocket = None
        self.is_running = False
        self.uid = self.parent.init_data.ai_person
        self.deviceid = f"H5-{self.parent.init_data.live_id}",
        self.queue_25d_audio = self.parent.runtime_data.queue_25d_audio
        self.asr_msg_queue = self.parent.runtime_data.asr_msg_queue
        self.pkl_name = self.parent.runtime_data.pkl_name
        self.send_index = 1
        self.priority = 1
        self.cache_dir = f"{config.AUDIO_CACHE}/{self.parent.init_data.live_id}"
        self.user_question = None
        self.client_id = str(uuid.uuid4())
        self.max_files = config.MAX_CAHCED_AUDIO
        self.msg_id = ""
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.current_message_id = ""
        self.speech_cnt = 0
        self.file_index = 0
    
    def is_punctuation(char):
        return char in string.punctuation

    async def get_nlp_resp(self, msg):
        nlp_info_map = self.build_nlp_info_map(msg)
        if nlp_info_map["text"] == "default":
            nlpmsg_chunk = "你好，有什么可以帮助你的吗"
            nlp_info_map["text"] = nlpmsg_chunk
            await asyncio.sleep(2)
            await self.say_hello(nlp_info_map)
        elif "你好小优" in nlp_info_map["text"]:
            self.parent.runtime_data.clear_current_session(nlp_info_map["msg_id"])
        elif "小优静音" in nlp_info_map["text"]:
            self.parent.runtime_data.clear_current_session(nlp_info_map["msg_id"])
            # 清空队列，插入break
            nlp_info_map["text"] = "嗯，好的"
            await self.say_hello(nlp_info_map)

    async def say_directly(self, msg):
        logger.info(f"说预置话术 {self.parent.init_data.live_id}  {msg}")
        msg_id = "predefined"+str(self.speech_cnt)
        data = {
                        "room_id": self.parent.init_data.live_id,
                        "msg_id": msg_id,
                        "sentence_id": msg_id,
                        "asr_text": msg,
                        "last_msg_id": self.current_message_id
        }
        nlp_info_map = self.build_nlp_info_map(data)
        await self.say_hello(nlp_info_map)
        self.current_message_id = msg_id


    async def put_nlp2tts(self, nlp_info: dict, answer_id: int, nlpmsg_chunk: str):
        """
        如果发生打断，清空后，这里也不应该继续放入上次的结果
        """
        if nlpmsg_chunk:

            data = {
                "room_id": nlp_info["room_id"],
                "msg_id": nlp_info["msg_id"],
                "sentence_id": f'{nlp_info["sentence_id"]}_{answer_id}',
                "text": nlpmsg_chunk,
                "last_msg_id": nlp_info["last_msg_id"]
            }
            logger.info(f"{self.parent.init_data.live_id} {data['sentence_id']} nlpmsg_chunk:{nlpmsg_chunk}")
            self.parent.runtime_data.nlp_answer_queue.put_nowait(data)
        logger.info(f"{self.parent.init_data.live_id} recv nlp answer and send to tts text is  {nlpmsg_chunk}")
            
    async def run(self):
        try:
            self.init()
            while self.parent.is_valid():
                # 获取
                logger.debug("NLP start")
                if self.default_msg_falg:
                    logger.info(f"nlp_队列收到消息 default")
                    data = {
                        "room_id": self.parent.init_data.live_id,
                        "msg_id": "default",
                        "sentence_id": "default",
                        "asr_text": "default",
                        "last_msg_id": "default"
                    }
                    await self.get_nlp_resp(data)
                    self.default_msg_falg = False
                    continue

                if not self.parent.is_valid():
                    self.is_running = False
                    break
                await self.connect()
                
                await asyncio.sleep(tick_time)
        except Exception:
            logger.error(f"nlp err :{traceback.format_exc()}")
        finally:
            await self.close()

        logger.info(f"stopping_room {self.parent.init_data.live_id}  exit nlp logic")

    async def connect(self):
        if self.websocket:
            return self.websocket
        self.is_running = True
        try:

            # async with websockets.connect(config.bota_url) as websocket:

            self.websocket = await websockets.connect(config.bota_url)
            initial_message = {"type": "initial", "client_id": self.client_id, "timestamp": time.time()}
            await self.websocket.send(json.dumps(initial_message))
            logger.debug(f"{self.parent.init_data.live_id} Sent initial message with client_id: {self.client_id}")
            await asyncio.gather(
                self.send_heartbeat_message(),
                self.send_message(),
                self.callback_node(),
                self.receive_message(),
            )
        except websockets.ConnectionClosed:
            # print(f"[{get_current_time()}]Connection closed by server.")
            # TODO 关闭直播间  当流式服务链接失败后暂时没有任何处理流程，需要后续补充
            pass
        except Exception as e:
            # print(f"[{get_current_time()}] An error occurred: {e}")
            # TODO 关闭直播间  当流式服务链接失败后暂时没有任何处理流程，需要后续补充
            pass

        return self.websocket
    
    async def receive_message(self):
        try:
            async for message in self.websocket:
                if isinstance(message,str) and (message.upper() == 'PING' or message.upper() == 'PONG'):
                    # logger.debug(f"Received: {message}")
                    pass
                else:
                    response = BotaResponse()
                    # 确保传入的是字节序列
                    if isinstance(message, bytes):
                        response.ParseFromString(message)
                    content = "No content"
                    tts_bytes = b''
                    logger.debug(f"[{self.parent.init_data.live_id}] received engine_returned_reponse: {response.content} and audio len {len(response.data)}")
                    msg_index = response.reqMsgIndex
                    # if len(response.data)>0 :
                    content = response.content # NLP的文本
                    tts_bytes = response.data   # TTS的结果PCM   ------wav
                    self.user_question["msg_id"] = response.reqMsgId[:-1] + str(msg_index)
                    file_index = self.file_index
                    self.file_index = self.file_index + 1
                    node_id = response.node_id
                    image_url = response.imageUrl
                    video_url = response.videoUrl
                    masg_index = response.msgIndex
                    status = response.states
                    logger.debug(f"收到botaserver数据：node_id：{node_id}， content: {content}, tts_bytes{len(tts_bytes)},video_url:{video_url},image_url:{image_url}")
                    if self.msg_id != response.reqMsgId:
                        await self.pcm_transition_wav(tts_bytes, content, file_index, node_id, image_url, video_url, masg_index, status)

        except websockets.ConnectionClosed:
            # TODO 如果发现流式服务连接失败，暂时先发送一句话术用于提示，后续处理，保证优雅退出并且通知客户端
            content = "服务器链接失败~请刷新页面"
            tts_bytes = b''
            file_index = 99999
            node_id = ""
            image_url = ""
            video_url = ""

            await self.pcm_transition_wav(tts_bytes, content, file_index,)
            logger.warning("{self.parent.init_data.live_id} Connection closed, stopping receive messages.")
        except Exception:
            logger.error(f"nlp receive_message err:{traceback.format_exc()}")
    async def say_hello(self, user_question):
        try:
            local_file = f'{self.cache_dir}/{user_question["room_id"]}_{user_question["msg_id"]}_{user_question["sentence_id"]}.wav'
            json_data = {
                "text": user_question["text"],
                "uid": self.uid,
                "audioType": "wav",
                "textID": "12123131231",
                "device_sn": "25d云渲染",
            }
            headers = {"Content-Type": "application/json"}
            t0 = time.time()
            async with aiohttp.ClientSession() as sess:
                async with sess.post(
                    config.tts_url, json=json_data, headers=headers
                ) as resp:
                    logger.info(f"tts resp cost:{time.time() - t0}")
                    total_len = 0
                    with open(local_file, "wb") as f:
                        while True:
                            chunk = await resp.content.read(1024)  # 每次读取 1024 字节
                            if not chunk:
                                break
                            total_len = total_len + len(chunk)
                            f.write(chunk)
            logger.info(
                f"{user_question['room_id']} tts {local_file} save audio cost:{time.time() - t0} total audio bytes: {total_len}"
            )
            self.queue_25d_audio.put_nowait(
                (
                    local_file,
                    self.pkl_name,
                    "infer",
                    self.priority,
                    False,
                    user_question["msg_id"],
                    user_question["sentence_id"],
                    user_question["text"],
                    "",
                    "", 
                    "",
                    0,
                    0
                )
            )
        except Exception:
            logger.error(
                f"TTS error of file {local_file} of question {user_question} error:{traceback.format_exc()}"
            )

    def interrupting(self,current_message_id):
        logger.debug(f'收到打断 即将通知对话服务打断{current_message_id}')
        if self.current_message_id == current_message_id:
            logger.debug(f'当前大脑正在思考和处理{current_message_id}，无需打断')
            return 
        logger.debug(f'收到打断 即将通知对话服务打断{current_message_id}')
        asyncio.run_coroutine_threadsafe(self.__interrupting(),self.parent.loop)
    async def __interrupting(self):
        await self.send(self.build_ws_break())
        self.send_index += 1
        await asyncio.sleep(0.01)

    async def send_message(self):
        while self.is_running or self.parent.is_valid():
            try:
                
                msg = await self.asr_msg_queue.get()
                if not msg:
                    break
                if msg["asr_text"] == "BREAK":
                    self.msg_id = msg["msg_id"]
                    await self.send(self.build_ws_break())
                else:
                    # logger.debug(f' {asr_result["asr_text"]}')
                    self.current_message_id = msg["msg_id"]
                    user_question = self.build_nlp_info_map(msg)
                    self.user_question = user_question
                    await self.send(self.build_ws_break()) #可能会连续两次打断，似乎目前server能处理所以暂时不改
                    self.send_index += 1
                    await self.send(self.build_ws_request(user_question["text"]))
                    self.send_index += 1
                
            except websockets.ConnectionClosed:
                logger.warning(f"Connection closed, stopping send messages.") 
                # 关闭直播间
        logger.info(f'任务退出，不再需要给llm发请求了。')

    async def send_heartbeat_message(self):
        while self.is_running or self.parent.is_valid():
            try:
                heartbeat_message = "PING"  # 心跳消息格式
                await self.websocket.send(heartbeat_message)
                logger.debug(f"Sent heartbeat: {heartbeat_message}")
                await asyncio.sleep(3)
            except websockets.ConnectionClosed:
                # 如果断开连接了：
                # 1.退出了，不需要重连
                # if not self.parent.is_valid():
                self.is_running
                break    
        logger.info(f'{self.parent.init_data.live_id} 不需要发心跳了，退出nlp心跳任务')

    def build_ws_request(self, query):
        query_request = BotaRequest()
        query_request.msgIndex = self.send_index
        query_request.msgType = BotaMessageType.BMT_ASR
        query_request.msgId = self.user_question["msg_id"]
        query_request.deviceSn = self.parent.init_data.live_id
        query_request.uid = self.uid
        # query_request.uid = "lRrUlLLf1uX8Iz0p"
        query_request.content = query
        logger.debug(f'{self.parent.init_data.live_id} send_query_2_llm {query} ')
        return query_request
    
    def build_ws_break(self):
        query_request = BotaRequest()
        query_request.msgIndex = self.send_index
        query_request.msgType = BotaMessageType.BMT_BREAK
        query_request.msgId = self.client_id
        query_request.deviceSn = self.parent.init_data.live_id
        query_request.uid = self.uid
        # query_request.uid = "lRrUlLLf1uX8Iz0p"
        query_request.content = ""
        logger.debug(f"即将发送打断{query_request.msgId} {self.send_index}")
        return query_request

    def build_nlp_info_map(self, asr_result):
        self.speech_cnt = self.speech_cnt + 1
        return {
            "room_id": asr_result["room_id"],
            "msg_id": asr_result["msg_id"],
            "sentence_id": asr_result["sentence_id"],
            "text": asr_result["asr_text"],
            "last_msg_id": asr_result["last_msg_id"]
        }
    
    def build_ws_callback(self,node_id):
        query_request = BotaRequest()
        query_request.msgIndex = self.send_index
        query_request.msgType = BotaMessageType.BMT_UPDATE
        query_request.msgId = self.client_id
        query_request.deviceSn = self.parent.init_data.live_id
        query_request.uid = self.uid
        # query_request.uid = "lRrUlLLf1uX8Iz0p"
        query_request.content = node_id
        logger.debug(f"发送回调：{node_id}")
        return query_request

    async def send(self, request):
        if self.websocket:
            logger.debug(f'向大脑发送数据：[{request}] [{request.msgType}] [{request.content}]')
            await self.websocket.send(request.SerializeToString())

    async def callback_node(self):
        while self.is_running:
            node_id = await self.parent.runtime_data.bota_server_callback.get()
            await self.send(self.build_ws_callback(node_id))

    async def close(self):

        if self.is_running:
            self.is_running = False
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        if self.websocket:
            await self.websocket.close()
    
    def is_connected(self):
        if self.websocket:
            return self.websocket.is_client
        return False
    
    async def pcm_transition_wav(self,pc_data, text, file_index, node_id, image_url, video_url, masg_index, status, sample_rate=16000, channels=1):
        sample_width = 2
        # 保存前先清理多余文件
        # self.clear_audio()
        local_file = f'{self.cache_dir}/{str(file_index).zfill(10)}.wav'
        # if pc_data == b'':
        with wave.open(local_file, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pc_data)

        self.queue_25d_audio.put_nowait(
        (
            local_file,
            self.pkl_name,
            "infer",
            self.priority,
            False,
            self.user_question["msg_id"],
            self.user_question["sentence_id"],
            text,
            node_id, 
            image_url, 
            video_url,
            masg_index,
            status
            )
        )

    def clear_audio(self):
        # 清除音频文件  --->   50个文件，多一个会删除最先保存的一个，最大保留50个音频文件
        files = [f for f in os.listdir(self.cache_dir) if os.path.isfile(os.path.join(self.cache_dir, f))]
        
        files.sort()
        num_files_to_keep = max(0, len(files) - self.max_files)
        for i in range(num_files_to_keep):
            file_path = os.path.join(self.cache_dir, files[i])
            os.remove(file_path)
    #     files = [os.path.join(self.cache_dir, f) for f in os.listdir(self.cache_dir) if os.path.isfile(os.path.join(self.cache_dir, f))]

    # # 按文件的修改时间进行排序（最早的在前）
    #     files.sort(key=os.path.getmtime)

        
    #     num_files_to_keep = max(0, len(files) - self.max_files)
    #     while len(files) > num_files_to_keep:
    #         oldest_file = files.pop(0)  # 删除最早的文件（最先保存的）
    #         os.remove(oldest_file)
