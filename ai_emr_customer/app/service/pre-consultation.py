import traceback
import subprocess
import pyaudio
import wave
from app.utils.log import logger
from app.tts.cosyvoice import CustomTTS
from app.model.llm.Qwen import QwenLLM
from queue import Queue
from threading import Event
from aiohttp import web
from app.asr.FunASR import AliFunASR
import asyncio
import aiohttp_cors
import json
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosedOK,ConnectionClosedError
from app.service.session import SessionProcessor

    

# 音频参数
FORMAT = pyaudio.paInt16  # 音频格式（16位整数）
CHANNELS = 1              # 单声道
RATE = 16000              # 采样率 16kHz
CHUNK = 1280              # 每个缓冲区的帧数

# 初始化 PyAudio
tick_time = 0.01
audio = pyaudio.PyAudio()

# 打开麦克风输入流
# stream = audio.open(format=FORMAT, channels=CHANNELS,
#                     rate=RATE, input=True,
#                     frames_per_buffer=CHUNK)

quit_event = asyncio.Event()
asr_msg_queue = asyncio.Queue()
nlp_answer_queue = asyncio.Queue()
tts_queue = asyncio.Queue()
# play_queue = asyncio.Queue()
vad_queue = asyncio.Queue()
play_queue = Queue()

tts = CustomTTS()
llm = QwenLLM()
asr = AliFunASR()
sp = SessionProcessor()
history = []
playing = False

# 打开播放流
# play_stream = audio.open(format=FORMAT, channels=CHANNELS,
#                          rate=RATE, output=True,
#                          frames_per_buffer=CHUNK)
async def play_task():
    global playing
    ws = ws_map["test"]
    while not quit_event.is_set():
        # logger.debug(f"play start")
        if play_queue.empty():
            await asyncio.sleep(tick_time)
            continue

        tts_chunk = play_queue.get()
        if tts_chunk is None:
            break
        # if tts_chunk == "silent":
        #     playing = False
        #     await asyncio.sleep(tick_time)
        #     continue
        await ws.send(tts_chunk)
        await asyncio.sleep(tick_time)
    logger.debug("play exist")

async def tts_task():
    loop = asyncio.get_running_loop()
    # print(loop)
    # import pdb
    # pdb.set_trace()
    while not quit_event.is_set():
        # logger.debug("tts start")
        if nlp_answer_queue.empty():
            await asyncio.sleep(tick_time)
            continue  # 如果 1 秒内没有新消息，跳出本轮，继续 while 监听
        nlp_chunk = await nlp_answer_queue.get()
        tts.stream_gen_cosyvoice(nlp_chunk, play_queue, loop)
            # await play_queue.put(tts_chunk)
            # await asyncio.sleep(0)  # 

        await asyncio.sleep(tick_time)
    logger.debug("tts exist")
            

async def nlp_task():
    global history
    ws = ws_map["test"]
    while not quit_event.is_set():
        # logger.debug("nlp start")
        if asr_msg_queue.empty():
            await asyncio.sleep(tick_time)
            continue
        asr_res = await asr_msg_queue.get()
        # 获取问卷形式
        # nlp_res = await sp.handle_user_message("test", asr_res)
        nlp_res, history= await llm.chat_stream(asr_res, history)
        # nlp_res = "行业指令数据在推动行业应用快速落地的场景中发挥着重要的作用。当前的行业指令数据存在着如下的挑战：1.行业数据缺失 2.数据质量参差不齐 3.数据维度相对单一，只存在某些特定场景下的数据"
        print(f"LLM answered>>: {nlp_res}")
        await nlp_answer_queue.put(nlp_res)
        await ws.send(json.dumps({"text": nlp_res}, ensure_ascii=False))        
        await asyncio.sleep(tick_time)
    logger.debug("nlp exist")

def decode_webm_to_pcm(webm_bytes: bytes) -> bytes:
    """将webm字节流转为16kHz的int16裸PCM数据"""
    process = subprocess.Popen(
        ['ffmpeg', '-i', 'pipe:0', '-f', 's16le', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', 'pipe:1'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )
    pcm_data, _ = process.communicate(input=webm_bytes)
    return pcm_data

async def asr_task():
    # global history
    # ws = ws_map["test"]
    while not quit_event.is_set():
        # logger.debug("nlp start")
        if vad_queue.empty():
            await asyncio.sleep(tick_time)
            continue  # 如果 1 秒内没有新消息，跳出本轮，继续 while 监听

        vad_audio = await vad_queue.get()
        if isinstance(vad_audio, str):
            asr_res = vad_audio
        elif isinstance(vad_audio, bytes):
            print(vad_audio[:100])
            pcm_bytes = decode_webm_to_pcm(vad_audio)
            asr_res = asr.recognize(pcm_bytes)
        else:
            asr_res = ""
        await asr_msg_queue.put(asr_res)
        # await ws.send(json.dumps({"text": nlp_res}, ensure_ascii=False))        
        await asyncio.sleep(tick_time)
    logger.debug("nlp exist")

async def close():
    if quit_event.is_set():
        return

    quit_event.set()
    # TODO 清理队列
    # play_stream.stop_stream()
    # play_stream.close()
    # 关闭 PyAudio
    audio.terminate()

    # close queue
    asr_msg_queue.put_nowait(None)
    nlp_answer_queue.put_nowait(None)
    # play_queue.put_nowait(None)
    tts_queue.put_nowait(None)

async def chat():
    global playing
    global history
    try:
        # loop = asyncio.get_running_loop()
        # threading.Thread(target=input_thread, args=(loop, asr_msg_queue), daemon=True).start()
        while not quit_event.is_set():
            if not playing:
                user_text = input("user>>:")
                await asr_msg_queue.put(user_text)
                playing = True
            # 后端获取麦克风输入
            # user_text = stream.read(CHUNK)
            await asyncio.sleep(tick_time)
        logger.debug(f"chatting exist")
        # await close()
    except Exception:
        logger.debug(traceback.format_exc())

ws_map = {}
# TODO: 把一次问诊存储为一个唯一会话记录 uid_sessionid_msgid, session_id 可以是时间戳
async def websocket_handler(websocket, path=None):
    logger.debug("🌐 WebSocket client connected")
    try:
        ws_map["test"] = websocket 
        async for message in websocket:
            try:
                print(f"**msg: {type(message)}")
                if isinstance(message, str):
                    data = json.loads(message)
                    if "text" or "audio" in data:
                        user_text = data.get("text") or data.get("audio")
                        await vad_queue.put(user_text)
                    elif "tongue_face_image" in data:
                        tongue_face_imgs = data["tongue_face_image"]
                    elif "check_img" in data:
                        check_imgs = data["check_img"]                 
                        
                elif isinstance(message, bytes):
                    user_text = message
                    await vad_queue.put(user_text)
                else:
                    logger.warning(f"收到无效消息: {message}")
            except Exception as e:
                logger.debug(f"消息处理失败: {e}")
    except (ConnectionClosedOK,  ConnectionClosedError):
        logger.debug("websocket client disconnected")
    except Exception as e:
        logger.debug(f"websocket handler debug: {e}")
    finally:
        await websocket.close()
        await websocket.wait_closed()
        logger.debug("🛑 websocket 服务已关闭")
        await close()

async def start_chat():
    status = start_tasks_in_main_thread()
    return status

async def thread_function(lp):
    # loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(loop)
    asyncio.create_task(asr_task())
    asyncio.create_task(nlp_task())
    asyncio.create_task(tts_task())
    asyncio.create_task(play_task())
    print(f"后台任务启动成功")
    return "ok"


def start_tasks_in_main_thread():
    loop = asyncio.get_event_loop()  # 获取主线程的事件循环
    # await self.thread_function()  
    asyncio.create_task(thread_function(loop))# 调度任务，不阻塞主线程
    return "ok"


async def start_chatting(request):
    # global chat_instance
    # global nerfreal
    try:
        params = await request.json()
        # ws = ws_map[params.get("uid", "test")]
        strat_res = await start_chat()
        logger.debug(f"chat_instance 初始化完成 {strat_res}")
        res = {"status": strat_res}
        return web.json_response(res)
    except exception:
        logger.debug(traceback.format_exc())
        return web.json_response({"status": "failed"})

async def websocket_server():
    async with serve(websocket_handler, "localhost", 8765) as server:
        logger.debug(f"ws server success started")
        await server.serve_forever()


async def aiohttp_server():
    async def handle(request):
        return web.response(text="hello from aiohttp")

    app = web.application()
    app.router.add_get('/', handle)
    app.router.add_static('/',path='app/web')
    # app.router.add_post('/offer', offer)
    app.router.add_post('/start', start_chatting) # 先调用这个
    # app.router.add_post('/stop', stop_chatting) # 先调用这个
    # Configure default CORS settings.
    cors = aiohttp_cors.setup(app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })
    # Configure CORS on all routes.
    for route in list(app.router.routes()):
        cors.add(route)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 8080)
    await site.start()
    logger.debug(f"http server success started")

# 主函数：统一启动所有任务
async def main():
    try:
        # 创建两个任务并行执行
        ws_task = asyncio.create_task(websocket_server())
        aiohttp_task = asyncio.create_task(aiohttp_server())

        # 等待两个任务完成（实际上会一直运行）
        await asyncio.gather(ws_task, aiohttp_task)
       
    except KeyboardInterrupt:
        print("⛔ 程序终止中...")
        await close()



# 启动方式
if __name__ == "__main__":
    asyncio.run(main())