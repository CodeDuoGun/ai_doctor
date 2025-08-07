#!/usr/bin/env python
import traceback
import os
import asyncio
from http import HTTPStatus
from websockets.asyncio.server import serve
from chat_manager.chat_instance import ChatInstance
from aiohttp import web
from utils.log import logger
import aiohttp_cors
import json
from aiortc import RTCPeerConnection, RTCSessionDescription
from human_player import HumanPlayer
from utils.tool import pcm_to_wav
global nerfreal
chat_instance = ChatInstance()
ws_map = {}

def health_check(connection, request):
    if request.path == "/health":
        return connection.respond(HTTPStatus.OK, "OK\n")


async def echo(websocket):
    global chat_instance
    global ws_map
    ws_map["test"] = websocket 
    async for message in websocket:
        logger.info(f"收到消息{type(message)}")
        # 暂时处理文本类型
        if isinstance(message, str):
            message = json.loads(message)
            asr_result = message.get("msg", "")
            await chat_instance.runtime_data.asr_msg_queue.put(asr_result)
        elif isinstance(message, bytes):
            # asr queue -> nlp
            print(len(message), message[:100])
            # cache_path = "data/cache/audio/media.wav"
            # pcm_to_wav(message, cache_path)
            await chat_instance.runtime_data.queue_vad.put(message)
            # asr_result = "北京天气"
            # await websocket.send(json.dumps({"asr": asr_result}, ensure_ascii=False))


async def websocket_server():
    async with serve(echo, "localhost", 8765, process_request=health_check) as server:
        logger.info(f"ws server success started")
        await server.serve_forever()

pcs = set()
from lipreal import LipReal

async def stop_chatting(request):
    global chat_instance
    try:
        params = await request.json()
        ws = ws_map[params.get("uid", "test")]
        # close ws
        await ws.close()
        stop_res = await chat_instance.stop_chat()
        if stop_res!= 200:
            logger.error(f"stop chat error for {traceback.format_exc()}")
        # close chat
    except Exception:
        logger.error(f"stop chat error for {traceback.format_exc()}")


async def start_chatting(request):
    global chat_instance
    global nerfreal
    try:
        params = await request.json()
        ws = ws_map[params.get("uid", "test")]
        strat_res = await chat_instance.start_chat(ws, nerfreal)
        logger.info(f"chat_instance 初始化完成 {strat_res}")
        res = {"status": strat_res}
        return web.json_response(res)
    except Exception:
        logger.error(traceback.format_exc())
        return web.json_response({"status": "failed"})
    # assert strat_res == "ok"


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    player = HumanPlayer(nerfreal)
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)

    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    #return jsonify({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )

async def aiohttp_server():
    async def handle(request):
        return web.Response(text="Hello from aiohttp")

    app = web.Application()
    app.router.add_get('/', handle)
    app.router.add_static('/',path='web')
    app.router.add_post('/offer', offer)
    app.router.add_post('/start', start_chatting) # 先调用这个
    app.router.add_post('/stop', stop_chatting) # 先调用这个
    # 静态资源映射 /static -> ./web
    app.router.add_static('/static', path=os.path.join(os.getcwd(), 'web'), name='static')
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
    logger.info(f"http server success started")
    

async def run_server():
    # 创建两个任务并行执行
    ws_task = asyncio.create_task(websocket_server())
    aiohttp_task = asyncio.create_task(aiohttp_server())

    # 等待两个任务完成（实际上会一直运行）
    await asyncio.gather(ws_task, aiohttp_task)


if __name__ == "__main__":
    nerfreal = LipReal()
    # import multiprocessing
    # multiprocessing.set_start_method("fork")  # 在 Mac 上推荐使用 fork
    asyncio.run(run_server())