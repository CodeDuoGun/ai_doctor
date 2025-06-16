#!/usr/bin/env python

import asyncio
from http import HTTPStatus
from websockets.asyncio.server import serve
from aiohttp import web
from utils.log import logger
import aiohttp_cors
import json
from aiortc import RTCPeerConnection, RTCSessionDescription
from human_player import HumanPlayer
from llm.LLM import LLM
global nerfreal

def health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(HTTPStatus.OK, "OK\n")

def llm_response(message):
    qwen = LLM().init_model('Qwen', model_path="")
    response = qwen.chat(message)
    return response

async def echo(websocket):
    async for message in websocket:
        logger.info(f"收到消息{type(message)}")
        # 暂时处理文本类型
        if isinstance(message, str):
            res=llm_response(message)                           
            nerfreal.put_msg_txt(res)
        await websocket.send(message)

async def websocket_server():
    async with serve(echo, "localhost", 8765, process_request=health_check) as server:
        logger.info(f"ws server success started")
        await server.serve_forever()

pcs = set()
from lipreal import LipReal


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
    asyncio.run(run_server())