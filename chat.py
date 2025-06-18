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
from asr.FunASR import AliFunASR
global nerfreal
asr = AliFunASR()

def health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(HTTPStatus.OK, "OK\n")

# TODO: need a queue to deal
async def llm_response(message):
    qwen = LLM().init_model('Qwen', model_path="")
    response = qwen.chat(message)
    return response

import wave

def pcm_to_wav(pcm_data, wav_file, channels=1, sample_rate=16000, bits_per_sample=16):
    wav = wave.open(wav_file, 'wb')
    wav.setnchannels(channels)
    wav.setsampwidth(2)
    wav.setframerate(sample_rate)
    wav.writeframes(pcm_data)
    wav.close()


# TODO: need a queue to deal
async def request_asr(audio_bytes):
    # cache
    cache_path = "data/cache/audio/audio.wav"
    # pcm_to_wav(audio_bytes, cache_path)
    with open(cache_path, "wb") as fp:
        fp.write(audio_bytes)
    return asr.recognize(audio_bytes)

async def echo(websocket):
    async for message in websocket:
        logger.info(f"收到消息{type(message)}")
        # 暂时处理文本类型
        if isinstance(message, str):
            res = await llm_response(message)
        elif isinstance(message, bytes):
            # asr queue -> nlp
            print(message[:100])
            asr_text = await request_asr(message)
            res = await llm_response(asr_text)
        await websocket.send(res)                     
        nerfreal.put_msg_txt(res)


        # await websocket.send(message)

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