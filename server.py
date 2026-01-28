"""
Async websocket server that captures microphone audio, streams it to
Aliyun realtime ASR, and pushes intermediate results to all clients.

Environment variables:
    ALIYUN_ASR_TOKEN   - required
    ALIYUN_ASR_APPKEY  - required
    ASR_SERVER_HOST    - default "0.0.0.0"
    ASR_SERVER_PORT    - default 50007
"""
from websockets.legacy.server import WebSocketServerProtocol
import json

import asyncio
import logging
import os
import threading
from typing import Set

import pyaudio
import websockets
from websockets.server import WebSocketServerProtocol

from asr.AliRealtime_ASR import AliRealtimeASR
from dotenv import load_dotenv

load_dotenv()
HOST = os.getenv("ASR_SERVER_HOST", "0.0.0.0")
PORT = int(os.getenv("ASR_SERVER_PORT", "50007"))
TOKEN = os.getenv("ALIYUN_ASR_TOKEN", "e57b77e9e370497e9c05e9cb11d37912")
APPKEY = os.getenv("ALIYUN_ASR_APPKEY")

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes
CHUNK_FRAMES = 320  # 320 frames * 2 bytes = 640 bytes per chunk

clients: Set[WebSocketServerProtocol] = set[WebSocketServerProtocol]()
clients_lock = asyncio.Lock()
logger = logging.getLogger("asr_server")
logging.basicConfig(level=logging.INFO)


def microphone_chunks():
    """
    Generator yielding raw PCM chunks from default microphone.
    """
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pa.get_format_from_width(SAMPLE_WIDTH),
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_FRAMES,
    )
    logger.info("Microphone stream opened")
    try:
        while True:
            yield stream.read(CHUNK_FRAMES, exception_on_overflow=False)
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        logger.info("Microphone stream closed")


async def broadcast(text: str) -> None:
    """
    Send text to all connected websocket clients; drop disconnected sockets.
    """
    async with clients_lock:
        dead = []
        for ws in clients:
            try:
                await ws.send(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)
            try:
                await ws.close()
            except Exception:
                pass


def start_asr_thread(loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> None:
    """
    Spin up blocking ASR in a thread; push results into asyncio queue.
    """
    if not TOKEN or not APPKEY:
        raise RuntimeError("ALIYUN_ASR_TOKEN and ALIYUN_ASR_APPKEY must be set")

    def on_asr_result(event: str, payload: dict) -> None:
        # Push into asyncio loop thread-safely
        loop.call_soon_threadsafe(
            queue.put_nowait,
            (event, payload),
        )

    def worker():
        asr = AliRealtimeASR(token=TOKEN, appkey=APPKEY)
        for audio in asr.run_stream(microphone_chunks(), on_result=on_asr_result):
            print(audio)
            pass  # events are handled via callback queue

    threading.Thread(target=worker, daemon=True).start()


async def dispatcher(queue: asyncio.Queue) -> None:
    """
    Consume ASR events and broadcast to websocket clients.
    """
    while True:
        event, payload = await queue.get()

        if isinstance(payload,list):
            payload = json.loads(payload)
        if event == "intermediate":
            await broadcast(f"[partial] {payload}")
        elif event == "sentence_end":
            await broadcast(f"[final] {payload}")
        elif event == "error":
            logger.error("ASR error: %s", payload)
            await broadcast(f"[error] {payload}")


async def ws_handler(websocket: WebSocketServerProtocol) -> None:
    """
    Register client and keep the connection open.
    """
    async with clients_lock:
        clients.add(websocket)
    logger.info("Websocket client connected: %s", websocket.remote_address)
    try:
        await websocket.send("Connected to ASR websocket server.")
        await websocket.wait_closed()
    finally:
        async with clients_lock:
            clients.discard(websocket)
        logger.info("Websocket client disconnected: %s", websocket.remote_address)


async def main_async() -> None:
    """
    Launch ASR worker thread and websocket server (asyncio).
    """
    loop = asyncio.get_running_loop()
    event_queue: asyncio.Queue = asyncio.Queue()
    start_asr_thread(loop, event_queue)

    async with websockets.serve(ws_handler, HOST, PORT):
        logger.info("ASR websocket server listening on %s:%s", HOST, PORT)
        await dispatcher(event_queue)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
