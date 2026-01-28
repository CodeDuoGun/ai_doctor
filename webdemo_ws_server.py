"""
Browser mic -> WebSocket audio chunks -> backend realtime ASR -> WebSocket text events.

This demo supports microphone from ANY device that can open the page (PC/phone),
because audio is captured in the browser and streamed to backend via WebSocket.

Requirements:
  - websockets
  - dashscope (and env DASHSCOPE_API_KEY)

Run:
  export DASHSCOPE_API_KEY="sk-xxx"
  python webdemo_ws_server.py

Then open:
  http://localhost:7861/
"""

from __future__ import annotations

import asyncio
import json
import os
import ssl
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import websockets
from websockets.server import WebSocketServerProtocol

from asr.RealtimeFunASR import RealtimeFunASR
from gradio_asr_demo import postprocess_asr


HTTP_HOST = "0.0.0.0"
HTTP_PORT = 7861
WS_HOST = "0.0.0.0"
WS_PORT = 7862

STATIC_DIR = Path(__file__).parent / "webdemo_static"


class _StaticHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)


def _start_http_server() -> None:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((HTTP_HOST, HTTP_PORT), _StaticHandler)
    certfile = os.getenv("SSL_CERTFILE")
    keyfile = os.getenv("SSL_KEYFILE")
    if certfile and keyfile:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile=certfile, keyfile=keyfile)
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
    server.serve_forever()


async def _ws_handler(ws: WebSocketServerProtocol) -> None:
    """
    Per-connection session:
      - receive binary PCM16LE frames (16kHz mono)
      - feed into DashScope RealtimeFunASR in a background thread
      - send JSON messages back: {"type":"intermediate"|"final"|"error", "text":"..."}
    """
    loop = asyncio.get_running_loop()
    audio_q: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue(maxsize=200)
    stop_evt = threading.Event()

    async def send_json(msg: dict) -> None:
        await ws.send(json.dumps(msg, ensure_ascii=False))

    def audio_iter():
        # blocking iterator for RealtimeFunASR thread
        while True:
            if stop_evt.is_set():
                return
            chunk = asyncio.run_coroutine_threadsafe(audio_q.get(), loop).result()
            if chunk is None:
                return
            yield chunk

    def on_result(event: str, text: str) -> None:
        if event == "intermediate":
            asyncio.run_coroutine_threadsafe(
                send_json({"type": "intermediate", "text": text}), loop
            )
        elif event == "sentence_end":
            print(f"before postprocess: {text}")
            text = postprocess_asr(text)
            print(f"after postprocess: {text}")
            asyncio.run_coroutine_threadsafe(
                send_json({"type": "final", "text": text}), loop
            )
        elif event == "error":
            asyncio.run_coroutine_threadsafe(
                send_json({"type": "error", "text": text}), loop
            )

    def asr_thread():
        try:
            asr = RealtimeFunASR()
            for _ in asr.run_stream(audio_iter(), on_result=on_result, chunk_interval=0.0):
                # callbacks handle sending
                if stop_evt.is_set():
                    break
        except Exception as e:
            asyncio.run_coroutine_threadsafe(
                send_json({"type": "error", "text": f"{e}"}), loop
            )

    t = threading.Thread(target=asr_thread, daemon=True)
    t.start()

    await send_json({"type": "status", "text": "connected"})

    try:
        async for message in ws:
            if isinstance(message, str):
                # control message
                try:
                    obj = json.loads(message)
                except Exception:
                    continue
                if obj.get("type") == "stop":
                    break
                continue

            # binary audio chunk
            try:
                audio_q.put_nowait(message)
            except asyncio.QueueFull:
                # drop if client sends too fast
                pass
    finally:
        stop_evt.set()
        try:
            audio_q.put_nowait(None)
        except asyncio.QueueFull:
            pass
        # best-effort wait a bit for thread to exit
        for _ in range(10):
            if not t.is_alive():
                break
            time.sleep(0.05)


async def main() -> None:
    # Start HTTP server in background thread
    threading.Thread(target=_start_http_server, daemon=True).start()

    certfile = os.getenv("SSL_CERTFILE")
    keyfile = os.getenv("SSL_KEYFILE")
    ssl_ctx = None
    scheme_http = "http"
    scheme_ws = "ws"
    if certfile and keyfile:
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(certfile=certfile, keyfile=keyfile)
        scheme_http = "https"
        scheme_ws = "wss"

    async with websockets.serve(
        _ws_handler, WS_HOST, WS_PORT, max_size=8 * 1024 * 1024, ssl=ssl_ctx
    ):
        print(f"HTTP: {scheme_http}://localhost:{HTTP_PORT}/")
        print(f"WS:   {scheme_ws}://localhost:{WS_PORT}/ws")
        if scheme_http == "http":
            print("iOS Safari 通常需要 HTTPS 才能使用麦克风。")
            print("可生成自签名证书并启动：")
            print('  openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost"')
            print("  export SSL_CERTFILE=cert.pem SSL_KEYFILE=key.pem")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())

