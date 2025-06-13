#!/usr/bin/env python

import asyncio
from http import HTTPStatus
from websockets.asyncio.server import serve
from aiohttp import web
from utils.log import logger
import aiohttp_cors


def health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(HTTPStatus.OK, "OK\n")

async def echo(websocket):
    async for message in websocket:
        await websocket.send(message)

async def websocket_server():
    async with serve(echo, "localhost", 8765, process_request=health_check) as server:
        logger.info(f"ws server success started")
        await server.serve_forever()


async def aiohttp_server():
    async def handle(request):
        return web.Response(text="Hello from aiohttp")

    app = web.Application()
    app.router.add_get('/', handle)
    app.router.add_static('/',path='web')
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
    asyncio.run(run_server())