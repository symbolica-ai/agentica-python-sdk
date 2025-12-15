import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator

import pytest
import pytest_asyncio
from agentica_internal.multiplex_protocol import (
    MultiplexClientMessage,
    MultiplexServerMessage,
    multiplex_to_json,
)
from agentica_internal.session_manager_messages import CreateAgentRequest
from aiohttp import WSMsgType, web

import agentica.client_session_manager.client_session_manager as csm_mod
from agentica.sdk_logging import enable_sdk_logging


@pytest.fixture
def global_csm(live_transport, testing_id_issuer):
    csm = csm_mod.ClientSessionManager(
        testing_id_issuer,
        base_url=live_transport["base_url"],
    )
    # Swap the global used by the decorator path
    old = getattr(csm_mod, "_GLOBAL_CSM", None)
    csm_mod._GLOBAL_CSM = csm
    try:
        yield csm
    finally:
        # cleanup
        csm_mod._GLOBAL_CSM = old


@pytest.fixture
def sdk_logging():
    reset = enable_sdk_logging()
    yield
    reset()


class TestingIDIssuer:
    def __init__(self) -> None:
        self._counter = 0

    def __call__(self) -> str:
        self._counter += 1
        return f"id_{self._counter}"


@pytest.fixture
def testing_id_issuer() -> TestingIDIssuer:
    return TestingIDIssuer()


@dataclass
class WSMessagePipe:
    to_client: asyncio.Queue[bytes]
    to_server: asyncio.Queue[bytes]


LOG: bool = False


@pytest_asyncio.fixture
async def live_transport_logging():
    global LOG
    LOG = True
    yield
    LOG = False


@pytest_asyncio.fixture
async def live_transport() -> AsyncIterator[dict[str, Any]]:
    # Minimal real HTTP server that asserts path and payload shape
    created_uids: list[str] = []
    pipes_by_uid: dict[str, WSMessagePipe] = {}
    conns: dict[str, Any] = {}

    async def handle_create_function(request: web.Request) -> web.Response:
        """Clean HTTP POST handler for agent creation"""
        payload = await request.json()
        print(f"[HTTP] POST {request.path} payload={payload}") if LOG else None

        # Validate required keys
        for key in CreateAgentRequest.__dataclass_fields__.keys():
            assert key in payload

        uid = payload.get("name", "uid-1")
        created_uids.append(uid)

        return web.Response(text=uid, content_type="text/plain")

    async def handle_health(request: web.Request) -> web.Response:
        """Clean health check handler"""
        print(f"[HTTP] GET {request.path}") if LOG else None
        return web.Response(text="", content_type="text/plain")

    async def handle_websocket(request: web.Request) -> web.WebSocketResponse:
        """Clean WebSocket handler with proper frame handling"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        uid = request.match_info['uid']  # Clean URL parameter extraction
        print(f"[WS] accept uid={uid}") if LOG else None

        conns[uid] = ws
        pipe = pipes_by_uid.setdefault(uid, WSMessagePipe(asyncio.Queue(), asyncio.Queue()))

        try:
            async for msg in ws:  # Proper async message loop
                if msg.type == WSMsgType.BINARY:
                    print(f"[WS] recv uid={uid} bytes={len(msg.data)}") if LOG else None
                    await pipe.to_server.put(msg.data)
                elif msg.type == WSMsgType.ERROR:
                    print(f"[WS] error uid={uid}: {ws.exception()}") if LOG else None
                    break
        finally:
            conns.pop(uid, None)

        return ws

    # Clean route setup with aiohttp
    app = web.Application()
    app.router.add_post("/agent/create", handle_create_function)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/socket", handle_websocket)  # URL parameter!

    # Clean async server startup using aiohttp's built-in approach
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    # Get port the server bound to
    port = next(socket_info[1] for socket_info in runner.addresses if socket_info)

    async def server_send(uid: str, msg: MultiplexServerMessage) -> None:
        ws = conns[uid]
        data = multiplex_to_json(msg)
        print(f"[WS] send uid={uid} bytes={len(data)} {msg=}") if LOG else None
        await ws.send_bytes(data)

    async def server_recv(uid: str) -> MultiplexClientMessage:
        raw = await pipes_by_uid[uid].to_server.get()
        from agentica_internal.multiplex_protocol import multiplex_from_json

        assert isinstance(raw, bytes)
        msg = multiplex_from_json(raw)
        assert msg is not None
        print(f"[WS] server_recv uid={uid} {msg=}") if LOG else None
        return msg  # type: ignore[return-value]

    async def wait_for_conn(uid: str, timeout_s: float = 2.0) -> None:
        end = asyncio.get_event_loop().time() + timeout_s
        while uid not in conns:
            if asyncio.get_event_loop().time() > end:
                raise TimeoutError(f"WebSocket connection for {uid} not established")
            await asyncio.sleep(0.01)

    try:
        yield {
            "base_url": f"http://127.0.0.1:{port}",
            "created_uids": created_uids,
            "server_send": server_send,
            "server_recv": server_recv,
            "wait_for_conn": wait_for_conn,
        }
    finally:
        await runner.cleanup()  # Clean async cleanup
