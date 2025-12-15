import asyncio
import json

# Suppress noisy external loggers
import logging
import uuid
from asyncio import CancelledError, Future, Queue, Task, create_task, sleep
from base64 import b64encode
from collections.abc import AsyncGenerator, Coroutine
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import Any, Callable, Never

import dotenv
import httpx
import websockets
from agentica_internal.core.log import LogBase
from agentica_internal.core.result import Result
from agentica_internal.multiplex_protocol import (
    MultiplexClientMessage,
    MultiplexDataMessage,
    MultiplexErrorMessage,
    MultiplexInvocationEventMessage,
    MultiplexInvokeMessage,
    MultiplexNewIIDResponse,
    MultiplexServerMessage,
    multiplex_from_json,
    multiplex_to_json,
)
from agentica_internal.session_manager_messages.session_manager_messages import CreateAgentRequest
from agentica_internal.telemetry.otel import *
from agentica_internal.warpc.worlds.base_world import QUIT
from opentelemetry import trace
from websockets import ClientConnection

from agentica.coming_soon import JsonModeComingSoon
from agentica.common import AgentRole, Chunk, make_role
from agentica.errors import (
    ClientServerOutOfSyncError,
    ConnectionError,
    InternalServerError,
    SDKUnsupportedError,
    ServerError,
    WebSocketConnectionError,
    enrich_error,
)
from agentica.model_notices import print_model_notice
from agentica.platform import AGENTICA_ERROR_LOG_DIR
from agentica.template import maybe_prompt_template
from agentica.version import __version__

from . import at_exit

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)

ERROR_LOG_PATH = AGENTICA_ERROR_LOG_DIR / '{pid_ts}__{pid}__{ts}.log'

_ = dotenv.load_dotenv()


def assert_unrecoverable(
    uid: str | None = None,
    iid: str | None = None,
    session_id: str | None = None,
) -> Never:
    error = ClientServerOutOfSyncError(
        "The server and client are out of sync and cannot be recovered."
    )
    enrich_error(error, uid=uid, iid=iid, session_id=session_id, error_log_path=ERROR_LOG_PATH)
    raise error


def raise_for_status(response: httpx.Response, session_id: str | None = None) -> None:
    if response.status_code == 426:
        try:
            error_data = response.json()
            detail = error_data.get("detail", "SDK version is no longer supported.")
            error = SDKUnsupportedError(detail)
            enrich_error(error, session_id=session_id, error_log_path=ERROR_LOG_PATH)
            raise error
        except (json.JSONDecodeError, KeyError):
            error = SDKUnsupportedError("SDK version is no longer supported.")
            enrich_error(error, session_id=session_id, error_log_path=ERROR_LOG_PATH)
            raise error

    try:
        _ = response.raise_for_status()
    except BaseException as e:
        detail = str(e)
        try:
            detail: str = str(json.loads(response.text)["detail"])
        except:
            detail = response.text

        e.add_note(f"HTTP {response.status_code}: {detail}")
        error = ServerError(detail, http_status_code=response.status_code)
        enrich_error(error, session_id=session_id, error_log_path=ERROR_LOG_PATH)
        raise error


@dataclass
class AgentInvocationHandle:
    send_message: Callable[[bytes], Coroutine[None, None, None]]
    recv_message: Callable[[], Coroutine[None, None, bytes]]
    future_result: Future[Any]
    iid: str
    _span: OSpan = None  # Internal: OTel span to end when invocation completes

    def __post_init__(self) -> None:
        """Set up callback to end span when invocation completes."""
        if self._span is not None and self.future_result is not None:
            # End the span when the invocation completes (success or failure)
            self.future_result.add_done_callback(lambda _: self._end_span())

    def _end_span(self) -> None:
        """End the span, recording success or error status."""
        if self._span is None:
            return
        from opentelemetry.trace import Status, StatusCode

        try:
            # Check if the future has an exception
            if (
                self.future_result.done()
                and not self.future_result.cancelled()
                and (exc := self.future_result.exception())
            ):
                self._span.record_exception(exc)
                self._span.set_status(Status(StatusCode.ERROR, str(exc)))
            else:
                self._span.set_status(Status(StatusCode.OK))
        except Exception:
            # Future might not be done or other issues
            pass
        finally:
            self._span.end()
            self._span = None


INVALID_UID = "INVALID"


class ClientSessionManager(LogBase):
    LOG_TAGS = 'csm'

    _websocket: ClientConnection | None
    _websocket_lock: asyncio.Lock  # Lock to prevent concurrent websocket creation
    _tasks: tuple[Task[None], Task[None]] | None
    _uid_iid_recv_queue: dict[str, dict[str, Queue[bytes]]]
    _uid_iid_future: dict[str, dict[str, Future[Result]]]
    _send_queue: Queue[MultiplexClientMessage]

    _match_iid: dict[str, Future[str]]
    _known_uids: set[str]
    id_issuer: Callable[[], str]
    _base_url: str
    _base_ws: str
    _agentica_api_key: str | None
    _tracer: OTracer
    _running: bool = False

    # OpenTelemetry spans for lifecycle tracking
    _session_span: OSpan  # Entire SDK session lifetime, and the WebSocket connection span

    # Cleanup management
    _cleanup_callback: Callable[[], None] | None

    # Session management
    _client_session_id: str  # Unique ID for this client session

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:2345",
        agentica_api_key: str | None = None,
        tracer: OTracer = None,
        id_issuer: Callable[[], str] = lambda: str(uuid.uuid4()),
        cleanup_callback: Callable[[], None] | None = None,
        *,
        enable_otel_logging: bool = False,
    ):
        super().__init__()
        self.log_name = 'ClientSessionManager'
        self._websocket = None
        self._websocket_lock = asyncio.Lock()
        self._tasks = None
        self._uid_iid_recv_queue = dict()
        self._uid_iid_future = dict()
        self._send_queue = Queue()
        self._match_iid = dict()

        self._known_uids = set()

        assert base_url.startswith("http://") or base_url.startswith("https://"), (
            "base_url must start with http:// or https://"
        )
        self._base_url = base_url
        self._base_ws = base_url.replace("http://", "ws://").replace("https://", "wss://")
        self._agentica_api_key = agentica_api_key

        # Initialize session management
        self._client_session_id = str(uuid.uuid4())

        if not enable_otel_logging:
            tracer = None

        # Initialize OpenTelemetry tracer
        self._tracer = tracer

        # Start session-level span for entire SDK lifetime
        self._session_span = span = None

        # Initialize OpenTelemetry logging (optional)
        if tracer is not None:
            span = span_start(tracer, "sdk.session")
            span_set_attribute(span, "sdk.base_url", base_url)
            span_set_attribute(span, "sdk.version", __version__)
            span_set_attribute(span, "sdk.session_id", self._client_session_id)

            if enable_otel_logging:
                # Generate a unique instance ID for this SDK instance
                instance_id = f"sdk-{id_issuer()[:8]}"
                from agentica_internal.otel_logging import CustomLogFW

                logFW = CustomLogFW(
                    service_name="agentica-sdk",
                    instance_id=instance_id,
                )
                otel_handler = logFW.setup_logging()
                logging.getLogger().addHandler(otel_handler)
                self.log(f"OpenTelemetry logging initialized for SDK instance: {instance_id}")

                span_set_attribute(span, "sdk.instance_id", instance_id)

        # Initialize cleanup management
        self._cleanup_callback = cleanup_callback

        self.id_issuer = id_issuer

        at_exit.register(self.close)

        # Register session
        _ = create_task(self._register_session(), name="CSM.register_session")
        self.log(
            f"Initialized ClientSessionManager with session ID: {self._client_session_id} on base URL: {self._base_url}"
        )

        self.__refresh_when_tags_change__()
        self._running = True

    @asynccontextmanager
    async def _http_client(self) -> AsyncGenerator[httpx.AsyncClient, None]:
        """Create a short-lived HTTP client with proper headers and base URL."""
        headers = {}
        if self._agentica_api_key:
            headers['Authorization'] = f'Bearer {self._agentica_api_key}'
        # Always include client session ID
        headers['X-Client-Session-ID'] = self._client_session_id

        async with httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=None,
        ) as client:
            yield client

    async def _register_session(self) -> None:
        """Register this client session with the server."""
        try:
            async with self._http_client() as client:
                response = await client.post("/session/register")
                if response.is_success:
                    self.log(f"Session {self._client_session_id} registered with server")
                else:
                    error = ConnectionError(
                        f"Failed to register session: HTTP {response.status_code} - {response.text}"
                    )
                    enrich_error(
                        error, session_id=self._client_session_id, error_log_path=ERROR_LOG_PATH
                    )
                    raise error
        except ConnectionError:
            raise
        except Exception as e:
            error = ConnectionError(f"Failed to register session: {e}")
            enrich_error(error, session_id=self._client_session_id, error_log_path=ERROR_LOG_PATH)
            raise error from e

    async def _ws_background_task_writer(self) -> None:
        ws = self._websocket
        assert ws is not None

        with self.log_as('writer') as ctx:
            send_queue = self._send_queue
            try:
                while True:
                    msg = await send_queue.get()
                    self.log(f"Sending {msg=}")
                    ctx.log(' ->', msg)
                    if not self._is_websocket_open():
                        break
                    try:
                        await ws.send(multiplex_to_json(msg))
                    except CancelledError:
                        self.log("Background writer task was cancelled")
                        break
                    except Exception as e:
                        self.log(f"Background writer task encountered an error {e}")
                        await sleep(0.25)
                        continue
            except CancelledError:
                pass

    async def _ws_background_task_reader(self) -> None:
        get_match = self._match_iid.get
        ws = self._websocket
        assert ws is not None

        def process_msg(msg: MultiplexServerMessage, ctx) -> None:
            match msg:
                case MultiplexNewIIDResponse(uid=uid, match_id=match_id, iid=iid):
                    recv_queues = self._uid_iid_recv_queue.get(uid)
                    futures = self._uid_iid_future.get(uid)
                    if recv_queues is None or futures is None:
                        ctx.warn('unknown uid', uid)
                        return
                    if iid not in recv_queues:
                        recv_queues[iid] = Queue()
                    else:
                        ctx.warn('iid queue reuse', iid)
                    if iid not in futures:
                        futures[iid] = Future()
                    else:
                        ctx.warn('iid future reuse', iid)

                    iid_request = get_match(match_id)
                    if iid_request is not None and not iid_request.done():
                        ctx.info('new iid', iid)
                        iid_request.set_result(iid)

                case MultiplexDataMessage(uid=uid, iid=iid, data=data):
                    recv_queues = self._uid_iid_recv_queue.get(uid)
                    if recv_queues is not None and iid in recv_queues:
                        ctx.vars(iid=iid, recv_queue=recv_queues[iid], data=data)
                        recv_queues[iid].put_nowait(data)

                case MultiplexInvocationEventMessage(uid=uid, iid=iid, event=event):
                    recv_queues = self._uid_iid_recv_queue.get(uid)
                    futures = self._uid_iid_future.get(uid)
                    ctx.info('iid =', iid, ', event =', event)
                    if recv_queues is not None and iid in recv_queues and event == 'ERROR':
                        error = InternalServerError(INTERNAL_ERROR_MSG)
                        enrich_error(
                            error,
                            uid=uid,
                            iid=iid,
                            session_id=self._client_session_id,
                            error_log_path=ERROR_LOG_PATH,
                        )
                        if futures is not None and iid in futures:
                            futures[iid].set_exception(error)
                        recv_queues[iid].put_nowait(QUIT)

                case MultiplexErrorMessage(iid=iid, uid=uid) as msg:
                    recv_queues = self._uid_iid_recv_queue.get(uid)
                    futures = self._uid_iid_future.get(uid)
                    ctx.info('iid =', iid, ', error =', msg.error_name)
                    error = msg.to_exception()
                    self.log(f"Prepared exception {error=}")
                    if recv_queues is not None and iid in recv_queues:
                        if futures is not None and iid in futures:
                            future = futures[iid]
                            self.log("Found future for exception")
                            if not future.done():
                                self.log("Future is not done")
                                future.set_exception(error)
                            else:
                                self.log("Future already done")
                    else:
                        self.log("Could not find future for exception")
                        iid_request = get_match(iid)
                        if iid_request is not None:
                            iid_request.set_exception(error)
                        else:
                            self.log(
                                f"Received an error message for an unknown uid/iid: {uid}/{iid}"
                            )

        with self.log_as('reader') as ctx:
            try:
                while True:
                    if not self._is_websocket_open():
                        break
                    msg_bytes = None
                    try:
                        msg_bytes = await ws.recv()
                    except CancelledError:
                        self.log("Background reader task was cancelled")
                        break
                    except Exception as e:
                        self.log(f"Background reader task encountered an error {e}")
                        await sleep(0.25)
                        continue
                    if type(msg_bytes) is not bytes:
                        continue
                    server_msg = None
                    try:
                        server_msg = multiplex_from_json(msg_bytes)
                    except:
                        continue
                    self.log(f"Received {server_msg=}")
                    if isinstance(server_msg, MultiplexServerMessage):
                        ctx.log(' <-', server_msg)
                        process_msg(server_msg, ctx)
            except CancelledError:
                pass

    async def _ensure_websocket_connection(self, uid: str) -> None:
        # Start connection-level span that will last for the WebSocket's lifetime
        # Make it a child of the SESSION span, not the new_agent span
        # This prevents parent-ends-before-child violations

        # Fast path: if websocket already exists, return immediately
        if self._is_websocket_open():
            return

        async with self._websocket_lock:
            if self._is_websocket_open():
                return

            connection_span = None
            websocket_uri = f"{self._base_ws}/socket"
            if tracer := self._tracer:
                self._session_span = span_start(tracer, "sdk.websocket_connection")
                connection_span = self._session_span
                span_set_attribute(connection_span, "agent.uid", uid)
                span_set_attribute(connection_span, "websocket.uri", websocket_uri)

            try:
                self.log(f"Dialing {websocket_uri=} for {uid=}")

                headers = (
                    {"Authorization": f"Bearer {self._agentica_api_key}"}
                    if self._agentica_api_key is not None
                    else {}
                )
                # For some reason websocket lowercases the header names
                headers["x-client-session-id"] = self._client_session_id

                if connection_span:
                    span_inject(connection_span, headers)

                self._websocket = await websockets.connect(
                    websocket_uri,
                    ping_interval=20,
                    ping_timeout=None,
                    max_size=None,
                    additional_headers=headers,
                )
                self.log(f"Spawning background tasks for {uid=}")
                self._tasks = (
                    create_task(self._ws_background_task_reader(), name="CSM.ws_reader"),
                    create_task(self._ws_background_task_writer(), name="CSM.ws_writer"),
                )

                if connection_span:
                    connection_span.set_attribute("websocket.state", "connected")

            except Exception as e:
                span_finish(self._session_span, e)
                raise

    async def new_agent(self, cmar: CreateAgentRequest) -> str:
        # Link to session span to build proper hierarchy
        span = span_start(self._tracer, "sdk.new_agent", context_span=self._session_span)

        try:
            with self.log_as('new_agent') as log_ctx:
                if cmar.json:
                    raise JsonModeComingSoon()

                cmar.protocol = f"python/{__version__}"

                print_model_notice(cmar.model)

                # Add span attributes
                if span:
                    span.set_attribute("agent.model", cmar.model)
                    span.set_attribute("agent.streaming", cmar.streaming)
                    if cmar.doc:
                        span.set_attribute("agent.doc_length", len(cmar.doc))

                uri = "/agent/create"
                self.log(f"Dialing {uri} for {str(cmar)[:128]}")
                request_dict = asdict(cmar)
                request_dict["warp_globals_payload"] = b64encode(cmar.warp_globals_payload).decode()
                self.log(f"Sending {str(request_dict)[:128]}")

                try:
                    async with self._http_client() as client:
                        response = await client.post(
                            uri,
                            json=request_dict,
                        )
                        raise_for_status(response, session_id=self._client_session_id)

                        if warning := response.headers.get("X-SDK-Warning"):
                            if warning == "deprecated":
                                message = response.headers.get("X-SDK-Upgrade-Message", "")
                                if message:
                                    self.log(message)

                        uid = response.text
                        self.log(f"Heard {uid=} for {str(cmar)[:128]}")

                        # Add agent UID to span
                        span.set_attribute("agent.uid", uid) if span else None

                        self._known_uids.add(uid)
                        # Initialize per-agent state
                        self._uid_iid_recv_queue[uid] = dict()
                        self._uid_iid_future[uid] = dict()

                        await self._ensure_websocket_connection(uid=uid)
                        log_ctx.log('uid =', uid)
                        return uid
                except Exception as e:
                    if span:
                        span.record_exception(e)
                    raise
        finally:
            span_end(span)

    async def destroy_agent(self, uid: str) -> None:
        """Destroy an agent on the server."""
        span = span_start(self._tracer, "sdk.destroy_agent", context_span=self._session_span)
        try:
            with self.log_as('destroy_agent') as log_ctx:
                span_set_attribute(span, "agent.uid", uid)

                uri = f"/agent/destroy/{uid}"
                self.log(f"Dialing {uri} for {uid}")

                try:
                    async with self._http_client() as client:
                        response = await client.delete(uri)

                        if response.is_success:
                            self.log(f"Destroyed agent {uid=}")
                        else:
                            if response.status_code == 404:
                                self.log(f"Agent {uid} not found, as expected.")
                            else:
                                self.log(
                                    f"Failed to destroy agent {uid}: HTTP {response.status_code}"
                                )

                        if warning := response.headers.get("X-SDK-Warning"):
                            if warning == "deprecated":
                                message = response.headers.get("X-SDK-Upgrade-Message", "")
                                if message:
                                    print(message)
                except Exception as e:
                    self.log(f"Failed to destroy agent {uid=} (server may be down): {e}")
                    span_finish(span, e)
        finally:
            span_end(span)

    async def invoke_agent(
        self,
        *,
        uid: str,
        warp_locals_payload: bytes,
        task_desc: str,
        streaming: bool,
        parent_uid: str | None = None,
        parent_iid: str | None = None,
    ) -> AgentInvocationHandle:
        # Start span as child of websocket_connection span
        # Use start_span() not start_as_current_span() to avoid polluting context

        invoke_span = span_start(self._tracer, "sdk.invoke_agent", context_span=self._session_span)

        with self.log_as('invoke_agent', uid) as ctx:
            # Add span attributes
            span_set_attributes(
                invoke_span,
                {"agent.uid": uid, "agent.task_desc": task_desc, "invocation.streaming": streaming},
            )

            if not self.uid_resource_exists(uid):
                error = WebSocketConnectionError(
                    "WebSocket for this agent is not open, the connection to the server has been lost."
                )
                enrich_error(
                    error,
                    uid=uid,
                    session_id=self._client_session_id,
                    error_log_path=ERROR_LOG_PATH,
                )
                span_finish(invoke_span, error)
                raise error

            try:
                match_id = self.id_issuer()
                span_set_attribute(invoke_span, "invocation.match_id", match_id)

                msg = MultiplexInvokeMessage(
                    match_id=match_id,
                    uid=uid,
                    parent_uid=parent_uid,
                    parent_iid=parent_iid,
                    warp_locals_payload=warp_locals_payload,
                    prompt=maybe_prompt_template(task_desc),
                    streaming=streaming,
                )

                match_iid = self._match_iid
                recv_queues = self._uid_iid_recv_queue
                send_queue = self._send_queue
                futures = self._uid_iid_future

                if not self._is_websocket_open() or send_queue is None:
                    assert_unrecoverable(uid=uid, session_id=self._client_session_id)

                match_iid[match_id] = iid_future = Future()
                send_queue.put_nowait(msg)
                iid = await iid_future
                _ = match_iid.pop(match_id, None)

                # Add invocation ID to span
                span_set_attribute(invoke_span, "invocation.iid", iid)

                if uid not in recv_queues or iid not in recv_queues[uid]:
                    assert_unrecoverable(uid=uid, iid=iid, session_id=self._client_session_id)

                if uid not in futures or iid not in futures[uid]:
                    assert_unrecoverable(uid=uid, iid=iid, session_id=self._client_session_id)

                recv_queue = recv_queues[uid][iid]
                future_result = futures[uid][iid]

                async def send_message(data: bytes) -> None:
                    send_queue.put_nowait(MultiplexDataMessage(uid=uid, iid=iid, data=data))

                # Pass the span to the handle so it ends when invocation completes
                handle = AgentInvocationHandle(
                    send_message=send_message,
                    recv_message=recv_queue.get,
                    future_result=future_result,
                    iid=iid,
                    _span=invoke_span,  # Span will be ended when future_result completes
                )

                ctx.vars(
                    iid=iid,
                    send_queue=send_queue,
                    recv_queue=recv_queue,
                    future_result=future_result,
                    handle=handle,
                )

                return handle
            except Exception as e:
                span_finish(invoke_span, e)
                raise

    def _is_websocket_open(self) -> bool:
        """Check if the websocket connection is open."""
        val = self._websocket is not None and self._websocket.state == websockets.State.OPEN
        self.log(f"Websocket open = {val}")
        return val

    def uid_resource_exists(self, uid: str) -> bool:
        return uid in self._known_uids and self._is_websocket_open()

    def agent_exists(self, uid: str) -> bool:
        return self.uid_resource_exists(uid)

    async def close_agent(self, uid: str) -> None:
        self.log(f"Closing agent {uid[:8]}")

        # Clean up per-agent state (shared websocket/tasks remain)
        self._uid_iid_recv_queue.pop(uid, None)
        self._uid_iid_future.pop(uid, None)
        self._known_uids.discard(uid)

        self.log(f"Closed agent {uid[:8]}")

        # Finally, destroy the agent on the server (best-effort, won't throw)
        await self.destroy_agent(uid)

    async def logs(
        self,
        uid: str,
        iid: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        endpoint = f"/logs/{uid}"
        if iid is not None:
            endpoint += f"/{iid}"
        async with self._http_client() as client:
            response = await client.get(endpoint, params=params)
            raise_for_status(response)
            return response.json()

    async def echo(
        self,
        uid: str,
        iid: str | None = None,
        *,
        connected: asyncio.Event | None = None,
    ) -> AsyncGenerator[Chunk, None]:
        endpoint = f"/echo/{uid}"
        if iid is not None:
            endpoint += f"/{iid}"

        async with self._http_client() as client:
            async with client.stream("GET", endpoint) as response:
                if connected is not None:
                    connected.set()

                is_streaming = False
                async for line in response.aiter_lines():
                    if not line:
                        continue

                    logmsg: dict[str, Any] = json.loads(line)

                    typ = logmsg.get("type", None)
                    if typ == "sm_invocation_exit":
                        # Only break if listening to a specific invocation.
                        # When iid is None, we're listening to all invocations
                        # for this agent and should continue across invocations.
                        if iid is not None:
                            break
                        continue

                    # yield back the raw streaming text chunks + other roles
                    if typ == "sm_monad" and (body := logmsg.get("body", None)):
                        body = json.loads(body)

                        if body.get("system", None):
                            continue

                        if body.get("type", None) == "stream_chunk":
                            is_streaming = True
                            delta = body["args"][0]
                            yield Chunk(AgentRole(), delta["content"])
                        elif body.get("type", None) == "delta":
                            delta = body["args"][0]
                            role = make_role(delta["role"], delta.get("username", None))
                            if role == "system":
                                # system: hidden, just clutter
                                continue
                            if is_streaming and role == "agent":
                                continue
                            yield Chunk(role, delta["content"])

    async def close(self) -> None:
        self._running = False
        agent_count = len(self._known_uids)
        self.log(f"Closing ClientSessionManager with {agent_count} agent(s)")

        # Close all agents (cleans up per-agent state and destroys on server)
        for uid in list(self._known_uids):
            await self.close_agent(uid)

        # Close the WebSocket connection first (this will cause reader/writer to exit)
        if self._is_websocket_open():
            ws = self._websocket
            assert ws is not None
            try:
                self.log("Closing the websocket")
                await ws.close()

                # End the connection span when closing
                if self._session_span:
                    span_set_attribute(self._session_span, "websocket.state", "closed")
                    span_end(self._session_span)
            except Exception as exc:
                # Record error on connection span before ending
                if self._session_span:
                    self._session_span.record_exception(exc)
                    self._session_span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
                    self._session_span.end()
                # websocket close failures are expected during atexit
                # (websockets are bound to the original event loop which is dead)
                self.log(f"Failed to close websocket: {exc}")
            finally:
                self._websocket = None

        # Cancel background tasks and await them to finish
        if self._tasks is not None:
            reader_task, writer_task = self._tasks
            reader_task.cancel()
            writer_task.cancel()
            try:
                await reader_task
            except CancelledError:
                pass
            except Exception:
                pass
            try:
                await writer_task
            except CancelledError:
                pass
            except Exception:
                pass
            self._tasks = None

        # Reset send queue for potential reuse
        self._send_queue = Queue()

        # End session span when closing the entire ClientSessionManager
        if session_span := self._session_span:
            session_span.set_attribute("sdk.agents_created", agent_count)
            session_span.end()
            self._session_span = None

    def __del__(self) -> None:
        """Cleanup when ClientSessionManager is garbage collected.

        This will trigger sandbox deletion via the cleanup callback if provided.
        """
        # Call sync cleanup callback if provided
        if self._cleanup_callback:
            try:
                self._cleanup_callback()
            except:
                pass


INTERNAL_ERROR_MSG = "Our server reached an unexpected state, this will be fixed soon."
