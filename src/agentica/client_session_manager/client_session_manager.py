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
from websockets import ClientConnection

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
class SessionManagerConnection:
    """Encapsulates state for a single session manager's WebSocket connection."""

    websocket: ClientConnection
    send_queue: Queue[MultiplexClientMessage]
    tasks: tuple[Task[None], Task[None]]  # (reader, writer)


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

    # Per-session-manager WebSocket state (replaces single global WebSocket)
    _session_managers: dict[str, SessionManagerConnection]  # sm_id -> connection state
    _sm_locks: dict[str, asyncio.Lock]  # sm_id -> creation lock
    _uid_to_sm: dict[str, str]  # agent uid -> sm_id

    # Per-agent state (unchanged - keyed by uid)
    _uid_iid_recv_queue: dict[str, dict[str, Queue[bytes]]]
    _uid_iid_future: dict[str, dict[str, Future[Result]]]

    _match_iid: dict[str, Future[str]]
    _known_uids: set[str]
    id_issuer: Callable[[], str]
    _base_url: str
    _base_ws: str
    _agentica_api_key: str | None
    _tracer: OTracer
    _running: bool = False

    # OpenTelemetry spans for lifecycle tracking
    _session_span: OSpan  # Entire SDK session lifetime

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
        # Per-session-manager WebSocket state
        self._session_managers = {}
        self._sm_locks = {}
        self._uid_to_sm = {}
        # Per-agent state
        self._uid_iid_recv_queue = dict()
        self._uid_iid_future = dict()
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

    def _is_websocket_open(self, smc: SessionManagerConnection | None) -> bool:
        """Check if the session manager connection and its websocket are open."""
        return (
            smc is not None
            and smc.websocket is not None
            and smc.websocket.state == websockets.State.OPEN
        )

    async def _sm_writer(self, sm_id: str) -> None:
        """Background writer task for a specific session manager's WebSocket."""
        smc = self._session_managers.get(sm_id)
        if smc is None:
            return

        with self.log_as('writer', sm_id) as ctx:
            try:
                while self._is_websocket_open(smc):
                    msg = await smc.send_queue.get()

                    # Re-check after await (socket may have closed while waiting)
                    if not self._is_websocket_open(smc):
                        break

                    self.log(f"Sending {msg=} to {sm_id=}")
                    ctx.log(' ->', msg)
                    try:
                        await smc.websocket.send(multiplex_to_json(msg))
                    except CancelledError:
                        self.log(f"Writer task for {sm_id=} was cancelled")
                        break
                    except Exception as e:
                        self.log(f"Writer task for {sm_id=} encountered an error {e}")
                        await sleep(0.25)
            except CancelledError:
                pass
            finally:
                await self._cleanup_session_manager(sm_id)

    async def _sm_reader(self, sm_id: str) -> None:
        """Background reader task for a specific session manager's WebSocket."""
        smc = self._session_managers.get(sm_id)
        if smc is None:
            return

        get_match = self._match_iid.get

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

        with self.log_as('reader', sm_id) as ctx:
            try:
                while self._is_websocket_open(smc):
                    msg_bytes = None
                    try:
                        msg_bytes = await smc.websocket.recv()
                    except CancelledError:
                        self.log(f"Reader task for {sm_id=} was cancelled")
                        break
                    except Exception as e:
                        self.log(f"Reader task for {sm_id=} encountered an error {e}")
                        await sleep(0.25)
                        continue
                    if type(msg_bytes) is not bytes:
                        continue
                    server_msg = None
                    try:
                        server_msg = multiplex_from_json(msg_bytes)
                    except:
                        continue
                    self.log(f"Received {server_msg=} from {sm_id=}")
                    if isinstance(server_msg, MultiplexServerMessage):
                        ctx.log(' <-', server_msg)
                        process_msg(server_msg, ctx)
            except CancelledError:
                pass
            finally:
                await self._cleanup_session_manager(sm_id)

    async def _ensure_sm_connection(self, sm_id: str) -> None:
        """Ensure a WebSocket connection exists for the given session manager.

        Each session manager gets its own WebSocket connection, send queue, and reader/writer tasks.
        Multiple agents on the same session manager share the same WebSocket.
        """
        # Fast path: session manager already connected
        if sm_id in self._session_managers:
            return

        # Lazy-create lock for this session manager
        lock = self._sm_locks.setdefault(sm_id, asyncio.Lock())
        async with lock:
            # Double-check after acquiring lock
            if sm_id in self._session_managers:
                return

            connection_span = None
            websocket_uri = f"{self._base_ws}/socket"
            if tracer := self._tracer:
                connection_span = span_start(tracer, "sdk.sm_connection")
                span_set_attribute(connection_span, "session_manager.id", sm_id)
                span_set_attribute(connection_span, "websocket.uri", websocket_uri)

            try:
                self.log(f"Dialing {websocket_uri=} for {sm_id=}")

                headers = (
                    {"Authorization": f"Bearer {self._agentica_api_key}"}
                    if self._agentica_api_key is not None
                    else {}
                )
                # For some reason websocket lowercases the header names
                headers["x-client-session-id"] = self._client_session_id

                if connection_span:
                    span_inject(connection_span, headers)

                ws = await websockets.connect(
                    websocket_uri,
                    ping_interval=20,
                    ping_timeout=None,
                    max_size=None,
                    additional_headers=headers,
                )

                # Create per-session-manager send queue
                send_queue: Queue[MultiplexClientMessage] = Queue()

                self.log(f"Spawning background tasks for {sm_id=}")
                tasks = (
                    create_task(self._sm_reader(sm_id), name=f"CSM.reader.{sm_id}"),
                    create_task(self._sm_writer(sm_id), name=f"CSM.writer.{sm_id}"),
                )

                # Store session manager connection state
                self._session_managers[sm_id] = SessionManagerConnection(
                    websocket=ws,
                    send_queue=send_queue,
                    tasks=tasks,
                )

                if connection_span:
                    connection_span.set_attribute("websocket.state", "connected")

            except Exception as e:
                if connection_span:
                    span_finish(connection_span, e)
                raise

    async def _cleanup_session_manager(self, sm_id: str) -> None:
        """Clean up a single session manager's resources without affecting others.

        This is called when a session manager's WebSocket connection closes or errors.
        It gracefully shuts down the session manager's resources and fails any pending
        futures for agents on that session manager.
        """
        smc = self._session_managers.pop(sm_id, None)
        self._sm_locks.pop(sm_id, None)
        if smc is None:
            return  # Already cleaned up or never existed

        self.log(f"Cleaning up session manager {sm_id}")

        # Cancel tasks (don't await - they may be the caller)
        for task in smc.tasks:
            if not task.done():
                task.cancel()

        # Close WebSocket
        try:
            await smc.websocket.close()
        except Exception as e:
            self.log(f"Error closing WebSocket for {sm_id}: {e}")

        # Find and clean up agents on this session manager
        affected_uids = [uid for uid, sid in self._uid_to_sm.items() if sid == sm_id]
        for uid in affected_uids:
            self._uid_to_sm.pop(uid, None)
            self._known_uids.discard(uid)

            # Fail pending futures for this agent
            if uid in self._uid_iid_future:
                for iid, future in self._uid_iid_future[uid].items():
                    if not future.done():
                        future.set_exception(
                            WebSocketConnectionError(f"Session manager {sm_id} connection closed")
                        )

        self.log(f"Cleaned up session manager {sm_id}, affected {len(affected_uids)} agent(s)")

    async def new_agent(self, cmar: CreateAgentRequest) -> str:
        # Link to session span to build proper hierarchy
        span = span_start(self._tracer, "sdk.new_agent", context_span=self._session_span)

        try:
            with self.log_as('new_agent') as log_ctx:
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

                        # Parse JSON response with uid and session_manager_id
                        data = response.json()
                        uid = data["uid"]
                        sm_id = data["session_manager_id"]
                        self.log(f"Heard {uid=}, {sm_id=} for {str(cmar)[:128]}")

                        # Add agent UID to span
                        span.set_attribute("agent.uid", uid) if span else None
                        span.set_attribute("agent.session_manager_id", sm_id) if span else None

                        # Track uid -> session manager mapping
                        self._uid_to_sm[uid] = sm_id
                        self._known_uids.add(uid)
                        # Initialize per-agent state
                        self._uid_iid_recv_queue[uid] = dict()
                        self._uid_iid_future[uid] = dict()

                        await self._ensure_sm_connection(sm_id=sm_id)
                        log_ctx.log('uid =', uid, ', sm_id =', sm_id)
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

            # Get session manager for this agent
            sm_id = self._uid_to_sm.get(uid)
            if sm_id is None or sm_id not in self._session_managers:
                error = WebSocketConnectionError(
                    "No connection for agent's session manager. The connection may have been lost."
                )
                enrich_error(
                    error,
                    uid=uid,
                    session_id=self._client_session_id,
                    error_log_path=ERROR_LOG_PATH,
                )
                span_finish(invoke_span, error)
                raise error

            smc = self._session_managers[sm_id]
            span_set_attribute(invoke_span, "agent.session_manager_id", sm_id)

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
                send_queue = smc.send_queue  # Use session-manager-specific send queue
                futures = self._uid_iid_future

                if sm_id not in self._session_managers:
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

    def _is_sm_connected(self, sm_id: str) -> bool:
        """Check if a specific session manager's WebSocket connection is open."""
        smc = self._session_managers.get(sm_id)
        if smc is None:
            return False
        return smc.websocket.state == websockets.State.OPEN

    def uid_resource_exists(self, uid: str) -> bool:
        """Check if an agent's resources exist and its session manager connection is open."""
        if uid not in self._known_uids:
            return False
        sm_id = self._uid_to_sm.get(uid)
        if sm_id is None:
            return False
        return self._is_sm_connected(sm_id)

    def agent_exists(self, uid: str) -> bool:
        return self.uid_resource_exists(uid)

    async def close_agent(self, uid: str) -> None:
        self.log(f"Closing agent {uid[:8]}")

        # Clean up per-agent state
        self._uid_iid_recv_queue.pop(uid, None)
        self._uid_iid_future.pop(uid, None)
        self._uid_to_sm.pop(uid, None)
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
                        # When listening across all invocations (iid is None),
                        # signal the end of this invocation without closing the stream.
                        yield Chunk(AgentRole(), "", "invocation_exit")
                        is_streaming = False  # if this invocation was streaming, we don't want to accidentally mark the next invocation as streaming
                        continue

                    # yield back the raw streaming text chunks + other roles
                    if typ == "sm_monad" and (body := logmsg.get("body", None)):
                        body = json.loads(body)

                        if body.get("system", None):
                            continue

                        if body.get("type", None) == "stream_chunk":
                            is_streaming = True
                            delta = body["args"][0]
                            yield Chunk(AgentRole(), delta["content"], delta.get("type"))
                        elif body.get("type", None) == "delta":
                            delta = body["args"][0]
                            role = make_role(delta["role"], delta.get("username", None))
                            if role == "system":
                                # system: hidden, just clutter
                                continue
                            if (
                                is_streaming
                                and role == "agent"
                                # exception: even when streaming we want to send usage because it's not sent any other way
                                and delta.get("type", None) != "usage"
                            ):
                                continue
                            yield Chunk(role, delta["content"], delta.get("type"))

    async def close(self) -> None:
        self._running = False
        agent_count = len(self._known_uids)
        sm_count = len(self._session_managers)
        self.log(
            f"Closing ClientSessionManager with {agent_count} agent(s) on {sm_count} session manager(s)"
        )

        # Close all agents (cleans up per-agent state and destroys on server)
        for uid in list(self._known_uids):
            await self.close_agent(uid)

        # Close all session manager connections
        for sm_id in list(self._session_managers.keys()):
            try:
                await self._cleanup_session_manager(sm_id)
            except Exception as exc:
                # Session manager cleanup failures are expected during atexit
                self.log(f"Failed to cleanup session manager {sm_id}: {exc}")

        # Clear any remaining state
        self._session_managers.clear()
        self._sm_locks.clear()
        self._uid_to_sm.clear()

        # End session span when closing the entire ClientSessionManager
        if session_span := self._session_span:
            session_span.set_attribute("sdk.agents_created", agent_count)
            session_span.set_attribute("sdk.session_managers_used", sm_count)
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
