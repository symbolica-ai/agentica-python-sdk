"""
Global ClientSessionManager singleton (per event loop).

Connects to the platform via AGENTICA_API_KEY. For development, also supports
direct connection to session manager via S_M_BASE_URL environment variable.

The CSM is keyed by event loop to support multi-threaded usage where each
thread runs its own asyncio event loop.
"""

import asyncio
import os
import threading
import weakref
from logging import getLogger

import dotenv

from agentica.agentica_client.agentica_client import Agentica
from agentica.client_session_manager.client_session_manager import ClientSessionManager

_ = dotenv.load_dotenv()

logger = getLogger(__name__)


# CSM instances keyed by event loop (weak references to allow loop cleanup)
_loop_csm_map: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, ClientSessionManager] = (
    weakref.WeakKeyDictionary()
)
_loop_csm_map_lock = threading.Lock()
# Two different threads with different event loops can't share the same asyncio.Lock instance
_init_locks = weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock](
    weakref.WeakKeyDictionary()
)

DEFAULT_BASE_URL = "https://api.platform.symbolica.ai"


def current_global_csm() -> ClientSessionManager | None:
    """
    Get the current global CSM entry for the current event loop.
    """
    try:
        loop = asyncio.get_running_loop()
        csm = _loop_csm_map.get(loop)
        return csm
    except RuntimeError:
        return None


async def clear_global_csm() -> None:
    """
    Clear the cached CSM for the current event loop.

    This is useful in tests when you need to switch between different session managers
    (e.g., switching from Chat Completions to Responses API).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    csm = _loop_csm_map.pop(loop, None)
    if csm is not None:
        await csm.close()
        logger.debug("Cleared global CSM for current event loop")


async def get_global_csm() -> ClientSessionManager:
    loop = asyncio.get_running_loop()
    csm = _loop_csm_map.get(loop)

    # Fastpath
    if csm is not None:
        return csm

    with _loop_csm_map_lock:
        csm = _loop_csm_map.get(loop)
        if csm is not None:
            return csm
        init_coro_lock = _init_locks.setdefault(loop, asyncio.Lock())

    async with init_coro_lock:
        with _loop_csm_map_lock:
            if csm := _loop_csm_map.get(loop):
                return csm

        otel_enabled = os.getenv("OTEL_ENABLED", "0").lower() not in ("0", "false")
        if (agentica_api_key := os.getenv("AGENTICA_API_KEY")) is not None:
            base_url = os.getenv("AGENTICA_BASE_URL", DEFAULT_BASE_URL)
            client = Agentica(base_url=base_url, api_key=agentica_api_key)
            csm = await client.get_session_manager(enable_otel_logging=otel_enabled)
            logger.debug(f"Used platform service to create session manager base_url={base_url}")

        # <DEVELOPMENT ONLY>
        # Check S_M_BASE_URL first to allow direct connection to session manager with optional auth
        elif (session_manager_base_url := os.getenv("S_M_BASE_URL")) is not None:
            logger.debug(f"Using static session manager base_url={session_manager_base_url}")
            agentica_api_key = os.getenv(
                "AGENTICA_API_KEY"
            )  # Optional: can be None for --disable-auth
            csm = ClientSessionManager(
                base_url=session_manager_base_url,
                agentica_api_key=agentica_api_key,
                enable_otel_logging=otel_enabled,
            )
            logger.debug(
                f"Connected directly to session manager base_url={session_manager_base_url} "
                + f"with_auth={agentica_api_key is not None}"
            )
        # </DEVELOPMENT ONLY>
        else:
            raise ValueError(
                "No AGENTICA_API_KEY found in environment variables, visit https://agentica.symbolica.ai to get an API key."
            )

        with _loop_csm_map_lock:
            assert _loop_csm_map.get(loop) is None, (
                f"CSM is already initialized for event loop {loop}, which means a data race."
            )
            _loop_csm_map[loop] = csm

    return csm
