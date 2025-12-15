"""
Global ClientSessionManager singleton.

Connects to the platform via AGENTICA_API_KEY. For development, also supports
direct connection to session manager via S_M_BASE_URL environment variable.
"""

import os
from logging import getLogger

import dotenv

from agentica.agentica_client.agentica_client import Agentica
from agentica.client_session_manager.client_session_manager import ClientSessionManager

_ = dotenv.load_dotenv()

logger = getLogger(__name__)

_global_csm: ClientSessionManager | None = None

DEFAULT_BASE_URL = "https://api.platform.symbolica.ai"


def current_global_csm() -> ClientSessionManager | None:
    return _global_csm


async def get_global_csm() -> ClientSessionManager:
    global _global_csm

    otel_enabled = os.getenv("OTEL_ENABLED", "0").lower() not in ("0", "false")

    if _global_csm is None or not _global_csm._running:
        if (agentica_api_key := os.getenv("AGENTICA_API_KEY")) is not None:
            base_url = os.getenv("AGENTICA_BASE_URL", DEFAULT_BASE_URL)
            client = Agentica(base_url=base_url, api_key=agentica_api_key)
            _global_csm = await client.get_session_manager(enable_otel_logging=otel_enabled)
            logger.debug(f"Used platform service to create session manager base_url={base_url}")

        # <DEVELOPMENT ONLY>
        # Check S_M_BASE_URL first to allow direct connection to session manager with optional auth
        elif (session_manager_base_url := os.getenv("S_M_BASE_URL")) is not None:
            logger.debug(f"Using static session manager base_url={session_manager_base_url}")
            agentica_api_key = os.getenv(
                "AGENTICA_API_KEY"
            )  # Optional: can be None for --disable-auth
            _global_csm = ClientSessionManager(
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

    return _global_csm
