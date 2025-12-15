import threading
import time
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager

import httpx
from agentica_internal.core.log import LogBase

from agentica.client_session_manager.client_session_manager import ClientSessionManager
from agentica.errors import InvalidAPIKey
from agentica.version import __version__


class Agentica(LogBase):
    __base_url: str
    __api_key: str

    def __init__(self, base_url: str, api_key: str, *, logging: bool = False):
        super().__init__(logging=logging)
        self.__base_url = base_url
        self.__api_key = api_key

    @asynccontextmanager
    async def _http_client(self) -> AsyncGenerator[httpx.AsyncClient, None]:
        """Create a short-lived HTTP client with proper headers and base URL."""
        headers = {
            "Authorization": f"Bearer {self.__api_key}",
            "Content-Type": "application/json",
            "X-Protocol": f"python/{__version__}",
        }
        timeout = httpx.Timeout(60.0, connect=10.0)  # Allow up to 60s. Used for sandbox creation
        async with httpx.AsyncClient(
            base_url=self.__base_url, headers=headers, timeout=timeout
        ) as client:
            yield client

    @contextmanager
    def _sync_http_client(self) -> Generator[httpx.Client, None, None]:
        """Create a short-lived synchronous HTTP client with proper headers and base URL."""
        headers = {
            "Authorization": f"Bearer {self.__api_key}",
            "Content-Type": "application/json",
            "X-Protocol": f"python/{__version__}",
        }
        # Increased timeout for keepalive and delete operations
        timeout = httpx.Timeout(30.0, connect=10.0)
        with httpx.Client(base_url=self.__base_url, headers=headers, timeout=timeout) as client:
            yield client

    async def get_session_manager(self, enable_otel_logging: bool = False) -> ClientSessionManager:
        self.log("Creating session manager")
        async with self._http_client() as client:
            response = await client.post("/sandbox", json={})

        self.log(f"Sandbox created in {response.elapsed.total_seconds()}s")
        if response.status_code == 401:
            raise InvalidAPIKey('Failed to authenticate your platform API key.')
        assert response.status_code in (201, 200), (
            f"Failed to create sandbox: {response.status_code} \"{response.text}\""
        )
        self.log(f"Response: {response.json()}")

        response_data = response.json()
        sandbox_id = response_data["sandbox_id"]
        assert isinstance(sandbox_id, str)
        assert isinstance(response_data, dict)

        sandbox_url = response_data.get("http_url")  # no sane default
        assert isinstance(sandbox_url, str)
        keepalive_interval_s = response_data["keepalive_interval_s"]
        assert isinstance(keepalive_interval_s, float)

        # Create ClientSessionManager with API key and cleanup callback
        csm = ClientSessionManager(
            base_url=sandbox_url,
            agentica_api_key=self.__api_key,
            cleanup_callback=lambda: self.__sandbox_delete_sync(sandbox_id),
            enable_otel_logging=enable_otel_logging,
        )

        # Start keepalive thread for this sandbox
        keepalive_thread = threading.Thread(
            target=self.__sandbox_keepalive,
            args=(sandbox_id, keepalive_interval_s),
            name=f"keepalive-{sandbox_id}",
            daemon=True,  # Thread will die when main program exits
        )
        keepalive_thread.start()

        self.log(f"Created sandbox {sandbox_id} with keepalive interval {keepalive_interval_s}s")

        return csm

    def __sandbox_keepalive(self, sandbox_id: str, keepalive_interval_s: float) -> None:
        """Send keepalive messages in a separate OS thread to prevent sandbox timeout.

        This runs in a genuine OS thread, so it won't be blocked by other Python code
        that might block the main thread or asyncio event loop.
        """
        try:
            while True:
                try:
                    with self._sync_http_client() as client:
                        _ = client.post(f"/sandbox/{sandbox_id}/keepalive")
                except Exception as e:
                    self.log(f"Keepalive request failed for sandbox {sandbox_id}: {e}")
                time.sleep(keepalive_interval_s)
        except Exception as e:
            self.log(f"Exception during keepalive thread for sandbox {sandbox_id}: {e}")

    def __sandbox_delete_sync(self, sandbox_id: str) -> None:
        """Delete a sandbox synchronously (for use in __del__).

        Handles all logging internally.
        """
        try:
            with self._sync_http_client() as client:
                _ = client.delete(f"/sandbox/{sandbox_id}")
        except:
            pass
