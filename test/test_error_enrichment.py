from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from agentica_internal.session_manager_messages import CreateAgentRequest

from agentica.client_session_manager.client_session_manager import ClientSessionManager
from agentica.errors import (
    AgenticaError,
    ConnectionError,
    ServerError,
    WebSocketConnectionError,
)


class TestClientSessionManagerErrorEnrichment:
    @pytest.mark.asyncio
    async def test_register_session_http_error_includes_session_id(self, sdk_logging):
        csm = ClientSessionManager(base_url="http://test.example.com")

        # Mock HTTP client to return 500 error
        with patch.object(csm, '_http_client') as mock_http:
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.is_success = False
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_client.post.return_value = mock_response
            mock_http.return_value.__aenter__.return_value = mock_client

            with pytest.raises(ConnectionError) as exc_info:
                await csm._register_session()

            error = exc_info.value

            assert error.session_id == csm._client_session_id
            assert error.error_timestamp is not None
            assert error.error_log_file is not None
            assert '__init__' in Path(error.error_log_file).read_text()
            assert "T" in error.error_timestamp

            assert f"Session: {csm._client_session_id}" in str(error)
            assert "hello@symbolica.ai" in str(error)

    @pytest.mark.asyncio
    async def test_register_session_network_error_includes_session_id(self, sdk_logging):
        csm = ClientSessionManager(base_url="http://test.example.com")

        # Mock HTTP client to raise network exception
        with patch.object(csm, '_http_client') as mock_http:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")
            mock_http.return_value.__aenter__.return_value = mock_client

            with pytest.raises(ConnectionError) as exc_info:
                await csm._register_session()

            error = exc_info.value
            assert error.session_id == csm._client_session_id
            assert error.error_timestamp is not None
            assert error.error_log_file is not None
            assert '__init__' in Path(error.error_log_file).read_text()

    @pytest.mark.asyncio
    async def test_invoke_agent_websocket_closed_includes_uid_and_session(self, sdk_logging):
        csm = ClientSessionManager(base_url="http://test.example.com")

        uid = "test-agent-uid"
        csm._uid_websocket = {}

        # Mock that websocket doesn't exist (agent not connected)
        with pytest.raises(WebSocketConnectionError) as exc_info:
            await csm.invoke_agent(
                uid=uid,
                warp_locals_payload=b"test",
                task_desc="test task",
                streaming=False,
            )

        error: AgenticaError = exc_info.value
        assert error.uid == uid
        assert error.session_id == csm._client_session_id
        assert error.error_timestamp is not None
        assert error.error_log_file is not None
        assert 'invoke_agent' in Path(error.error_log_file).read_text()

        assert f"UID: {uid}" in str(error)
        assert f"Session: {csm._client_session_id}" in str(error)
        assert "hello@symbolica.ai" in str(error)

    @pytest.mark.asyncio
    async def test_new_agent_http_error_includes_session_id(self, sdk_logging):
        csm = ClientSessionManager(base_url="http://test.example.com")

        cmar = CreateAgentRequest(
            model="test-model",
            warp_globals_payload=b"test",
            streaming=False,
            doc="test",
            system="test",
            json=False,
        )

        # Mock HTTP response with 500 error
        with patch.object(csm, '_http_client') as mock_http:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Server Error", request=Mock(), response=mock_response
            )
            mock_client.post.return_value = mock_response
            mock_http.return_value.__aenter__.return_value = mock_client

            with pytest.raises(ServerError) as exc_info:
                await csm.new_agent(cmar)

            error = exc_info.value
            print(error)
            assert error.session_id == csm._client_session_id
            assert error.error_timestamp is not None
            assert error.error_log_file is not None
            assert 'new_agent' in Path(error.error_log_file).read_text()
            assert error.http_status_code == 500
