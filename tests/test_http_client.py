import asyncio
import json
import sys
from types import SimpleNamespace

import httpx
import pytest

from jev_adapter.http_client import AiohttpCompatClient
from jev_adapter.sglang import SGLangBackend


@pytest.fixture
def aiohttp_stub(monkeypatch):
    sessions = []

    class ClientError(Exception):
        pass

    class Response:
        status = 200
        raw_headers = ((b"Content-Type", b"application/json"),)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def read(self):
            return b'{"ok":true}'

    class Session:
        def __init__(self, **kwargs):
            asyncio.get_running_loop()
            self.settings = kwargs
            self.calls = []
            self.closed = False
            self.error = None
            self.response = Response()
            sessions.append(self)

        def request(self, method, url, **kwargs):
            self.calls.append((method, url, kwargs))
            if self.error:
                raise self.error
            return self.response

        async def close(self):
            self.closed = True

    module = SimpleNamespace(
        ClientError=ClientError,
        ClientSession=Session,
        TCPConnector=lambda **kwargs: SimpleNamespace(**kwargs),
        ClientTimeout=lambda **kwargs: SimpleNamespace(**kwargs),
        sessions=sessions,
    )
    monkeypatch.setitem(sys.modules, "aiohttp", module)
    return module


def test_aiohttp_is_optional_and_has_install_guidance(monkeypatch):
    monkeypatch.setitem(sys.modules, "aiohttp", None)
    with pytest.raises(ImportError, match=r"jev-adapter\[performance\]"):
        AiohttpCompatClient()


def test_constructor_outside_loop_is_lazy(aiohttp_stub):
    client = AiohttpCompatClient(concurrency=128, timeout=17)
    assert aiohttp_stub.sessions == []

    async def exercise():
        first, second = await asyncio.gather(
            client.get("http://engine/info"),
            client.post(
                "http://engine/generate",
                json={"input_ids": [1, 2]},
                headers={"Authorization": "Bearer test-fixture"},
                timeout=9,
            ),
        )
        assert first.json() == second.json() == {"ok": True}
        assert second.request.method == "POST"
        assert second.request.url.path == "/generate"
        session = aiohttp_stub.sessions[0]
        assert len(aiohttp_stub.sessions) == 1
        connector = session.settings["connector"]
        assert connector.limit == connector.limit_per_host == 128
        assert session.settings["timeout"].total == 17
        assert session.settings["trust_env"] is False
        assert session.settings["auto_decompress"] is False
        post = session.calls[1][2]
        assert json.loads(post["data"]) == {"input_ids": [1, 2]}
        assert post["headers"]["authorization"] == "Bearer test-fixture"
        assert post["timeout"].total == 9
        assert post["allow_redirects"] is False
        session.response.status = 503
        response = await client.get("http://engine/info", timeout=None)
        assert session.calls[-1][2]["timeout"].total is None
        with pytest.raises(httpx.HTTPStatusError):
            response.raise_for_status()
        await client.aclose()
        assert session.closed
        with pytest.raises(RuntimeError, match="closed"):
            await client.get("http://engine/info")

    asyncio.run(exercise())


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [True, False])
async def test_aiohttp_errors_preserve_httpx_error_contract(aiohttp_stub, timeout):
    async with AiohttpCompatClient() as client:
        await client.get("http://engine/info")
        session = aiohttp_stub.sessions[0]
        session.error = (
            TimeoutError("deadline")
            if timeout
            else aiohttp_stub.ClientError("connection failed")
        )
        expected = httpx.TimeoutException if timeout else httpx.RequestError
        with pytest.raises(expected) as caught:
            await client.post("http://engine/generate", json={})
        assert caught.value.request.url.path == "/generate"
    assert session.closed


@pytest.mark.asyncio
async def test_close_before_first_request_does_not_create_session(aiohttp_stub):
    client = AiohttpCompatClient()
    await client.aclose()
    assert aiohttp_stub.sessions == []


@pytest.mark.asyncio
async def test_backend_selects_owned_transport_and_preserves_injected_client(
    aiohttp_stub, monkeypatch
):
    backend = SGLangBackend(
        "http://engine", "m", max_connections=128, http_transport="aiohttp"
    )
    assert isinstance(backend._client, AiohttpCompatClient)
    assert await backend._request("GET", "/info") == {"ok": True}
    assert aiohttp_stub.sessions[0].settings["connector"].limit == 128
    await backend.close()
    assert aiohttp_stub.sessions[0].closed

    monkeypatch.setitem(sys.modules, "aiohttp", None)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"ok": True})
        )
    ) as injected:
        backend = SGLangBackend(
            "http://engine", "m", client=injected, http_transport="aiohttp"
        )
        assert backend._client is injected
        await backend.close()
        assert not injected.is_closed
        assert await backend._request("GET", "/info") == {"ok": True}


def test_backend_rejects_unknown_transport():
    with pytest.raises(ValueError, match="http_transport"):
        SGLangBackend("http://engine", "m", http_transport="unknown")
