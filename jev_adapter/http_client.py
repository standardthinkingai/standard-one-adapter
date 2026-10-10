"""Optional aiohttp transport with buffered HTTPX response/error contracts."""

from __future__ import annotations

import importlib
import math

import httpx

_DEFAULT_TIMEOUT = object()


class AiohttpCompatClient:
    """Small AsyncClient-compatible surface for JSON inference requests.

    Construction is safe before an event loop starts: the ClientSession and
    connector are created on the first async request. Timeout values are total
    request deadlines, unlike HTTPX's individual inactivity timeouts. The client
    buffers responses, does not follow redirects, and ignores proxy environment
    variables. aiohttp is loaded only when this optional client is selected.
    """

    def __init__(self, concurrency: int = 32, timeout: float = 120):
        if (
            type(concurrency) is not int
            or concurrency < 1
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("concurrency and timeout must be positive")
        try:
            self._aiohttp = importlib.import_module("aiohttp")
        except ImportError as error:
            raise ImportError(
                "aiohttp transport requires the optional performance dependency; "
                "install with: pip install 'jev-adapter[performance]' "
                "(or pip install '.[performance]' from the source directory)."
            ) from error
        self._concurrency = concurrency
        self._timeout = timeout
        self._session = None
        self._closed = False

    def _get_session(self):
        if self._closed:
            raise RuntimeError("Cannot send a request, as the client has been closed.")
        if self._session is None:
            # No await separates the check from assignment, so concurrent first
            # requests on the same event loop create exactly one session.
            self._session = self._aiohttp.ClientSession(
                connector=self._aiohttp.TCPConnector(
                    limit=self._concurrency, limit_per_host=self._concurrency
                ),
                timeout=self._aiohttp.ClientTimeout(total=self._timeout),
                trust_env=False,
                auto_decompress=False,
            )
        return self._session

    async def request(
        self, method, url, *, headers=None, json=None, timeout=_DEFAULT_TIMEOUT
    ):
        request = httpx.Request(method, url, headers=headers, json=json)
        options = {}
        if timeout is not _DEFAULT_TIMEOUT:
            if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
                raise ValueError("timeout must be positive or None")
            options["timeout"] = self._aiohttp.ClientTimeout(total=timeout)
        try:
            async with self._get_session().request(
                method,
                str(request.url),
                headers=dict(request.headers),
                data=request.content or None,
                allow_redirects=False,
                **options,
            ) as response:
                content = await response.read()
                return httpx.Response(
                    response.status,
                    headers=response.raw_headers,
                    content=content,
                    request=request,
                )
        except TimeoutError as error:
            raise httpx.TimeoutException(str(error), request=request) from error
        except self._aiohttp.ClientError as error:
            raise httpx.RequestError(str(error), request=request) from error

    async def get(self, url, *, headers=None, timeout=_DEFAULT_TIMEOUT):
        return await self.request("GET", url, headers=headers, timeout=timeout)

    async def post(self, url, *, headers=None, json=None, timeout=_DEFAULT_TIMEOUT):
        return await self.request(
            "POST", url, headers=headers, json=json, timeout=timeout
        )

    async def aclose(self):
        self._closed = True
        if self._session is not None:
            await self._session.close()

    async def __aenter__(self):
        if self._closed:
            raise RuntimeError("Cannot reopen a closed client.")
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.aclose()
