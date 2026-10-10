"""A completed score must not wait forever for a cancelled disconnect poll."""

import asyncio

import pytest

from jev_adapter.backend import AdapterError, ScoringResult
from jev_adapter.protocol import SystemOneRequest
from jev_adapter.server import create_app


class ControlledBackend:
    model = "test-model"

    def __init__(self):
        self.release = asyncio.Event()
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()

    def labels(self, count):
        return tuple("AB"[:count]), tuple(range(count))

    async def evaluate(self, prompt, images, labels, token_ids, assistant_prefix):
        self.entered.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return ScoringResult((-0.1, -2.0), 10)


class CancellationSwallowingRequest:
    """Reproduce a request probe whose internal cancellation scope wins a race.

    CancelledError can be suppressed within a probe, leaving its caller alive.
    A real Request.is_disconnected uses a self-cancelling AnyIO CancelScope;
    deliberately control that boundary here rather than rely on scheduler luck.
    """

    def __init__(self):
        self.entered = asyncio.Event()
        self.swallowed = asyncio.Event()
        self.probe_task = None
        self.calls = 0

    async def is_disconnected(self):
        self.probe_task = asyncio.current_task()
        self.calls += 1
        self.entered.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.swallowed.set()
            return False


def endpoint_and_body(backend):
    app = create_app(backend)
    endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/v1/systemone"
    )
    body = SystemOneRequest.model_validate(
        {
            "model": backend.model,
            "state": "A billing question.",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Choose the team.",
                    "criteria": {"billing": None, "technical": None},
                }
            },
        }
    )
    return endpoint, body


@pytest.mark.asyncio
async def test_completed_score_returns_when_disconnect_probe_swallows_cancel():
    backend = ControlledBackend()
    endpoint, body = endpoint_and_body(backend)
    request = CancellationSwallowingRequest()
    response = asyncio.create_task(endpoint(body, request))
    try:
        await asyncio.wait_for(request.entered.wait(), timeout=1)
        await asyncio.wait_for(backend.entered.wait(), timeout=1)
        backend.release.set()
        await asyncio.wait_for(request.swallowed.wait(), timeout=1)
        done, _ = await asyncio.wait({response}, timeout=0.1)
        assert response in done, "completed inference is stuck cleaning up its watcher"
        result = response.result()
        assert result["answers"]["route"]["choice"] == "billing"
        assert request.calls == 1
        assert request.probe_task.done()
    finally:
        # Do not let a deliberately cancellation-resistant fake leak after red.
        request.is_disconnected = disconnected_now
        if request.probe_task is not None:
            request.probe_task.cancel()
        response.cancel()
        await asyncio.gather(response, return_exceptions=True)


async def disconnected_now():
    return True


@pytest.mark.asyncio
async def test_client_disconnect_still_cancels_unfinished_inference():
    backend = ControlledBackend()
    endpoint, body = endpoint_and_body(backend)

    class DisconnectedRequest:
        async def is_disconnected(self):
            await backend.entered.wait()
            return True

    with pytest.raises(AdapterError) as error:
        await asyncio.wait_for(endpoint(body, DisconnectedRequest()), timeout=1)
    assert error.value.code == "client_disconnected"
    assert error.value.status == 499
    assert backend.cancelled.is_set()


@pytest.mark.asyncio
async def test_cancelled_handler_joins_work_and_cancellation_resistant_watcher():
    backend = ControlledBackend()
    endpoint, body = endpoint_and_body(backend)
    request = CancellationSwallowingRequest()
    response = asyncio.create_task(endpoint(body, request))
    try:
        await asyncio.wait_for(request.entered.wait(), timeout=1)
        await asyncio.wait_for(backend.entered.wait(), timeout=1)
        response.cancel()
        done, _ = await asyncio.wait({response}, timeout=0.1)
        assert response in done, "cancelled handler did not release its child tasks"
        with pytest.raises(asyncio.CancelledError):
            response.result()
        assert backend.cancelled.is_set()
        assert request.swallowed.is_set()
        assert request.probe_task.done()
    finally:
        request.is_disconnected = disconnected_now
        if request.probe_task is not None:
            request.probe_task.cancel()
        response.cancel()
        await asyncio.gather(response, return_exceptions=True)
