"""Full adapter stack, with only the remote engine's HTTP transport simulated."""

import httpx
import pytest
from test_service import payload
from test_sglang import Engine

from jev_adapter.server import create_app
from jev_adapter.sglang import SGLangBackend


@pytest.mark.asyncio
async def test_public_request_through_real_service_and_sglang_client():
    engine = Engine()
    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as upstream:
        backend = SGLangBackend("http://engine", "qwen", client=upstream)
        app = create_app(backend, api_key="client-key")
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app),
                base_url="http://adapter",
            ) as client,
        ):
            body = payload()
            body["model"] = "jev-latest"
            body["images"] = ["data:image/png;base64,YQ=="]
            body["options"] = {"permutations": 2, "return_logprobs": True}
            response = await client.post(
                "/v1/systemone",
                json=body,
                headers={"Authorization": "Bearer client-key"},
            )
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["model"] == "qwen"
        assert set(result["answers"]) == set(body["questions"])
        assert result["answers"]["route"]["probabilities"]["billing"] == pytest.approx(
            0.5
        )
        assert result["usage"] == {"input_tokens": 426, "output_tokens": 0}
        assert result["metadata"]["evaluations"] == 6
        generations = [data for path, data, _ in engine.calls if path == "/generate"]
        assert len(generations) == 6
        assert len({data["rid"] for data in generations}) == 6
        assert all(data["image_data"] == body["images"] for data in generations)
        assert all(
            data["sampling_params"]["max_new_tokens"] == 0 for data in generations
        )
