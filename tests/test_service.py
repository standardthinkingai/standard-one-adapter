import asyncio
import math
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from jev_adapter.__main__ import positive_temperature, temperature_map
from jev_adapter.backend import AdapterError, ScoringResult
from jev_adapter.protocol import SystemOneRequest, probabilities_from_logprobs
from jev_adapter.server import create_app
from jev_adapter.service import SystemOneService


def payload():
    return {
        "model": "test-model",
        "state": {"text": "결제가 두 번 되었어요. 환불해 주세요."},
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "담당 부서",
                "criteria": {"billing": "청구", "technical": "기술"},
            },
            "refund": {"type": "noul", "instructions": "환불 요청인가?"},
            "urgency": {
                "type": "score",
                "instructions": "긴급도",
                "criteria": ["낮음", "중간", "높음"],
            },
        },
    }


class FakeBackend:
    model = "test-model"

    def __init__(self, delay=0, failure=False, cached_tokens=None):
        self.calls = []
        self.active = self.peak = self.cancelled = 0
        self.started = self.closed = False
        self.delay, self.failure = delay, failure
        self.cached_tokens = cached_tokens

    async def start(self):
        self.started = True

    async def close(self):
        self.closed = True

    def labels(self, count):
        return tuple(chr(65 + i) for i in range(count)), tuple(range(65, 65 + count))

    async def evaluate(self, prompt, images, labels, token_ids, assistant_prefix):
        self.calls.append((prompt, images, labels, token_ids, assistant_prefix))
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if self.failure:
                raise AdapterError(
                    "backend_unavailable", "Engine unavailable.", status=502
                )
            await asyncio.sleep(self.delay)
            return ScoringResult(
                tuple(math.log(0.8 if i == 0 else 0.1) for i in range(len(labels))),
                20,
                self.cached_tokens,
            )
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.active -= 1


class TestService(unittest.IsolatedAsyncioTestCase):
    async def test_mixed_questions_keep_images_and_build_typed_response(self):
        backend = FakeBackend()
        body = payload()
        body["images"] = ["data:image/png;base64,aGVsbG8="]
        result = await SystemOneService(backend).score(
            SystemOneRequest.model_validate(body)
        )
        self.assertEqual(result["answers"]["route"]["choice"], "billing")
        self.assertAlmostEqual(result["answers"]["refund"]["noul"], 8 / 9)
        self.assertAlmostEqual(result["answers"]["urgency"]["score"], 0.3)
        self.assertEqual(result["usage"], {"input_tokens": 60, "output_tokens": 0})
        self.assertEqual(result["metadata"]["evaluations"], 3)
        self.assertEqual(result["metadata"]["service_max_concurrency"], 32)
        for call in backend.calls:
            self.assertEqual(call[1], body["images"])
        self.assertNotIn("담당 부서", backend.calls[1][0])

    async def test_cache_usage_is_aggregated_across_questions(self):
        result = await SystemOneService(FakeBackend(cached_tokens=2)).score(
            SystemOneRequest.model_validate(payload())
        )
        self.assertEqual(result["metadata"]["cached_tokens"], 6)

    async def test_rotations_preserve_canonical_option_mapping(self):
        body = payload()
        body["questions"] = {"route": body["questions"]["route"]}
        body["options"] = {"permutations": 2, "return_logprobs": True}
        result = await SystemOneService(FakeBackend()).score(
            SystemOneRequest.model_validate(body)
        )
        route = result["answers"]["route"]
        self.assertAlmostEqual(route["probabilities"]["billing"], 0.5)
        self.assertAlmostEqual(route["confidence"], 0)
        self.assertEqual(len(route["logprobs"]), 2)

    async def test_concurrency_limit_applies_across_requests(self):
        backend = FakeBackend(delay=0.01)
        service = SystemOneService(backend, max_concurrency=2)
        results = await asyncio.gather(
            *(
                service.score(SystemOneRequest.model_validate(payload()))
                for _ in range(3)
            )
        )
        self.assertEqual(len(backend.calls), 9)
        self.assertEqual(backend.peak, 2)
        self.assertTrue(
            all(result["metadata"]["service_max_concurrency"] == 2 for result in results)
        )

    async def test_cancellation_closes_all_pending_work(self):
        backend = FakeBackend(delay=10)
        task = asyncio.create_task(
            SystemOneService(backend, max_concurrency=2).score(
                SystemOneRequest.model_validate(payload())
            )
        )
        while backend.active != 2:
            await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(backend.cancelled, 2)
        self.assertEqual(backend.active, 0)
        self.assertEqual(len(backend.calls), 2)

    async def test_wrong_model_rejected_before_inference(self):
        backend = FakeBackend()
        body = payload()
        body["model"] = "wrong-model"
        with self.assertRaises(AdapterError) as error:
            await SystemOneService(backend).score(SystemOneRequest.model_validate(body))
        self.assertEqual(error.exception.status, 404)
        self.assertEqual(backend.calls, [])


class ImageBackend(FakeBackend):
    async def evaluate(self, prompt, images, labels, token_ids, assistant_prefix):
        result = await super().evaluate(prompt, images, labels, token_ids, assistant_prefix)
        return ScoringResult(result.logprobs, 20, None, 7 if images else 0)


class PreparingBackend(FakeBackend):
    def __init__(self, refuse=None, **kwargs):
        super().__init__(**kwargs)
        self.prepared = []
        self.refuse = refuse

    async def prepare(self, evaluations, images, assistant_prefix):
        self.prepared.append([evaluation.question for evaluation in evaluations])
        for evaluation in evaluations:
            if evaluation.question == self.refuse:
                raise AdapterError(
                    "context_length_exceeded", "Too long.", f"questions.{evaluation.question}"
                )
        return [(evaluation, tuple(images)) for evaluation in evaluations]

    async def evaluate_prepared(self, prepared):
        evaluation, images = prepared
        return await self.evaluate(
            evaluation.prompt, list(images), evaluation.labels, evaluation.token_ids, None
        )


class TestEngineLimits(unittest.IsolatedAsyncioTestCase):
    async def test_deadline_cancels_all_work_and_answers_504(self):
        backend = FakeBackend(delay=10)
        service = SystemOneService(backend, max_concurrency=2, deadline_seconds=0.05)
        with self.assertRaises(AdapterError) as error:
            await service.score(SystemOneRequest.model_validate(payload()))
        self.assertEqual((error.exception.code, error.exception.status), ("engine_timeout", 504))
        self.assertEqual(backend.cancelled, 2)
        self.assertEqual(backend.active, 0)
        self.assertEqual(service._pending, 0)

    async def test_backend_timeout_is_not_mistaken_for_the_deadline(self):
        backend = FakeBackend()

        async def timed_out(*args):
            raise TimeoutError("socket read timed out")

        backend.evaluate = timed_out
        service = SystemOneService(backend, deadline_seconds=30)
        with self.assertRaises(TimeoutError):
            await service.score(SystemOneRequest.model_validate(payload()))

    async def test_py310_deadline_path_matches_the_contract(self):
        # SGLang ROCm images run Python 3.10, which has no asyncio.timeout.
        class NoTimeout:
            def __getattr__(self, name):
                if name == "timeout":
                    raise AttributeError(name)
                return getattr(asyncio, name)

        with mock.patch("jev_adapter.service.asyncio", NoTimeout()):
            backend = FakeBackend(delay=10)
            service = SystemOneService(backend, max_concurrency=2, deadline_seconds=0.05)
            with self.assertRaises(AdapterError) as error:
                await service.score(SystemOneRequest.model_validate(payload()))
            self.assertEqual(
                (error.exception.code, error.exception.status), ("engine_timeout", 504)
            )
            self.assertEqual(backend.cancelled, 2)
            self.assertEqual(backend.active, 0)
            self.assertEqual(service._pending, 0)

            backend = FakeBackend()

            async def timed_out(*args):
                raise TimeoutError("socket read timed out")

            backend.evaluate = timed_out
            service = SystemOneService(backend, deadline_seconds=30)
            with self.assertRaises(TimeoutError):
                await service.score(SystemOneRequest.model_validate(payload()))

            backend = FakeBackend(delay=10)
            service = SystemOneService(backend, max_concurrency=2, deadline_seconds=30)
            task = asyncio.create_task(
                service.score(SystemOneRequest.model_validate(payload()))
            )
            while backend.active != 2:
                await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(backend.active, 0)
            self.assertEqual(service._pending, 0)

            backend = FakeBackend()
            service = SystemOneService(backend, deadline_seconds=30)
            response = await service.score(SystemOneRequest.model_validate(payload()))
            self.assertEqual(set(response["answers"]), {"route", "refund", "urgency"})

    async def test_queue_limit_refuses_before_work_and_admits_when_idle(self):
        backend = FakeBackend(delay=10)
        service = SystemOneService(backend, max_concurrency=2, max_queued_evaluations=1)
        first = asyncio.create_task(service.score(SystemOneRequest.model_validate(payload())))
        while backend.active != 2:
            await asyncio.sleep(0)
        with self.assertRaises(AdapterError) as error:
            await service.score(SystemOneRequest.model_validate(payload()))
        self.assertEqual((error.exception.code, error.exception.status), ("engine_overloaded", 503))
        self.assertEqual(len(backend.calls), 2)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        self.assertEqual(service._pending, 0)
        backend.delay = 0
        result = await service.score(SystemOneRequest.model_validate(payload()))
        self.assertEqual(result["metadata"]["evaluations"], 3)

    async def test_remote_images_can_be_refused(self):
        backend = FakeBackend()
        body = payload()
        body["images"] = ["https://example.test/a.png"]
        with self.assertRaises(AdapterError) as error:
            await SystemOneService(backend, remote_images=False).score(
                SystemOneRequest.model_validate(body)
            )
        self.assertEqual((error.exception.code, error.exception.field), ("invalid_image", "images"))
        self.assertEqual(backend.calls, [])
        body["images"] = ["data:image/webp;base64,UklGRg=="]
        await SystemOneService(backend, remote_images=False).score(
            SystemOneRequest.model_validate(body)
        )
        self.assertEqual(len(backend.calls), 3)

    async def test_image_tokens_are_reported_only_for_image_requests(self):
        body = payload()
        body["images"] = ["data:image/png;base64,aGVsbG8="]
        result = await SystemOneService(ImageBackend()).score(SystemOneRequest.model_validate(body))
        self.assertEqual(
            result["usage"],
            {"input_tokens": 60, "output_tokens": 0, "input_tokens_details": {"image_tokens": 21}},
        )
        text = await SystemOneService(ImageBackend()).score(SystemOneRequest.model_validate(payload()))
        self.assertEqual(text["usage"], {"input_tokens": 60, "output_tokens": 0})

    async def test_prepare_checks_every_evaluation_before_any_inference(self):
        backend = PreparingBackend(refuse="urgency")
        body = payload()
        body["options"] = {"permutations": 2}
        with self.assertRaises(AdapterError) as error:
            await SystemOneService(backend).score(SystemOneRequest.model_validate(body))
        self.assertEqual(error.exception.field, "questions.urgency")
        self.assertEqual(
            backend.prepared,
            [["route", "route", "refund", "refund", "urgency", "urgency"]],
        )
        self.assertEqual(backend.calls, [])

        backend = PreparingBackend()
        result = await SystemOneService(backend).score(SystemOneRequest.model_validate(body))
        self.assertEqual(result["metadata"]["evaluations"], 6)
        self.assertEqual(len(backend.calls), 6)

    def test_invalid_limits_rejected_at_construction(self):
        for kwargs in (
            {"deadline_seconds": 0},
            {"deadline_seconds": float("inf")},
            {"deadline_seconds": True},
            {"max_queued_evaluations": -1},
            {"max_queued_evaluations": 1.5},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SystemOneService(FakeBackend(), **kwargs)


class TestDefaultTemperature(unittest.IsolatedAsyncioTestCase):
    """A server-side default fills an omitted options.temperature only."""

    vector = tuple(math.log(0.8 if i == 0 else 0.1) for i in range(3))

    async def score(self, body, **service_options):
        service = SystemOneService(FakeBackend(), **service_options)
        return await service.score(SystemOneRequest.model_validate(body))

    def route_probabilities(self, result):
        return list(result["answers"]["urgency"]["probabilities"].values())

    async def test_unset_default_keeps_request_default_of_one(self):
        result = await self.score(payload())
        self.assertEqual(result["metadata"]["temperature"], 1.0)
        expected = probabilities_from_logprobs(self.vector, 1.0)
        for actual, wanted in zip(self.route_probabilities(result), expected):
            self.assertAlmostEqual(actual, wanted)

    async def test_omitted_temperature_takes_server_default(self):
        for body in (payload(), {**payload(), "options": {"permutations": 1}}):
            result = await self.score(body, default_temperature=2.4)
            self.assertEqual(result["metadata"]["temperature"], 2.4)
            expected = probabilities_from_logprobs(self.vector, 2.4)
            for actual, wanted in zip(self.route_probabilities(result), expected):
                self.assertAlmostEqual(actual, wanted)
            self.assertNotAlmostEqual(
                self.route_probabilities(result)[0],
                probabilities_from_logprobs(self.vector, 1.0)[0],
            )

    async def test_explicit_request_temperature_wins_over_default(self):
        body = {**payload(), "options": {"temperature": 1.0}}
        result = await self.score(body, default_temperature=2.4)
        self.assertEqual(result["metadata"]["temperature"], 1.0)
        expected = probabilities_from_logprobs(self.vector, 1.0)
        for actual, wanted in zip(self.route_probabilities(result), expected):
            self.assertAlmostEqual(actual, wanted)
        body = {**payload(), "options": {"temperature": 3.0}}
        result = await self.score(body, default_temperature=2.4)
        self.assertEqual(result["metadata"]["temperature"], 3.0)

    async def test_disabled_scaling_ignores_default(self):
        body = {**payload(), "options": {"temperature_scaling": False}}
        result = await self.score(body, default_temperature=2.4)
        self.assertEqual(result["metadata"]["temperature"], 1.0)
        expected = probabilities_from_logprobs(self.vector, 1.0)
        for actual, wanted in zip(self.route_probabilities(result), expected):
            self.assertAlmostEqual(actual, wanted)

    async def test_default_does_not_mutate_the_request(self):
        request = SystemOneRequest.model_validate(payload())
        service = SystemOneService(FakeBackend(), default_temperature=2.4)
        await service.score(request)
        self.assertEqual(request.options.temperature, 1.0)
        self.assertNotIn("temperature", request.options.model_fields_set)
        self.assertEqual(service.effective_options(request).temperature, 2.4)

    def test_invalid_default_temperature_rejected_at_construction(self):
        for value in (0, -1.0, float("inf"), float("nan"), True, "2.4", None):
            with self.assertRaises((ValueError, TypeError)):
                SystemOneService(FakeBackend(), default_temperature=value)
        self.assertEqual(
            SystemOneService(FakeBackend(), default_temperature=2).default_temperature,
            2.0,
        )

    async def test_temperature_by_type_applies_per_answer_type(self):
        result = await self.score(
            payload(), default_temperature=2.4, temperature_by_type={"score": 3.0, "noul": 1.5}
        )
        urgency = list(result["answers"]["urgency"]["probabilities"].values())
        for actual, wanted in zip(urgency, probabilities_from_logprobs(self.vector, 3.0)):
            self.assertAlmostEqual(actual, wanted)
        route = list(result["answers"]["route"]["probabilities"].values())
        for actual, wanted in zip(route, probabilities_from_logprobs(self.vector[:2], 2.4)):
            self.assertAlmostEqual(actual, wanted)
        self.assertEqual(result["metadata"]["temperature"], 2.4)
        self.assertEqual(result["metadata"]["temperature_by_type"], {"score": 3.0, "noul": 1.5})

    async def test_explicit_request_temperature_wins_over_temperature_by_type(self):
        body = {**payload(), "options": {"temperature": 1.0}}
        result = await self.score(body, temperature_by_type={"score": 3.0})
        urgency = list(result["answers"]["urgency"]["probabilities"].values())
        for actual, wanted in zip(urgency, probabilities_from_logprobs(self.vector, 1.0)):
            self.assertAlmostEqual(actual, wanted)
        self.assertNotIn("temperature_by_type", result["metadata"])

    async def test_disabled_scaling_ignores_temperature_by_type(self):
        body = {**payload(), "options": {"temperature_scaling": False}}
        result = await self.score(body, temperature_by_type={"score": 3.0})
        urgency = list(result["answers"]["urgency"]["probabilities"].values())
        for actual, wanted in zip(urgency, probabilities_from_logprobs(self.vector, 1.0)):
            self.assertAlmostEqual(actual, wanted)
        self.assertNotIn("temperature_by_type", result["metadata"])

    def test_invalid_temperature_by_type_rejected_at_construction(self):
        for value in ({"judge": 1.2}, {"score": 0}, {"score": float("nan")}, {"noul": True}):
            with self.assertRaises(ValueError):
                SystemOneService(FakeBackend(), temperature_by_type=value)
        self.assertEqual(SystemOneService(FakeBackend()).temperature_by_type, {})

    def test_cli_temperature_map_type(self):
        self.assertEqual(temperature_map("choice=1.6, noul=1.2,score=1.8"), {"choice": 1.6, "noul": 1.2, "score": 1.8})
        self.assertEqual(temperature_map(""), {})
        import argparse

        for text in ("choice", "judge=1.2", "choice=0", "choice=1.2,choice=1.3", "noul=abc"):
            with self.assertRaises(argparse.ArgumentTypeError):
                temperature_map(text)

    def test_cli_default_temperature_type(self):
        self.assertEqual(positive_temperature("2.4"), 2.4)
        self.assertEqual(positive_temperature("1"), 1.0)
        import argparse

        for text in ("0", "-1", "inf", "nan", "abc", ""):
            with self.assertRaises(argparse.ArgumentTypeError):
                positive_temperature(text)


class TestPromptWording(unittest.IsolatedAsyncioTestCase):
    """--prompt-wording is a server-wide setting (CLI/env, not per-request);
    the default keeps today's served text unchanged."""

    async def test_default_served_wording_is_unchanged(self):
        backend = FakeBackend()
        body = payload()
        body["questions"] = {"route": body["questions"]["route"]}
        await SystemOneService(backend).score(SystemOneRequest.model_validate(body))
        prompt = backend.calls[0][0]
        self.assertTrue(prompt.startswith("Context:\n"))

    async def test_native_wording_is_plumbed_into_build_prompt(self):
        backend = FakeBackend()
        body = payload()
        body["questions"] = {"route": body["questions"]["route"]}
        service = SystemOneService(backend, prompt_wording="native")
        await service.score(SystemOneRequest.model_validate(body))
        prompt = backend.calls[0][0]
        self.assertTrue(prompt.startswith("Read the state and question."))
        self.assertIn("\n\nState:\n", prompt)
        self.assertNotIn("Context:\n", prompt)

    async def test_native_wording_requires_canonical_az_labels(self):
        # FakeBackend.labels() already returns canonical A, B, C, ... so this
        # documents the happy path; build_prompt itself enforces the
        # requirement (see TestNativePromptWording in test_protocol.py) when a
        # backend's labels are not canonical.
        backend = FakeBackend()
        labels, _ = backend.labels(3)
        self.assertEqual(labels, ("A", "B", "C"))

    def test_invalid_prompt_wording_rejected_at_construction(self):
        for value in ("", "SERVED", "native ", None, 1):
            with self.assertRaises(ValueError):
                SystemOneService(FakeBackend(), prompt_wording=value)


class TestHTTP(unittest.TestCase):
    def test_standalone_app_lifecycle_schema_and_alias(self):
        backend = FakeBackend()
        with TestClient(create_app(backend)) as client:
            self.assertTrue(backend.started)
            self.assertEqual(client.get("/health").json(), {"status": "ok"})
            self.assertEqual(
                client.get("/v1/models").json()["data"][0]["id"], backend.model
            )
            body = payload()
            body["model"] = "jev-latest"
            response = client.post("/v1/systemone", json=body)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["model"], backend.model)
            self.assertEqual(
                set(response.json()["answers"]), {"route", "refund", "urgency"}
            )
            bad = client.post("/v1/systemone", json={"model": backend.model})
            self.assertEqual(bad.status_code, 422)
            self.assertEqual(bad.json()["error"]["code"], "invalid_request")
        self.assertTrue(backend.closed)

    def test_app_default_temperature_applies_when_request_omits_it(self):
        with TestClient(create_app(FakeBackend(), default_temperature=2.4)) as client:
            response = client.post("/v1/systemone", json=payload())
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["metadata"]["temperature"], 2.4)
            body = {**payload(), "options": {"temperature": 1.5}}
            response = client.post("/v1/systemone", json=body)
            self.assertEqual(response.json()["metadata"]["temperature"], 1.5)
        with self.assertRaises(ValueError):
            create_app(FakeBackend(), default_temperature=0)

    def test_app_prompt_wording_native_reaches_the_backend_over_http(self):
        backend = FakeBackend()
        with TestClient(create_app(backend, prompt_wording="native")) as client:
            body = payload()
            body["questions"] = {"route": body["questions"]["route"]}
            response = client.post("/v1/systemone", json=body)
            self.assertEqual(response.status_code, 200)
            self.assertTrue(backend.calls[0][0].startswith("Read the state and question."))

    def test_auth_guards_inference_and_model_discovery(self):
        backend = FakeBackend()
        with TestClient(create_app(backend, api_key="test-secret")) as client:
            self.assertEqual(client.get("/v1/models").status_code, 401)
            self.assertEqual(
                client.post("/v1/systemone", json=payload()).status_code, 401
            )
            self.assertEqual(
                client.post(
                    "/v1/systemone",
                    json=payload(),
                    headers={b"Authorization": b"Bearer caf\xe9"},
                ).status_code,
                401,
            )
            self.assertEqual(backend.calls, [])
            response = client.post(
                "/v1/systemone",
                json=payload(),
                headers={"Authorization": "Bearer test-secret"},
            )
            self.assertEqual(response.status_code, 200)

    def test_upstream_failure_is_not_returned_as_probabilities(self):
        with TestClient(create_app(FakeBackend(failure=True))) as client:
            response = client.post("/v1/systemone", json=payload())
            self.assertEqual(response.status_code, 502)
            self.assertEqual(response.json()["error"]["code"], "backend_unavailable")
            self.assertNotIn("answers", response.json())
