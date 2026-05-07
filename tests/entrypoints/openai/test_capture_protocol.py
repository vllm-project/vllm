# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase F protocol tests: ``capture`` request field + ``capture_results``
response field + admission validation path.

These tests intentionally do not spin up a real engine. They cover:

- Pydantic round-trip: ``ChatCompletionRequest.capture`` / ``capture_results``
  and the legacy ``CompletionRequest`` / ``CompletionResponse`` mirror.
- The serving-layer admission validator (``_admit_capture``) against a
  fake in-memory consumer cache. Missing name → HTTP 400. Invalid raw
  spec (:class:`CaptureValidationError`) → HTTP 400 with
  ``param=capture.<name>``.
- The response-building helper (``_build_capture_results_response``)
  emitting ``None`` for empty dicts and a serializable dict for
  populated ones.
"""

from __future__ import annotations

from vllm.entrypoints.openai.chat_completion.protocol import (
    CaptureResultResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
)
from vllm.entrypoints.openai.chat_completion.serving import (
    _build_capture_results_response,
    _capture_result_to_response_payload,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionStreamResponse,
)
from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.v1.capture import (
    CaptureConsumer,
    CaptureContext,
    CaptureResult,
    CaptureSpec,
    CaptureValidationError,
)

# ---------------------------------------------------------------------------
# Pydantic round-trip: ChatCompletionRequest + ChatCompletionResponse
# ---------------------------------------------------------------------------


class TestChatProtocolRoundTrip:
    """Serialize/parse the new capture fields without engine involvement."""

    def test_capture_request_field_accepted(self) -> None:
        payload = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "capture": {
                "filesystem": {
                    "request_id": "r1",
                    "tag": "t1",
                    "hooks": {"post_mlp": [0]},
                    "positions": "last_prompt",
                },
            },
        }
        req = ChatCompletionRequest.model_validate(payload)
        assert req.capture == payload["capture"]

    def test_capture_request_default_none(self) -> None:
        req = ChatCompletionRequest.model_validate(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert req.capture is None

    def test_capture_response_field_serializes(self) -> None:
        resp = ChatCompletionResponse(
            model="m",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="hello"),
                )
            ],
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            capture_results={
                "filesystem": CaptureResultResponse(
                    status="ok",
                    error=None,
                    payload={"paths": ["/tmp/x.bin"]},
                ),
            },
        )
        dumped = resp.model_dump(exclude_unset=True, exclude_none=True)
        assert "capture_results" in dumped
        assert dumped["capture_results"]["filesystem"]["status"] == "ok"
        assert dumped["capture_results"]["filesystem"]["payload"] == {
            "paths": ["/tmp/x.bin"]
        }

    def test_capture_response_field_omitted_when_none(self) -> None:
        resp = ChatCompletionResponse(
            model="m",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="hello"),
                )
            ],
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        dumped = resp.model_dump(exclude_unset=True, exclude_none=True)
        # ``capture_results`` defaults to ``None`` and must be stripped
        # from the wire payload so OpenAI clients see a clean envelope.
        assert "capture_results" not in dumped

    def test_stream_response_capture_results_optional(self) -> None:
        # The SSE-terminal frame uses ``ChatCompletionStreamResponse``;
        # the capture_results field sits alongside ``usage`` so the
        # streaming envelope stays additive.
        frame = ChatCompletionStreamResponse(
            model="m",
            choices=[],
            capture_results={
                "filesystem": CaptureResultResponse(
                    status="error",
                    error="disk full",
                    payload={"paths": []},
                ),
            },
        )
        dumped = frame.model_dump(exclude_unset=True, exclude_none=True)
        assert dumped["capture_results"]["filesystem"]["status"] == "error"
        assert dumped["capture_results"]["filesystem"]["error"] == "disk full"


class TestLegacyCompletionProtocolRoundTrip:
    """Same round-trip for ``CompletionRequest`` / ``CompletionResponse``."""

    def test_capture_request_field_accepted(self) -> None:
        req = CompletionRequest.model_validate(
            {
                "model": "m",
                "prompt": "hi",
                "capture": {"logging": {"verbosity": "high"}},
            }
        )
        assert req.capture == {"logging": {"verbosity": "high"}}

    def test_capture_response_roundtrip(self) -> None:
        resp = CompletionResponse(
            model="m",
            choices=[
                CompletionResponseChoice(
                    index=0,
                    text="hi",
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            capture_results={
                "filesystem": CaptureResultResponse(
                    status="ok",
                    payload={"paths": ["/tmp/a.bin"]},
                ),
            },
        )
        dumped = resp.model_dump(exclude_unset=True, exclude_none=True)
        assert dumped["capture_results"]["filesystem"]["status"] == "ok"

    def test_stream_response_capture_results_optional(self) -> None:
        frame = CompletionStreamResponse(
            model="m",
            choices=[],
            capture_results={
                "fs": CaptureResultResponse(status="ok"),
            },
        )
        dumped = frame.model_dump(exclude_unset=True, exclude_none=True)
        assert "capture_results" in dumped


# ---------------------------------------------------------------------------
# _build_capture_results_response: empty-dict coalescing
# ---------------------------------------------------------------------------


class _FakeFinal:
    """Minimal stand-in for ``RequestOutput`` exposing only the fields
    ``_build_capture_results_response`` reads.
    """

    def __init__(self, capture_results: dict | None = None) -> None:
        # The helper reads via getattr with a None default, so both
        # presence and absence of the attribute are valid shapes to
        # exercise.
        if capture_results is not None:
            self.capture_results = capture_results


class TestBuildCaptureResultsResponse:
    def test_none_when_attribute_missing(self) -> None:
        result = _build_capture_results_response(
            ChatCompletionRequest(
                model="m", messages=[{"role": "user", "content": "x"}]
            ),
            _FakeFinal(),  # type: ignore[arg-type]
        )
        assert result is None

    def test_none_when_empty_dict(self) -> None:
        result = _build_capture_results_response(
            ChatCompletionRequest(
                model="m", messages=[{"role": "user", "content": "x"}]
            ),
            _FakeFinal(capture_results={}),  # type: ignore[arg-type]
        )
        assert result is None

    def test_populated_dict_builds_response_models(self) -> None:
        req = ChatCompletionRequest(
            model="m", messages=[{"role": "user", "content": "x"}]
        )
        final = _FakeFinal(
            capture_results={
                "fs": CaptureResult(
                    key=("r1", 0, "post_mlp"),
                    status="ok",
                    error=None,
                    payload=["/tmp/a.bin", "/tmp/a.json"],
                ),
                "log": CaptureResult(
                    key=("r1", 0, "post_mlp"),
                    status="partial_error",
                    error="dropped",
                    payload=None,
                ),
            }
        )
        result = _build_capture_results_response(req, final)  # type: ignore[arg-type]
        assert result is not None
        assert set(result) == {"fs", "log"}
        assert isinstance(result["fs"], CaptureResultResponse)
        assert result["fs"].status == "ok"
        # list payload is wrapped under ``items`` to keep the JSON schema
        # stable across consumers whose payload type varies.
        assert result["fs"].payload == {"items": ["/tmp/a.bin", "/tmp/a.json"]}
        assert result["log"].status == "partial_error"
        assert result["log"].error == "dropped"
        # ``None`` payload becomes an empty dict so the response schema
        # is never ``payload: null``.
        assert result["log"].payload == {}


class TestPayloadCoercion:
    """Direct coverage of ``_capture_result_to_response_payload``."""

    def test_none_becomes_empty_dict(self) -> None:
        assert _capture_result_to_response_payload(None) == {}

    def test_dict_passes_through(self) -> None:
        assert _capture_result_to_response_payload({"k": 1}) == {"k": 1}

    def test_list_wrapped_under_items(self) -> None:
        assert _capture_result_to_response_payload([1, 2]) == {"items": [1, 2]}

    def test_fallback_to_value_key(self) -> None:
        assert _capture_result_to_response_payload(42) == {"value": 42}


# ---------------------------------------------------------------------------
# _admit_capture: unknown name + validator error paths
# ---------------------------------------------------------------------------


class _FakeConsumerAccepts(CaptureConsumer):
    reads_client_spec = True

    def validate_client_spec(self, raw_spec, ctx):  # type: ignore[override]
        # Returns a valid CaptureSpec derived from the raw payload.
        return CaptureSpec(hooks={"post_mlp": [0]}, positions="last_prompt")

    def on_capture(self, key, tensor, sidecar):  # pragma: no cover - unused
        pass


class _FakeConsumerRejects(CaptureConsumer):
    reads_client_spec = True

    def validate_client_spec(self, raw_spec, ctx):  # type: ignore[override]
        raise CaptureValidationError("layer out of range")

    def on_capture(self, key, tensor, sidecar):  # pragma: no cover - unused
        pass


def _build_admit_ctx_factories(monkeypatch):
    """Prepare a servingchat instance without calling ``__init__``.

    Doing full ``__init__`` would require a real engine client; instead
    we manually install the attributes ``_admit_capture`` reads and
    borrow the method from the class.
    """
    from vllm.entrypoints.openai.chat_completion.serving import (
        OpenAIServingChat,
    )

    # Plain namespace object mimicking the subset of ``OpenAIServingChat``
    # that ``_admit_capture`` touches.
    class _StubEngineClient:
        class _VllmConfig:
            class _ParallelConfig:
                tensor_parallel_size = 1
                pipeline_parallel_size = 1

            class _ModelConfig:
                @staticmethod
                def get_total_num_hidden_layers() -> int:
                    return 32

                @staticmethod
                def get_hidden_size() -> int:
                    return 4096

                # torch.bfloat16 exposes itemsize=2 on modern torches;
                # stubbing a simple object keeps this test torch-free.
                class _Dtype:
                    itemsize = 2

                dtype = _Dtype()

            parallel_config = _ParallelConfig()
            model_config = _ModelConfig()

        vllm_config = _VllmConfig()

    class _StubServing:
        engine_client = _StubEngineClient()
        _capture_consumers: dict = {}

        def _extract_prompt_len(self, engine_input) -> int:  # noqa: ARG002
            return 8

        def create_error_response(self, message, status_code, param=None):
            # Mirror the shape OpenAIServing returns so assertions can
            # check the surfaced field without importing the full
            # OpenAIServing base class.
            return {
                "message": message,
                "status_code": int(status_code),
                "param": param,
            }

    admit = OpenAIServingChat._admit_capture

    return _StubServing, admit


class TestAdmitCaptureValidation:
    def test_unknown_consumer_name_returns_400(self, monkeypatch) -> None:
        from vllm.sampling_params import SamplingParams

        stub_cls, admit = _build_admit_ctx_factories(monkeypatch)
        stub = stub_cls()
        stub._capture_consumers = {}  # no consumers registered

        sp = SamplingParams(capture={"does_not_exist": {}})

        result = admit(
            stub,
            sampling_params=sp,
            engine_input=object(),
            request_id="req-1",
        )

        assert result is not None
        # HTTP 400 + param anchored at the consumer key so clients can
        # map the error back to which capture entry failed.
        assert result["status_code"] == 400
        assert result["param"] == "capture.does_not_exist"
        assert "does_not_exist" in result["message"]

    def test_validation_error_returns_400_with_param(self, monkeypatch) -> None:
        from vllm.sampling_params import SamplingParams

        stub_cls, admit = _build_admit_ctx_factories(monkeypatch)
        stub = stub_cls()
        # Use a bare-object params dict so the fake consumer accepts it.
        consumer = _FakeConsumerRejects.__new__(_FakeConsumerRejects)
        stub._capture_consumers = {"filesystem": consumer}

        sp = SamplingParams(capture={"filesystem": {"bad": True}})

        result = admit(
            stub,
            sampling_params=sp,
            engine_input=object(),
            request_id="req-1",
        )

        assert result is not None
        assert result["status_code"] == 400
        assert result["param"] == "capture.filesystem"
        assert "layer out of range" in result["message"]

    def test_happy_path_mutates_sampling_params_to_spec(self, monkeypatch) -> None:
        from vllm.sampling_params import SamplingParams

        stub_cls, admit = _build_admit_ctx_factories(monkeypatch)
        stub = stub_cls()
        consumer = _FakeConsumerAccepts.__new__(_FakeConsumerAccepts)
        stub._capture_consumers = {"filesystem": consumer}

        sp = SamplingParams(
            capture={"filesystem": {"tag": "t", "hooks": {"post_mlp": [0]}}}
        )

        result = admit(
            stub,
            sampling_params=sp,
            engine_input=object(),
            request_id="req-1",
        )

        assert result is None
        # The validator's CaptureSpec replaces the raw payload in place.
        assert isinstance(sp.capture["filesystem"], CaptureSpec)

    def test_noop_when_capture_is_none(self, monkeypatch) -> None:
        from vllm.sampling_params import SamplingParams

        stub_cls, admit = _build_admit_ctx_factories(monkeypatch)
        stub = stub_cls()

        sp = SamplingParams(capture=None)
        result = admit(
            stub,
            sampling_params=sp,
            engine_input=object(),
            request_id="req-1",
        )
        assert result is None
        assert sp.capture is None

    def test_capture_context_populated_correctly(self, monkeypatch) -> None:
        """The admission validator receives the model shape from
        ``vllm_config`` and the request-derived prompt length.
        """
        from vllm.sampling_params import SamplingParams

        stub_cls, admit = _build_admit_ctx_factories(monkeypatch)
        stub = stub_cls()

        received: list[CaptureContext] = []

        class _Capturing(CaptureConsumer):
            reads_client_spec = True

            def validate_client_spec(self, raw_spec, ctx):  # type: ignore[override]
                received.append(ctx)
                return CaptureSpec(hooks={"post_mlp": [0]}, positions="last_prompt")

            def on_capture(self, key, tensor, sidecar):  # pragma: no cover
                pass

        stub._capture_consumers = {"fs": _Capturing.__new__(_Capturing)}
        sp = SamplingParams(capture={"fs": {}})

        result = admit(
            stub,
            sampling_params=sp,
            engine_input=object(),
            request_id="req-42",
        )
        assert result is None
        assert len(received) == 1
        ctx = received[0]
        assert ctx.vllm_internal_request_id == "req-42"
        assert ctx.num_prompt_tokens == 8
        assert ctx.num_computed_tokens == 0
        assert ctx.num_hidden_layers == 32
        assert ctx.hidden_size == 4096
        assert ctx.element_size_bytes == 2
        assert ctx.tensor_parallel_size == 1
        assert ctx.pipeline_parallel_size == 1
