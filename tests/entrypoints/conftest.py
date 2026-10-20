# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest


@pytest.fixture
def inference_spans(monkeypatch):
    sdk = pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from vllm.sampling_params import SamplingParams
    from vllm.tracing import otel
    from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
    from vllm.v1.engine.output_processor import OutputProcessor
    from vllm.v1.metrics.stats import IterationStats

    if not otel.is_otel_available():
        pytest.skip("OTel required")
    provider = sdk.TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(otel.trace, "get_tracer", provider.get_tracer)
    processor = OutputProcessor(None, log_stats=False, tracing_enabled=True)

    def finish_engine(request_id="request", trace_headers=None):
        request = EngineCoreRequest(
            request_id=request_id,
            external_req_id=request_id,
            prompt_token_ids=[1],
            mm_features=None,
            sampling_params=SamplingParams(detokenize=False),
            pooling_params=None,
            arrival_time=1,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
            trace_headers=trace_headers,
        )
        processor.add_request(request, None)
        span = processor.request_states[request_id].trace_span
        processor.process_outputs(
            [
                EngineCoreOutput(
                    request_id=request_id,
                    new_token_ids=[2],
                    finish_reason=FinishReason.STOP,
                )
            ],
            1,
            IterationStats(),
        )
        return span

    yield exporter, finish_engine
    processor.shutdown()
    provider.shutdown()


@pytest.fixture
def tool_trace(inference_spans):
    from vllm.tracing import RequestTrace, use_request_trace

    exporter, finish_engine = inference_spans
    trace = RequestTrace()
    with use_request_trace(trace):
        finish_engine()
        yield trace, exporter
    trace.end()


@pytest.fixture
def sample_prompts():
    return [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]


@pytest.fixture
def sample_regex():
    return (
        r"((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.){3}"
        r"(25[0-5]|(2[0-4]|1\d|[1-9]|)\d)"
    )


@pytest.fixture
def sample_json_schema():
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "skills": {
                "type": "array",
                "items": {"type": "string", "maxLength": 10},
                "minItems": 3,
            },
            "work_history": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "company": {"type": "string"},
                        "duration": {"type": "number"},
                        "position": {"type": "string"},
                    },
                    "required": ["company", "position"],
                },
            },
        },
        "required": ["name", "age", "skills", "work_history"],
    }


@pytest.fixture
def sample_structured_outputs_choices():
    return [
        "Python",
        "Java",
        "JavaScript",
        "C++",
        "C#",
        "PHP",
        "TypeScript",
        "Ruby",
        "Swift",
        "Kotlin",
    ]


@pytest.fixture(scope="session")
def qwen3_lora_files():
    """Download Qwen3 LoRA files once per test session."""
    from vllm.transformers_utils.repo_utils import hf_api

    return hf_api().snapshot_download(repo_id="charent/self_cognition_Alice")


@pytest.fixture(scope="session")
def qwen3_meowing_lora_files():
    """Download Qwen3 LoRA files once per test session."""
    from vllm.transformers_utils.repo_utils import hf_api

    return hf_api().snapshot_download(repo_id="Jackmin108/Qwen3-0.6B-Meow-LoRA")


@pytest.fixture(scope="session")
def qwen3_woofing_lora_files():
    """Download Qwen3 LoRA files once per test session."""
    from vllm.transformers_utils.repo_utils import hf_api

    return hf_api().snapshot_download(repo_id="Jackmin108/Qwen3-0.6B-Woof-LoRA")


@pytest.fixture(scope="session")
def opt125_lora_files() -> str:
    """Download opt-125m LoRA files once per test session."""
    from vllm.transformers_utils.repo_utils import hf_api

    return hf_api().snapshot_download(
        repo_id="peft-internal-testing/opt-125m-dummy-lora"
    )
