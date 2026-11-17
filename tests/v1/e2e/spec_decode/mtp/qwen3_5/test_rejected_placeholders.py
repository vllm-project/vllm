# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest

from tests.utils import single_gpu_only
from vllm import SamplingParams
from vllm.config import CompilationConfig, CUDAGraphMode
from vllm.platforms import current_platform

from ...utils import get_spec_decode_metric_value

PLACEHOLDER_REGRESSION_MODEL = "Qwen/Qwen3.5-0.8B-Base"
PLACEHOLDER_REGRESSION_PROMPT = "The capital of France is"
PLACEHOLDER_REGRESSION_MAX_TOKENS = 16
PLACEHOLDER_REGRESSION_ALLOWED_TOKEN_ID = 42


def _flashinfer_mtp_decode_state(worker):
    from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder
    from vllm.v1.kv_cache_interface import KVQuantMode

    speculator = worker.model_runner.speculator
    builders = [
        (
            builder.use_xqa,
            str(builder.kv_cache_spec.dtype),
            builder.kv_cache_spec.kv_quant_mode == KVQuantMode.NONE,
            builder.supports_draft_decode_metadata_update,
        )
        for groups in speculator.attn_groups
        for group in groups
        for builder in group.metadata_builders
        if isinstance(builder, FlashInferMetadataBuilder)
    ]
    return speculator.supports_decode_metadata_update, builders


@pytest.mark.parametrize(
    "cudagraph_mode,draft_config",
    [
        pytest.param(CUDAGraphMode.PIECEWISE, None, id="piecewise"),
        pytest.param(CUDAGraphMode.FULL_AND_PIECEWISE, None, id="full-and-piecewise"),
    ]
    + (
        [
            pytest.param(
                CUDAGraphMode.FULL_AND_PIECEWISE, (1, "bfloat16", False), id="xqa-k1"
            ),
            pytest.param(CUDAGraphMode.PIECEWISE, (2, "bfloat16", False), id="xqa-k2"),
            pytest.param(
                CUDAGraphMode.FULL_AND_PIECEWISE, (3, "bfloat16", False), id="xqa-k3"
            ),
            pytest.param(
                CUDAGraphMode.FULL_AND_PIECEWISE,
                (3, "float16", False),
                id="xqa-fp16-fallback-k3",
            ),
            pytest.param(
                CUDAGraphMode.FULL_AND_PIECEWISE,
                (3, "bfloat16", True),
                id="xqa-dynamic-full-k123",
            ),
            pytest.param(
                CUDAGraphMode.NONE, (3, "bfloat16", True), id="xqa-dynamic-eager-k123"
            ),
        ]
        if current_platform.is_cuda()
        and (
            current_platform.is_device_capability(90)
            or current_platform.is_device_capability_family(120)
        )
        else []
    ),
)
@single_gpu_only
def test_mtp_rejected_placeholders_match_greedy_token_ids(
    cudagraph_mode: CUDAGraphMode,
    draft_config: tuple[int, str, bool] | None,
    vllm_runner,
    enable_pickle,
):
    """All-rejected MTP padding must preserve constrained greedy token IDs."""
    num_speculative_tokens, model_dtype, dynamic = draft_config or (3, None, False)
    attention_backend = "FLASHINFER" if draft_config else "auto"
    common_kwargs: dict[str, Any] = {
        "block_size": None,
        "max_model_len": 256,
        "max_num_seqs": 3 if dynamic else 1,
        "enable_chunked_prefill": None,
        "limit_mm_per_prompt": {"image": 0, "video": 0},
        "attention_backend": (
            "TRITON_ATTN" if current_platform.is_rocm() else attention_backend
        ),
    }
    if attention_backend == "FLASHINFER":
        common_kwargs.update(
            dtype=model_dtype,
            kv_cache_dtype="auto",
            kv_cache_memory_bytes=256 * 1024**2,
            gpu_memory_utilization=0.3,
            enable_extensible_kv_cache=False,
        )
    if dynamic:
        common_kwargs.update(enable_prefix_caching=False, kv_cache_memory_bytes=1024**3)
    if cudagraph_mode == CUDAGraphMode.NONE:
        common_kwargs["enforce_eager"] = True
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=PLACEHOLDER_REGRESSION_MAX_TOKENS,
        ignore_eos=True,
        allowed_token_ids=[PLACEHOLDER_REGRESSION_ALLOWED_TOKEN_ID],
        logprobs=1,
    )

    def generate_token_ids(
        runner, batch_size: int, expected_drafts: int | None = None
    ) -> tuple[tuple[int, ...], ...]:
        outputs = runner.llm.generate(
            [PLACEHOLDER_REGRESSION_PROMPT] * batch_size, sampling_params
        )
        assert len(outputs) == batch_size
        token_ids = tuple(tuple(output.outputs[0].token_ids) for output in outputs)
        assert all(len(ids) == PLACEHOLDER_REGRESSION_MAX_TOKENS for ids in token_ids)
        if expected_drafts is not None:
            for output in outputs:
                metrics = output.outputs[0].spec_decode_metrics
                assert metrics is not None
                assert metrics.num_spec_tokens == num_speculative_tokens
                assert expected_drafts in metrics.per_step_drafted
                assert all(0 <= k <= expected_drafts for k in metrics.per_step_drafted)
        return token_ids

    def compilation_config() -> CompilationConfig:
        if dynamic:
            return CompilationConfig(
                mode=0 if cudagraph_mode == CUDAGraphMode.NONE else None,
                cudagraph_mode=cudagraph_mode,
                prefill_cudagraph_capture_token_sizes=[],
                decode_cudagraph_capture_request_sizes=[1, 2, 3],
            )
        return CompilationConfig(
            cudagraph_mode=cudagraph_mode,
            cudagraph_capture_sizes=[1, num_speculative_tokens + 1],
        )

    batch_sizes = (1, 2, 3) if dynamic else (1,)
    expected_token_ids = (PLACEHOLDER_REGRESSION_ALLOWED_TOKEN_ID,) * (
        PLACEHOLDER_REGRESSION_MAX_TOKENS
    )
    ref_token_ids = {batch: (expected_token_ids,) * batch for batch in batch_sizes}
    if draft_config is None:
        with vllm_runner(
            PLACEHOLDER_REGRESSION_MODEL,
            compilation_config=compilation_config(),
            **common_kwargs,
        ) as ref_runner:
            ref_mode = (
                ref_runner.llm.llm_engine.vllm_config.compilation_config.cudagraph_mode
            )
            assert ref_mode == cudagraph_mode
            assert {
                batch: generate_token_ids(ref_runner, batch) for batch in batch_sizes
            } == ref_token_ids

    speculative_config: dict[str, Any] = {
        "method": "mtp",
        "num_speculative_tokens": num_speculative_tokens,
        "max_model_len": 256,
        "rejection_sample_method": "synthetic",
        "synthetic_acceptance_rates": [0.0] * num_speculative_tokens,
    }
    if dynamic:
        speculative_config["num_speculative_tokens_per_batch_size"] = [
            (1, 1, 1),
            (2, 2, 2),
            (3, 8, 3),
        ]
    with vllm_runner(
        PLACEHOLDER_REGRESSION_MODEL,
        speculative_config=speculative_config,
        disable_log_stats=False,
        per_request_spec_decode_metrics="detailed",
        compilation_config=compilation_config(),
        **common_kwargs,
    ) as spec_runner:
        spec_mode = (
            spec_runner.llm.llm_engine.vllm_config.compilation_config.cudagraph_mode
        )
        assert spec_mode == cudagraph_mode
        spec_token_ids = {
            batch: generate_token_ids(
                spec_runner, batch, batch if dynamic else num_speculative_tokens
            )
            for batch in batch_sizes
        }
        if attention_backend == "FLASHINFER":
            states = spec_runner.llm.collective_rpc(_flashinfer_mtp_decode_state)
            expected_builder = model_dtype == "bfloat16" and (
                current_platform.is_device_capability_family(120)
            )
            expected_aggregate = expected_builder and num_speculative_tokens > 1
            assert states
            for supports_update, builders in states:
                assert supports_update is expected_aggregate
                assert builders
                for use_xqa, kv_dtype, unquantized, builder_support in builders:
                    assert use_xqa and unquantized
                    assert kv_dtype == f"torch.{model_dtype}"
                    if num_speculative_tokens > 1:
                        assert builder_support is expected_builder
        metrics = spec_runner.llm.get_metrics()
        num_drafts = get_spec_decode_metric_value(
            metrics, "vllm:spec_decode_num_drafts"
        )
        num_draft_tokens = get_spec_decode_metric_value(
            metrics, "vllm:spec_decode_num_draft_tokens"
        )
        num_accepted_tokens = get_spec_decode_metric_value(
            metrics, "vllm:spec_decode_num_accepted_tokens"
        )

    assert num_drafts > 0, "MTP drafter did not run"
    assert num_draft_tokens > 0, "MTP drafter did not produce draft tokens"
    assert num_accepted_tokens == 0, "Synthetic sampler accepted a draft token"
    assert spec_token_ids == ref_token_ids, (
        f"MTP changed greedy token ids under {cudagraph_mode.name}:\n"
        f"  reference={ref_token_ids}\n"
        f"  speculative={spec_token_ids}"
    )
