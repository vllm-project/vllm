# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end ``VLLM_META_MODE`` capability tests for frontier model families,
CPU-only (no GPU).

Each case builds the arch and runs ``LLM.generate`` on the ``meta`` device
(zero-memory weights + KV, simulated GPU async-output path): validates
construct + run (config, TP/EP, attention/MoE, quant, sampling), not numerics.

Entries (single ``test_meta_model_e2e`` parametrization):
* Real families by hub ID.
* Quant variants: ``quantization_config`` layered via ``hf_overrides`` +
  ``expected_quantization``, asserted to remain the requested scheme.

All cases enable ``SimpleCPUOffloadConnector``; the prompt clears the scheduler
block size (LCM of KV groups, inflated by hybrid GDN/mamba) so a full KV block
offloads. ``expected_offload=True`` replays the same prompt 4x to verify the
two-tier round-trip via ``num_cached_tokens`` + the connector load counter:
cold miss -> L1 (GPU) hit -> [drop L1] L2 (CPU) load hit -> [drop L1+L2] miss.
Other cases only assert the connector is wired.

``VLLM_META_MODE`` forces ``load_format="meta"`` and eager execution; with
``skip_tokenizer_init=True`` this avoids all downloads except ``config.json``
(hub cache / ``HF_HUB_OFFLINE`` / ``VLLM_USE_MODELSCOPE`` for offline CI).

Run: ``VLLM_META_MODE=1 CUDA_VISIBLE_DEVICES="" python -m pytest <this file>``.
The env vars below are also set at import so the file can run directly.
"""

import os

# Must precede the vllm import: VLLM_META_MODE gates the meta branches, empty
# CUDA_VISIBLE_DEVICES keeps torch off any GPU, and the in-process EngineCore
# keeps its scheduler inspectable while model ranks use MultiprocExecutor.
os.environ.setdefault("VLLM_META_MODE", "1")
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import pytest  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402
from vllm.meta_mode import META_TOKEN_ID  # noqa: E402

_MAX_TOKENS = 8
# > LCM scheduler block size so offload stores >=1 block. Hybrid GDN/mamba
# inflates it (Qwen3.8: 8320); 8704 clears that, plain MoEs offload far below.
_PROMPT_LEN = 8704
_MAX_MODEL_LEN = 16384

# eager (lazy_offload=False) stores blocks as computed. cpu_bytes sized so the
# offload pool has >=2 blocks (block 0 is the reserved null) even for hybrid
# per-block KV; worker transfers are meta-stubbed (pure bookkeeping, no memory).
_OFFLOAD_KV_TRANSFER = {
    "kv_connector": "SimpleCPUOffloadConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "cpu_bytes_to_use_per_rank": 1024 * 1024 * 1024,
        "lazy_offload": False,
    },
}

# Per-run kwargs. Flags mirror production recipes; parallel ranks are real
# processes using Gloo, while weights and KV remain on meta.
_MODELS = [
    pytest.param(
        dict(
            model="moonshotai/Kimi-K3",
            tensor_parallel_size=8,
            enable_expert_parallel=True,
            language_model_only=True,  # native multimodal; text LM only
            # KDA "align" mode needs aligned Mamba state indices (normally a
            # triton/CUDA kernel in serving; MRV2 computes equivalent indices
            # from its value-carrying host block table in meta mode.
            expected_offload=True,
        ),
        id="Kimi-K3",
    ),
    pytest.param(
        dict(
            model="Qwen/Qwen3.8-2.4T-A95B-FP8",
            tensor_parallel_size=8,
            enable_expert_parallel=True,
            # Native block-wise [128,128] FP8 on a hybrid GDN + full-attention
            # text MoE; full-attention KV group offloads past the LCM (8320).
            expected_quantization="fp8",
            expected_offload=True,
        ),
        id="Qwen3.8-2.4T-A95B-FP8",
    ),
    pytest.param(
        dict(
            model="deepseek-ai/DeepSeek-V4-Pro",
            tensor_parallel_size=8,
            enable_expert_parallel=True,
            # fp8_ds_mla KV layout needs an fp8 kv-cache dtype; fp8 checkpoint
            # runs the native deepseek_v4_fp8 path. CuteDSL/FA4 launches are
            # intercepted below the model, so the full prompt runs.
            kv_cache_dtype="fp8",
            expected_quantization="deepseek_v4_fp8",
        ),
        id="DeepSeek-V4-Pro",
    ),
    pytest.param(
        dict(
            model="zai-org/GLM-5",
            tensor_parallel_size=8,
            enable_expert_parallel=True,
            # GlmMoeDsaForCausalLM (DeepSeek Sparse Attention). Sparse-MLA decode
            # uses the FA3 fwd op (_vllm_fa3_C::fwd), now meta-faked.
        ),
        id="GLM-5",
    ),
    # Quant coverage: text MoE (no vision tower, so the scheme applies cleanly)
    # with the scheme injected via hf_overrides, asserted live. Plain
    # full-attention MoEs -> deterministic offload + full L1/L2 round-trip.
    pytest.param(
        dict(
            model="Qwen/Qwen3-30B-A3B",
            tensor_parallel_size=8,
            enable_expert_parallel=True,
            hf_overrides={
                "quantization_config": {
                    "quant_method": "fp8",
                    "activation_scheme": "dynamic",
                }
            },
            expected_quantization="fp8",
            expected_offload=True,
        ),
        id="quant-fp8-per-tensor",
    ),
    pytest.param(
        dict(
            model="Qwen/Qwen3-30B-A3B",
            tensor_parallel_size=8,
            enable_expert_parallel=True,
            hf_overrides={
                "quantization_config": {
                    "quant_method": "modelopt",
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                }
            },
            expected_quantization="modelopt_fp4",
            expected_offload=True,
        ),
        id="quant-nvfp4",
    ),
    pytest.param(
        dict(
            model="Qwen/Qwen3-30B-A3B",
            pipeline_parallel_size=2,
            hf_overrides={"num_hidden_layers": 2},
            prompt_len=16,
            offload=False,
        ),
        id="parallel-pp",
    ),
    pytest.param(
        dict(
            model="deepseek-ai/DeepSeek-V2-Lite",
            prefill_context_parallel_size=2,
            hf_overrides={"num_hidden_layers": 1},
            prompt_len=16,
            offload=False,
        ),
        id="parallel-pcp",
    ),
    pytest.param(
        dict(
            model="Qwen/Qwen3-30B-A3B",
            tensor_parallel_size=2,
            decode_context_parallel_size=2,
            hf_overrides={
                "num_hidden_layers": 1,
                "num_key_value_heads": 1,
            },
            prompt_len=16,
            offload=False,
        ),
        id="parallel-dcp",
    ),
]


@pytest.fixture(autouse=True)
def _cleanup_dist_env():
    """Clear any process groups retained in the parent process."""
    yield
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    cleanup_dist_env_and_memory()


def _generate(llm: LLM, prompt: TokensPrompt, sampling: SamplingParams) -> int:
    """Generate once, assert the constant fake output, return num_cached_tokens
    (local L1 + external L2 prefix hits)."""
    outputs = llm.generate(prompt, sampling)
    token_ids = list(outputs[0].outputs[0].token_ids)
    assert token_ids == [META_TOKEN_ID] * _MAX_TOKENS
    return outputs[0].num_cached_tokens


@pytest.mark.parametrize("llm_kwargs", _MODELS)
def test_meta_model_e2e(llm_kwargs: dict, request: pytest.FixtureRequest) -> None:
    """Build + generate on meta with CPU KV-offload triggered."""
    assert os.environ.get("VLLM_META_MODE") == "1", (
        "VLLM_META_MODE=1 must be set before importing vllm for this test."
    )

    # Copy, then peel off the test-only keys.
    llm_kwargs = dict(llm_kwargs)
    requested_parallel = {
        name: llm_kwargs.get(name, 1)
        for name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
            "data_parallel_size",
        )
    }
    expected_quantization = llm_kwargs.pop("expected_quantization", None)
    expected_offload = llm_kwargs.pop("expected_offload", False)
    prompt_len = llm_kwargs.pop("prompt_len", _PROMPT_LEN)
    offload = llm_kwargs.pop("offload", True)
    # Offload requires prefix caching; opting out also disables it (v1 default
    # is on) to keep the connector/prefix-cache paths out of the way.
    if offload:
        llm_kwargs.setdefault("enable_prefix_caching", True)
        llm_kwargs.setdefault("kv_transfer_config", _OFFLOAD_KV_TRANSFER)
    else:
        llm_kwargs.setdefault("enable_prefix_caching", False)

    llm = LLM(
        trust_remote_code=True,
        skip_tokenizer_init=True,  # no tokenizer download; pass token ids
        max_model_len=_MAX_MODEL_LEN,
        **llm_kwargs,
    )
    request.addfinalizer(llm.llm_engine.engine_core.shutdown)
    assert llm.llm_engine.vllm_config.use_v2_model_runner

    parallel_config = llm.llm_engine.vllm_config.parallel_config
    for name, expected in requested_parallel.items():
        assert getattr(parallel_config, name) == expected
    model_world_size = (
        parallel_config.tensor_parallel_size
        * parallel_config.pipeline_parallel_size
        * parallel_config.prefill_context_parallel_size
    )
    expected_backend = "mp" if model_world_size > 1 else "uni"
    assert parallel_config.distributed_executor_backend == expected_backend

    if expected_quantization is not None:
        # The configured scheme must remain unchanged.
        model_config = llm.llm_engine.vllm_config.model_config
        assert model_config.quantization == expected_quantization

    prompt = TokensPrompt(prompt_token_ids=list(range(1, prompt_len + 1)))
    sampling = SamplingParams(max_tokens=_MAX_TOKENS, temperature=0.0)

    if expected_offload:
        # Same prompt 4x: cold miss (stores L2) -> L1 hit -> [drop L1] L2 load
        # hit -> [drop L1+L2] cold miss. num_cached_tokens shows a hit; the load
        # counter distinguishes L1 (unchanged) from L2 (incremented).
        scheduler = llm.llm_engine.engine_core.engine_core.scheduler
        offload_mgr = scheduler.connector.scheduler_manager
        assert offload_mgr is not None, (
            "CPU offload connector scheduler_manager missing"
        )

        assert _generate(llm, prompt, sampling) == 0, "R1 should be a cold miss"

        load_before = offload_mgr._load_event_counter
        assert _generate(llm, prompt, sampling) > 0, "R2 should hit the L1 cache"
        assert offload_mgr._load_event_counter == load_before, (
            "R2 must be served from the GPU prefix cache (L1), not a CPU load"
        )

        llm.reset_prefix_cache()  # drop L1 only
        load_before = offload_mgr._load_event_counter
        assert _generate(llm, prompt, sampling) > 0, "R3 should hit the L2 cache"
        assert offload_mgr._load_event_counter > load_before, (
            "R3 must be served from the CPU offload tier (L2 external load)"
        )

        assert llm.reset_prefix_cache(reset_connector=True), (
            "reset_prefix_cache(reset_connector=True) should wipe L1 + L2"
        )
        assert _generate(llm, prompt, sampling) == 0, (
            "R4 should be a cold miss after both tiers are cleared"
        )
    else:
        # Run once; if offload is enabled, confirm the connector was wired.
        _generate(llm, prompt, sampling)
        if offload:
            scheduler = llm.llm_engine.engine_core.engine_core.scheduler
            assert scheduler.connector.scheduler_manager is not None, (
                "CPU offload connector scheduler_manager missing"
            )


if __name__ == "__main__":
    # Keep this standalone suite independent of repository-level test fixtures.
    import sys

    sys.exit(pytest.main([__file__, "-v", "--noconftest", "-p", "no:cacheprovider"]))
