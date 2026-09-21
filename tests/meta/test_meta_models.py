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
  ``expected_quantization``, asserted to stay live on meta (not neutralized to
  bf16).

All cases enable ``SimpleCPUOffloadConnector``; the prompt clears the scheduler
block size (LCM of KV groups, inflated by hybrid GDN/mamba) so a full KV block
offloads. ``expected_offload=True`` replays the same prompt 4x to verify the
two-tier round-trip via ``num_cached_tokens`` + the connector load counter:
cold miss -> L1 (GPU) hit -> [drop L1] L2 (CPU) load hit -> [drop L1+L2] miss.
Other cases only assert the connector is wired.

``load_format="dummy"`` + ``skip_tokenizer_init=True`` avoid all downloads
except ``config.json`` (hub cache / ``HF_HUB_OFFLINE`` / ``VLLM_USE_MODELSCOPE``
for offline CI).

Run: ``VLLM_META_MODE=1 CUDA_VISIBLE_DEVICES="" python -m pytest <this file>``.
The env vars below are also set at import so the file can run directly.
"""

import os

# Must precede the vllm import: VLLM_META_MODE gates the meta branches, empty
# CUDA_VISIBLE_DEVICES keeps torch off any GPU, and in-process engine
# (multiprocessing off) lets the test reach the offload scheduler.
os.environ.setdefault("VLLM_META_MODE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import pytest  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

_META_TOKEN_ID = 100  # constant fake token from the meta sampler
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

# Per-run kwargs. Flags mirror production recipes (TP=8 + EP), single-process
# under meta.
_MODELS = [
    # Kimi-K3 first as a precaution: its KDA/mamba builder was historically
    # sensitive to global state from a prior in-process build (e.g. DeepSeek-V4).
    pytest.param(
        dict(
            model="moonshotai/Kimi-K3",
            tensor_parallel_size=8,
            enable_expert_parallel=True,
            language_model_only=True,  # native multimodal; text LM only
            # KDA "align" mode needs aligned Mamba state indices (normally a
            # triton/CUDA kernel, no-op on meta); a meta branch in kda_metadata.py
            # computes them via pure-torch gather, enabling full offload.
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
            # runs the native deepseek_v4_fp8 path. Sparse-MLA CuteDSL indexer +
            # CuteDSL/FA4 forward are meta-gated, so the full prompt runs.
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
]


@pytest.fixture(autouse=True)
def _cleanup_dist_env():
    """Tear down the in-process fake-TP group after each case, else the next
    LLM build trips the "tp group already initialized" assert."""
    yield
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    cleanup_dist_env_and_memory()


def _generate(llm: LLM, prompt: TokensPrompt, sampling: SamplingParams) -> int:
    """Generate once, assert the constant fake output, return num_cached_tokens
    (local L1 + external L2 prefix hits)."""
    outputs = llm.generate(prompt, sampling)
    token_ids = list(outputs[0].outputs[0].token_ids)
    assert token_ids == [_META_TOKEN_ID] * _MAX_TOKENS
    return outputs[0].num_cached_tokens


@pytest.mark.parametrize("llm_kwargs", _MODELS)
def test_meta_model_e2e(llm_kwargs: dict) -> None:
    """Build + generate on meta with CPU KV-offload triggered."""
    assert os.environ.get("VLLM_META_MODE") == "1", (
        "VLLM_META_MODE=1 must be set before importing vllm for this test."
    )

    # Copy, then peel off the test-only keys.
    llm_kwargs = dict(llm_kwargs)
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
        load_format="dummy",  # no weight download; synthesized on meta
        trust_remote_code=True,
        skip_tokenizer_init=True,  # no tokenizer download; pass token ids
        enforce_eager=True,
        max_model_len=_MAX_MODEL_LEN,
        **llm_kwargs,
    )

    if expected_quantization is not None:
        # Scheme must stay live (not neutralized to bf16).
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
