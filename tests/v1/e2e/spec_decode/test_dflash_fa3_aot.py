# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.models.utils import check_logprobs_close

from .utils import compute_acceptance_rate


@pytest.mark.parametrize(
    ("draft_window", "full_allocation"), [(None, False), (128, False), (128, True)]
)
def test_draft_aot_uses_own_window_after_group_merge(
    dist_init, draft_window, full_allocation
):
    from vllm.config import (
        AttentionConfig,
        CacheConfig,
        CompilationConfig,
        ModelConfig,
        SchedulerConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.attention.backends.flash_attn import (
        FlashAttentionBackend,
        FlashAttentionImpl,
        FlashAttentionMetadataBuilder,
    )
    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
    from vllm.v1.core.kv_cache_planning import get_kv_cache_groups
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        get_kv_cache_spec_sliding_window,
    )
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
        _disable_aot_schedule_for_sliding_window_drafts,
    )
    from vllm.v1.worker.utils import AttentionGroup

    config = VllmConfig(
        model_config=ModelConfig(
            model="Qwen/Qwen3-4B",
            revision="1cfa9a7208912126459214e8b04321603b3df60c",
            dtype="bfloat16",
            max_model_len=512,
        ),
        cache_config=CacheConfig(block_size=16),
        compilation_config=CompilationConfig(cudagraph_mode="NONE"),
        scheduler_config=SchedulerConfig(
            max_model_len=512,
            is_encoder_decoder=False,
            max_num_seqs=2,
            max_num_batched_tokens=512,
            disable_hybrid_kv_cache_manager=True,
        ),
        attention_config=AttentionConfig(flash_attn_version=3),
    )
    with set_current_vllm_config(config):
        layers = {
            name: Attention(
                1,
                128,
                128**-0.5,
                num_kv_heads=1,
                cache_config=config.cache_config,
                per_layer_sliding_window=window,
                prefix=name,
                attn_backend=backend,
            )
            for name, window, backend in (
                ("target", None if draft_window else 128, TritonAttentionBackend),
                ("draft", draft_window, FlashAttentionBackend),
            )
        }
        if full_allocation:
            # Match DFlashLagunaModel's allocation-only override.
            layers["draft"].sliding_window = None
        for impl in layers["draft"].impl.get_impl_variants():
            assert isinstance(impl, FlashAttentionImpl)
            assert impl.sliding_window == ((127, 0) if draft_window else (-1, -1))
        specs = {
            name: spec
            for name, layer in layers.items()
            if (spec := layer.get_kv_cache_spec(config)) is not None
        }
        assert len(specs) == 2
        groups = get_kv_cache_groups(config, specs)
        assert len(groups) == 1
        spec = groups[0].kv_cache_spec
        assert isinstance(spec, FullAttentionSpec)
        assert spec.sliding_window == (None if full_allocation else 128)
        assert get_kv_cache_spec_sliding_window(spec) is None
        builder = FlashAttentionMetadataBuilder(
            spec, ["draft"], config, torch.device("cpu")
        )
        group = AttentionGroup(
            FlashAttentionBackend, ["draft"], spec, 0, metadata_builders=[builder]
        )
        assert builder.aot_schedule
        _disable_aot_schedule_for_sliding_window_drafts(config, [[group]])
        assert builder.aot_schedule is (draft_window is None)


def _windowed_dflash(config):
    # Configure a real trained full-attention head as a windowed regression fixture.
    if getattr(config, "dflash_config", None):
        config.sliding_window = 128
        config.use_sliding_window = True
        config.dflash_config = {**config.dflash_config, "use_swa": True}
    return config


def _dflash_aot_state(worker):
    from vllm.config import get_layers_from_vllm_config
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.kv_cache_interface import get_kv_cache_spec_sliding_window

    draft = worker.model_runner.speculator
    layers = get_layers_from_vllm_config(
        draft.vllm_config, Attention, draft.draft_attn_layer_names
    )
    builders = [
        group.get_metadata_builder() for groups in draft.attn_groups for group in groups
    ]
    manager = draft.query_cudagraph_manager
    return {
        "windows": [layer.sliding_window for layer in layers.values()],
        "builders": [
            {
                "type": type(builder).__name__,
                "spec": type(builder.kv_cache_spec).__name__,
                "spec_window": builder.kv_cache_spec.sliding_window,
                "allocation_window": get_kv_cache_spec_sliding_window(
                    builder.kv_cache_spec
                ),
                "aot": builder.aot_schedule,
            }
            for builder in builders
        ],
        "graphs": len(manager.graphs),
        "dispatch": [
            manager.dispatch(
                num_reqs=n,
                num_tokens=n * draft.num_query_per_req,
                uniform_token_count=draft.num_query_per_req,
                num_active_loras=0,
            ).cg_mode.name
            for n in (1, 2)
        ],
    }


def test_dflash_fa3_window_guard_survives_cache_promotion(monkeypatch, vllm_runner):
    """Keep the real FA3 guard through full allocation and FULL graph decoding."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    outputs = []
    for window, promote in ((None, True), (128, False), (128, True)):
        with vllm_runner(
            "Qwen/Qwen3-4B",
            revision="1cfa9a7208912126459214e8b04321603b3df60c",
            dtype="bfloat16",
            max_model_len=4096,
            max_num_batched_tokens=512,
            max_num_seqs=2,
            kv_cache_memory_bytes=2 * 1024**3,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            async_scheduling=False,
            enforce_eager=False,
            disable_hybrid_kv_cache_manager=promote,
            disable_log_stats=False,
            hf_overrides=_windowed_dflash if window else None,
            attention_config={"backend": "TRITON_ATTN", "flash_attn_version": 3},
            compilation_config={"cudagraph_capture_sizes": [16, 32]},
            speculative_config={
                "method": "dflash",
                "model": "z-lab/Qwen3-4B-DFlash-b16",
                "revision": "b74e3a329c4d963783143b1e970d95b002be72bd",
                "num_speculative_tokens": 15,
                "attention_backend": "FLASH_ATTN",
            },
        ) as model:
            tokenizer = model.llm.get_tokenizer()
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for content in (
                    "What is 2 + 2? Give the answer only.",
                    "The quick brown fox jumps over the lazy dog. " * 260
                    + "What animal jumps? Answer briefly.",
                )
            ]
            assert len(tokenizer.encode(prompts[1])) > 128
            outputs.append(model.generate_greedy_logprobs(prompts, 32, 5))
            state = model.llm.collective_rpc(_dflash_aot_state)[0]
            assert state["windows"] and set(state["windows"]) == {window}
            assert state["builders"]
            for builder in state["builders"]:
                assert builder["type"] == "FlashAttentionMetadataBuilder"
                assert builder["spec"] == (
                    "SlidingWindowSpec"
                    if window and not promote
                    else "FullAttentionSpec"
                )
                assert builder["spec_window"] == window
                assert builder["allocation_window"] == (window if not promote else None)
                assert builder["aot"] is (window is None)
            assert state["graphs"] >= 2
            assert state["dispatch"] == ["FULL", "FULL"]
            metrics = model.llm.get_metrics()
            assert (
                next(
                    metric.value
                    for metric in metrics
                    if metric.name == "vllm:spec_decode_num_drafts"
                )
                > 0
            )
            assert compute_acceptance_rate(metrics) > 0
    for i in (1, 2):
        check_logprobs_close(
            outputs_0_lst=outputs[0],
            outputs_1_lst=outputs[i],
            name_0="full-attention draft control",
            name_1="windowed draft with "
            + ("full" if i == 2 else "native")
            + " allocation",
            always_check_logprobs=True,
        )
