# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest

from vllm.model_executor.warmup.kernel_warmup import (
    _run_flashinfer_autotune_dummy_runs,
    _run_flashinfer_deferred_moe_autotune,
    _run_flashinfer_mla_decode_autotune,
)

pytestmark = pytest.mark.cpu_test


class _Backend:
    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA"


def _make_runner(
    *,
    use_v2_model_runner: bool,
    flashinfer_mla: bool = True,
    query_len: int = 8,
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    max_model_len: int = 1_048_576,
    dcp_size: int = 1,
):
    backend = _Backend() if flashinfer_mla else SimpleNamespace(get_name=lambda: "X")
    runner = SimpleNamespace(
        _dummy_run=Mock(),
        get_model=Mock(return_value=SimpleNamespace(modules=Mock(return_value=[]))),
        attn_groups=[[SimpleNamespace(backend=backend)]],
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
        ),
        max_model_len=max_model_len,
        model_state=SimpleNamespace(max_model_len=max_model_len),
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(linear_backend=None),
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=dcp_size,
            ),
            use_v2_model_runner=use_v2_model_runner,
        ),
    )
    runner.decode_query_len = query_len
    runner.uniform_decode_query_len = query_len
    return runner


@pytest.mark.parametrize(
    (
        "query_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "max_decode_batch_size",
        "tuning_buckets",
    ),
    [
        (1, 16, 8192, 16, (1, 2, 4, 8, 16)),
        (8, 16, 8192, 16, (1, 2, 4, 8, 16)),
        (8, 16, 100, 12, (1, 2, 4, 8, 12)),
        (8, 128, 800, 100, (1, 2, 4, 8, 16, 32, 64, 100)),
    ],
)
def test_flashinfer_autotune_adds_mla_decode_run_for_v2(
    query_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    max_decode_batch_size: int,
    tuning_buckets: tuple[int, ...],
):
    runner = _make_runner(
        use_v2_model_runner=True,
        query_len=query_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
    )

    with (
        patch("vllm.utils.flashinfer.autotune") as autotune,
        patch(
            "vllm.utils.flashinfer.flashinfer_get_hybrid_num_tokens_buckets",
            return_value=tuning_buckets,
        ) as get_buckets,
        patch(
            "vllm.model_executor.warmup.kernel_warmup."
            "_run_flashinfer_mla_decode_autotune"
        ) as mla_decode_autotune,
    ):
        _run_flashinfer_autotune_dummy_runs(runner)

    # The generic pass needs one full-model dummy. The MLA-specific pass must
    # not run the model again, otherwise it retriggers MoE autotuning.
    runner._dummy_run.assert_called_once_with(
        num_tokens=max_num_batched_tokens,
        skip_eplb=True,
        is_profile=True,
        randomize_inputs=True,
    )
    mla_decode_autotune.assert_called_once_with(
        runner, max_decode_batch_size, query_len, 1_048_576
    )
    get_buckets.assert_called_once_with(max_decode_batch_size)
    autotune.assert_called_once_with(tuning_buckets=tuning_buckets)


def test_flashinfer_autotune_adds_mla_decode_run_for_v1():
    runner = _make_runner(use_v2_model_runner=False)

    with (
        patch("vllm.utils.flashinfer.autotune") as autotune,
        patch(
            "vllm.utils.flashinfer.flashinfer_get_hybrid_num_tokens_buckets",
            return_value=(1, 2, 4, 8, 16),
        ),
        patch(
            "vllm.model_executor.warmup.kernel_warmup."
            "_run_flashinfer_mla_decode_autotune"
        ) as mla_decode_autotune,
    ):
        _run_flashinfer_autotune_dummy_runs(runner)

    assert runner._dummy_run.call_count == 1
    mla_decode_autotune.assert_called_once_with(runner, 16, 8, 1_048_576)
    autotune.assert_called_once_with(tuning_buckets=(1, 2, 4, 8, 16))


class _FakeMoERunner:
    moe_config: Any
    routed_experts: Any


def _make_deferred_moe(hidden_dim: int = 3584):
    import torch

    moe_config = SimpleNamespace(
        use_deferred_moe_finalize=True,
        defer_moe_finalize_max_num_tokens=128,
        should_defer_moe_finalize=Mock(return_value=True),
        hidden_dim=hidden_dim,
        num_experts=896,
        in_dtype=torch.bfloat16,
        router_logits_dtype=torch.bfloat16,
    )
    moe_kernel = SimpleNamespace(supports_deferred_moe_finalize=Mock(return_value=True))
    routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(is_monolithic=True, moe_kernel=moe_kernel),
        w13_weight=torch.empty(0),
        forward_monolithic=Mock(),
    )
    moe = _FakeMoERunner()
    moe.moe_config = moe_config
    moe.routed_experts = routed_experts
    return moe


def _run_deferred_moe_autotune(modules, buckets):
    runner = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
        get_model=Mock(
            return_value=SimpleNamespace(modules=Mock(return_value=modules))
        ),
    )
    with (
        patch("vllm.model_executor.layers.fused_moe.MoERunner", _FakeMoERunner),
        patch(
            "vllm.utils.flashinfer.flashinfer_get_hybrid_num_tokens_buckets",
            return_value=buckets,
        ) as get_buckets,
        patch("vllm.utils.flashinfer.autotune") as autotune,
    ):
        _run_flashinfer_deferred_moe_autotune(runner)
    return get_buckets, autotune


def test_flashinfer_autotune_directly_tunes_deferred_moe_buckets():
    moe = _make_deferred_moe()
    buckets = (1, 2, 4, 8, 16, 32, 64, 128)

    get_buckets, autotune = _run_deferred_moe_autotune([object(), moe], buckets)

    get_buckets.assert_called_once_with(128)
    autotune.assert_called_once_with(tuning_buckets=buckets)
    moe_kernel = moe.routed_experts.quant_method.moe_kernel
    moe.moe_config.should_defer_moe_finalize.assert_called_once_with(128)
    moe_kernel.supports_deferred_moe_finalize.assert_called_once_with()
    moe.routed_experts.forward_monolithic.assert_called_once()
    call = moe.routed_experts.forward_monolithic.call_args.kwargs
    assert call["x"].shape == (128, 3584)
    assert call["router_logits"].shape == (128, 896)


def test_flashinfer_autotune_tunes_each_deferred_moe_geometry_once():
    # Identical layers share one cache key, so only the first needs a
    # dispatcher call; a differently shaped layer still gets its own.
    same_a, same_b = _make_deferred_moe(), _make_deferred_moe()
    other = _make_deferred_moe(hidden_dim=1792)
    buckets = (1, 2, 4, 8, 16, 32, 64, 128)

    _run_deferred_moe_autotune([same_a, same_b, other], buckets)

    same_a.routed_experts.forward_monolithic.assert_called_once()
    same_b.routed_experts.forward_monolithic.assert_not_called()
    other.routed_experts.forward_monolithic.assert_called_once()


@pytest.mark.parametrize("use_v2_model_runner", [False, True])
def test_flashinfer_mla_decode_autotune_uses_initialized_attention_geometry(
    use_v2_model_runner: bool,
):
    import torch

    layer = object()
    block_table = torch.empty((16, 37), dtype=torch.int32)
    group = SimpleNamespace(
        backend=_Backend(),
        layer_names=["model.layers.0.self_attn.attn"],
        kv_cache_group_id=0,
    )
    runner = _make_runner(use_v2_model_runner=use_v2_model_runner)
    runner.attn_groups = [[group]]
    runner.vllm_config.compilation_config = SimpleNamespace(
        static_forward_context={group.layer_names[0]: layer}
    )
    if use_v2_model_runner:
        runner.block_tables = SimpleNamespace(
            get_dummy_block_tables=Mock(return_value=(block_table,))
        )
    else:
        table = SimpleNamespace(get_device_tensor=Mock(return_value=block_table))
        runner.input_batch = SimpleNamespace(
            block_table=SimpleNamespace(block_tables=[table])
        )

    with patch(
        "vllm.v1.attention.backends.mla.flashinfer_mla.flashinfer_mla_decode_autotune"
    ) as decode_autotune:
        _run_flashinfer_mla_decode_autotune(runner, 16, 8, 1_048_576)

    decode_autotune.assert_called_once_with(layer, block_table, 16, 8, 1_048_576)
    if use_v2_model_runner:
        runner.block_tables.get_dummy_block_tables.assert_called_once_with(16)
    else:
        table.get_device_tensor.assert_called_once_with(16)


def test_flashinfer_mla_decode_autotune_builds_uniform_decode_metadata():
    import torch

    from vllm.v1.attention.backends.mla.flashinfer_mla import (
        FlashInferMLAImpl,
        flashinfer_mla_decode_autotune,
    )

    impl = object.__new__(FlashInferMLAImpl)
    impl.forward_mqa = Mock()
    impl.num_heads = 12
    layer = SimpleNamespace(
        impl=impl,
        kv_cache=torch.empty((3, 64, 576), dtype=torch.bfloat16),
        kv_cache_dtype="auto",
        # K3 keeps the global head count on the layer while the backend impl
        # owns the TP-local runtime count used by the decode dispatcher.
        num_heads=96,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    block_table = torch.ones((16, 16392), dtype=torch.int32)

    flashinfer_mla_decode_autotune(layer, block_table, 16, 8, 1_048_576)

    query, kv_cache, metadata, called_layer = impl.forward_mqa.call_args.args
    assert query.shape == (128, 12, 576)
    assert query.dtype == torch.bfloat16
    assert kv_cache is layer.kv_cache
    assert called_layer is layer
    assert metadata.num_decodes == 16
    assert metadata.num_decode_tokens == 128
    assert metadata.max_query_len == 8
    assert metadata.max_seq_len == 1_048_576
    assert metadata.query_start_loc.tolist() == list(range(0, 129, 8))
    assert metadata.decode is not None
    assert metadata.decode.block_table is block_table
    assert metadata.decode.block_table.shape == (16, 16392)
    assert not metadata.decode.block_table.any()
    assert metadata.decode.seq_lens.tolist() == [1_048_576] * 16


def test_flashinfer_autotune_skips_decode_run_for_other_attention_backend():
    runner = _make_runner(use_v2_model_runner=True, flashinfer_mla=False)

    _run_flashinfer_autotune_dummy_runs(runner)

    assert runner._dummy_run.call_count == 1


def test_flashinfer_autotune_skips_mla_decode_run_for_native_dcp():
    runner = _make_runner(use_v2_model_runner=True, dcp_size=2)

    _run_flashinfer_autotune_dummy_runs(runner)

    assert runner._dummy_run.call_count == 1


@pytest.mark.parametrize("query_len", [1, 8])
def test_mla_tuning_keys_cover_full_graph_capture_keys(query_len: int):
    # This intentionally couples to FlashInfer internals. A failure means its
    # MLA cache-key contract changed and this warmup must be revalidated.
    import torch

    mla_core = pytest.importorskip("flashinfer.mla._core")
    from flashinfer.autotuner import AutoTuner
    from flashinfer.fused_moe.utils import get_hybrid_num_tokens_buckets

    def _inputs(batch_size: int) -> list[torch.Tensor]:
        return [
            torch.empty((batch_size, query_len, 8, 576), dtype=torch.bfloat16),
            torch.empty((batch_size, 8192), dtype=torch.int32),
            torch.empty((batch_size,), dtype=torch.int32),
            torch.empty((batch_size, query_len, 8, 512), dtype=torch.bfloat16),
        ]

    def _make_runners(max_seq_len: int):
        kv_cache = torch.empty((1, 1, 64, 576), dtype=torch.bfloat16)
        workspace_buffer = torch.empty(1024, dtype=torch.int8)

        common_attrs = {
            "kv_cache": kv_cache,
            "workspace_buffer": workspace_buffer,
            "qk_nope_head_dim": 128,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "page_size": 64,
            "max_seq_len": max_seq_len,
            "sinks": None,
            "enable_pdl": False,
            "is_var_seq": False,
            "uses_shared_paged_kv_idx": False,
        }

        trt = object.__new__(mla_core.TrtllmGenMlaDecodeRunner)
        trt.__dict__.update(common_attrs)
        trt.sparse_mla_top_k = 0
        trt.bmm1_scale = 1.0
        trt.bmm2_scale = 1.0
        trt.skip_softmax_threshold_scale_factor = None
        trt.return_lse = False

        cute = object.__new__(mla_core.CuteDslMlaDecodeRunner)
        cute.__dict__.update(common_attrs)
        cute._resolved_cute_dsl_impl = "monolithic"
        cute.cute_dsl_impl = "auto"
        cute.enable_dcp = False
        cute.cp_world = 1
        return trt, cute

    default_config = mla_core._mla_decode_tuning_config(
        get_hybrid_num_tokens_buckets(8192),
        num_pages=1,
        profile_seq_len=1_048_576,
    )
    capture_buckets = get_hybrid_num_tokens_buckets(16)
    tuning_config = mla_core._mla_decode_tuning_config(
        capture_buckets,
        num_pages=1,
        profile_seq_len=1_048_576,
    )
    full_context_runners = _make_runners(max_seq_len=1_048_576)
    full_context_cute = full_context_runners[1]
    tuner = AutoTuner()

    for batch_size in capture_buckets:
        inputs = _inputs(batch_size)
        cache_key = AutoTuner._get_cache_key(
            "trtllm_batch_decode_mla",
            full_context_cute,
            tuple(tuple(t.shape) for t in inputs),
            tuning_config,
            full_context_cute.get_cache_key_extras(inputs),
        )
        tuner.profiling_cache[cache_key] = (-1, None)

    # Outside the nested override, every FULL graph capture batch hits the
    # actual FlashInfer CuTe runner key instead of runner 0 (TRTLLM-GEN).
    for batch_size in range(1, 17):
        inputs = _inputs(batch_size)
        is_hit, runner_id, tactic, _ = tuner.search_cache(
            "trtllm_batch_decode_mla",
            list(full_context_runners),
            tuple(tuple(t.shape) for t in inputs),
            default_config,
            inputs,
        )
        assert (is_hit, runner_id, tactic) == (True, 1, -1)

    # The old max_seq_len=8 dummy generates a real CuTe key that cannot cover
    # a FULL capture lookup at max_seq_len=1,048,576. The miss selects runner
    # 0, which is exactly FlashInfer's TRTLLM-GEN fallback behavior.
    short_context_cute = _make_runners(max_seq_len=8)[1]
    inputs = _inputs(8)
    full_context_extras = full_context_cute.get_cache_key_extras(inputs)
    assert (1_048_576 + 127) // 128 in full_context_extras
    short_context_key = AutoTuner._get_cache_key(
        "trtllm_batch_decode_mla",
        short_context_cute,
        tuple(tuple(t.shape) for t in inputs),
        tuning_config,
        short_context_cute.get_cache_key_extras(inputs),
    )
    short_context_tuner = AutoTuner()
    short_context_tuner.profiling_cache[short_context_key] = (-1, None)
    is_hit, runner_id, tactic, _ = short_context_tuner.search_cache(
        "trtllm_batch_decode_mla",
        list(full_context_runners),
        tuple(tuple(t.shape) for t in inputs),
        default_config,
        inputs,
    )
    assert (is_hit, runner_id, tactic) == (False, 0, -1)
