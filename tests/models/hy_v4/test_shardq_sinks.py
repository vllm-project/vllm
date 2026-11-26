# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

from vllm.config import (
    AttentionConfig,
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.model_executor.model_loader.weight_utils import finalize_model_weights
from vllm.models.hy_v4.nvidia.attention import HYV4MLAAttention
from vllm.models.hy_v4.nvidia.flashmla_sparse import HYV4FlashMLASparseImpl
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.hy_v4 import HYV4Config
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

_SINK_CASES = [(1, 64, False), (2, 32, True), (8, 64, False)]
if current_platform.is_rocm():
    # TP2 at 32 heads checks the head-tile gate; 16 keeps ShardQ on at TP2.
    _SINK_CASES.append((2, 16, True))


def _rocm_sink_check(layer, rank, world, device):
    """Returns a check that the real `forward_mqa` applies this rank's sinks to
    local heads and, from a graph captured once, the reloaded full sinks to
    ShardQ's all-head query."""
    from vllm import _custom_ops as ops
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseImpl,
    )

    impl = layer.mla_attn.impl
    assert type(impl) is ROCMAiterMLASparseImpl
    num_heads = impl.num_heads * world
    block_size, num_slots, rank_dim = 64, 128, impl.kv_lora_rank
    gen = torch.Generator(device).manual_seed(0)
    kv_c = torch.randn(num_slots, rank_dim, device=device, generator=gen) * 0.5
    k_pe = torch.randn(num_slots, 64, device=device, generator=gen) * 0.5
    cache = torch.zeros(num_slots // block_size, block_size, 656, dtype=torch.uint8)
    ops.concat_and_cache_mla(
        kv_c.bfloat16(),
        k_pe.bfloat16(),
        cache,
        torch.arange(num_slots),
        kv_cache_dtype="fp8_ds_mla",
        scale=torch.ones(1),
    )
    rows = cache.view(num_slots, 656)
    nope = rows[:, :rank_dim].contiguous().view(torch.float8_e4m3fn).float()
    scales = rows[:, rank_dim : rank_dim + 16].contiguous().view(torch.float32)
    rope = rows[:, rank_dim + 16 :].contiguous().view(torch.bfloat16).float()
    keys = torch.cat((nope * scales.repeat_interleave(128, dim=1), rope), dim=1)

    # Few live keys keep the sink's share above BF16 noise.
    live_counts = [64, 1, 64, 17]
    num_tokens = len(live_counts)
    topk = impl.topk_indices_buffer
    assert topk is not None
    topk[:num_tokens] = -1
    for i, count in enumerate(live_counts):
        topk[i, :count] = torch.randperm(num_slots, device=device)[:count]
    live = [topk[i, :count].long() for i, count in enumerate(live_counts)]
    metadata = SimpleNamespace(
        num_actual_tokens=num_tokens,
        num_prefills=0,
        topk_start=0,
        topk_tokens=topk.shape[1],
        block_size=block_size,
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        req_id_per_token=torch.zeros(num_tokens, dtype=torch.int32),
        paged_kv_indptr=torch.tensor([0, *live_counts]).cumsum(0).int(),
        paged_kv_indices=torch.zeros(num_tokens * topk.shape[1], dtype=torch.int32),
        attn_out_dtype=torch.bfloat16,
        remapped_buf=None,
        remapped_epoch=-1,
    )
    full_q = torch.randn(num_tokens, num_heads, 576, device=device, generator=gen)
    full_q = (full_q * 0.5).bfloat16()
    heads = slice(rank * impl.num_heads, (rank + 1) * impl.num_heads)
    local_q = full_q[:, heads].contiguous()

    def reference(q, sinks):
        out = torch.empty(q.shape[:2] + (rank_dim,))
        for i, slots in enumerate(live):
            scores = q[i].float() @ keys[slots].T * impl.scale
            lse = torch.logsumexp(torch.cat((scores, sinks[:, None]), dim=1), dim=1)
            out[i] = (torch.exp(scores - lse[:, None]) @ keys[slots, :rank_dim]).cpu()
        return out

    graph = None
    graph_out = None

    def check(checkpoint_sinks):
        nonlocal graph, graph_out
        local = impl._attn_sink(local_q)
        assert local is not None
        torch.testing.assert_close(local, checkpoint_sinks[heads])
        full = impl._attn_sink(full_q)
        assert full is not None
        torch.testing.assert_close(full, checkpoint_sinks)
        with torch.no_grad():
            out, _ = impl.forward_mqa(local_q, cache, metadata, layer.mla_attn)
            if graph is None:
                impl.forward_mqa(full_q, cache, metadata, layer.mla_attn)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_out, _ = impl.forward_mqa(
                        full_q, cache, metadata, layer.mla_attn
                    )
            graph.replay()
        tol = dict(atol=2e-2, rtol=2e-2)
        expected = reference(local_q, checkpoint_sinks[heads])
        torch.testing.assert_close(out.float().cpu(), expected, **tol)
        expected = reference(full_q, checkpoint_sinks)
        torch.testing.assert_close(graph_out.float().cpu(), expected, **tol)

    return check


def _check_sinks(rank, world, num_heads, offload, port, model_path):
    attention_cls = HYV4MLAAttention
    if current_platform.is_rocm():
        from vllm.models.hy_v4.amd.attention import HYV4MLAAttention as AMDAttention

        attention_cls = AMDAttention
    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)
    config = VllmConfig(
        model_config=ModelConfig(
            model=model_path, dtype="bfloat16", skip_tokenizer_init=True
        ),
        cache_config=CacheConfig(cache_dtype="fp8_ds_mla"),
        attention_config=AttentionConfig(sparse_mla_shard_q=True),
        parallel_config=ParallelConfig(tensor_parallel_size=world),
        scheduler_config=SchedulerConfig(
            max_model_len=8192,
            is_encoder_decoder=False,
            max_num_batched_tokens=128,
            max_num_seqs=8,
        ),
    )
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=world,
            rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            local_rank=rank,
            backend="nccl",
        )
        ensure_model_parallel_initialized(world, 1)
        init_workspace_manager(device)
        heads = num_heads // world
        with set_default_torch_dtype(torch.bfloat16), torch.device(device):
            layer = attention_cls(
                vllm_config=config,
                config=config.model_config.hf_text_config,
                hidden_size=256,
                num_heads=num_heads,
                q_lora_rank=128,
                kv_lora_rank=512,
                qk_nope_head_dim=192,
                qk_rope_head_dim=64,
                v_head_dim=256,
                cache_config=config.cache_config,
                prefix="model.layers.0.self_attn",
                topk_indices_buffer=torch.empty(128, 2048, dtype=torch.int32),
            )
        sinks = layer.learnable_sink_param
        impl = layer.mla_attn.impl
        # ROCm turns ShardQ off when local heads fill AITER's 16-head tile.
        gated = current_platform.is_rocm() and world > 1 and heads >= 16
        assert layer.mla_attn.use_shardq == (world > 1 and not gated)
        rocm_check = None
        if current_platform.is_rocm() and not gated:
            with torch.device(device):
                rocm_check = _rocm_sink_check(layer, rank, world, device)
        local_query = torch.zeros(2, 1, heads, 576, device=device)
        full_query = torch.zeros(2, 1, num_heads, 576, device=device)
        graph = None
        for offset in () if gated else (0, 2):
            if offload:
                layer.cpu()
            checkpoint_sinks = torch.linspace(-2, 2, num_heads, device=device) + offset
            sinks.weight_loader(sinks, checkpoint_sinks.to(sinks.device))
            finalize_model_weights(layer, device)
            layer.to(device)
            if rocm_check is not None:
                rocm_check(checkpoint_sinks)
                continue
            assert isinstance(impl, HYV4FlashMLASparseImpl)
            local = impl._sinks_for_query(local_query, 2, 64)
            assert local is not None
            torch.testing.assert_close(
                local[:heads], checkpoint_sinks[rank * heads : (rank + 1) * heads]
            )
            assert torch.isneginf(local[heads:]).all()
            full = impl._sinks_for_query(full_query, 2, 64)
            assert full is not None
            torch.testing.assert_close(full[:num_heads], checkpoint_sinks)
            assert torch.isneginf(full[num_heads:]).all()
            if graph is None:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_sinks = impl._sinks_for_query(full_query, 2, 64)
                    assert graph_sinks is not None
                    captured_sinks = graph_sinks.clone()
            graph.replay()
            torch.testing.assert_close(captured_sinks[:num_heads], checkpoint_sinks)
            assert torch.isneginf(captured_sinks[num_heads:]).all()
        if not current_platform.is_rocm():
            assert isinstance(impl, HYV4FlashMLASparseImpl)
            with pytest.raises(ValueError, match="head count must match"):
                impl._sinks_for_query(full_query[:, :, :3], 2, 64)
        reset_workspace_manager()
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize("world,num_heads,offload", _SINK_CASES)
def test_local_and_shardq_sinks_follow_checkpoint_reload(
    world, num_heads, offload, tmp_path, monkeypatch
):
    if torch.accelerator.device_count() < world:
        pytest.skip(f"Requires {world} GPUs")
    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx950

        if not on_gfx950():
            pytest.skip("fp8_ds_mla sparse MLA requires gfx950")
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    elif not current_platform.is_cuda():
        pytest.skip("Requires CUDA or ROCm")
    config = HYV4Config(
        hidden_size=256,
        num_hidden_layers=1,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        q_lora_rank=128,
        max_position_embeddings=8192,
        layer_types=["deepseek_sparse_attention"],
    )
    config.architectures = ["HYV4ForCausalLM"]
    config.save_pretrained(tmp_path)
    mp.spawn(
        _check_sinks,
        args=(world, num_heads, offload, get_open_port(), str(tmp_path)),
        nprocs=world,
    )
