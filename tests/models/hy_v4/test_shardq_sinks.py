# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.multiprocessing as mp

from vllm.config import (
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


def _check_sinks(rank, world, num_heads, offload, port, model_path):
    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)
    config = VllmConfig(
        model_config=ModelConfig(
            model=model_path, dtype="bfloat16", skip_tokenizer_init=True
        ),
        cache_config=CacheConfig(cache_dtype="fp8_ds_mla"),
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
            layer = HYV4MLAAttention(
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
        assert isinstance(impl, HYV4FlashMLASparseImpl)
        assert layer.mla_attn.use_shardq == (world > 1)
        local_query = torch.zeros(2, 1, heads, 576, device=device)
        full_query = torch.zeros(2, 1, num_heads, 576, device=device)
        graph = None
        for offset in (0, 2):
            if offload:
                layer.cpu()
            checkpoint_sinks = torch.arange(num_heads, device=device).float() + offset
            sinks.weight_loader(sinks, checkpoint_sinks.to(sinks.device))
            finalize_model_weights(layer, device)
            layer.to(device)
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
        with pytest.raises(ValueError, match="head count must match"):
            impl._sinks_for_query(full_query[:, :, :3], 2, 64)
        reset_workspace_manager()
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize(
    "world,num_heads,offload", [(1, 64, False), (2, 32, True), (8, 64, False)]
)
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_local_and_shardq_sinks_follow_checkpoint_reload(
    world, num_heads, offload, tmp_path
):
    if torch.accelerator.device_count() < world:
        pytest.skip(f"Requires {world} GPUs")
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
