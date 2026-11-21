# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the model constructors, weight loader and rank-local Q view."""

import pytest
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn

from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm() or torch.accelerator.device_count() < 2,
    reason="Requires two ROCm GPUs",
)


def _check_q_replication(rank, model_path, port, dcp):
    from vllm.config import (
        CacheConfig,
        ModelConfig,
        ParallelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed import (
        destroy_model_parallel,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        DCPGroupColumnParallelLinear,
    )
    from vllm.models.kimi_k3.amd.dsa import KimiK3DSAAttention
    from vllm.models.kimi_k3.amd.linear import KimiMLAAttention
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

    torch.accelerator.set_device_index(rank)
    device = torch.device(f"cuda:{rank}")
    init_workspace_manager(device)
    config = VllmConfig(
        model_config=ModelConfig(model=model_path, dtype="bfloat16"),
        cache_config=CacheConfig(cache_dtype="bfloat16"),
        parallel_config=ParallelConfig(
            tensor_parallel_size=2,
            decode_context_parallel_size=dcp,
            dcp_q_replicate=True,
        ),
    )
    with set_current_vllm_config(config), set_default_torch_dtype(torch.bfloat16):
        init_distributed_environment(
            world_size=2,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend=current_platform.dist_backend,
        )
        initialize_model_parallel(
            tensor_model_parallel_size=2, decode_context_model_parallel_size=dcp
        )
        try:
            with torch.device(device), torch.inference_mode():
                for attention in (KimiMLAAttention, KimiK3DSAAttention):
                    for q_lora_rank in (None, 256):
                        kwargs = (
                            {"use_nope": True}
                            if attention is KimiMLAAttention
                            else {
                                "topk_indices_buffer": torch.empty(
                                    (16, 2048), dtype=torch.int32
                                )
                            }
                        )
                        layer = attention(
                            config.model_config.hf_text_config,
                            hidden_size=256,
                            num_heads=96,
                            qk_nope_head_dim=128,
                            qk_rope_head_dim=64,
                            v_head_dim=128,
                            q_lora_rank=q_lora_rank,
                            kv_lora_rank=512,
                            cache_config=CacheConfig(
                                cache_dtype="bfloat16"
                                if attention is KimiMLAAttention
                                else "fp8_ds_mla"
                            ),
                            prefix=f"model.layers.0.{attention.__name__}_{q_lora_rank}",
                            **kwargs,
                        )
                        projection = (
                            layer.q_proj if q_lora_rank is None else layer.q_b_proj
                        )
                        assert isinstance(projection, DCPGroupColumnParallelLinear) == (
                            dcp > 1
                        )
                        assert layer.mla_attn.dcp_q_replicate == (dcp > 1)
                        reference = ColumnParallelLinear(256, 96 * 192, bias=False)
                        weights = torch.zeros(96 * 192, 256, dtype=torch.bfloat16)
                        # Distinct, exact heads detect wrong-group loading.
                        weights[:, 0] = (torch.arange(96 * 192) // 192 + 1) / 128
                        projection.weight.weight_loader(projection.weight, weights)
                        reference.weight.weight_loader(reference.weight, weights)
                        x = torch.zeros(3, 256, dtype=torch.bfloat16)
                        for multiplier in (1, 2):
                            x[:, 0] = multiplier
                            grouped = projection(x)[0].view(3, -1, 192)
                            local = (
                                projection._local_view(grouped) if dcp > 1 else grouped
                            )
                            local = local.reshape(3, -1)
                            torch.testing.assert_close(
                                local, reference(x)[0], rtol=0, atol=0
                            )
                            gathered = get_tp_group().all_gather(local, dim=-1)
                            torch.testing.assert_close(
                                gathered, x @ weights.T, rtol=0, atol=0
                            )
        finally:
            destroy_model_parallel()
            dist.destroy_process_group()
            reset_workspace_manager()


@pytest.mark.parametrize("dcp", [1, 2])
def test_amd_model_q_replication(tmp_path, monkeypatch, dcp):
    from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

    monkeypatch.delenv("VLLM_DCP_Q_REPLICATE", raising=False)
    KimiLinearConfig(
        architectures=["KimiLinearForCausalLM"],
        hidden_size=256,
        num_hidden_layers=1,
        num_attention_heads=96,
        q_lora_rank=256,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mla_use_nope=True,
        linear_attn_config={
            "kda_layers": [],
            "full_attn_layers": [0],
            "num_heads": 96,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
        },
        dsa_config={"dsa_layers": [0], "index_num_heads": 32, "index_head_dim": 128},
    ).save_pretrained(tmp_path)
    spawn(
        _check_q_replication,
        args=(str(tmp_path), get_open_port(), dcp),
        nprocs=2,
        join=True,
    )
