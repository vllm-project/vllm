# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Distributed QSA DCP regression coverage."""

from typing import Literal

import pytest
import torch
import torch.multiprocessing as mp

from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

mp.set_start_method("spawn", force=True)

Backend = Literal["ag_rs", "a2a"]


def _run_distributed(fn, world_size: int, backend: Backend) -> None:
    port = str(get_open_port())
    processes: list[mp.Process] = []
    for rank in range(world_size):
        env = {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": port,
        }
        process = mp.Process(target=fn, args=(env, backend))
        processes.append(process)
        process.start()

    for process in processes:
        process.join(timeout=180)
        if process.is_alive():
            process.kill()
            process.join()
        assert process.exitcode == 0


def _qsa_config(world_size: int, backend: Backend):
    from dataclasses import replace

    from vllm.config import ModelConfig, VllmConfig
    from vllm.config.parallel import ParallelConfig
    from vllm.models.qwen4_exp.config import Qwen4ExpTextConfig

    text_config = Qwen4ExpTextConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        partial_rotary_factor=0.5,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
    )
    model_config = ModelConfig(max_model_len=32, dtype="bfloat16")
    model_config.hf_config = text_config
    model_config.hf_text_config = text_config
    model_config.model_arch_config = replace(
        model_config.model_arch_config,
        hidden_size=256,
        total_num_hidden_layers=1,
        total_num_attention_heads=2,
        head_size=128,
        total_num_kv_heads=1,
    )
    config = VllmConfig(
        model_config=model_config,
        parallel_config=ParallelConfig(
            tensor_parallel_size=world_size,
            decode_context_parallel_size=world_size,
            dcp_comm_backend=backend,
        ),
    )
    config.cache_config.block_size = 16
    config.scheduler_config.max_num_batched_tokens = 32
    config.scheduler_config.max_num_seqs = 2
    return config, text_config


def _make_common_metadata(block_table: torch.Tensor):
    from vllm.v1.attention.backend import CommonAttentionMetadata

    query_start_loc = torch.tensor([0, 12, 32], dtype=torch.int32, device="cuda")
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.tensor([12, 20], dtype=torch.int32, device="cuda"),
        num_reqs=2,
        num_actual_tokens=32,
        max_query_len=20,
        max_seq_len=20,
        block_table_tensor=block_table,
        slot_mapping=torch.full((32,), -1, dtype=torch.int64, device="cuda"),
    )


def _assert_replicated_values(
    cache: torch.Tensor,
    metadata,
    group,
) -> None:
    valid = metadata.slot_mapping >= 0
    values = cache.reshape(-1, cache.shape[-1])[metadata.slot_mapping[valid].long()]
    gathered = group.all_gather(values.contiguous(), dim=0).view(
        group.world_size, values.shape[0], values.shape[1]
    )
    assert torch.equal(gathered, gathered[0].unsqueeze(0).expand_as(gathered))


def _assert_localization_reconstructs_global(
    packed: torch.Tensor,
    local: torch.Tensor,
    group,
) -> None:
    gathered = group.all_gather(local.contiguous(), dim=0).view(
        group.world_size, *local.shape
    )
    for row in range(packed.shape[0]):
        expected_count = int(packed[row, -1].item())
        expected = set(packed[row, :expected_count].tolist())
        recovered: set[int] = set()
        for rank in range(group.world_size):
            count = int(gathered[rank, row, -1].item())
            recovered.update(
                int(value) * group.world_size + rank
                for value in gathered[rank, row, :count].tolist()
            )
        assert recovered == expected


def _qsa_dcp_model_path_worker(env: dict[str, str], backend: Backend) -> None:
    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import (
        get_dcp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.models.qwen4_exp.common.qsa_cache import QSAMetadataBuilder
    from vllm.models.qwen4_exp.nvidia import model as _qwen4_exp_model  # noqa: F401
    from vllm.models.qwen4_exp.nvidia.indexer_qsa import QSAIndexer
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention
    from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import qsa_dcp_empty_owner_rows
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAFlashAttentionImpl
    from vllm.v1.attention.ops.dcp import (
        cp_lse_ag_out_rs,
        dcp_a2a_lse_reduce,
    )

    update_environment_variables(env)
    rank = int(env["RANK"])
    world_size = int(env["WORLD_SIZE"])
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    torch.set_default_device(device)
    init_distributed_environment()

    config, text_config = _qsa_config(world_size, backend)
    with set_current_vllm_config(config):
        initialize_model_parallel(
            tensor_model_parallel_size=world_size,
            decode_context_model_parallel_size=world_size,
        )
        rotary = get_rope(
            128,
            32,
            rope_parameters=text_config.rope_parameters,
            dtype=torch.bfloat16,
        )
        indexer = QSAIndexer(
            vllm_config=config,
            config=text_config,
            layer_id=0,
            rotary_emb=rotary,
            prefix="qsa",
        )
        indexer.to(dtype=torch.bfloat16)
        with torch.no_grad():
            indexer.index_qk_proj.weight.fill_(0.01)
            indexer.q_layernorm.weight.fill_(1)
            indexer.k_layernorm.weight.fill_(1)

        raw_spec = indexer.raw_key_cache.get_kv_cache_spec(config)
        compressed_spec = indexer.compressed_key_cache.get_kv_cache_spec(config)
        raw_storage = torch.zeros(
            4,
            1,
            raw_spec.num_states,
            indexer.raw_key_cache.head_size,
            dtype=torch.bfloat16,
            device=device,
        )
        compressed_storage = torch.zeros(
            4,
            1,
            compressed_spec.num_states,
            indexer.compressed_key_cache.head_size,
            dtype=indexer.indexer_dtype,
            device=device,
        )
        indexer.raw_key_cache.bind_kv_cache(raw_storage)
        indexer.compressed_key_cache.bind_kv_cache(compressed_storage)

        raw_builder = QSAMetadataBuilder(
            raw_spec, [indexer.raw_key_cache.prefix], config, device
        )
        compressed_builder = QSAMetadataBuilder(
            compressed_spec, [indexer.compressed_key_cache.prefix], config, device
        )
        raw_metadata = raw_builder.build(
            0,
            _make_common_metadata(
                torch.tensor(
                    [[rank * 2], [rank * 2 + 1]], dtype=torch.int32, device=device
                )
            ),
        )
        compressed_metadata = compressed_builder.build(
            0,
            _make_common_metadata(
                torch.tensor(
                    [[rank * 2], [rank * 2 + 1]], dtype=torch.int32, device=device
                )
            ),
        )
        hidden = (
            torch.arange(32 * 256, device=device, dtype=torch.float32).reshape(32, 256)
            / 1024
        ).to(torch.bfloat16)
        positions = torch.cat(
            (torch.arange(12, device=device), torch.arange(20, device=device))
        )
        with set_forward_context(
            {
                indexer.raw_key_cache.prefix: raw_metadata,
                indexer.compressed_key_cache.prefix: compressed_metadata,
            },
            config,
        ):
            projected_qk, _ = indexer.index_qk_proj(hidden)
            packed = indexer(projected_qk, positions)

        group = get_dcp_group()
        _assert_replicated_values(indexer.raw_key_cache.kv_cache, raw_metadata, group)
        _assert_replicated_values(
            indexer.compressed_key_cache.kv_cache, compressed_metadata, group
        )
        gathered_packed = group.all_gather(packed.contiguous(), dim=0).view(
            world_size, *packed.shape
        )
        assert torch.equal(
            gathered_packed, gathered_packed[0].unsqueeze(0).expand_as(gathered_packed)
        )
        packed[:2, :-1] = -1
        packed[0, 0] = 0
        packed[1, 0] = 1
        packed[:2, -1] = 1

        page_size = 16
        local_main_block_table = torch.tensor(
            [[0], [1]], dtype=torch.int32, device=device
        )
        full_main_block_table = torch.tensor(
            [[0, -1], [1, 2]], dtype=torch.int32, device=device
        )
        full_key = torch.arange(
            3 * page_size * 128, dtype=torch.float32, device=device
        ).reshape(3, page_size, 1, 128)
        full_value = (full_key + 1).neg()
        full_key = (full_key / 1024).to(torch.bfloat16)
        full_value = (full_value / 1024).to(torch.bfloat16)
        local_key = torch.zeros(
            2, page_size, 1, 128, dtype=torch.bfloat16, device=device
        )
        local_value = torch.zeros_like(local_key)
        for request, length in enumerate((12, 20)):
            for position in range(length):
                if position % world_size == rank:
                    local_position = position // world_size
                    full_block = full_main_block_table[request, position // page_size]
                    local_key[request, local_position] = full_key[
                        full_block, position % page_size
                    ]
                    local_value[request, local_position] = full_value[
                        full_block, position % page_size
                    ]

        query = (
            torch.arange(32 * 128, dtype=torch.float32, device=device).reshape(
                32, 1, 128
            )
            / 2048
            + rank
        ).to(torch.bfloat16)
        output_gate = torch.zeros_like(query)
        impl = object.__new__(Qwen4ExpQSAFlashAttentionImpl)
        impl.dcp_world_size = world_size
        impl.dcp_rank = rank
        impl.cp_kv_cache_interleave_size = 1
        impl._dcp_local_indices = None
        combine_fn = dcp_a2a_lse_reduce if backend == "a2a" else cp_lse_ag_out_rs
        combine_called = False

        def selected_combine(*args, **kwargs):
            nonlocal combine_called
            combine_called = True
            assert kwargs["is_lse_base_on_e"] is False
            return combine_fn(*args, **kwargs)

        impl.dcp_combine = selected_combine
        output = torch.empty_like(query)
        impl._forward_qsa_dcp(
            query,
            local_key,
            local_value,
            packed,
            32,
            local_main_block_table,
            compressed_metadata.token_to_req,
            True,
            output,
            output_gate,
            None,
            None,
        )
        assert combine_called

        local_indices = impl._dcp_local_indices[:32]
        _assert_localization_reconstructs_global(packed, local_indices, group)
        query_all_heads = group.all_gather(query.contiguous(), dim=1)
        empty_rows = qsa_dcp_empty_owner_rows(local_indices)
        assert bool(empty_rows[0]) == (rank == 1)
        assert bool(empty_rows[1]) == (rank == 0)

        reference = qsa_sparse_paged_attention(
            query_all_heads,
            full_key,
            full_value,
            packed,
            full_main_block_table,
            compressed_metadata.token_to_req,
            True,
            output_gate=torch.zeros_like(query_all_heads).contiguous(),
        )
        torch.testing.assert_close(
            output.float(),
            reference[:, rank : rank + 1].float(),
            rtol=3e-2,
            atol=3e-2,
        )


@pytest.mark.parametrize("backend", ["ag_rs", "a2a"])
@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="QSA DCP model-path test requires CUDA"
)
def test_qsa_dcp_model_path_matches_dcp1_reference(backend: Backend) -> None:
    if torch.accelerator.device_count() < 2:
        pytest.skip("QSA DCP model-path test requires two GPUs")
    _run_distributed(_qsa_dcp_model_path_worker, world_size=2, backend=backend)
