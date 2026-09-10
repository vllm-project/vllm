# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram tables shared by co-located DP replicas.

Four GPUs on one node split the hash heads four ways, whether as DEP
(TP1 x DP4, the single-node MoE layout) or TP2 x DP2. A lookup only returns
the right rows if the ids of every replica are gathered, each rank looks up
its own heads, and each replica trades those tokens back for the heads its
peers own.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.parallel import ParallelConfig
from vllm.distributed import cleanup_dist_env_and_memory, parallel_state
from vllm.distributed.parallel_state import (
    get_engram_dp_group,
    get_engram_dp_size,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import set_forward_context
from vllm.models.deepseek_v4_1.common import engram as engram_ops
from vllm.models.deepseek_v4_1.common.engram import (
    Engram,
    ParallelEngramEmbedding,
    engram_head_shard_rank,
    gather_engram_hashes,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

WORLD_SIZE = 4
DIM, BLOCK = 64, 32
# Prime-sized buckets, as the model's layout builds them.
HEAD_SIZES = (97, 101, 103, 107, 109, 113, 127, 131)


def _token_counts(dp_size: int) -> tuple[tuple[int, ...], ...]:
    """Even loads, then uneven ones with an idle replica to pad around."""
    return ((8,) * dp_size, (7, 0, 5, 1)[:dp_size], (7,) * dp_size)


def _full_table(rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The unsharded table, identical on every rank."""
    torch.manual_seed(0)
    weight = (torch.randn(rows, DIM) * 4).to(torch.float8_e4m3fn)
    scale_inv = torch.randint(120, 134, (rows, DIM // BLOCK), dtype=torch.uint8)
    return weight, scale_inv


def _make_ids(head_sizes: tuple[int, ...], num_tokens: int, seed: int) -> torch.Tensor:
    """Ids of each head, drawn from that head's own bucket range."""
    torch.manual_seed(seed)
    offset = 0
    columns = []
    for size in head_sizes:
        columns.append(torch.randint(offset, offset + size, (num_tokens, 1)))
        offset += size
    return torch.cat(columns, dim=1).to(torch.int32).cuda()


def _reference(weight, scale_inv, ids) -> torch.Tensor:
    """Dequantized lookup against the whole table."""
    valid = ids >= 0
    safe_ids = ids.clamp_min(0).long()
    values = torch.nn.functional.embedding(safe_ids, weight).float()
    scales = torch.nn.functional.embedding(safe_ids, scale_inv)
    scales = (scales.to(torch.int32) << 23).view(torch.float32)
    values = values.unflatten(-1, (-1, BLOCK)) * scales.unsqueeze(-1)
    return values.flatten(-2).to(torch.bfloat16).masked_fill(~valid.unsqueeze(-1), 0)


def _worker(
    rank: int,
    tp_size: int,
    dp_size: int,
    cpu_offload: bool,
    n_heads: int,
    sequence_parallel: bool,
    port: int,
) -> None:
    monkeypatch = pytest.MonkeyPatch()
    with monkeypatch.context() as m:
        # Any DeepSeek V4.1 config would do; only the answer matters here.
        m.setattr("vllm.config.engram.model_has_engram_layers", lambda config: True)
        torch.accelerator.set_device_index(torch.device(f"cuda:{rank}"))
        update_environment_variables(
            {
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "WORLD_SIZE": str(WORLD_SIZE),
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": str(port),
            }
        )
        dp_rank, tp_rank = divmod(rank, tp_size)
        vllm_config = VllmConfig(
            parallel_config=ParallelConfig(
                tensor_parallel_size=tp_size,
                data_parallel_size=dp_size,
                data_parallel_rank=dp_rank,
            )
        )
        init_distributed_environment()
        with set_current_vllm_config(vllm_config):
            initialize_model_parallel(tensor_model_parallel_size=tp_size)

            assert get_engram_dp_size() == dp_size
            # Replicas group per TP rank: {0, 2} and {1, 3} at TP2 x DP2.
            assert get_engram_dp_group().ranks == [
                tp_rank + replica * tp_size for replica in range(dp_size)
            ]
            # Head shards run TP-major, so the DP gather brings in neighbours.
            assert engram_head_shard_rank() == tp_rank * dp_size + dp_rank

            head_sizes = HEAD_SIZES[:n_heads]
            rows = sum(head_sizes)
            weight, scale_inv = _full_table(rows)
            with torch.device("cuda"):
                layer = ParallelEngramEmbedding(
                    rows, DIM, head_sizes, block_size=BLOCK, cpu_offload=cpu_offload
                )
            layer.weight.weight_loader(layer.weight, weight)
            layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, scale_inv)

            engram = Engram.__new__(Engram)
            nn.Module.__init__(engram)
            engram.embed_tokens = layer
            engram.use_sequence_parallel = sequence_parallel
            token_counts = _token_counts(dp_size)
            engram.staged_rows = torch.empty(
                max(max(counts) for counts in token_counts) * dp_size,
                layer.part_n_hash_cols,
                DIM,
                dtype=torch.bfloat16,
                device="cuda",
            )

            for batch_idx, counts in enumerate(token_counts):
                num_tokens = counts[dp_rank]
                # Ids differ per replica but must agree across its TP ranks.
                ids = _make_ids(head_sizes, num_tokens, seed=100 + dp_rank)
                if batch_idx == 2 and dp_rank == 1:
                    ids.fill_(engram_ops.DEAD_ID)
                with set_forward_context(
                    None,
                    vllm_config,
                    num_tokens=num_tokens,
                    num_tokens_across_dp=torch.tensor(counts, dtype=torch.int32),
                ):
                    gathered = gather_engram_hashes(ids)
                    assert gathered.shape[0] == max(counts) * dp_size
                    engram.prepare_embeddings(gathered)
                    looked_up = engram.embed(ids)
                    standalone = layer(ids)
                expected = _reference(weight.cuda(), scale_inv.cuda(), ids)
                torch.testing.assert_close(standalone, expected, atol=0, rtol=0)
                if sequence_parallel:
                    # SP hands back only this rank's token shard, zero-padded.
                    chunk = -(-num_tokens // tp_size)
                    expected = torch.nn.functional.pad(
                        expected, (0, 0, 0, 0, 0, chunk * tp_size - num_tokens)
                    )
                    expected = expected[tp_rank * chunk : (tp_rank + 1) * chunk]
                torch.testing.assert_close(looked_up, expected, atol=0, rtol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.parametrize("cpu_offload", [False, True])
# 8 heads split evenly; 7 leave one padded column on the last shard.
@pytest.mark.parametrize("n_heads", [8, 7])
# DEP first: expert parallel over the node keeps the dense layers at TP1.
# Sequence parallelism only exists above TP1, where it replaces the head
# gather with a token gather.
@pytest.mark.parametrize(
    "tp_size,dp_size,sequence_parallel",
    [(1, 4, False), (2, 2, False), (2, 2, True)],
)
def test_engram_table_shared_across_dp_replicas(
    tp_size: int,
    dp_size: int,
    sequence_parallel: bool,
    cpu_offload: bool,
    n_heads: int,
) -> None:
    if torch.accelerator.device_count() < WORLD_SIZE:
        pytest.skip(f"Need {WORLD_SIZE} GPUs to run the test.")
    mp.spawn(
        _worker,
        args=(
            tp_size,
            dp_size,
            cpu_offload,
            n_heads,
            sequence_parallel,
            get_open_port(),
        ),
        nprocs=WORLD_SIZE,
    )
    cleanup_dist_env_and_memory()


@pytest.mark.parametrize(
    "node_ids,dp_size,replica_size,expected",
    [
        ([0] * 4, 4, 1, 4),
        ([0] * 4, 2, 2, 2),
        ([0] * 8 + [1] * 8, 8, 2, 4),
        ([0] * 8 + [1] * 8 + [2] * 8, 8, 3, 1),
        ([0, 1] * 4, 4, 2, 1),
        ([0] * 3 + [1] * 5, 4, 2, 1),
    ],
)
def test_engram_dp_shard_size_respects_node_boundaries(
    node_ids, dp_size, replica_size, expected, monkeypatch
):
    """Sharing must fall back when replicas or node rank ranges do not align."""
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            architecture="DeepseekV41ForCausalLM",
            hf_text_config=SimpleNamespace(engram_layer_ids=[1, 14]),
        )
    )
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(parallel_state, "get_node_count", lambda: len(set(node_ids)))
    monkeypatch.setattr(
        parallel_state, "get_world_group", lambda: SimpleNamespace(cpu_group=None)
    )
    monkeypatch.setattr(
        parallel_state,
        "in_the_same_node_as",
        lambda group, src: [node == node_ids[src] for node in node_ids],
    )
    assert (
        parallel_state._engram_dp_shard_size(
            config, len(node_ids), dp_size, replica_size, False
        )
        == expected
    )


@pytest.mark.parametrize("num_tokens", [0, 2, 4, 5])
def test_engram_hash_padding_has_no_valid_rows(num_tokens, monkeypatch):
    """Idle/padded slots must not query row zero; oversized batches are rejected."""
    monkeypatch.setattr(engram_ops, "engram_gathered_num_tokens", lambda: 4)
    monkeypatch.setattr(
        engram_ops,
        "get_engram_dp_group",
        lambda: SimpleNamespace(all_gather=lambda ids, dim: torch.cat([ids, ids], dim)),
    )
    ids = torch.full((num_tokens, 2, 3), 17, dtype=torch.int32)
    if num_tokens > 4:
        with pytest.raises(ValueError, match="exceeds the DP token slot"):
            gather_engram_hashes(ids)
        return
    gathered = gather_engram_hashes(ids).reshape(2, 4, 2, 3)
    for replica in gathered:
        torch.testing.assert_close(replica[:num_tokens], ids)
        assert torch.all(replica[num_tokens:] == engram_ops.DEAD_ID)
