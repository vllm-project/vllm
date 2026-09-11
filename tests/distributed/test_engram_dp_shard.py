# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram tables shared by co-located DP replicas.

Four GPUs on one node split the hash heads four ways, whether as DEP
(TP1 x DP4, the single-node MoE layout) or TP2 x DP2. A lookup only returns
the right rows if the ids of every replica are gathered, each rank looks up
its own heads, and each replica trades those tokens back for the heads its
peers own.

With shared host storage, DP replicas instead map the same TP slice and
prefetch only their own tokens without DP gathers.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

from vllm.config import (
    EngramConfig,
    VllmConfig,
    get_current_vllm_config_or_none,
    set_current_vllm_config,
)
from vllm.config.parallel import ParallelConfig
from vllm.config.scheduler import SchedulerConfig
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
    EngramLayout,
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
    shared_memory: bool = False,
) -> None:
    monkeypatch = pytest.MonkeyPatch()
    with monkeypatch.context() as m:
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
        head_sizes = HEAD_SIZES[:n_heads]
        rows = sum(head_sizes)
        config = SimpleNamespace(
            hidden_size=DIM,
            hc_mult=2,
            rms_norm_eps=1e-6,
            engram_layer_ids=[0],
            engram_num_embeddings=[rows],
            engram_max_ngram_size=2,
            engram_n_heads=n_heads,
            engram_head_dim=DIM,
            engram_compressed_vocab_size=32,
            engram_pad_token_id=0,
            engram_vocab_size=HEAD_SIZES[0],
        )
        vllm_config = VllmConfig(
            parallel_config=ParallelConfig(
                tensor_parallel_size=tp_size,
                data_parallel_size=dp_size,
                data_parallel_rank=dp_rank,
            ),
            scheduler_config=SchedulerConfig(
                max_model_len=8,
                is_encoder_decoder=False,
                max_num_batched_tokens=8,
                max_num_seqs=8,
            ),
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

            weight, scale_inv = _full_table(rows)
            layout = EngramLayout(config)
            # Stub configuration inputs, not Engram's initialization or execution.
            component_config = SimpleNamespace(
                engram_config=EngramConfig(
                    cpu_offload=cpu_offload,
                    enable_engram_shared_memory=shared_memory,
                ),
                parallel_config=ParallelConfig(ubatch_size=2 if shared_memory else 1),
                scheduler_config=vllm_config.scheduler_config,
                load_config=vllm_config.load_config,
            )
            with m.context() as init_patch, torch.device("cuda"):
                init_patch.setattr(
                    engram_ops, "get_current_vllm_config", lambda: component_config
                )
                engram = Engram(
                    config,
                    quant_config=None,
                    layout=layout,
                    layer_hash_index=0,
                    use_sequence_parallel=sequence_parallel,
                    prefix="model.layers.0.engram",
                )
            layer = engram.embed_tokens
            # Only the leader has valid checkpoint payload in shared mode.
            loader_weight = weight if not shared_memory or dp_rank == 0 else None
            loader_scale = scale_inv if not shared_memory or dp_rank == 0 else None
            layer.weight.weight_loader(layer.weight, loader_weight)
            layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, loader_scale)
            if shared_memory:
                assert layer.dp_size == 1
                assert layer.head_start == tp_rank * layer.part_n_hash_cols
                assert layer.weight.is_pinned() and layer.weight_scale_inv.is_pinned()

                def unexpected_collective(*args, **kwargs):
                    raise AssertionError("shared lookup must not use DP collectives")

                m.setattr(get_engram_dp_group(), "all_gather", unexpected_collective)

            token_counts = _token_counts(dp_size)

        # Production forward runs after the construction-time config has exited.
        assert get_current_vllm_config_or_none() is None
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
                gathered = gather_engram_hashes(ids, shared_memory=layer.shared_memory)
                expected_tokens = num_tokens if shared_memory else max(counts) * dp_size
                assert gathered.shape[0] == expected_tokens
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

        if shared_memory:
            _check_shared_prefetch_replay(engram, head_sizes, weight, scale_inv, m)


def _check_shared_prefetch_replay(engram, head_sizes, weight, scale_inv, patch):
    """Check buffer isolation with controlled IDs, not the full DBO scheduler."""
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    ids = [_make_ids(head_sizes, 8, seed=seed) for seed in (501, 502)]
    tp_rank = engram_ops.get_tensor_model_parallel_rank()
    tp_size = engram.embed_tokens.tp_size
    tokens = 8 // tp_size if engram.use_sequence_parallel else 8
    outputs = [
        torch.empty(tokens, len(head_sizes), DIM, device="cuda", dtype=torch.bfloat16)
        for _ in ids
    ]

    def step():
        for ubatch in (0, 1):
            patch.setattr(engram_ops, "dbo_current_ubatch_id", lambda i=ubatch: i)
            engram.prepare_embeddings(ids[ubatch].clone())
        for ubatch in (1, 0):
            patch.setattr(engram_ops, "dbo_current_ubatch_id", lambda i=ubatch: i)
            outputs[ubatch].copy_(engram.embed(ids[ubatch]))

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        step()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = BreakableCUDAGraphCapture()
    with torch.cuda.stream(warmup), graph:
        step()
    torch.cuda.current_stream().wait_stream(warmup)
    assert graph.num_eager_breaks == 4  # Two prefetch starts and two waits.
    for iteration in range(3):
        for ubatch in (0, 1):
            ids[ubatch].copy_(
                _make_ids(head_sizes, 8, seed=600 + 2 * iteration + ubatch)
            )
        graph.replay()
        for actual, indices in zip(outputs, ids):
            expected = _reference(weight.cuda(), scale_inv.cuda(), indices)
            if engram.use_sequence_parallel:
                expected = expected[tp_rank * tokens : (tp_rank + 1) * tokens]
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.accelerator.synchronize()


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


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.parametrize("n_heads", [8, 7])
@pytest.mark.parametrize(
    "tp_size,dp_size,sequence_parallel",
    [(1, 4, False), (2, 2, False), (2, 2, True)],
)
def test_engram_shared_host_memory_prefetch(
    tp_size: int,
    dp_size: int,
    sequence_parallel: bool,
    n_heads: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All DP readers prefetch leader-loaded pages without exchanging tokens/rows."""
    if torch.accelerator.device_count() < WORLD_SIZE:
        pytest.skip(f"Need {WORLD_SIZE} GPUs to run the test.")
    # Spawned workers must import Engram with its production decorators enabled.
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    mp.spawn(
        _worker,
        args=(
            tp_size,
            dp_size,
            True,
            n_heads,
            sequence_parallel,
            get_open_port(),
            True,
        ),
        nprocs=WORLD_SIZE,
    )
    cleanup_dist_env_and_memory()


def test_engram_shared_memory_requires_cpu_offload():
    with pytest.raises(ValueError, match="requires cpu_offload"):
        EngramConfig(cpu_offload=False, enable_engram_shared_memory=True)


def test_engram_shared_memory_requires_dp_peers(monkeypatch):
    monkeypatch.setattr(engram_ops, "get_engram_dp_size", lambda: 1)
    monkeypatch.setattr(engram_ops, "get_tensor_model_parallel_world_size", lambda: 4)
    with pytest.raises(AssertionError, match="requires co-located DP replicas"):
        ParallelEngramEmbedding(
            sum(HEAD_SIZES), DIM, HEAD_SIZES, cpu_offload=True, shared_memory=True
        )


@pytest.mark.parametrize("load_format,multithread", [("dummy", False), ("auto", True)])
def test_engram_shared_memory_rejects_uncoordinated_loaders(
    monkeypatch, load_format, multithread
):
    """Reject loaders that bypass the single writer or reorder collective loads."""
    monkeypatch.setattr(engram_ops, "get_engram_dp_size", lambda: 4)
    monkeypatch.setattr(engram_ops, "get_tensor_model_parallel_world_size", lambda: 1)
    config = SimpleNamespace(
        load_config=SimpleNamespace(
            load_format=load_format,
            model_loader_extra_config={"enable_multithread_load": multithread},
        )
    )
    monkeypatch.setattr(engram_ops, "get_current_vllm_config", lambda: config)
    with pytest.raises(AssertionError, match="Shared Engram"):
        ParallelEngramEmbedding(
            sum(HEAD_SIZES), DIM, HEAD_SIZES, cpu_offload=True, shared_memory=True
        )


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


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("uniform", [False, True])
def test_engram_hash_padding_ignores_other_nodes(tp_size, rank, uniform, monkeypatch):
    """Two four-GPU nodes pad only to their own maximum, including TP x DP."""
    dp_rank = rank // tp_size
    group_size = 4 // tp_size
    counts = (
        [4] * (8 // tp_size)
        if uniform
        else [4096] + [2] * (group_size - 1) + [1, 3] + [0] * (group_size - 2)
    )
    slot = 4 if uniform else (4096 if rank < 4 else 3)
    ids = torch.full((counts[dp_rank], 2, 3), 17, dtype=torch.int32)
    monkeypatch.setattr(
        engram_ops, "get_dp_group", lambda: SimpleNamespace(rank_in_group=dp_rank)
    )
    monkeypatch.setattr(
        engram_ops,
        "get_engram_dp_group",
        lambda: SimpleNamespace(
            rank_in_group=dp_rank % group_size,
            world_size=group_size,
            all_gather=lambda ids, dim: torch.cat([ids] * group_size, dim),
        ),
    )
    monkeypatch.setattr(
        engram_ops,
        "get_forward_context",
        lambda: SimpleNamespace(
            dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor(counts))
        ),
    )
    gathered = gather_engram_hashes(ids)
    expected = ids.new_full((slot, 2, 3), engram_ops.DEAD_ID)
    expected[: ids.shape[0]] = ids
    torch.testing.assert_close(gathered, expected.repeat(group_size, 1, 1))
