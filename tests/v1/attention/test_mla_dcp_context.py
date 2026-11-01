# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MLA chunked-context loop under DCP against the unsharded context.

Every rank's paged cache holds its interleaved shard of each request's context.
The loop's local gather, DCP all-gather and row gather must hand ``kv_b_proj``
and the prefill backend every context row of the chunk's requests exactly once
and no padding: request by request, each rank's rows of the chunk in turn.
"""

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from tests.utils import multi_gpu_only
from vllm.platforms import current_platform

_KV_LORA_RANK = 512
_QK_ROPE = 64
_QK_NOPE = 128
_V_HEAD_DIM = 128
_ENTRY = _KV_LORA_RANK + _QK_ROPE
_NUM_HEADS = 2
_BLOCK_SIZE = 16
_K_SCALE = 0.37
# (interleave, context lengths, workspace rows): one and several chunks, a
# request split over chunks, empty contexts and lengths that leave some ranks
# with fewer rows than others.
_LAYOUTS = (
    (1, [37], 1024),
    (1, [1000] + [5] * 20, 512),
    (16, [145, 0, 3, 129], 1024),
    (64, [1500, 0, 3, 129], 1024),
)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="requires GPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("dims", [(512, 64), (384, 48), (512, 0)])
@pytest.mark.parametrize("num_rows", [0, 1, 33, 1000])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_gather_dcp_context_rows_copies_rows(num_rows, dims, dtype, index_dtype):
    """The gather equals indexing the latent and rope parts of the rows, also
    for NoPE MLA layers, which have no rope part."""
    from vllm.v1.attention.ops.mla_dcp_context import gather_dcp_context_rows

    kv_lora_rank, rope_dim = dims
    generator = torch.Generator(device="cuda").manual_seed(num_rows)
    gathered = torch.randn(
        (1200, kv_lora_rank + rope_dim), device="cuda", generator=generator
    ).to(dtype)
    row_indices = torch.randint(
        0, 1200, (num_rows,), device="cuda", generator=generator
    ).to(index_dtype)

    latent, k_pe = gather_dcp_context_rows(gathered, row_indices, kv_lora_rank)

    expected = gathered[row_indices.long()].unsqueeze(1)
    assert latent.shape == (num_rows, 1, kv_lora_rank)
    assert k_pe.shape == (num_rows, 1, rope_dim)
    assert latent.dtype == k_pe.dtype == dtype
    assert latent.is_contiguous() and k_pe.is_contiguous()
    assert torch.equal(_bits(latent), _bits(expected[..., :kv_lora_rank]))
    assert torch.equal(_bits(k_pe), _bits(expected[..., kv_lora_rank:]))


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="requires GPU")
@pytest.mark.parametrize(
    "dtype, out_dtype",
    [
        (torch.bfloat16, None),
        (torch.float16, None),
        # fp8 rows dequantized into the model dtype.
        (current_platform.fp8_dtype(), torch.bfloat16),
        (current_platform.fp8_dtype(), torch.float16),
    ],
)
@pytest.mark.parametrize("rope_dim", [_QK_ROPE, 0])
def test_gather_dcp_context_rows_warmup_covers_the_context_loop(
    dtype, out_dtype, rope_dim
):
    """The JIT warmup compiles the variant that the context loop launches: rows
    of the all-gather part of the workspace, int64 row indices taken at any
    offset into the batch's indices, any number of rows and, for fp8 rows, the
    layer's 0-dim k_scale."""
    from vllm.v1.attention.ops.mla_dcp_context import (
        _gather_context_rows,
        gather_dcp_context_rows,
    )

    kernel = _gather_context_rows.kernel

    def compiled_variants() -> set:
        return {key for caches in kernel.device_caches.values() for key in caches[0]}

    kernel.device_caches.clear()
    _gather_context_rows.warmup(
        dtype=dtype, kv_lora_rank=_KV_LORA_RANK, rope_dim=rope_dim, out_dtype=out_dtype
    )
    torch.accelerator.synchronize()
    warmed = compiled_variants()
    assert warmed

    workspace_rows, dcp_world_size = 1024, 4
    workspace = torch.zeros(
        (workspace_rows + workspace_rows // dcp_world_size, _KV_LORA_RANK + rope_dim),
        dtype=dtype,
        device="cuda",
    )
    gathered = workspace[workspace_rows // dcp_world_size :]
    row_indices = torch.arange(workspace_rows, device="cuda")
    dequant = (
        {}
        if out_dtype is None
        else dict(
            k_scale=torch.ones((), dtype=torch.float32, device="cuda"),
            out_dtype=out_dtype,
        )
    )
    for start, num_rows in ((0, 1), (1, 33), (3, 1000)):
        gather_dcp_context_rows(
            gathered, row_indices[start : start + num_rows], _KV_LORA_RANK, **dequant
        )
    torch.accelerator.synchronize()

    assert compiled_variants() == warmed


@pytest.mark.parametrize(
    "cache_dtype, variants",
    [
        ("auto", [(torch.float16, None)]),
        # A plain fp8 cache may also take the fp8 path, which dequantizes the
        # rows into the model dtype in the row gather.
        (
            "fp8",
            [(torch.float16, None), (current_platform.fp8_dtype(), torch.float16)],
        ),
        ("fp8_ds_mla", [(torch.bfloat16, None)]),
    ],
)
def test_dcp_manager_registers_the_context_row_gather_warmup(
    cache_dtype, variants, monkeypatch
):
    """Every MLA layer under DCP, model-specific ones included, builds an
    MLADCPManager while the model is built, so the manager registers the row
    gather variants the chunked-context loop may launch with the JIT warmup."""
    from vllm.v1.attention.ops import dcp
    from vllm.v1.attention.ops.mla_dcp_context import _gather_context_rows

    register = MagicMock()
    monkeypatch.setattr(_gather_context_rows, "register_warmup", register)
    monkeypatch.setattr(dcp, "get_dcp_group", MagicMock())
    monkeypatch.setattr(dcp, "get_dcp_workspace_max_num_tokens", lambda config: 64)
    monkeypatch.setattr(dcp.MLADCPManager, "_init_combine", MagicMock())
    monkeypatch.setattr(dcp.MLADCPManager, "_init_query_gather", MagicMock())
    config = MagicMock()
    config.parallel_config.num_ubatches = 1
    config.parallel_config.dcp_comm_backend = "ag_rs"
    config.cache_config.cache_dtype = cache_dtype
    config.model_config.dtype = torch.float16

    dcp.MLADCPManager(
        config,
        torch.device("cpu"),
        num_heads=_NUM_HEADS,
        query_head_dim=_ENTRY,
        output_head_dim=_KV_LORA_RANK,
        query_dtype=torch.float16,
        output_dtype=torch.float16,
        padded_num_heads=None,
        is_lse_base_on_e=True,
        use_pcp=False,
    )

    dims = dict(kv_lora_rank=_KV_LORA_RANK, rope_dim=_QK_ROPE)
    assert register.call_args_list == [
        call(dtype=dtype, **dims)
        if out_dtype is None
        else call(dtype=dtype, out_dtype=out_dtype, **dims)
        for dtype, out_dtype in variants
    ]


class _RecordingKVBProj:
    """Records the latent ``kv_b_proj`` receives and projects it."""

    def __init__(self, weight: torch.Tensor) -> None:
        self.weight = weight
        self.inputs: list[torch.Tensor] = []

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        self.inputs.append(x.clone())
        out = x.float().reshape(x.shape[0], -1) @ self.weight.float().T
        return out.to(self.weight.dtype), None


class _RecordingPrefillBackend:
    """Records the K of every context chunk."""

    def __init__(self) -> None:
        self.keys: list[torch.Tensor] = []

    @staticmethod
    def get_name() -> str:
        return "recording"

    def supports_out(self) -> bool:
        return False

    def run_prefill_context_chunk(self, *, chunk, q, k, v, out=None):
        self.keys.append(k.clone())
        num_q = q.shape[0]
        out = q.new_zeros((num_q, _NUM_HEADS, _V_HEAD_DIM))
        lse = torch.zeros((_NUM_HEADS, num_q), dtype=torch.float32, device=q.device)
        return out, lse


def _local_token_ids(context_len, rank, dcp_world_size, interleave) -> torch.Tensor:
    """The context tokens a rank holds, in its local order: blocks of
    ``interleave`` tokens go to the ranks in turn."""
    tokens = torch.arange(context_len)
    return tokens[(tokens // interleave) % dcp_world_size == rank]


class _Context:
    """Random unsharded contexts and every rank's paged cache of its shard."""

    def __init__(
        self, context_lens, dcp_world_size, interleave, kv_cache_dtype, dtype, seed
    ):
        device = torch.device("cuda", torch.accelerator.current_device_index())
        generator = torch.Generator(device=device).manual_seed(seed)
        self.kv_cache_dtype = kv_cache_dtype
        self.dtype = dtype
        self.dcp_world_size = dcp_world_size
        self.interleave = interleave
        self.rows = [self._random((n, _ENTRY), generator, device) for n in context_lens]
        max_local = -(-max(context_lens) // (interleave * dcp_world_size)) * interleave
        blocks_per_request = -(-max_local // _BLOCK_SIZE) + 1
        num_blocks = len(context_lens) * blocks_per_request
        self.block_table = torch.arange(
            num_blocks, dtype=torch.int32, device=device
        ).view(len(context_lens), blocks_per_request)
        # The rows past a rank's shard are left random: the loop must drop
        # them as padding.
        self.caches = []
        for rank in range(dcp_world_size):
            cache = self._random((num_blocks, _BLOCK_SIZE, _ENTRY), generator, device)
            for request, rows in enumerate(self.rows):
                local = _local_token_ids(
                    rows.shape[0], rank, dcp_world_size, interleave
                )
                self._request_rows(cache, request)[: local.numel()] = rows[
                    local.to(device)
                ]
            self.caches.append(cache)

    def _random(self, shape, generator, device) -> torch.Tensor:
        if self.kv_cache_dtype == "auto":
            return torch.randn(shape, device=device, generator=generator).to(self.dtype)
        raw = torch.randint(
            0, 256, shape, dtype=torch.uint8, device=device, generator=generator
        )
        # Finite fp8 codes only.
        raw[(raw & 0x7F) == 0x7F] = 0
        return raw.view(current_platform.fp8_dtype())

    def _request_rows(self, cache, request) -> torch.Tensor:
        blocks = self.block_table.shape[1]
        return cache[request * blocks : (request + 1) * blocks].view(-1, _ENTRY)

    def dequantize(self, rows: torch.Tensor) -> torch.Tensor:
        if self.kv_cache_dtype == "auto":
            return rows
        return (rows.float() * _K_SCALE).to(self.dtype)

    def local_rows(self, rank, chunk) -> torch.Tensor:
        """What a rank's local gather writes for a chunk, padding included."""
        return torch.cat(
            [
                self.dequantize(
                    self._request_rows(self.caches[rank], request)[
                        start : start + padded
                    ]
                )
                for request, start, padded in zip(
                    range(chunk.request_slice.start, chunk.request_slice.stop),
                    chunk.local_starts,
                    chunk.padded_local_seq_lens,
                )
            ]
        )

    def expected_rows(self, chunks) -> list[torch.Tensor]:
        """Each chunk's context rows: request by request, each rank's rows of
        the chunk in turn. They are the chunk's span of the request's context,
        and the chunks cover every context row once."""
        cursor = [0] * len(self.rows)
        expected = []
        for chunk in chunks:
            parts = []
            for request, start, padded, length in zip(
                range(chunk.request_slice.start, chunk.request_slice.stop),
                chunk.local_starts,
                chunk.padded_local_seq_lens,
                chunk.seq_lens.tolist(),
            ):
                tokens = torch.cat(
                    [
                        _local_token_ids(
                            self.rows[request].shape[0],
                            rank,
                            self.dcp_world_size,
                            self.interleave,
                        )[start : start + padded]
                        for rank in range(self.dcp_world_size)
                    ]
                )
                span = torch.arange(cursor[request], cursor[request] + length)
                assert torch.equal(tokens.sort().values, span)
                cursor[request] += length
                parts.append(self.rows[request][tokens.to(self.rows[request].device)])
            expected.append(self.dequantize(torch.cat(parts)))
        assert cursor == [rows.shape[0] for rows in self.rows]
        return expected


def _run_context_loop(
    context, rank, dcp_world_size, interleave, workspace_size, manager
):
    """One pass of ``_context_parallel_compute_prefill_context`` on ``rank``.

    Asserts that ``kv_b_proj`` gets the unsharded latent rows and the prefill
    backend the unsharded rope rows, chunk by chunk, bit for bit.
    """
    from vllm.model_executor.layers.attention import mla_attention
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonImpl,
        build_mla_chunked_context_metadata,
    )

    device = context.block_table.device
    dtype = context.dtype
    context_lens = [rows.shape[0] for rows in context.rows]
    query_lens = [i + 2 for i in range(len(context_lens))]
    query_start_loc = torch.zeros(len(query_lens) + 1, dtype=torch.int32)
    query_start_loc[1:] = torch.tensor(query_lens).cumsum(0)
    workspace = torch.empty(
        (workspace_size + workspace_size // dcp_world_size, _ENTRY),
        dtype=dtype,
        device=device,
    )
    manager.init_kv_gather(workspace, workspace_size)
    chunked = build_mla_chunked_context_metadata(
        context_lens_cpu=torch.tensor(context_lens, dtype=torch.int32),
        prefill_query_start_loc_cpu=query_start_loc,
        chunked_prefill_workspace=workspace,
        chunked_prefill_workspace_size=workspace_size,
        block_size=_BLOCK_SIZE,
        align_chunk_to_block=True,
        device=device,
        dcp_world_size=dcp_world_size,
        dcp_local_block_size=interleave,
        dcp_virtual_block_size=interleave * dcp_world_size,
        dcp_manager=manager,
    )
    assert chunked is not None
    if isinstance(manager, _SimulatedKVGather):
        manager.chunks = iter(chunked.chunks)
    for chunk in chunked.chunks:
        for request, local_lens in zip(
            range(chunk.request_slice.start, chunk.request_slice.stop),
            chunk.local_context_lens_allranks,
        ):
            assert local_lens == [
                _local_token_ids(
                    context_lens[request], r, dcp_world_size, interleave
                ).numel()
                for r in range(dcp_world_size)
            ]

    weight = (
        torch.randn(_NUM_HEADS * (_QK_NOPE + _V_HEAD_DIM), _KV_LORA_RANK, device=device)
        * 0.02
    ).to(dtype)
    kv_b_proj = _RecordingKVBProj(weight)
    impl = SimpleNamespace(
        kv_cache_dtype=context.kv_cache_dtype,
        kv_lora_rank=_KV_LORA_RANK,
        qk_rope_head_dim=_QK_ROPE,
        qk_nope_head_dim=_QK_NOPE,
        v_head_dim=_V_HEAD_DIM,
        num_heads=_NUM_HEADS,
        kv_b_proj=kv_b_proj,
        _use_flashinfer_concat_mla_k=False,
    )
    impl._concat_k_nope_k_pe = MLACommonImpl._concat_k_nope_k_pe.__get__(impl)
    backend = _RecordingPrefillBackend()
    attn_metadata = SimpleNamespace(
        prefill=SimpleNamespace(
            prefill_backend=backend,
            chunked_context=chunked,
            q_data_type=dtype,
            block_table=context.block_table,
        )
    )
    q = torch.zeros(
        (sum(query_lens), _NUM_HEADS, _QK_NOPE + _QK_ROPE), dtype=dtype, device=device
    )
    with patch.object(mla_attention, "_get_kv_b_proj_input_dtype", lambda *_: None):
        MLACommonImpl._context_parallel_compute_prefill_context(
            impl,
            q,
            context.caches[rank],
            attn_metadata,
            k_scale=torch.tensor(_K_SCALE, dtype=torch.float32, device=device),
            dcp_world_size=dcp_world_size,
        )
    torch.accelerator.synchronize()

    expected = context.expected_rows(chunked.chunks)
    assert len(kv_b_proj.inputs) == len(backend.keys) == len(expected) > 0
    for latent, k, rows in zip(kv_b_proj.inputs, backend.keys, expected):
        assert latent.dtype == k.dtype == dtype
        assert torch.equal(_bits(latent), _bits(rows[:, None, :_KV_LORA_RANK]))
        k_pe = rows[:, None, _KV_LORA_RANK:].expand(-1, _NUM_HEADS, -1)
        assert torch.equal(_bits(k[..., _QK_NOPE:]), _bits(k_pe))


class _SimulatedKVGather:
    """The DCP all-gather on one GPU: the other ranks' local rows are what
    their local gather would write from their own caches."""

    def __init__(self, context, dcp_world_size) -> None:
        self.context = context
        self.dcp_world_size = dcp_world_size
        self.chunks: Iterator[Any] = iter(())

    def init_kv_gather(self, workspace, workspace_size) -> None:
        pass

    def kv_gather(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        chunk = next(self.chunks)
        assert torch.equal(_bits(src), _bits(self.context.local_rows(0, chunk)))
        dst = dst.view(self.dcp_world_size, *src.shape)
        dst[0].copy_(src)
        for rank in range(1, self.dcp_world_size):
            dst[rank].copy_(self.context.local_rows(rank, chunk))


def _workspace_size(rows, dcp_world_size, interleave) -> int:
    alignment = max(_BLOCK_SIZE, interleave * dcp_world_size)
    return -(-rows // alignment) * alignment


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("dcp_world_size", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@torch.inference_mode()
def test_dcp_context_loop_unshards_the_context(dcp_world_size, dtype, kv_cache_dtype):
    """On one GPU, with the other ranks simulated, the loop hands the
    projection and the prefill backend the unsharded context rows."""
    for seed, (interleave, context_lens, rows) in enumerate(_LAYOUTS):
        context = _Context(
            context_lens, dcp_world_size, interleave, kv_cache_dtype, dtype, seed
        )
        _run_context_loop(
            context,
            0,
            dcp_world_size,
            interleave,
            _workspace_size(rows, dcp_world_size, interleave),
            _SimulatedKVGather(context, dcp_world_size),
        )


_DCP = 2


@torch.inference_mode()
def _dcp_context_worker(rank, init_method, direct_kv_gather):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.v1.attention.ops import dcp

    torch.accelerator.set_device_index(rank)
    device = torch.device(f"cuda:{rank}")
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=_DCP,
            rank=rank,
            local_rank=rank,
            distributed_init_method=init_method,
        )
        initialize_model_parallel(
            tensor_model_parallel_size=_DCP,
            decode_context_model_parallel_size=_DCP,
        )
        try:
            config = MagicMock()
            config.parallel_config.num_ubatches = 1
            config.parallel_config.dcp_comm_backend = "ag_rs"
            config.scheduler_config.max_num_batched_tokens = 64
            config.scheduler_config.max_num_seqs = 8
            config.num_speculative_tokens = 0
            config.speculative_config = None
            config.compilation_config.max_cudagraph_capture_size = 0
            manager = dcp.MLADCPManager(
                config,
                device,
                num_heads=_NUM_HEADS,
                query_head_dim=_ENTRY,
                output_head_dim=_V_HEAD_DIM,
                query_dtype=torch.bfloat16,
                output_dtype=torch.bfloat16,
                padded_num_heads=None,
                is_lse_base_on_e=True,
                use_pcp=False,
            )
            # The single-GPU test covers the model dtypes; this one covers the
            # real KV gather.
            for kv_cache_dtype in ("auto", "fp8"):
                for seed, (interleave, context_lens, rows) in enumerate(_LAYOUTS):
                    # Every rank builds the same contexts and caches and serves
                    # its own cache.
                    context = _Context(
                        context_lens,
                        _DCP,
                        interleave,
                        kv_cache_dtype,
                        torch.bfloat16,
                        seed,
                    )
                    _run_context_loop(
                        context,
                        rank,
                        _DCP,
                        interleave,
                        _workspace_size(rows, _DCP, interleave),
                        manager,
                    )
            gather = manager._kv_gather
            uses_direct = isinstance(
                getattr(gather, "__self__", None), dcp.DirectDCPKVGatherWorkspace
            )
            assert uses_direct == direct_kv_gather
        finally:
            cleanup_dist_env_and_memory()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@multi_gpu_only(num_gpus=2)
@pytest.mark.parametrize("direct_kv_gather", [False, True])
def test_dcp_context_loop_unshards_the_context_on_two_ranks(
    direct_kv_gather, monkeypatch, tmp_path
):
    """With either DCP KV gather, every rank hands the projection and the
    prefill backend the unsharded context rows, with an unquantized and an
    fp8 cache."""
    import torch.multiprocessing as mp

    from vllm.distributed.device_communicators.custom_all_reduce import (
        _has_local_multicast_support,
    )

    if direct_kv_gather and not _has_local_multicast_support(torch.device("cuda", 0)):
        pytest.skip("The direct DCP KV gather needs symmetric-memory multicast.")
    # The spawned ranks inherit the environment.
    monkeypatch.setenv(
        "VLLM_USE_DIRECT_DCP_KV_GATHER", "1" if direct_kv_gather else "0"
    )
    mp.spawn(
        _dcp_context_worker,
        args=(f"file://{tmp_path / 'dist_init'}", direct_kv_gather),
        nprocs=_DCP,
    )
