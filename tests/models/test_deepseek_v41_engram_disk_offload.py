# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram disk offload: the mapped shard must behave exactly like the pinned one.

The disk path replaces an in-kernel UVA gather with a host gather plus a
separate dequantization kernel, so the thing worth testing is that the two
produce identical bytes -- including for heads this rank does not own, and for
ids that fall outside its vocab slice.
"""

import glob
import json
import os

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("Engram is CUDA-only", allow_module_level=True)

from vllm.models.deepseek_v4_1.common.engram import (  # noqa: E402
    ParallelEngramEmbedding,
)


@pytest.fixture(scope="module", autouse=True)
def _tp1_world():
    """ParallelEngramEmbedding reads the TP group at construction."""
    import os
    import tempfile

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    # Bringing the groups up reads the current config, so the context has to
    # wrap the setup as well as each construction.
    with tempfile.TemporaryDirectory() as tmp, set_current_vllm_config(VllmConfig()):
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29591")
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{tmp}/init",
            local_rank=0,
            backend="nccl",
        )
        ensure_model_parallel_initialized(1, 1)
        yield


DIM = 256
BLOCK = 32
HEAD_SIZES = (1024, 1024)


def _build(tmp_dir: str | None) -> ParallelEngramEmbedding:
    from vllm.config import VllmConfig, set_current_vllm_config

    # Construction reads the current config; the context must be live *here*,
    # not merely entered by a fixture, because it is a ContextVar.
    with set_current_vllm_config(VllmConfig()):
        return ParallelEngramEmbedding(
            num_embeddings=sum(HEAD_SIZES),
            dim=DIM,
            head_sizes=HEAD_SIZES,
            block_size=BLOCK,
            cpu_offload=True,
            disk_offload_dir=tmp_dir,
        )


def _fill(module: ParallelEngramEmbedding, seed: int = 0) -> None:
    gen = torch.Generator().manual_seed(seed)
    rows = module.weight.shape[0]
    raw = torch.randint(0, 255, (rows, DIM), generator=gen, dtype=torch.uint8)
    # 0x7F and 0xFF are the e4m3 NaN encodings. Keep them out so the comparison
    # below can stay bitwise: NaN != NaN would fail even on identical bytes.
    raw[raw == 0x7F] = 0x7E
    module.weight.data.copy_(raw.view(torch.float8_e4m3fn))
    # ue8m0 exponents around 1.0 keep the dequantized values in a sane range.
    module.weight_scale_inv.data.copy_(
        torch.randint(120, 134, (rows, DIM // BLOCK), generator=gen, dtype=torch.uint8)
    )


def _indices(num_tokens: int, n_hash_cols: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(7)
    ids = torch.randint(
        0, sum(HEAD_SIZES), (num_tokens, n_hash_cols), generator=gen, dtype=torch.int32
    )
    # Repeats are the common case and exercise the gather's deduplication.
    ids[1::3] = ids[0]
    return ids.cuda()


@pytest.mark.parametrize("num_tokens", [1, 17, 512])
def test_disk_matches_pinned(tmp_path, num_tokens):
    pinned = _build(None)
    _fill(pinned)

    disk = _build(str(tmp_path))
    disk.weight.data.copy_(pinned.weight.data)
    disk.weight_scale_inv.data.copy_(pinned.weight_scale_inv.data)

    ids = _indices(num_tokens, pinned.n_hash_cols)
    shape = (num_tokens, pinned.part_n_hash_cols, DIM)
    want = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    got = torch.empty_like(want)
    pinned.lookup(ids, want)
    disk.lookup(ids, got)

    torch.testing.assert_close(got, want, atol=0, rtol=0)


def test_unowned_ids_write_zeros(tmp_path):
    disk = _build(str(tmp_path))
    _fill(disk)
    # Ids past this rank's vocab slice are not its rows to serve.
    ids = torch.full(
        (8, disk.n_hash_cols), disk.vocab_end_idx, dtype=torch.int32, device="cuda"
    )
    out = torch.full(
        (8, disk.part_n_hash_cols, DIM), 7.0, dtype=torch.bfloat16, device="cuda"
    )
    disk.lookup(ids, out)
    assert torch.count_nonzero(out) == 0


def test_second_boot_reuses_the_file(tmp_path):
    first = _build(str(tmp_path))
    _fill(first)
    ids = _indices(32, first.n_hash_cols)
    out = torch.empty(
        (32, first.part_n_hash_cols, DIM), dtype=torch.bfloat16, device="cuda"
    )
    first.lookup(ids, out)  # finalizes: flush, sidecar, remap copy-on-write

    # Glob rather than derive the shard id: it is the TP rank or the EDP head
    # rank depending on the sharding in force.
    assert glob.glob(os.path.join(str(tmp_path), "engram_r*.done.json"))

    second = _build(str(tmp_path))
    # A finished shard is mapped as-is, with no load pass.
    assert second._disk_finalized
    again = torch.empty_like(out)
    second.lookup(ids, again)
    torch.testing.assert_close(again, out, atol=0, rtol=0)


def test_geometry_change_rebuilds(tmp_path):
    first = _build(str(tmp_path))
    _fill(first)
    first.lookup(
        _indices(4, first.n_hash_cols),
        torch.empty(
            (4, first.part_n_hash_cols, DIM), dtype=torch.bfloat16, device="cuda"
        ),
    )
    meta = glob.glob(os.path.join(str(tmp_path), "engram_r*.done.json"))[0]
    with open(meta) as fh:
        recorded = json.load(fh)
    recorded["vocab_end"] += 1
    with open(meta, "w") as fh:
        json.dump(recorded, fh)

    # A sidecar that no longer describes this shard must not be gathered from.
    rebuilt = _build(str(tmp_path))
    assert not rebuilt._disk_finalized


def test_disk_requires_cpu_offload(tmp_path):
    from vllm.config import VllmConfig, set_current_vllm_config

    with (
        pytest.raises(ValueError, match="cpu_offload"),
        set_current_vllm_config(VllmConfig()),
    ):
        ParallelEngramEmbedding(
            num_embeddings=sum(HEAD_SIZES),
            dim=DIM,
            head_sizes=HEAD_SIZES,
            block_size=BLOCK,
            cpu_offload=False,
            disk_offload_dir=str(tmp_path),
        )
