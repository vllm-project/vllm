# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hetero-TP conv-state transfer moves every head of every conv sub-projection
to the decode rank that owns it.

The test builds the full conv state with values that name their sub-projection,
head, channel and row, shards it for the prefill TP, moves it with the byte
ranges from `MambaConvSplitInfo` the way the NIXL connector does, and compares
each decode page with the full state sharded directly for the decode TP, in
both conv state layouts.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    compute_nixl_compatibility_hash,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
    NixlConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
    derive_mamba_conv_split,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    get_conv_state_layout,
    is_conv_state_dim_first,
)
from vllm.models.nemotron_shortcut.nvidia.gdp import (
    gdp_conv_proj_dims,
    gdp_state_shape,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

from .utils import create_vllm_config

pytestmark = pytest.mark.cpu_test

CONV_ROWS = 3  # conv window - 1
GDP_VALUE_HEAD, GDP_KEY_HEAD = 64, 128


@dataclass(frozen=True)
class Proj:
    """One conv sub-projection, sharded across TP ranks by heads."""

    num_heads: int
    head_width: int


@dataclass(frozen=True)
class Layer:
    name: str
    projs: tuple[Proj, ...]
    temporal: tuple[int, ...]  # global temporal state shape; dim 0 is sharded
    rows: int = CONV_ROWS  # conv rows per request, more with speculative decoding

    def spec(self, tp: int) -> MambaSpec:
        raise NotImplementedError


@dataclass(frozen=True)
class Mamba2Layer(Layer):
    def spec(self, tp: int) -> MambaSpec:
        return _spec(self, tp, MambaAttentionBackendEnum.MAMBA2)


@dataclass(frozen=True)
class GDNLayer(Layer):
    def spec(self, tp: int) -> MambaSpec:
        return _spec(self, tp, MambaAttentionBackendEnum.GDN_ATTN)


@dataclass(frozen=True)
class GDPLayer(Layer):
    n: int = 1
    num_heads: int = 8
    num_spec: int = 0

    def spec(self, tp: int) -> MambaSpec:
        # The GDP layer's own state shapes and declared sub-projections.
        shapes = gdp_state_shape(
            tp,
            self.num_heads,
            self.num_heads,
            GDP_KEY_HEAD,
            GDP_VALUE_HEAD,
            CONV_ROWS + 1,
            self.n,
            self.num_spec,
        )
        proj_dims = gdp_conv_proj_dims(
            tp, self.num_heads, self.num_heads, GDP_KEY_HEAD, GDP_VALUE_HEAD, self.n
        )
        return MambaSpec(
            block_size=16,
            shapes=shapes,
            dtypes=(torch.float32, torch.float32),
            mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
            conv_proj_dims=proj_dims,
        )


def _spec(layer: Layer, tp: int, mamba_type) -> MambaSpec:
    local_conv_dim = sum(p.num_heads // tp * p.head_width for p in layer.projs)
    temporal = (layer.temporal[0] // tp, *layer.temporal[1:])
    conv_shape = (local_conv_dim, layer.rows)
    if not is_conv_state_dim_first():
        conv_shape = conv_shape[::-1]
    return MambaSpec(
        block_size=16,
        shapes=(conv_shape, temporal),
        dtypes=(torch.float32, torch.float32),
        mamba_type=mamba_type,
    )


def gdp_layer(n: int, num_heads: int = 8, num_spec: int = 0) -> GDPLayer:
    # Checkpoint order of the conv input: [V_1..V_n, K_1..K_n, Q].
    value, key = Proj(num_heads, GDP_VALUE_HEAD), Proj(num_heads, GDP_KEY_HEAD)
    return GDPLayer(
        name=f"gdp_n{n}" + (f"_spec{num_spec}" if num_spec else ""),
        projs=(value,) * n + (key,) * n + (key,),
        temporal=(num_heads, GDP_VALUE_HEAD, GDP_KEY_HEAD),
        rows=CONV_ROWS + num_spec,
        n=n,
        num_heads=num_heads,
        num_spec=num_spec,
    )


LAYERS = [
    Mamba2Layer(
        name="mamba2",
        projs=(Proj(8, 16), Proj(4, 8), Proj(4, 8)),  # x, B, C
        temporal=(8, 16),
    ),
    GDNLayer(
        name="gdn",
        projs=(Proj(4, 128), Proj(4, 128), Proj(8, 64)),  # q, k, v
        temporal=(8, 64, 128),
    ),
    gdp_layer(n=1),
    gdp_layer(n=2),
    gdp_layer(n=3),
    gdp_layer(n=3, num_spec=2),
]

TP_PAIRS = [(1, 1), (2, 2), (1, 2), (2, 1), (2, 4), (4, 2), (1, 4), (4, 1)]


def full_conv_state(layer: Layer) -> list[torch.Tensor]:
    """Per sub-projection (channels, rows); each value encodes its position."""
    states = []
    for p_idx, proj in enumerate(layer.projs):
        head, chan, row = torch.meshgrid(
            torch.arange(proj.num_heads),
            torch.arange(proj.head_width),
            torch.arange(layer.rows),
            indexing="ij",
        )
        code = ((p_idx * 16 + head) * 256 + chan) * 8 + row
        states.append(code.reshape(-1, layer.rows).float())
    return states


def page(layer: Layer, state: list[torch.Tensor], tp: int, rank: int) -> bytes:
    """Rank's conv page: its heads of each sub-projection, in order, stored
    (channels, rows) for DS or (rows, channels) for SD."""
    shards = []
    for proj, proj_state in zip(layer.projs, state):
        width = proj.num_heads // tp * proj.head_width
        shards.append(proj_state[rank * width : (rank + 1) * width])
    conv = torch.cat(shards)
    if not is_conv_state_dim_first():
        conv = conv.T
    return conv.contiguous().numpy().tobytes()


def transfer(layer: Layer, p_pages: list[bytes], p_tp: int, d_tp: int, d_rank: int):
    """Assemble a decode page from prefill pages like the NIXL connector:
    remote ranges from `remote_conv_offsets`, local ranges from
    `local_conv_offsets`, split per source rank when P_TP > D_TP."""
    info = derive_mamba_conv_split(layer.spec(d_tp), d_tp, require_dim_first=False)
    out = bytearray(info.ssm_sizes[0])
    local = info.local_conv_offsets
    if d_tp >= p_tp:
        tp_ratio = d_tp // p_tp
        src = p_pages[d_rank // tp_ratio]
        remote = info.remote_conv_offsets(d_rank % tp_ratio, tp_ratio)
        for (r_off, r_size), (l_off, l_size) in zip(remote, local, strict=True):
            assert r_size == l_size
            out[l_off : l_off + l_size] = src[r_off : r_off + r_size]
    else:
        num_sources = p_tp // d_tp
        remote = info.remote_conv_offsets(0, -num_sources)
        for j in range(num_sources):
            src = p_pages[d_rank * num_sources + j]
            for (r_off, r_size), (l_off, l_size) in zip(remote, local, strict=True):
                chunk = l_size // num_sources
                assert r_size == chunk
                dst = l_off + j * chunk
                out[dst : dst + chunk] = src[r_off : r_off + r_size]
    return bytes(out)


@pytest.fixture(autouse=True, params=["DS", "SD"])
def _conv_layout(monkeypatch, request):
    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", request.param)
    get_conv_state_layout.cache_clear()
    assert is_conv_state_dim_first() == (request.param == "DS")
    yield
    get_conv_state_layout.cache_clear()


@pytest.mark.parametrize("layer", LAYERS, ids=lambda layer: layer.name)
@pytest.mark.parametrize("p_tp,d_tp", TP_PAIRS)
def test_decode_ranks_get_their_heads(layer: Layer, p_tp: int, d_tp: int):
    state = full_conv_state(layer)
    p_pages = [page(layer, state, p_tp, r) for r in range(p_tp)]
    for d_rank in range(d_tp):
        got = transfer(layer, p_pages, p_tp, d_tp, d_rank)
        assert got == page(layer, state, d_tp, d_rank), (
            f"{layer.name}: decode rank {d_rank} got the wrong conv channels "
            f"for P_TP={p_tp} -> D_TP={d_tp}"
        )


@pytest.mark.parametrize("n", [1, 2, 3])
def test_gdp_sub_projections(n: int):
    # 4 heads per rank: value pieces of 4*64, key and query pieces of 4*128.
    info = derive_mamba_conv_split(gdp_layer(n).spec(2), 2, require_dim_first=False)
    assert info.local_proj_dims == (256,) * n + (512,) * (n + 1)


NUM_BLOCKS = 3


def _nixl_worker(spec: MambaSpec, tp: int, rank: int) -> MagicMock:
    """The attributes NIXL's Mamba descriptor builders read, for one layer."""
    worker = MagicMock(spec=NixlConnectorWorker)
    worker._conv_decomp = derive_mamba_conv_split(spec, tp, require_dim_first=False)
    worker._mamba_ssm_size = worker._conv_decomp.ssm_sizes
    worker._logical_num_blocks = NUM_BLOCKS
    worker._physical_blocks_per_logical_kv_block = 1
    worker.device_id = 0
    worker.tp_rank = rank
    worker._has_mamba = True
    worker.num_regions = 0
    worker.region_num_blocks = []
    worker._ssm_region_indices = None
    worker.block_stride_per_layer = [spec.page_size_bytes]
    worker.block_len_per_layer = [spec.page_size_bytes]
    worker._ple_region_index = None
    worker._ple_group_index = None
    worker._group_spec_types = [AttentionSpec, MambaSpec]
    worker._fa_desc_replicated = lambda starts, *_: np.zeros(len(starts), dtype=bool)
    worker._pack_strided_descs = NixlConnectorWorker._pack_strided_descs
    worker._strided_desc_starts = NixlConnectorWorker._strided_desc_starts
    for name in (
        "_build_mamba_local",
        "_build_mamba_remote",
        "_build_local_splits_from_plan",
        "_compute_desc_ids",
    ):
        method = getattr(NixlConnectorWorker, name)
        setattr(worker, name, method.__get__(worker, NixlConnectorWorker))
    return worker


def _expand(descs: np.ndarray) -> list[tuple[int, int]]:
    """Strided descriptors (addr, len, device, stride, count) -> one per block."""
    return [
        (addr + k * stride, length)
        for addr, length, _, stride, count in descs.tolist()
        for k in range(count)
    ]


def _full_page(layer: GDPLayer, tp: int, rank: int) -> bytes:
    """Conv state, then this rank's heads of the recurrent state."""
    heads = layer.num_heads // tp
    temporal = torch.arange(
        layer.num_heads * GDP_VALUE_HEAD * GDP_KEY_HEAD, dtype=torch.float32
    ).reshape(layer.num_heads, GDP_VALUE_HEAD, GDP_KEY_HEAD)
    ssm = temporal[rank * heads : (rank + 1) * heads].contiguous().numpy()
    return page(layer, full_conv_state(layer), tp, rank) + ssm.tobytes()


@pytest.mark.parametrize("p_tp,d_tp", [(1, 1), (1, 2), (2, 1), (2, 4), (4, 2)])
def test_nixl_descriptors_move_each_rank_its_heads(p_tp: int, d_tp: int):
    """NIXL's own descriptor builders and descriptor-id math, driven through a
    pull read of block 1 on every prefill rank into block 0 on each decode
    rank, deliver the decode rank's whole page: conv and recurrent state."""
    layer = gdp_layer(n=2, num_spec=2)
    p_spec = layer.spec(p_tp)
    p_mem = []
    for rank in range(p_tp):
        mem = bytearray(p_spec.page_size_bytes * NUM_BLOCKS)
        mem[p_spec.page_size_bytes : 2 * p_spec.page_size_bytes] = _full_page(
            layer, p_tp, rank
        )
        p_mem.append(mem)
    for d_rank in range(d_tp):
        d_spec = layer.spec(d_tp)
        d_mem = bytearray(d_spec.page_size_bytes * NUM_BLOCKS)
        worker = _nixl_worker(d_spec, d_tp, d_rank)
        local = worker._build_mamba_local([0])
        if d_tp >= p_tp:
            tp_ratio = d_tp // p_tp
            sources = [(d_rank // tp_ratio, local)]
        else:
            tp_ratio = -(p_tp // d_tp)
            ranks = [d_rank * -tp_ratio + j for j in range(-tp_ratio)]
            plan = SimpleNamespace(
                source_ranks_per_group=[[0], ranks],
                all_source_ranks=ranks,
                rank_to_attention_slot={},
            )
            sources = list(
                zip(ranks, worker._build_local_splits_from_plan(plan, local, 0))
            )
        local_ids, remote_ids = (
            worker._compute_desc_ids(
                block_ids=([], [block]),
                dst_num_blocks=NUM_BLOCKS,
                block_size_ratio=None,
                physical_blocks_per_logical=1,
                pairs=([0], [1]),
                region_num_blocks=[0],
            ).tolist()
            for block in (0, 1)
        )
        for p_rank, local_descs in sources:
            remote_descs = worker._build_mamba_remote(
                SimpleNamespace(
                    kv_caches_base_addr=[0],
                    ssm_sizes=derive_mamba_conv_split(
                        p_spec, p_tp, require_dim_first=False
                    ).ssm_sizes,
                    num_blocks=NUM_BLOCKS,
                    device_id=0,
                    block_strides=[p_spec.page_size_bytes],
                ),
                tp_ratio,
                SimpleNamespace(remote_physical_blocks_per_logical=1),
                SimpleNamespace(remote_ssm_regions=[0], remote_ple_region=None),
            )
            dst, src = _expand(local_descs), _expand(remote_descs)
            for d_id, s_id in zip(local_ids, remote_ids, strict=True):
                (d_addr, d_len), (s_addr, s_len) = dst[d_id], src[s_id]
                assert d_len == s_len
                d_mem[d_addr : d_addr + d_len] = p_mem[p_rank][s_addr : s_addr + s_len]
        assert bytes(d_mem[: d_spec.page_size_bytes]) == _full_page(layer, d_tp, d_rank)


def test_conv_layout_is_part_of_the_nixl_compatibility_hash():
    """A DS prefill and an SD decode must fail the handshake."""
    config = create_vllm_config()
    layout_hash = compute_nixl_compatibility_hash(config, "FLASH_ATTN")
    other = "SD" if is_conv_state_dim_first() else "DS"
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_SSM_CONV_STATE_LAYOUT", other)
        get_conv_state_layout.cache_clear()
        other_hash = compute_nixl_compatibility_hash(config, "FLASH_ATTN")
    get_conv_state_layout.cache_clear()
    assert layout_hash != other_hash
