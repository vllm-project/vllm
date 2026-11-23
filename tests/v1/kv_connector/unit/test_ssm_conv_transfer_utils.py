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

import pytest
import torch

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
from vllm.v1.kv_cache_interface import MambaSpec

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
