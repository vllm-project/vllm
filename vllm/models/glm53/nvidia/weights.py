# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 decode segment 1's weight forms, built by vLLM when it loads the model."""

from __future__ import annotations

import logging
import math
import re
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.models.glm53.nvidia.ops.cute_dsl.fused_moe import LAYOUT
from vllm.models.glm53.nvidia.ops.launch import Glm53Geometry

logger = logging.getLogger(__name__)

_LAYER_NAME = re.compile(r"(?:^|\.)layers\.(\d+)\.(.+)$")


@dataclass(frozen=True)
class WorkerPages:
    """The worker's streamed weight pages: an E4M3 register-A tile of ``rows`` x
    ``columns``, then FP32 scales.
    """

    rows: int = 32
    columns: int = 1024
    trailer_bytes: int = 128
    scale_columns: int = 128
    absorbed_scale_rows: int = 16
    chunk_bytes: int = 16
    swizzle_bytes: int = 128


PAGES = WorkerPages()


def enabled() -> bool:
    """Return whether the latency-specialized path asks for the segment's weight
    forms.
    """
    from vllm.models.glm53.nvidia import switch

    return switch.current().enabled


def pack_register_a_records(
    matrix: torch.Tensor, *, rows_per_cta: int, columns_per_page: int
) -> torch.Tensor:
    """Return ``uint8 [cta, page, bytes]`` register-A records of ``mma.m16n8k16``'s 16 x
    16 A tiles.
    """
    tile = 16
    if matrix.ndim != 2 or matrix.element_size() not in (1, 2):
        raise ValueError("expected a rank-two matrix of one- or two-byte elements")
    rows, columns = matrix.shape
    if (
        rows_per_cta % tile
        or columns_per_page % tile
        or rows % rows_per_cta
        or columns % columns_per_page
    ):
        raise ValueError("matrix must contain complete CTA/page tiles of 16-multiples")
    element = torch.uint8 if matrix.element_size() == 1 else torch.int16
    ctas, m_tiles = rows // rows_per_cta, rows_per_cta // tile
    pages, k_tiles = columns // columns_per_page, columns_per_page // tile
    tiles = matrix.contiguous().view(element)
    tiles = tiles.reshape(ctas, m_tiles, tile, pages, k_tiles, tile).permute(
        0, 3, 1, 4, 2, 5
    )
    # A tile's fragments: row halves of 8 rows; column halves of 4 lanes x 2 elements.
    tiles = tiles.reshape(ctas, pages, m_tiles, k_tiles, 2, 8, 2, 4, 2)
    tiles = tiles.permute(0, 1, 2, 3, 5, 7, 6, 4, 8)
    records = tiles.contiguous().reshape(ctas, pages, rows_per_cta * columns_per_page)
    return records.view(torch.uint8)


def _with_trailer(
    records: torch.Tensor, trailer: torch.Tensor, trailer_bytes: int
) -> torch.Tensor:
    ctas, pages, matrix_bytes = records.shape
    if trailer.shape[:2] != (ctas, pages) or trailer.dtype != torch.float32:
        raise ValueError("scale trailer must be FP32 [cta, page, scales]")
    if trailer.shape[2] * trailer.element_size() > trailer_bytes:
        raise ValueError(f"scale trailer exceeds {trailer_bytes} bytes")
    output = torch.zeros(
        (ctas, pages, matrix_bytes + trailer_bytes),
        dtype=torch.uint8,
        device=records.device,
    )
    output[..., :matrix_bytes] = records
    scale_bytes = trailer.contiguous().view(torch.uint8).reshape(ctas, pages, -1)
    output[..., matrix_bytes : matrix_bytes + scale_bytes.shape[2]] = scale_bytes
    return output


def _block_rows(first_row: int, rows: int, block: int) -> int:
    index = first_row // block
    if (first_row + rows - 1) // block != index:
        raise ValueError(
            f"row tile [{first_row}, {first_row + rows}) straddles a {block}-row scale "
            "block"
        )
    return index


def _row_block_scales(
    scales: torch.Tensor, rows: list[int], tile: int, block: int
) -> torch.Tensor:
    groups = []
    for first in range(0, len(rows), tile):
        members = rows[first : first + tile]
        blocks = {row // block for row in members}
        if len(blocks) != 1:
            raise ValueError(
                f"row tile starting at {members[0]} straddles scale blocks"
            )
        groups.append(scales[blocks.pop()].float())
    return torch.stack(groups)


def dequantize_blocks(
    matrix: torch.Tensor, scales: torch.Tensor, block: int
) -> torch.Tensor:
    """Return FP32 ``matrix`` times its ``block`` x ``block`` ``scales``."""
    rows, columns = matrix.shape
    expanded = (
        scales.float()
        .repeat_interleave(block, 0)[:rows]
        .repeat_interleave(block, 1)[:, :columns]
    )
    return matrix.float() * expanded


def _e4m3_blocks(
    blocks: torch.Tensor, dims: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``blocks`` to E4M3 with one scale per block over ``dims``; return the
    values and scales.
    """
    e4m3_max = torch.finfo(torch.float8_e4m3fn).max
    peak = blocks.abs().amax(dim=dims)
    scales = torch.where(peak > 0, peak / e4m3_max, torch.ones_like(peak))
    expanded = scales
    for dim in sorted(dims):
        expanded = expanded.unsqueeze(dim)
    quantized = (blocks / expanded).clamp(-e4m3_max, e4m3_max).to(torch.float8_e4m3fn)
    return quantized, scales


def absorbed_pages(
    query: torch.Tensor,
    query_scales: torch.Tensor,
    keys: torch.Tensor,
    key_scales: torch.Tensor,
    geometry: Glm53Geometry,
) -> torch.Tensor:
    """Return the absorbed-query pages ``[18 H, 2, 32896]`` of ``H`` heads."""
    g, page = geometry, PAGES
    heads = query.shape[0]
    query_fp32 = dequantize_blocks(
        query.reshape(heads * g.qk_head, g.q_lora), query_scales, g.block
    )
    query_fp32 = query_fp32.reshape(query.shape)
    row_tile, column_tile = (keys.shape[dim] // key_scales.shape[dim] for dim in (1, 2))
    keys_fp32 = keys.float() * key_scales.float().repeat_interleave(
        row_tile, dim=1
    ).repeat_interleave(column_tile, dim=2)
    matrix = torch.matmul(
        keys_fp32.double(), query_fp32[:, : g.nope_dim].double()
    ).float()
    block_rows, block_columns = page.absorbed_scale_rows, page.scale_columns
    blocks = matrix.reshape(
        heads,
        g.kv_lora // block_rows,
        block_rows,
        g.q_lora // block_columns,
        block_columns,
    )
    quantized, scales = _e4m3_blocks(blocks, (2, 4))
    quantized = quantized.reshape(heads * g.kv_lora, g.q_lora)
    pe = query[:, g.nope_dim :].reshape(heads * g.rope_dim, g.q_lora)
    packed_rows = torch.cat((quantized.view(torch.uint8), pe.view(torch.uint8)))
    records = pack_register_a_records(
        packed_rows, rows_per_cta=page.rows, columns_per_page=page.columns
    )
    # The RoPE rows' scales: each head's Q-B scale tile that holds them.
    pe_scales = query_scales.float().reshape(heads, g.qk_head // g.block, -1)[
        :, g.nope_dim // g.block
    ]
    row_scales = torch.cat(
        (
            scales.float()
            .repeat_interleave(block_rows, 1)
            .reshape(heads * g.kv_lora, -1),
            pe_scales.repeat_interleave(g.rope_dim, 0),
        )
    )
    count = packed_rows.shape[0] // page.rows
    pages = g.q_lora // page.columns
    trailer = (
        row_scales.reshape(
            count,
            page.rows // block_rows,
            block_rows,
            pages,
            page.columns // block_columns,
        )[:, :, 0]
        .permute(0, 2, 1, 3)
        .reshape(count, pages, -1)
    )
    return _with_trailer(records, trailer.contiguous(), page.trailer_bytes)


def plane_pages(rows: torch.Tensor) -> torch.Tensor:
    """Lay E4M3 row pages ``[..., rows, columns]`` out as the worker's input pages land
    in smem.
    """
    *lead, height, width = rows.shape
    chunk = PAGES.chunk_bytes
    chunks = PAGES.swizzle_bytes // chunk
    planes = rows.reshape(
        *lead, height, width // PAGES.swizzle_bytes, chunks, chunk
    ).movedim(-3, -4)
    row = torch.arange(height, device=rows.device).view(height, 1)
    column = torch.arange(chunks, device=rows.device).view(1, chunks)
    source = (
        (column ^ (row % chunks)).view(height, chunks, 1).expand(height, chunks, chunk)
    )
    swizzled = torch.gather(planes, -2, source.expand(*planes.shape))
    return swizzled.reshape(*lead, height * width)


def router_pages(
    router: torch.Tensor, gamma_post: torch.Tensor, gamma_input: torch.Tensor
) -> torch.Tensor:
    """Pack a MoE router as input-projection pages for the route predictor."""
    page = PAGES
    block = page.scale_columns
    experts, hidden = router.shape
    if experts % block or hidden % page.columns or gamma_post.shape != (hidden,):
        raise ValueError(
            f"router pages expect W[{block}k, {page.columns}k] and two [{hidden}] "
            "RMSNorm weights"
        )
    if bool((gamma_input == 0).any()):
        raise ValueError("the attention's input RMSNorm weight has a zero entry")
    folded = router.float() * (gamma_post.float() / gamma_input.float())[None, :]
    blocks = folded.reshape(experts // block, block, hidden // block, block)
    quantized, scales = _e4m3_blocks(blocks, (1, 3))
    ctas, pages = experts // page.rows, hidden // page.columns
    records = plane_pages(
        quantized.reshape(experts, hidden)
        .view(torch.uint8)
        .reshape(ctas, page.rows, pages, page.columns)
        .permute(0, 2, 1, 3)
    )
    scale_rows = torch.tensor(
        [_block_rows(page.rows * cta, page.rows, block) for cta in range(ctas)],
        device=scales.device,
    )
    trailer = scales[scale_rows].reshape(ctas, pages, page.columns // block)
    return _with_trailer(records, trailer, page.trailer_bytes)


def head_groups(geometry: Glm53Geometry, layer: int, rank: int) -> tuple[int, ...]:
    """Return the first heads of the head groups ``rank`` needs (a push rank also a
    selector rank's).
    """
    heads = geometry.heads
    if geometry.is_push_rank(layer, rank):
        return (heads * rank, heads * (rank - geometry.selector_ranks))
    return (heads * rank,)


Slicer = Callable[[torch.Tensor], torch.Tensor]


class CheckpointStash:
    """This rank's slices of checkpoint tensors vLLM does not keep as-is."""

    def __init__(
        self, geometry: Glm53Geometry, rank: int, device: torch.device
    ) -> None:
        self.geometry: Glm53Geometry = geometry
        self.rank: int = rank
        self.device: torch.device = device
        self._kept: dict[tuple[int, str], torch.Tensor] = {}
        self.seconds: float = 0.0
        self.bytes: int = 0

    def _slicer(self, layer: int, leaf: str) -> dict[str, Slicer]:
        g = self.geometry
        whole: dict[str, Slicer] = {leaf: lambda tensor: tensor}
        attention = "self_attn."
        if leaf.startswith(attention):
            name = leaf.removeprefix(attention)
            if name in ("q_b_proj.weight_scale_inv", "kv_b_proj.weight_scale_inv"):
                return whole
            groups = head_groups(g, layer, self.rank)
            stride = {
                "q_b_proj.weight": g.qk_head,
                "kv_b_proj.weight": g.kv_b_stride,
            }.get(name)
            if stride is None:
                return {}
            return {
                f"{leaf}@{head}": (
                    lambda tensor, head=head: tensor[
                        stride * head : stride * (head + g.heads)
                    ]
                )
                for head in groups
            }
        return {}

    def _keep(self, name: str, tensor: torch.Tensor) -> None:
        match = _LAYER_NAME.search(name)
        if match is None:
            return
        layer = int(match.group(1))
        if layer >= self.geometry.layers:
            return
        for key, cut in self._slicer(layer, match.group(2)).items():
            # CPU FP8 indexing support varies: slice as bytes.
            raw = tensor.view(torch.uint8) if tensor.element_size() == 1 else tensor
            part = cut(raw).contiguous().to(self.device, copy=True)
            self._kept[(layer, key)] = (
                part.view(tensor.dtype) if tensor.element_size() == 1 else part
            )
            self.bytes += part.numel() * part.element_size()

    def tee(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield ``weights`` unchanged, keeping this rank's slices on the way."""
        for name, tensor in weights:
            start = time.perf_counter()
            self._keep(name, tensor)
            self.seconds += time.perf_counter() - start
            yield name, tensor

    def take(self, layer: int, key: str) -> torch.Tensor:
        """Remove and return one kept slice."""
        try:
            return self._kept.pop((layer, key))
        except KeyError:
            raise KeyError(
                f"the checkpoint stash has no layers.{layer}.{key} (not yielded by "
                "vLLM's iterator)"
            ) from None

    def remaining(self) -> list[str]:
        """Return the names of slices not taken."""
        return [f"layers.{layer}.{key}" for layer, key in self._kept]


@dataclass(frozen=True)
class HeadShard:
    """One head group's Q-B weights and KV-B keys of one layer."""

    query: torch.Tensor
    query_scales: torch.Tensor
    keys: torch.Tensor
    key_scales: torch.Tensor


def _head_rows(
    heads: int, first_head: int, rows_per_head: int, count: int
) -> list[int]:
    return [
        head * rows_per_head + row
        for head in range(first_head, first_head + heads)
        for row in range(count)
    ]


def head_shard(
    stash: CheckpointStash,
    layer: int,
    first_head: int,
    q_b_scale: torch.Tensor,
    kv_b_scale: torch.Tensor,
) -> HeadShard:
    """Return a head group's Q-B and KV-B weights of ``layer``."""
    g = stash.geometry
    query = stash.take(layer, f"self_attn.q_b_proj.weight@{first_head}").reshape(
        g.heads, g.qk_head, g.q_lora
    )
    query_rows = _head_rows(g.heads, first_head, g.qk_head, g.qk_head)
    query_scales = _row_block_scales(q_b_scale, query_rows, g.block, g.block)
    kv_b = stash.take(layer, f"self_attn.kv_b_proj.weight@{first_head}").reshape(
        g.heads, g.kv_b_stride, g.kv_lora
    )
    keys = kv_b[:, : g.nope_dim].contiguous()
    keys_t = (
        keys.view(torch.uint8).transpose(1, 2).contiguous().view(torch.float8_e4m3fn)
    )
    # The largest row tile inside one scale block at every head's first row.
    key_tile = math.gcd(g.kv_b_stride, g.block)
    key_rows = _head_rows(g.heads, first_head, g.kv_b_stride, g.nope_dim)
    key_scales = (
        _row_block_scales(kv_b_scale, key_rows, key_tile, g.block)
        .reshape(g.heads, g.nope_dim // key_tile, -1)
        .transpose(1, 2)
        .contiguous()
    )
    return HeadShard(query, query_scales, keys_t, key_scales)


def _norm(module: torch.nn.Module) -> torch.Tensor:
    weight = module.weight
    assert isinstance(weight, torch.Tensor)
    return weight.data


def attention_forms(
    model: torch.nn.Module, stash: CheckpointStash, layer: int, rank: int
) -> dict[str, torch.Tensor]:
    """Return ``layer``'s attention forms on ``rank``."""
    g = stash.geometry
    decoder = model.layers[layer]
    attention = decoder.self_attn
    take = stash.take
    prefix = f"layers.{layer}.attention."
    gamma_x = _norm(decoder.input_layernorm).float().contiguous()
    gamma_q = _norm(attention.q_a_layernorm).float().contiguous()
    gamma_kv = _norm(attention.kv_a_layernorm).float().contiguous()
    q_b_scale = take(layer, "self_attn.q_b_proj.weight_scale_inv")
    kv_b_scale = take(layer, "self_attn.kv_b_proj.weight_scale_inv")
    groups = {
        head: head_shard(stash, layer, head, q_b_scale, kv_b_scale)
        for head in head_groups(g, layer, rank)
    }
    absorbed = torch.cat(
        [
            absorbed_pages(s.query, s.query_scales, s.keys, s.key_scales, g)
            for s in groups.values()
        ]
    )
    worker = {
        "gamma_x": gamma_x,
        "gamma_q": gamma_q,
        "absorb_weight": absorbed,
        "gamma_kv": gamma_kv,
    }
    if g.is_shared(layer) or rank >= g.selector_ranks:
        return {prefix + name: tensor for name, tensor in worker.items()}
    indexer = attention.indexer
    wk_weights_proj = indexer.wk_weights_proj.weight.data
    shape = (g.index_dim + g.index_heads, g.hidden)
    if (
        wk_weights_proj.dtype != torch.bfloat16
        or tuple(wk_weights_proj.shape) != shape
        or not wk_weights_proj.is_contiguous()
    ):
        raise ValueError(
            f"layer {layer}: vLLM's wk_weights_proj is not a contiguous BF16 "
            f"{list(shape)} tensor"
        )
    selector = {
        "gamma_x": gamma_x,
        "wis_weight": wk_weights_proj,
        "key_gamma": _norm(indexer.k_norm).float().contiguous(),
        "key_bias": indexer.k_norm.bias.data.float().contiguous(),
        "gamma_q": gamma_q,
    }
    return {
        **{prefix + name: tensor for name, tensor in selector.items()},
        **{prefix + "probe." + name: tensor for name, tensor in worker.items()},
    }


def ffn_forms(
    model: torch.nn.Module, geometry: Glm53Geometry, layer: int
) -> dict[str, torch.Tensor]:
    """Return ``layer``'s feed-forward forms except the weights vLLM holds in place."""
    decoder = model.layers[layer]
    prefix = f"layers.{layer}.ffn."
    forms = {prefix + "gamma": _norm(decoder.post_attention_layernorm)}
    if layer < geometry.dense_layers:
        return forms
    gate_module = decoder.mlp.gate
    forms[prefix + "router_weight"] = gate_module.weight.data
    forms[prefix + "correction_bias"] = gate_module.e_score_correction_bias.data
    return forms


@dataclass(frozen=True)
class RoutedBuffers:
    """One layer's colocated experts in two ``uint8`` allocations
    (``fused_moe.LAYOUT``).
    """

    routed_up: torch.Tensor
    routed_down: torch.Tensor
    slots: int = LAYOUT.routed_experts

    def _slot_views(
        self, first: int, count: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = self.routed_up.view(torch.float8_e4m3fn)
        w13 = weights[
            first * LAYOUT.w13_expert_bytes : (first + count) * LAYOUT.w13_expert_bytes
        ]
        w2_start = LAYOUT.w2_offset + first * LAYOUT.w2_expert_bytes
        w2 = self.routed_down[
            w2_start : w2_start + count * LAYOUT.w2_expert_bytes
        ].view(torch.float8_e4m3fn)
        w13_scale_start = (
            LAYOUT.w13_scale_offset + first * LAYOUT.w13_scale_expert_bytes
        )
        w13_scale = self.routed_down[
            w13_scale_start : w13_scale_start + count * LAYOUT.w13_scale_expert_bytes
        ]
        w2_scale_start = LAYOUT.w2_scale_offset + first * LAYOUT.w2_scale_expert_bytes
        w2_scale = self.routed_down[
            w2_scale_start : w2_scale_start + count * LAYOUT.w2_scale_expert_bytes
        ]
        return (
            w13.view(count, *LAYOUT.w13_shape),
            w13_scale.view(torch.float32).view(count, *LAYOUT.w13_scale_shape),
            w2.view(count, *LAYOUT.w2_shape),
            w2_scale.view(torch.float32).view(count, *LAYOUT.w2_scale_shape),
        )

    def views(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the routed experts' (w13, w13_scale, w2, w2_scale) views."""
        return self._slot_views(0, LAYOUT.routed_experts)

    def shared_views(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the shared expert slot's (w13, w13_scale, w2, w2_scale) views."""
        if self.slots != LAYOUT.slots:
            raise ValueError("these buffers have no shared-expert slot")
        return tuple(view[0] for view in self._slot_views(LAYOUT.shared_slot, 1))  # type: ignore[return-value]

    def kernel_inputs(self) -> dict[str, torch.Tensor]:
        """Return the MoE inputs these buffers provide, by name."""
        routed = LAYOUT.routed_experts
        inputs = {
            "routed_up": self.routed_up[: LAYOUT.routed_up_bytes(routed)],
            "routed_down": self.routed_down[: LAYOUT.routed_down_bytes(routed)],
        }
        if self.slots == LAYOUT.slots:
            up = LAYOUT.routed_up_bytes(LAYOUT.shared_slot)
            down = LAYOUT.routed_down_bytes(LAYOUT.shared_slot)
            inputs["shared_up"] = self.routed_up[up : up + LAYOUT.w13_expert_bytes]
            inputs["shared_down"] = self.routed_down[
                down : down + LAYOUT.w2_expert_bytes
            ]
        return inputs


def expert_block_scales(
    name: str, weights: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """Return FP32 block scales of vLLM's experts in either backend's form."""
    experts, rows, columns = weights.shape
    block = LAYOUT.block
    if scales.dtype == torch.float32:
        expected = (experts, -(-rows // block), -(-columns // block))
        if tuple(scales.shape) != expected:
            raise ValueError(
                f"{name}: FP32 block scales are {tuple(scales.shape)}, expected "
                f"{expected}"
            )
        return scales
    return ue8m0_block_scales(name, weights, scales, block)


def materialize_experts(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    slots: int = LAYOUT.routed_experts,
) -> RoutedBuffers:
    """Copy one layer's expert tensors into two colocated buffers."""
    routed = LAYOUT.routed_experts
    if slots not in (routed, LAYOUT.slots):
        raise ValueError(
            f"colocated buffers hold {routed} or {LAYOUT.slots} expert slots, not "
            f"{slots}"
        )
    for name, tensor, shape in (
        ("w13", w13, LAYOUT.w13_shape),
        ("w2", w2, LAYOUT.w2_shape),
    ):
        if (
            tuple(tensor.shape) != (routed, *shape)
            or tensor.dtype != torch.float8_e4m3fn
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"routed experts: {name} must be contiguous float8_e4m3fn "
                f"{(routed, *shape)}, got {tensor.dtype} "
                f"{tuple(tensor.shape)}; start vLLM with --moe-backend deep_gemm"
            )
    scales = (
        expert_block_scales("w13 scales", w13, w13_scale),
        expert_block_scales("w2 scales", w2, w2_scale),
    )
    device = w13.device
    buffers = RoutedBuffers(
        torch.empty(LAYOUT.routed_up_bytes(slots), dtype=torch.uint8, device=device),
        torch.empty(LAYOUT.routed_down_bytes(slots), dtype=torch.uint8, device=device),
        slots,
    )
    buffers.routed_down[: LAYOUT.w2_offset].zero_()
    if slots > routed:
        buffers.routed_up[LAYOUT.routed_up_bytes(routed) :].zero_()
        buffers.routed_down[LAYOUT.routed_down_bytes(routed) :].zero_()
    w13_view, w13_scale_view, w2_view, w2_scale_view = buffers.views()
    for view, tensor in (
        (w13_view, w13),
        (w13_scale_view, scales[0]),
        (w2_view, w2),
        (w2_scale_view, scales[1]),
    ):
        view.copy_(tensor)
    return buffers


def _routed_experts(mlp: torch.nn.Module, layer: int) -> torch.nn.Module:
    found = [
        module
        for _, module in mlp.named_modules()
        if "w13_weight" in dict(module.named_parameters(recurse=False))
    ]
    if len(found) != 1:
        raise ValueError(
            f"layer {layer}: {len(found)} vLLM modules hold routed experts, expected "
            "one"
        )
    return found[0]


def share_shared_expert(
    mlp: torch.nn.Module, buffers: RoutedBuffers, layer: int
) -> None:
    """Move vLLM's shared expert of one layer into its slot of the colocated buffers."""
    shared = mlp.shared_experts
    linears = ("gate_up_proj", "down_proj")
    views = buffers.shared_views()
    if not is_deepgemm_form(shared.gate_up_proj):
        raise ValueError(
            f"layer {layer}: vLLM's shared expert is not in the DeepGEMM form (its "
            "default FP8 linear)"
        )
    weights = [getattr(shared, linear).weight for linear in linears]
    for weight, view in zip(weights, views[0::2]):
        if (
            weight.shape != view.shape
            or weight.dtype != view.dtype
            or not weight.data.is_contiguous()
        ):
            raise ValueError(
                f"layer {layer}: vLLM's shared expert weight is {weight.dtype} "
                f"{tuple(weight.shape)}"
            )
    decoded = [
        ue8m0_block_scales(
            f"layer {layer} shared {linear}",
            weight,
            getattr(shared, linear).weight_scale_inv,
            LAYOUT.block,
        )
        for linear, weight in zip(linears, weights)
    ]
    views[1].copy_(decoded[0])
    views[3].copy_(decoded[1])
    for weight, view in zip(weights, views[0::2]):
        view.copy_(weight.data)
        weight.data = view


def repoint_routed_experts(
    experts: torch.nn.Module, layer: int, slots: int
) -> RoutedBuffers:
    """Move one layer's routed experts into colocated buffers."""
    names = ("w13_weight", "w13_weight_scale_inv", "w2_weight", "w2_weight_scale_inv")
    params = [getattr(experts, name) for name in names]
    buffers = materialize_experts(*(param.data for param in params), slots=slots)
    for param, view in zip(params, buffers.views(), strict=True):
        if param.data.dtype != view.dtype:
            continue
        if view.shape != param.data.shape:
            raise RuntimeError(
                f"layer {layer}: a colocated expert view is {tuple(view.shape)}"
            )
        param.data = view
    return buffers


def share_experts(
    model: torch.nn.Module, layers: Iterable[int]
) -> dict[int, RoutedBuffers]:
    """Move each MoE layer's experts into colocated buffers."""
    shared: dict[int, RoutedBuffers] = {}
    for layer in layers:
        experts = _routed_experts(model.layers[layer].mlp, layer)
        buffers = repoint_routed_experts(experts, layer, LAYOUT.slots)
        share_shared_expert(model.layers[layer].mlp, buffers, layer)
        torch.cuda.empty_cache()
        shared[layer] = buffers
    return shared


def is_deepgemm_form(module: torch.nn.Module) -> bool:
    """Return whether vLLM holds ``module``'s FP8 linear in DeepGEMM form."""
    scale = getattr(module, "weight_scale_inv", None)
    return scale is not None and scale.dtype == torch.int32


def ue8m0_block_scales(
    name: str, weight: torch.Tensor, scale: torch.Tensor, block: int
) -> torch.Tensor:
    """Return FP32 ``block`` x ``block`` scales decoded from vLLM's packed UE8M0
    scales.
    """
    *batch, rows, columns = weight.shape
    per_word = torch.int32.itemsize  # one exponent byte each
    blocks_k = -(-columns // block)
    words = -(-blocks_k // per_word)
    if (
        weight.dtype != torch.float8_e4m3fn
        or not weight.is_contiguous()
        or tuple(scale.shape) != (*batch, rows, words)
        or scale.dtype != torch.int32
        or scale.stride(-2) != 1
    ):
        raise ValueError(
            f"vLLM's {name} is {tuple(weight.shape)} {weight.dtype} with scales "
            f"{tuple(scale.shape)} {scale.dtype} "
            f"strides {tuple(scale.stride())}: not the DeepGEMM form (float8_e4m3fn "
            "weight, "
            f"{(*batch, rows, words)} int32 MN-major scales)"
        )
    # contiguous() would keep a one-column tensor's MN-major strides.
    exponents = (
        scale.new_empty(scale.shape)
        .copy_(scale)
        .view(torch.uint8)
        .view(*batch, rows, per_word * words)
    )
    if bool((exponents[..., blocks_k:] != 0).any()):
        raise ValueError(
            f"vLLM's {name}: packed UE8M0 scales have non-zero padding K blocks"
        )
    exponents = exponents[..., :blocks_k]
    firsts = exponents[..., ::block, :]
    if not torch.equal(exponents, firsts.repeat_interleave(block, -2)[..., :rows, :]):
        raise ValueError(
            f"vLLM's {name}: packed UE8M0 scales differ within a {block}-row block"
        )
    if bool((firsts == 0).any()) or bool(
        (firsts == torch.iinfo(torch.uint8).max).any()
    ):
        raise ValueError(
            f"vLLM's {name}: packed UE8M0 scales hold a zero or NaN exponent"
        )
    mantissa_bits = round(-math.log2(torch.finfo(torch.float32).eps))
    return (firsts.to(torch.int32) << mantissa_bits).view(torch.float32).contiguous()


@dataclass(frozen=True)
class VllmLinear:
    """A worker linear read in place: weight, FP32 scales, padded to an aligned size."""

    ALIGNMENT: ClassVar[int] = 128

    key: str
    operand: str
    rows: int
    columns: int
    block: int

    @property
    def scale_shape(self) -> tuple[int, int]:
        """Return the block-scale shape ``[ceil(N/block), K/block]``."""
        return (-(-self.rows // self.block), -(-self.columns // self.block))

    @property
    def scale_offset(self) -> int:
        """Return the scales' byte offset (after the E4M3 weight)."""
        return self.rows * self.columns * torch.float8_e4m3fn.itemsize

    @property
    def bytes(self) -> int:
        """Return the buffer size (the scales' end, aligned)."""
        end = self.scale_offset + torch.float32.itemsize * math.prod(self.scale_shape)
        return -(-end // self.ALIGNMENT) * self.ALIGNMENT

    def _place(self, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        buffer = torch.zeros(self.bytes, dtype=torch.uint8, device=weight.device)
        buffer[: self.scale_offset].copy_(weight.view(torch.uint8).reshape(-1))
        scale_bytes = scale.view(torch.uint8).reshape(-1)
        buffer[self.scale_offset : self.scale_offset + scale_bytes.numel()].copy_(
            scale_bytes
        )
        return buffer

    def adopt(self, module: torch.nn.Module, layer: int) -> torch.Tensor:
        """Move vLLM's weight and scales into one buffer and re-point them."""
        weight = module.weight
        scale = getattr(module, "weight_scale_inv", None)
        if not is_deepgemm_form(module):
            raise ValueError(
                f"layer {layer}: vLLM's {self.key} is not in the DeepGEMM form (its "
                "default FP8 linear)"
            )
        if tuple(weight.shape) != (self.rows, self.columns):
            raise ValueError(
                f"layer {layer}: vLLM's {self.key} weight is {tuple(weight.shape)}"
            )
        decoded = ue8m0_block_scales(
            f"layer {layer} {self.key}", weight, scale, self.block
        )
        buffer = self._place(weight.data, decoded)
        weight.data = buffer[: self.scale_offset].view(weight.dtype).view(weight.shape)
        return buffer.view(1, 1, self.bytes)


def worker_linears(geometry: Glm53Geometry) -> dict[str, VllmLinear]:
    """Return the worker's linears read in place, by family."""
    g = geometry
    return {
        "input": VllmLinear(
            "fused_qkv_a_proj",
            "input_weight",
            g.q_lora + g.kv_lora + g.rope_dim,
            g.hidden,
            g.block,
        ),
        "o": VllmLinear("o_proj", "o_weight", g.hidden, g.value_dim * g.heads, g.block),
        "kvb": VllmLinear(
            "kv_b_proj", "kvb_weight", g.kv_b_stride * g.heads, g.kv_lora, g.block
        ),
    }


def selector_linear(geometry: Glm53Geometry) -> VllmLinear:
    """Return the selector's replicated ``indexer.wq_b``, read in place like the worker
    linears.
    """
    g = geometry
    return VllmLinear(
        "indexer.wq_b", "wqb_weight", g.index_heads * g.index_dim, g.q_lora, g.block
    )


def adopt_selector_linears(
    model: torch.nn.Module,
    geometry: Glm53Geometry,
    forms: dict[str, torch.Tensor],
    rank: int,
) -> None:
    """Bind each full layer's vLLM ``indexer.wq_b`` as the selector's ``wqb_weight`` on
    selector ranks, one copy.
    """
    if rank >= geometry.selector_ranks:
        return
    linear = selector_linear(geometry)
    for layer in range(geometry.layers):
        if not geometry.is_shared(layer):
            wq_b = model.layers[layer].self_attn.indexer.wq_b
            forms[f"layers.{layer}.attention.{linear.operand}"] = linear.adopt(
                wq_b, layer
            )
    torch.cuda.empty_cache()


def adopt_worker_linears(
    model: torch.nn.Module,
    geometry: Glm53Geometry,
    forms: dict[str, torch.Tensor],
    rank: int,
) -> None:
    """Bind each target layer's vLLM worker linears in place."""
    for linear in worker_linears(geometry).values():
        for layer in range(geometry.layers):
            buffer = linear.adopt(
                getattr(model.layers[layer].self_attn, linear.key), layer
            )
            probe = (
                "probe."
                if not geometry.is_shared(layer) and rank < geometry.selector_ranks
                else ""
            )
            forms[f"layers.{layer}.attention.{probe}{linear.operand}"] = buffer
        torch.cuda.empty_cache()


def dense_shapes(geometry: Glm53Geometry) -> dict[str, tuple[int, int]]:
    """Return the dense FFN's weight and block-scale shapes on one rank."""
    g = geometry
    return {
        "gate_up_proj": (2 * g.dense_shard, g.hidden),
        "gate_up_scale": (2 * g.dense_shard // g.block, g.hidden // g.block),
        "down_proj": (g.hidden, g.dense_shard),
        "down_scale": (g.hidden // g.block, g.dense_shard // g.block),
    }


def adopt_dense(
    model: torch.nn.Module, geometry: Glm53Geometry, forms: dict[str, torch.Tensor]
) -> None:
    """Move each dense layer's FFN weights into its two buffers: ``dense_up`` (weight,
    scales), ``dense_down`` (scales, weight).
    """
    shapes = dense_shapes(geometry)
    fp8, f32 = torch.float8_e4m3fn.itemsize, torch.float32.itemsize
    up_scales = fp8 * math.prod(shapes["gate_up_proj"])
    down_weight = f32 * math.prod(shapes["down_scale"])
    up_bytes = up_scales + f32 * math.prod(shapes["gate_up_scale"])
    down_bytes = down_weight + fp8 * math.prod(shapes["down_proj"])
    for layer in range(geometry.dense_layers):
        mlp = model.layers[layer].mlp
        params = []
        if not is_deepgemm_form(mlp.gate_up_proj):
            raise ValueError(
                f"layer {layer}: vLLM's dense mlp is not in the DeepGEMM form (its "
                "default FP8 linear)"
            )
        for name in ("gate_up_proj", "down_proj"):
            module = getattr(mlp, name)
            weight, scale = module.weight, getattr(module, "weight_scale_inv", None)
            if tuple(weight.shape) != shapes[name]:
                raise ValueError(
                    f"layer {layer}: vLLM's mlp.{name} weight is {tuple(weight.shape)}"
                )
            params += [
                weight,
                ue8m0_block_scales(
                    f"layer {layer} mlp.{name}", weight, scale, geometry.block
                ),
            ]
        gate_up = params[0]
        device = gate_up.device
        up_buffer = torch.zeros(up_bytes, dtype=torch.uint8, device=device)
        down_buffer = torch.zeros(down_bytes, dtype=torch.uint8, device=device)
        views = (
            up_buffer[:up_scales]
            .view(torch.float8_e4m3fn)
            .view(shapes["gate_up_proj"]),
            up_buffer[up_scales:].view(torch.float32).view(shapes["gate_up_scale"]),
            down_buffer[down_weight:]
            .view(torch.float8_e4m3fn)
            .view(shapes["down_proj"]),
            down_buffer[:down_weight].view(torch.float32).view(shapes["down_scale"]),
        )
        for view, param in zip(views, params, strict=True):
            view.copy_(param.data)
        for view, param in zip(views, params, strict=True):
            if isinstance(param, torch.nn.Parameter):
                param.data = view
        forms[f"layers.{layer}.ffn.dense_up"] = up_buffer
        forms[f"layers.{layer}.ffn.dense_down"] = down_buffer
    torch.cuda.empty_cache()


class SegmentWeights:
    """One rank's segment-1 weight tensors by name, and the geometry they were built
    for.
    """

    def __init__(
        self, geometry: Glm53Geometry, rank: int, tensors: dict[str, torch.Tensor]
    ) -> None:
        self.geometry: Glm53Geometry = geometry
        self.rank: int = rank
        self._tensors: dict[str, torch.Tensor] = dict(tensors)

    def names(self) -> tuple[str, ...]:
        """Return every form's name, sorted."""
        return tuple(sorted(self._tensors))

    def get(self, name: str) -> torch.Tensor:
        """Return form ``name``."""
        try:
            return self._tensors[name]
        except KeyError:
            raise KeyError(
                f"no GLM-5.3 segment weight {name!r} on rank {self.rank}"
            ) from None

    def tensors(self) -> dict[str, torch.Tensor]:
        """Return a copy of the name-to-tensor mapping."""
        return dict(self._tensors)


def build_segment_weights(
    model: torch.nn.Module, stash: CheckpointStash
) -> SegmentWeights:
    """Build this rank's segment-1 weight forms and empty the stash."""
    rank, geometry = stash.rank, stash.geometry
    kernel = (LAYOUT.hidden, LAYOUT.routed_experts, LAYOUT.shard, LAYOUT.block)
    if (
        geometry.hidden,
        geometry.experts,
        geometry.expert_shard,
        geometry.block,
    ) != kernel:
        raise ValueError(
            "GLM-5.3 decode kernels are built for hidden, experts, shard, block "
            f"{kernel}"
        )
    start = time.perf_counter()
    forms: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for layer in range(geometry.layers):
            forms.update(attention_forms(model, stash, layer, rank))
            forms.update(ffn_forms(model, geometry, layer))
        left = stash.remaining()
        if left:
            raise RuntimeError(f"checkpoint slices not used by any form: {left[:8]}")
        torch.cuda.synchronize(stash.device)
        packed = time.perf_counter()
        adopt_worker_linears(model, geometry, forms, rank)
        adopt_selector_linears(model, geometry, forms, rank)
        adopt_dense(model, geometry, forms)
        torch.cuda.synchronize(stash.device)
        linears = time.perf_counter()
        experts = share_experts(model, range(geometry.dense_layers, geometry.layers))
        for layer, buffers in experts.items():
            forms.update(
                {
                    f"layers.{layer}.ffn.{name}": tensor
                    for name, tensor in buffers.kernel_inputs().items()
                }
            )
        torch.cuda.synchronize(stash.device)
    done = time.perf_counter()
    timing: dict[str, object] = {
        "stash_seconds": round(stash.seconds, 2),
        "stash_gib": round(stash.bytes / 2**30, 3),
        "pack_seconds": round(packed - start, 2),
        "linears_seconds": round(linears - packed, 2),
        "experts_seconds": round(done - linears, 2),
    }
    logger.info(
        "GLM-5.3 segment weights, rank %d: %d tensors, %s", rank, len(forms), timing
    )
    return SegmentWeights(geometry, rank, forms)
