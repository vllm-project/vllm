# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 decode segment 1's weight forms, built by vLLM when it loads the model."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, fields
from typing import ClassVar

import regex as re
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


def head_groups(
    geometry: Glm53Geometry, layer: int, rank: int, narrow: bool = True
) -> tuple[int, ...]:
    """Return the first heads of the head groups ``rank`` needs (a push rank also a
    selector rank's; ``narrow``: a deployment that serves a build with push roles).
    """
    heads = geometry.heads
    if narrow and geometry.is_push_rank(layer, rank):
        return (heads * rank, heads * (rank - geometry.selector_ranks))
    return (heads * rank,)


Slicer = Callable[[torch.Tensor], torch.Tensor]


class CheckpointStash:
    """This rank's slices of checkpoint tensors vLLM does not keep as-is."""

    def __init__(
        self,
        geometry: Glm53Geometry,
        rank: int,
        device: torch.device,
        narrow: bool = True,
        wide: bool = False,
    ) -> None:
        self.geometry: Glm53Geometry = geometry
        self.rank: int = rank
        self.device: torch.device = device
        self.narrow: bool = narrow
        self.wide: bool = wide
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
            groups = head_groups(g, layer, self.rank, self.narrow)
            stride = {
                "q_b_proj.weight": g.qk_head,
                "kv_b_proj.weight": g.kv_b_stride,
            }.get(name)
            if stride is None:
                return {}
            return {
                f"{leaf}@{head}": (
                    lambda tensor, head=head: tensor[  # type: ignore[misc]
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


def form_sets(
    geometry: Glm53Geometry, layer: int, rank: int, narrow: bool, wide: bool
) -> tuple[bool, bool]:
    """Return which attention forms of full layer ``layer`` ``rank`` holds: ``(plain,
    selector)``.
    """
    if geometry.is_shared(layer):
        return True, False
    selects = rank < geometry.selector_ranks
    return narrow and not selects, selects or wide


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
        for head in head_groups(g, layer, rank, stash.narrow)
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
    plain, selects = form_sets(g, layer, rank, stash.narrow, stash.wide)
    forms = {prefix + name: tensor for name, tensor in worker.items() if plain}
    if not selects:
        return forms
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
    probe = {**worker, "absorb_weight": absorbed[: absorbed.shape[0] // len(groups)]}
    return {
        **forms,
        **{prefix + name: tensor for name, tensor in selector.items()},
        **{prefix + "probe." + name: tensor for name, tensor in probe.items()},
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
class ExpertInputs:
    """One MoE layer's expert inputs of the decode MoE kernels (``fused_moe.LAYOUT``),
    in the kernels' parameter order: vLLM's routed and shared expert weights and its
    FP32 routed scales, read where vLLM allocated them, then the header (the shared
    expert's FP32 scales), the only tensor of our own.
    """

    routed_up: torch.Tensor
    routed_down: torch.Tensor
    shared_up: torch.Tensor
    shared_down: torch.Tensor
    routed_up_scales: torch.Tensor
    routed_down_scales: torch.Tensor
    shared_scales: torch.Tensor

    def __post_init__(self) -> None:
        routed, fp8 = (LAYOUT.routed_experts,), torch.float8_e4m3fn
        forms = {
            "routed_up": (routed + LAYOUT.w13_block_shape, fp8),
            "routed_down": (routed + LAYOUT.w2_block_shape, fp8),
            "shared_up": (LAYOUT.w13_shape, fp8),
            "shared_down": (LAYOUT.w2_shape, fp8),
            "routed_up_scales": (routed + LAYOUT.w13_scale_shape, torch.float32),
            "routed_down_scales": (routed + LAYOUT.w2_scale_shape, torch.float32),
            "shared_scales": ((LAYOUT.header_words,), torch.float32),
        }
        for name, (shape, dtype) in forms.items():
            tensor = getattr(self, name)
            if (
                tuple(tensor.shape) != shape
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
                or tensor.data_ptr() % 16
            ):
                raise ValueError(
                    f"MoE input {name} must be a contiguous 16-byte aligned {dtype} "
                    f"{shape}, got {tensor.dtype} {tuple(tensor.shape)}"
                )

    def kernel_inputs(self) -> dict[str, torch.Tensor]:
        """Return the inputs by name, in the kernels' parameter order."""
        return {field.name: getattr(self, field.name) for field in fields(self)}


def expert_header(w13_scale: torch.Tensor, w2_scale: torch.Tensor) -> torch.Tensor:
    """Return a new header: the shared expert's FP32 ``w13``, then ``w2``, scales."""
    if (
        tuple(w13_scale.shape) != LAYOUT.w13_scale_shape
        or tuple(w2_scale.shape) != LAYOUT.w2_scale_shape
    ):
        raise ValueError(
            f"shared expert scales {tuple(w13_scale.shape)}, {tuple(w2_scale.shape)}; "
            f"expected {LAYOUT.w13_scale_shape}, {LAYOUT.w2_scale_shape}"
        )
    return torch.cat((w13_scale.flatten(), w2_scale.flatten())).float()


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


def layer_experts(mlp: torch.nn.Module, layer: int) -> ExpertInputs:
    """Return one MoE layer's expert inputs: vLLM's tensors in place, a new header."""
    experts = _routed_experts(mlp, layer)
    shared = mlp.shared_experts
    linears = {"gate_up_proj": shared.gate_up_proj, "down_proj": shared.down_proj}
    if not all(is_deepgemm_form(linear) for linear in linears.values()):
        raise ValueError(
            f"layer {layer}: vLLM's shared expert is not in the DeepGEMM form (its "
            "default FP8 linear)"
        )
    scales = [
        ue8m0_block_scales(
            f"layer {layer} shared {name}",
            linear.weight,
            linear.weight_scale_inv,
            LAYOUT.block,
        )
        for name, linear in linears.items()
    ]
    try:
        return ExpertInputs(
            experts.w13_weight.data,
            experts.w2_weight.data,
            shared.gate_up_proj.weight.data,
            shared.down_proj.weight.data,
            experts.w13_weight_scale_inv.data,
            experts.w2_weight_scale_inv.data,
            expert_header(*scales),
        )
    except ValueError as error:
        raise ValueError(
            f"layer {layer}: {error}; start vLLM with --moe-backend flashinfer_trtllm"
        ) from None


def is_deepgemm_form(module: torch.nn.Module) -> bool:
    """Return whether vLLM holds ``module``'s FP8 linear in DeepGEMM form."""
    scale = getattr(module, "weight_scale_inv", None)
    return scale is not None and scale.dtype == torch.int32


def ue8m0_block_scales(
    name: str, weight: torch.Tensor, scale: torch.Tensor, block: int
) -> torch.Tensor:
    """Return FP32 ``block`` x ``block`` scales decoded from vLLM's packed
    UE8M0 scales.
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
    """A linear the kernels read in place: vLLM's E4M3 weight stays where vLLM put it,
    and its block scales are decoded to FP32 into a small buffer of their own.
    """

    ALIGNMENT: ClassVar[int] = 128

    key: str
    operand: str
    rows: int
    columns: int
    block: int

    @property
    def scales_operand(self) -> str:
        return self.operand.removesuffix("_weight") + "_scales"

    @property
    def scale_shape(self) -> tuple[int, int]:
        return (-(-self.rows // self.block), -(-self.columns // self.block))

    @property
    def scale_bytes(self) -> int:
        raw = torch.float32.itemsize * math.prod(self.scale_shape)
        return -(-raw // self.ALIGNMENT) * self.ALIGNMENT

    def adopt(
        self, module: torch.nn.Module, layer: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return vLLM's weight, left in place, and a new buffer of its FP32 scales."""
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
        if not weight.data.is_contiguous() or weight.data_ptr() % 16:
            raise ValueError(
                f"layer {layer}: vLLM's {self.key} weight is not contiguous and "
                "16-byte aligned"
            )
        decoded = ue8m0_block_scales(
            f"layer {layer} {self.key}", weight, scale, self.block
        )
        return weight.data, scale_buffer(decoded, self.scale_bytes)


def scale_buffer(decoded: torch.Tensor, size: int) -> torch.Tensor:
    """Return ``decoded`` FP32 scales in a new zero-padded byte buffer of ``size``
    bytes.
    """
    flat = decoded.contiguous().view(torch.uint8).reshape(-1)
    buffer = torch.zeros(size, dtype=torch.uint8, device=decoded.device)
    buffer[: flat.numel()].copy_(flat)
    return buffer


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
    wide: bool = False,
) -> None:
    """Bind each full layer's vLLM ``indexer.wq_b`` in place as the selector's
    ``wqb_weight`` (and its ``wqb_scales``) on selector ranks (every rank when
    ``wide``).
    """
    if rank >= geometry.selector_ranks and not wide:
        return
    linear = selector_linear(geometry)
    for layer in range(geometry.layers):
        if not geometry.is_shared(layer):
            wq_b = model.layers[layer].self_attn.indexer.wq_b
            prefix = f"layers.{layer}.attention."
            weight, scales = linear.adopt(wq_b, layer)
            forms[prefix + linear.operand] = weight
            forms[prefix + linear.scales_operand] = scales


def adopt_worker_linears(
    model: torch.nn.Module,
    geometry: Glm53Geometry,
    forms: dict[str, torch.Tensor],
    rank: int,
    narrow: bool = True,
    wide: bool = False,
) -> None:
    """Bind each target layer's vLLM worker linears in place (:func:`form_sets` says
    under which names: plain, ``probe.``, or both).
    """
    for linear in worker_linears(geometry).values():
        for layer in range(geometry.layers):
            weight, scales = linear.adopt(
                getattr(model.layers[layer].self_attn, linear.key), layer
            )
            plain, selects = form_sets(geometry, layer, rank, narrow, wide)
            for probe, wanted in (("", plain), ("probe.", selects)):
                if wanted:
                    prefix = f"layers.{layer}.attention.{probe}"
                    forms[prefix + linear.operand] = weight
                    forms[prefix + linear.scales_operand] = scales


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
    """Bind each dense layer's vLLM FFN weights in place as ``dense_up`` (gate, up rows)
    and ``dense_down``, with their decoded FP32 scales as ``dense_up_scales`` and
    ``dense_down_scales``.
    """
    shapes = dense_shapes(geometry)
    for layer in range(geometry.dense_layers):
        mlp = model.layers[layer].mlp
        if not is_deepgemm_form(mlp.gate_up_proj):
            raise ValueError(
                f"layer {layer}: vLLM's dense mlp is not in the DeepGEMM form (its "
                "default FP8 linear)"
            )
        prefix = f"layers.{layer}.ffn."
        for name, form, scales in (
            ("gate_up_proj", "dense_up", "gate_up_scale"),
            ("down_proj", "dense_down", "down_scale"),
        ):
            module = getattr(mlp, name)
            weight, scale = module.weight, getattr(module, "weight_scale_inv", None)
            if tuple(weight.shape) != shapes[name]:
                raise ValueError(
                    f"layer {layer}: vLLM's mlp.{name} weight is {tuple(weight.shape)}"
                )
            if not weight.data.is_contiguous() or weight.data_ptr() % 16:
                raise ValueError(
                    f"layer {layer}: vLLM's mlp.{name} weight is not contiguous and "
                    "16-byte aligned"
                )
            decoded = ue8m0_block_scales(
                f"layer {layer} mlp.{name}", weight, scale, geometry.block
            )
            if tuple(decoded.shape) != shapes[scales]:
                raise ValueError(
                    f"layer {layer}: vLLM's mlp.{name} has {tuple(decoded.shape)} "
                    "scale blocks"
                )
            forms[prefix + form] = weight.data
            # One spare row: the last Down CTA copies two scale rows from its first.
            rows, columns = decoded.shape
            forms[prefix + form + "_scales"] = scale_buffer(
                decoded, (rows + 1) * columns * decoded.element_size()
            )


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
        torch.accelerator.synchronize(stash.device)
        packed = time.perf_counter()
        adopt_worker_linears(model, geometry, forms, rank, stash.narrow, stash.wide)
        adopt_selector_linears(model, geometry, forms, rank, stash.wide)
        adopt_dense(model, geometry, forms)
        torch.accelerator.synchronize(stash.device)
        linears = time.perf_counter()
        for layer in range(geometry.dense_layers, geometry.layers):
            experts = layer_experts(model.layers[layer].mlp, layer)
            forms.update(
                {
                    f"layers.{layer}.ffn.{name}": tensor
                    for name, tensor in experts.kernel_inputs().items()
                }
            )
        torch.accelerator.synchronize(stash.device)
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
