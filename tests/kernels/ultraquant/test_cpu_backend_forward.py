# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whole-backend tests for CPU UltraQuant attention.

The kernel tests next to this one hold the store and decode ops to the
reference in isolation. What is left to cover is the backend around them: the
builder's split of a batch into the impl's three routes, the staging page it
sizes for the whole KV cache group, and the placement of each route's result
in the shared output tensor. So these drive a real `Attention` layer through
`do_kv_cache_update` and `forward` the way the model runner does, over batches
that mix prefills, decodes and continuation chunks.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import NamedTuple

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm import _cpu_ops as ops
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import set_default_torch_dtype, set_random_seed
from vllm.v1.attention.backends.cpu_ultraquant_attn import (
    _CONTINUATION_DECODE_RATIO,
    CPUUltraQuantAttentionImpl,
    CPUUltraQuantAttentionMetadata,
    CPUUltraQuantAttentionMetadataBuilder,
)
from vllm.v1.attention.ops.ultraquant.format import DEFAULT_CONSTANT_C, slot_size
from vllm.v1.attention.ops.ultraquant.reference import (
    hadamard_matrix,
    ultraquant_encode_decode,
)
from vllm.v1.kv_cache_interface import AttentionSpec, get_kv_quant_mode

pytestmark = pytest.mark.skipif(
    not torch.cpu._is_avx512_supported(), reason="requires AVX-512"
)

BLOCK_SIZE = 32
# The quantization cancels against the reference below, so a row that reads
# nothing back out of the cache is left with only bf16 staging and the
# softmax's vectorised exp. A row that does read the cache back also carries
# the store's bf16 K rotation against the reference's fp32 one, which is the
# term test_decode_in_backend_precision measures.
_TOL_EXACT = 5e-3
_TOL_QUANTIZED = 8e-2


class _Route(NamedTuple):
    """How the impl is expected to serve one row of the batch.

    `num_quantized` is how many of the row's context tokens it reads back out
    of the 4-bit cache rather than from the K/V it was handed, and `fp8_query`
    whether it goes through the decode kernel, which rounds Q to FP8 E4M3.
    """

    kind: str
    num_quantized: int
    fp8_query: bool

    @property
    def tol(self) -> float:
        return _TOL_EXACT if self.num_quantized == 0 else _TOL_QUANTIZED


def _expected_route(query_len: int, context_len: int, max_query_len: int) -> _Route:
    """The contract the impl's routing is held to, restated independently.

    A first prefill chunk has nothing cached to read, so it attends its raw
    K/V. A continuation reads its prefix back, and reads its own chunk back
    too unless the chunk is long enough to earn the prefix's dequantization.
    """
    if max_query_len == 1:
        # A pure-decode batch skips the split, so even a one-token prompt in it
        # is served from the cache its store just wrote.
        return _Route("decode", num_quantized=context_len, fp8_query=True)
    if query_len == context_len:
        return _Route("prefill", num_quantized=0, fp8_query=False)
    if query_len == 1:
        return _Route("decode", num_quantized=context_len, fp8_query=True)
    if query_len * _CONTINUATION_DECODE_RATIO <= context_len:
        return _Route("cont_decode", num_quantized=context_len, fp8_query=True)
    return _Route(
        "cont_dequant", num_quantized=context_len - query_len, fp8_query=False
    )


def _reference_row(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float,
    route: _Route,
    sinks: torch.Tensor | None,
) -> torch.Tensor:
    """One request's causal attention, in the Hadamard basis, in fp32.

    Scores are taken in the rotated basis because that is the only place the
    FP8 query rounding and the cache's rotated K meet; the rotation being
    orthonormal, it leaves the rows that skip both unchanged.
    """
    ctx, num_kv_heads, head_dim = key.shape
    q_len, num_heads, _ = query.shape
    group = num_heads // num_kv_heads
    quantized = route.num_quantized
    rot = hadamard_matrix(head_dim, query.device, torch.float32)

    q = query.float() @ rot
    if route.fp8_query:
        q = q.to(torch.float8_e4m3fn).float()

    k = torch.cat(
        [
            ultraquant_encode_decode(key[:quantized], rotate=True),
            key[quantized:].float() @ rot,
        ]
    ).repeat_interleave(group, dim=1)
    v = torch.cat(
        [
            ultraquant_encode_decode(value[:quantized], rotate=False),
            value[quantized:].float(),
        ]
    ).repeat_interleave(group, dim=1)

    scores = torch.einsum("qhd,chd->qhc", q, k) * scale
    positions = torch.arange(ctx - q_len, ctx)
    causal = torch.arange(ctx)[None, :] > positions[:, None]
    scores.masked_fill_(causal[:, None, :], float("-inf"))

    if sinks is not None:
        sink = sinks.float().view(1, num_heads, 1).expand(q_len, num_heads, 1)
        probs = torch.cat([scores, sink], dim=-1).softmax(dim=-1)[..., :ctx]
    else:
        probs = scores.softmax(dim=-1)
    return torch.einsum("qhc,chd->qhd", probs, v)


def _check_staging_page(
    attn_metadata: CPUUltraQuantAttentionMetadata,
    routes: list[_Route],
    seq_lens: list[int],
    query_lens: list[int],
) -> None:
    """The page is sized once for the whole group, so it has to cover whichever
    of the two staging routes wants the most blocks.

    Asserted directly rather than left to the output comparison, because a page
    one block short is an out-of-bounds write whose damage is not reliably
    visible in the result.
    """
    prefill = sum(
        cdiv(q, BLOCK_SIZE) for q, r in zip(query_lens, routes) if r.kind == "prefill"
    )
    cont = max(
        (
            cdiv(c, BLOCK_SIZE)
            for c, r in zip(seq_lens, routes)
            if r.kind == "cont_dequant"
        ),
        default=0,
    )
    needed = max(prefill, cont)
    if needed:
        assert attn_metadata.scratch is not None
        assert attn_metadata.scratch.size(0) >= needed, (
            f"staging page holds {attn_metadata.scratch.size(0)} blocks, needs {needed}"
        )


def _rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a, b = actual.float().reshape(-1), expected.float().reshape(-1)
    return ((a - b).norm() / b.norm()).item()


@contextmanager
def _ultraquant_layers(
    layer_names: list[str],
    *,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    sinks: torch.Tensor | None = None,
) -> Iterator[tuple[list[Attention], CPUUltraQuantAttentionMetadataBuilder]]:
    """Real `Attention` layers on the UltraQuant backend, plus their builder.

    Real layers rather than stand-ins because the builder writes the chosen ISA
    onto them and reads the group's sliding window back off them.
    """
    vllm_config = create_vllm_config(
        dtype=torch.bfloat16, block_size=BLOCK_SIZE, max_model_len=4096
    )
    vllm_config.cache_config.cache_dtype = "ultraquant_4bit"
    vllm_config.cache_config.kv_cache_layout = "LBHNC"

    # The backend is chosen from the process default dtype, which in a real run
    # is set around model loading.
    with set_current_vllm_config(vllm_config), set_default_torch_dtype(torch.bfloat16):
        extra = {"sinks": sinks} if sinks is not None else {}
        layers = [
            Attention(
                num_heads,
                head_size,
                head_size**-0.5,
                num_kv_heads=num_kv_heads,
                cache_config=vllm_config.cache_config,
                prefix=name,
                **extra,
            )
            for name in layer_names
        ]
        backend = layers[0].get_attn_backend()
        assert backend.get_name() == "CPU_ULTRAQUANT"

        spec = backend.customize_spec(
            AttentionSpec(
                block_size=BLOCK_SIZE,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=torch.bfloat16,
                kv_quant_mode=get_kv_quant_mode("ultraquant_4bit"),
            )
        )
        builder = backend.get_builder_cls()(
            spec, layer_names, vllm_config, torch.device("cpu")
        )
        yield layers, builder


def _run_batch(
    batch_spec: BatchSpec,
    *,
    num_heads: int = 8,
    num_kv_heads: int = 2,
    head_size: int = 128,
    num_layers: int = 1,
    with_sinks: bool = False,
) -> None:
    """Drive `batch_spec` through the backend and hold every row to the
    reference, on every layer of the group."""
    seq_lens, query_lens = batch_spec.seq_lens, batch_spec.query_lens
    scale = head_size**-0.5
    sinks = (
        torch.randn(num_heads, dtype=torch.float32).clamp(-2, 2) if with_sinks else None
    )
    names = [f"model.layers.{i}.self_attn.attn" for i in range(num_layers)]

    with _ultraquant_layers(
        names,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        sinks=sinks,
    ) as (layers, builder):
        common = create_common_attn_metadata(
            batch_spec,
            BLOCK_SIZE,
            torch.device("cpu"),
            arange_block_indices=True,
        )
        # arange_block_indices gives each request a contiguous run of blocks but
        # a slot mapping that ignores the context already in them, so the new
        # tokens' slots are recomputed here.
        blocks_per_req = common.block_table_tensor.size(1)
        prefix_slots, new_slots = [], []
        for req, (ctx, q_len) in enumerate(zip(seq_lens, query_lens)):
            base = req * blocks_per_req * BLOCK_SIZE
            prefix_slots += [base + p for p in range(ctx - q_len)]
            new_slots += [base + p for p in range(ctx - q_len, ctx)]
        common.slot_mapping = torch.tensor(new_slots, dtype=torch.int64)

        attn_metadata = builder.build(0, common)
        routes = [
            _expected_route(q, c, attn_metadata.max_query_len)
            for q, c in zip(query_lens, seq_lens)
        ]
        if attn_metadata.max_query_len > 1:
            for kind, rows in (
                ("prefill", attn_metadata.prefill_rows),
                ("decode", attn_metadata.decode_rows),
            ):
                assert rows == [i for i, r in enumerate(routes) if r.kind == kind]
            assert attn_metadata.cont_rows == [
                i for i, r in enumerate(routes) if r.kind.startswith("cont")
            ]
            _check_staging_page(attn_metadata, routes, seq_lens, query_lens)

        kv_cache = torch.zeros(
            batch_spec.batch_size * blocks_per_req,
            num_kv_heads,
            BLOCK_SIZE,
            slot_size(head_size),
            dtype=torch.uint8,
        )
        key = [
            torch.randn(n, num_kv_heads, head_size, dtype=torch.bfloat16)
            for n in seq_lens
        ]
        value = [torch.randn_like(k) for k in key]

        # Everything before this step's chunk was stored by an earlier step.
        if prefix_slots:
            ops.ultraquant_store(
                torch.cat([k[: c - q] for k, c, q in zip(key, seq_lens, query_lens)]),
                torch.cat([v[: c - q] for v, c, q in zip(value, seq_lens, query_lens)]),
                kv_cache,
                torch.tensor(prefix_slots, dtype=torch.int64),
                hadamard_matrix(head_size, kv_cache.device, torch.float32),
                DEFAULT_CONSTANT_C,
            )

        key_new = torch.cat([k[c - q :] for k, c, q in zip(key, seq_lens, query_lens)])
        value_new = torch.cat(
            [v[c - q :] for v, c, q in zip(value, seq_lens, query_lens)]
        )
        query = torch.randn(
            batch_spec.compute_num_tokens(),
            num_heads,
            head_size,
            dtype=torch.bfloat16,
        )

        for layer in layers:
            impl = layer.impl
            assert isinstance(impl, CPUUltraQuantAttentionImpl)
            output = torch.zeros_like(query)
            # The runner stores this step's K/V before attending, so a
            # continuation can read its own chunk back out of the cache.
            impl.do_kv_cache_update(
                layer, key_new, value_new, kv_cache, common.slot_mapping
            )
            impl.forward(
                layer, query, key_new, value_new, kv_cache, attn_metadata, output
            )

            starts = common.query_start_loc_cpu.tolist()
            for req, route in enumerate(routes):
                lo, hi = starts[req], starts[req + 1]
                expected = _reference_row(
                    query[lo:hi],
                    key[req],
                    value[req],
                    scale=scale,
                    route=route,
                    sinks=sinks,
                )
                err = _rel_err(output[lo:hi], expected)
                assert err < route.tol, (
                    f"{layer.layer_name} req {req} ({route.kind}, "
                    f"q={query_lens[req]} ctx={seq_lens[req]}): rel err {err:.4f}"
                )


@pytest.mark.parametrize(
    "batch_spec",
    [
        BatchSpec(seq_lens=[64], query_lens=[64], name="single_prefill"),
        BatchSpec(seq_lens=[40, 128, 97], query_lens=[40, 128, 97], name="prefills"),
        BatchSpec(seq_lens=[33, 256, 1], query_lens=[1, 1, 1], name="decodes"),
        BatchSpec(seq_lens=[96], query_lens=[32], name="cont_dequant"),
        BatchSpec(seq_lens=[512], query_lens=[8], name="cont_decode"),
        BatchSpec(seq_lens=[64, 128, 96, 512], query_lens=[64, 1, 32, 8], name="mixed"),
        BatchSpec(
            seq_lens=[200, 97, 1, 320], query_lens=[1, 97, 1, 40], name="mixed_ragged"
        ),
        # Several decodes behind one prefill: the decodes are gathered out of
        # the batch and scattered back, so their rows have to stay paired with
        # their tokens.
        BatchSpec(
            seq_lens=[64, 100, 33, 257], query_lens=[64, 1, 1, 1], name="many_decodes"
        ),
        # Prefills that do not sit next to each other, which the impl stages one
        # request at a time rather than as one dense range.
        BatchSpec(
            seq_lens=[64, 100, 96, 33],
            query_lens=[64, 1, 96, 1],
            name="scattered_prefills",
        ),
    ],
    ids=lambda s: s.name,
)
@torch.inference_mode()
def test_forward_matches_reference(batch_spec: BatchSpec) -> None:
    set_random_seed(0xBE11)
    _run_batch(batch_spec)


@pytest.mark.parametrize("head_size", [32, 64, 256])
@torch.inference_mode()
def test_forward_across_head_sizes(head_size: int) -> None:
    set_random_seed(0xF00D)
    _run_batch(
        BatchSpec(seq_lens=[64, 128, 96], query_lens=[64, 1, 32]),
        head_size=head_size,
    )


@torch.inference_mode()
def test_forward_shares_staging_page_across_layers() -> None:
    """The staging page belongs to the group, not the layer, so a second layer
    must not see what the first left in the tail of a partly written block."""
    set_random_seed(0x5CA7)
    _run_batch(
        BatchSpec(seq_lens=[97, 128, 96], query_lens=[97, 1, 32]),
        num_layers=3,
    )


@torch.inference_mode()
def test_forward_applies_sinks() -> None:
    set_random_seed(0x51)
    _run_batch(
        BatchSpec(seq_lens=[64, 128, 96, 512], query_lens=[64, 1, 32, 8]),
        with_sinks=True,
    )
