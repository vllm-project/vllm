# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA layers on FlashInfer's kernels: the fused decode (conv +
SiLU, the recurrence and the gated RMSNorm in one launch) and the CuTe DSL
chunked prefill.

Each test runs the same layer, weights, caches and step once on the Triton
path FlashInfer replaces and once on FlashInfer: conv states must match
exactly, recurrent states and outputs within kernel rounding, and null or
unused slots must stay untouched.
"""

from math import prod
from typing import Any

import pytest
import torch
from transformers import Glm5NextTextConfig

from tests.v1.attention.utils import create_vllm_config
from vllm.config import SpeculativeConfig, VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.mamba.mamba_utils import get_conv_state_layout
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.models.glm5next.common.kda import (
    Glm5NextLinearAttention,
    _is_flashinfer_kda_prefill_supported,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

if not current_platform.is_cuda():
    pytest.skip("FlashInfer KDA kernels are CUDA-only", allow_module_level=True)

pytestmark = pytest.mark.usefixtures("dist_init")

HIDDEN = 256
D, W = 128, 4
NUM_SLOTS = 24
PREFIX = "model.layers.0.self_attn"


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory) -> str:
    """A one-layer GLM-5.3-Flash config; the layer needs no checkpoint."""
    path = tmp_path_factory.mktemp("glm5next")
    config = Glm5NextTextConfig(
        hidden_size=HIDDEN, num_hidden_layers=1, layer_types=["linear_attention"]
    )
    config.architectures = ["Glm5NextForCausalLM"]
    config.save_pretrained(path)
    return str(path)


@pytest.fixture
def conv_state_layout(request, monkeypatch):
    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", request.param)
    get_conv_state_layout.cache_clear()
    yield request.param
    get_conv_state_layout.cache_clear()


def _make_layer(
    model_dir: str,
    num_heads: int,
    decode: str = "triton",
    prefill: str = "triton",
    num_spec: int = 0,
    state_dtype: torch.dtype = torch.float32,
) -> tuple[Glm5NextLinearAttention, VllmConfig]:
    """A layer with seeded weights and caches, post-processed as after a load."""
    vllm_config = create_vllm_config(
        model_name=model_dir,
        dtype=torch.bfloat16,
        hf_config_override={"linear_num_heads": num_heads},
    )
    vllm_config.cache_config.mamba_ssm_cache_dtype = (
        "float32" if state_dtype == torch.float32 else "bfloat16"
    )
    vllm_config.additional_config = {
        "kda_decode_backend": decode,
        "kda_spec_decode_backend": decode,
        "kda_prefill_backend": prefill,
    }
    if num_spec:
        vllm_config.speculative_config = SpeculativeConfig(
            method="ngram", num_speculative_tokens=num_spec
        )
    with (
        set_current_vllm_config(vllm_config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cuda"),
    ):
        layer = Glm5NextLinearAttention(
            vllm_config.model_config.hf_text_config, vllm_config, PREFIX
        )

    g = torch.Generator(device="cuda").manual_seed(0)
    for param in layer.parameters():
        weight = torch.randn(param.shape, device="cuda", generator=g)
        default_weight_loader(param, weight / param.shape[-1] ** 0.5)
    heads = torch.linspace(1, 16, layer.local_num_heads, device="cuda")
    default_weight_loader(layer.A_log, heads.log().view_as(layer.A_log))
    layer.o_norm.weight.data.add_(1)
    layer.process_weights_after_loading()

    shapes, dtypes = layer.get_state_shape(), layer.get_state_dtype()
    page_bytes = sum(prod(s) * d.itemsize for s, d in zip(shapes, dtypes))
    layer.bind_kv_cache(
        torch.zeros(NUM_SLOTS, 1, 1, page_bytes, dtype=torch.int8, device="cuda")
    )
    for cache, scale in zip(layer.kv_cache, (1.0, 0.2)):
        cache.copy_(scale * torch.randn(cache.shape, device="cuda", generator=g))
    return layer, vllm_config


def _require_flashinfer_decode(layer: Glm5NextLinearAttention) -> None:
    backend = (
        layer.kda_spec_decode_backend if layer.num_spec else layer.kda_decode_backend
    )
    if backend != "flashinfer":
        pytest.skip("FlashInfer fused KDA decode does not support this config")


def _require_flashinfer_prefill(state_dtype: torch.dtype) -> None:
    if not _is_flashinfer_kda_prefill_supported(D, torch.bfloat16, state_dtype, -5.0):
        pytest.skip("FlashInfer KDA prefill does not support this platform")


def _hidden(num_tokens: int, seed: int = 1) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(num_tokens, HIDDEN, device="cuda", generator=g).bfloat16()


def _forward(
    layer: Glm5NextLinearAttention,
    vllm_config: VllmConfig,
    meta: GDNAttentionMetadata,
    hidden: torch.Tensor,
) -> torch.Tensor:
    positions = torch.arange(hidden.size(0), device="cuda")
    with (
        set_current_vllm_config(vllm_config),
        set_forward_context({PREFIX: meta}, vllm_config),
    ):
        return layer(hidden, positions)


def _non_spec_metadata(
    query_lens: list[int],
    num_decodes: int,
    slots: list[int],
    has_state: list[bool],
    block_table_width: int = 1,
) -> GDNAttentionMetadata:
    """Non-spec metadata as the builder makes it: decodes first, then the
    prefill tail rebased for the peeled path. The slot indices are column 0
    of a [N, block_table_width] block table (1 + num_spec columns with
    speculative decoding, so a strided view)."""
    qsl_cpu = torch.tensor([0, *torch.tensor(query_lens).cumsum(0).tolist()]).int()
    qsl = qsl_cpu.cuda()
    block_table = torch.zeros(
        len(slots), block_table_width, dtype=torch.int32, device="cuda"
    )
    block_table[:, 0] = torch.tensor(slots, dtype=torch.int32)
    indices = block_table[:, 0]
    has_initial_state = torch.tensor(has_state, device="cuda")
    num_tokens = int(qsl_cpu[-1])
    prefill: dict[str, Any] = {}
    if num_decodes < len(query_lens):
        nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
            qsl_cpu, device=qsl.device
        )
        prefill = dict(
            has_initial_state=has_initial_state,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
            prefill_query_start_loc=qsl[num_decodes:] - num_decodes,
            prefill_state_indices=indices[num_decodes:],
            prefill_has_initial_state=has_initial_state[num_decodes:],
        )
    return GDNAttentionMetadata(
        num_prefills=len(query_lens) - num_decodes,
        num_prefill_tokens=num_tokens - num_decodes,
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=num_tokens,
        non_spec_query_start_loc=qsl,
        non_spec_state_indices_tensor=indices,
        **prefill,
    )


def _spec_metadata(
    num_spec: int,
    accepted: list[int],
    prefill_lens: list[int],
    slots: list[list[int]],
) -> GDNAttentionMetadata:
    """Draft-verify rows first (1 + num_spec tokens each), then prefills."""
    T = 1 + num_spec
    num_spec_rows = len(accepted)
    lens = [T] * num_spec_rows + prefill_lens
    is_spec = torch.tensor([1] * num_spec_rows + [0] * len(prefill_lens)).bool()
    token_is_spec = torch.repeat_interleave(is_spec, torch.tensor(lens))
    prefill: dict[str, Any] = {}
    if prefill_lens:
        qsl_cpu = torch.tensor(
            [0, *torch.tensor(prefill_lens).cumsum(0).tolist()]
        ).int()
        qsl = qsl_cpu.cuda()
        indices = torch.tensor(
            [s[0] for s in slots[num_spec_rows:]], dtype=torch.int32, device="cuda"
        )
        has_initial_state = torch.tensor([False, True], device="cuda")[
            : len(prefill_lens)
        ]
        nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
            qsl_cpu, device=qsl.device
        )
        prefill = dict(
            non_spec_query_start_loc=qsl,
            non_spec_state_indices_tensor=indices,
            has_initial_state=has_initial_state,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
            prefill_query_start_loc=qsl,
            prefill_state_indices=indices,
            prefill_has_initial_state=has_initial_state,
        )
    return GDNAttentionMetadata(
        num_prefills=len(prefill_lens),
        num_prefill_tokens=sum(prefill_lens),
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=num_spec_rows,
        num_spec_decode_tokens=T * num_spec_rows,
        num_actual_tokens=sum(lens),
        spec_query_start_loc=torch.arange(
            0, T * num_spec_rows + 1, T, dtype=torch.int32, device="cuda"
        ),
        spec_state_indices_tensor=torch.tensor(
            slots[:num_spec_rows], dtype=torch.int32, device="cuda"
        ),
        spec_sequence_masks=is_spec.cuda(),
        spec_token_indx=token_is_spec.nonzero().flatten().cuda(),
        non_spec_token_indx=(~token_is_spec).nonzero().flatten().cuda(),
        num_accepted_tokens=torch.tensor(accepted, dtype=torch.int32, device="cuda"),
        **prefill,
    )


@torch.inference_mode()
def test_backend_selection(model_dir):
    """``auto`` runs GLM-5.3's decode shapes (64 heads at TP1, 16 at TP4) on
    FlashInfer on SM10x, keeps Triton for 8-head draft-verify rows, and an
    explicit ``flashinfer`` fails at startup where it is unsupported."""
    if not current_platform.is_device_capability_family(100):
        with pytest.raises(RuntimeError, match="SM10x"):
            _make_layer(model_dir, 64, decode="flashinfer")
        with pytest.raises(RuntimeError, match="SM100/SM103/SM107"):
            _make_layer(model_dir, 64, prefill="flashinfer")
        return
    for num_heads in (16, 64):
        layer, _ = _make_layer(model_dir, num_heads, decode="auto")
        assert layer.kda_decode_backend == "flashinfer"
        layer, _ = _make_layer(model_dir, num_heads, decode="auto", num_spec=2)
        assert layer.kda_spec_decode_backend == "flashinfer"
    layer, _ = _make_layer(model_dir, 8, decode="auto", num_spec=2)
    assert layer.kda_spec_decode_backend == "triton"
    with pytest.raises(RuntimeError, match="SM10x"):
        _make_layer(model_dir, 8, decode="flashinfer", num_spec=2)
    layer, _ = _make_layer(model_dir, 64, prefill="flashinfer")
    assert layer.kda_prefill_backend == "flashinfer"


@pytest.mark.parametrize("conv_state_layout", ["SD", "DS"], indirect=True)
@pytest.mark.parametrize(
    "state_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"]
)
@pytest.mark.parametrize("num_heads", [16, 64])
@torch.inference_mode()
def test_decode_step_matches_triton(
    model_dir, num_heads, state_dtype, conv_state_layout
):
    flashinfer = _make_layer(model_dir, num_heads, "auto", state_dtype=state_dtype)
    _require_flashinfer_decode(flashinfer[0])
    triton = _make_layer(model_dir, num_heads, state_dtype=state_dtype)
    before = [cache.clone() for cache in triton[0].kv_cache]
    # Null rows (slot 0, also CUDA-graph padding) must not touch any cache.
    slots = [3, 0, 7, 11, 0, 19, 1, 22]
    meta = _non_spec_metadata([1] * len(slots), len(slots), slots, [True] * 8)
    hidden = _hidden(len(slots))
    out_t = _forward(*triton, meta, hidden)
    out_f = _forward(*flashinfer, meta, hidden)

    (conv_t, state_t), (conv_f, state_f) = triton[0].kv_cache, flashinfer[0].kv_cache
    torch.testing.assert_close(conv_f, conv_t, atol=0, rtol=0)
    tol = 2e-2 if state_dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(state_f, state_t, atol=tol, rtol=tol)
    live = torch.tensor(slots) > 0
    torch.testing.assert_close(out_f[live], out_t[live], atol=3e-2, rtol=3e-2)
    untouched = [s for s in range(NUM_SLOTS) if s not in slots or s == 0]
    for cache, original in zip(flashinfer[0].kv_cache, before):
        torch.testing.assert_close(
            cache[untouched], original[untouched], atol=0, rtol=0
        )


@pytest.mark.parametrize("prefill", ["triton", "flashkda", "flashinfer"])
@torch.inference_mode()
def test_mixed_step_matches_triton(model_dir, workspace_init, prefill):
    """The decode rows of a mixed step go through the fused decode; the
    prefill rows (fresh and continued) and their states do not change."""
    num_heads = 16
    if prefill == "flashinfer":
        _require_flashinfer_prefill(torch.float32)
    # FlashKDA is the ``auto`` prefill backend where it is available.
    requested = "auto" if prefill == "flashkda" else prefill
    flashinfer = _make_layer(model_dir, num_heads, "auto", requested)
    _require_flashinfer_decode(flashinfer[0])
    if flashinfer[0].kda_prefill_backend != prefill:
        pytest.skip("FlashKDA is not available")
    triton = _make_layer(model_dir, num_heads, "triton", requested)
    decode_slots = [5, 2, 20, 9]
    n = len(decode_slots)
    query_lens = [1] * n + [37, 1, 130]
    slots = decode_slots + [12, 14, 17]
    meta = _non_spec_metadata(query_lens, n, slots, [True] * n + [False, True, True])
    hidden = _hidden(sum(query_lens))
    out_t = _forward(*triton, meta, hidden)
    out_f = _forward(*flashinfer, meta, hidden)

    (conv_t, state_t), (conv_f, state_f) = triton[0].kv_cache, flashinfer[0].kv_cache
    torch.testing.assert_close(conv_f, conv_t, atol=0, rtol=0)
    # Without the fused decode, FlashInfer's prefill also runs the decode rows
    # (as one-token sequences), and it is not batch invariant.
    tol = 1e-2 if prefill == "flashinfer" else 0
    torch.testing.assert_close(out_f[n:], out_t[n:], atol=tol, rtol=tol)
    prefill_slots = slots[n:]
    torch.testing.assert_close(
        state_f[prefill_slots], state_t[prefill_slots], atol=tol, rtol=tol
    )
    # The Triton path runs a mixed step's decode rows through the chunked
    # kernel, so they differ by chunked-vs-recurrent rounding.
    torch.testing.assert_close(out_f[:n], out_t[:n], atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(
        state_f[decode_slots], state_t[decode_slots], atol=2e-2, rtol=2e-2
    )


@pytest.mark.parametrize("with_prefill", [False, True], ids=["verify", "mixed"])
@pytest.mark.parametrize(
    "state_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"]
)
@torch.inference_mode()
def test_spec_verify_rows_match_triton(model_dir, state_dtype, with_prefill):
    num_heads, num_spec = 64, 2
    flashinfer = _make_layer(
        model_dir, num_heads, "auto", "triton", num_spec, state_dtype
    )
    _require_flashinfer_decode(flashinfer[0])
    triton = _make_layer(
        model_dir, num_heads, "triton", "triton", num_spec, state_dtype
    )
    # Three verify rows (the third a null row), then a fresh and a continued
    # prefill.
    slots = [[1, 2, 3], [4, 5, 6], [0, 0, 0], [9, 9, 9], [10, 10, 10]]
    meta = _spec_metadata(num_spec, [1, 3, 1], [21, 6] if with_prefill else [], slots)
    hidden = _hidden(meta.num_actual_tokens)
    out_t = _forward(*triton, meta, hidden)
    out_f = _forward(*flashinfer, meta, hidden)

    (conv_t, state_t), (conv_f, state_f) = triton[0].kv_cache, flashinfer[0].kv_cache
    torch.testing.assert_close(conv_f, conv_t, atol=0, rtol=0)
    tol = 2e-2 if state_dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(state_f, state_t, atol=tol, rtol=tol)
    live = torch.ones(meta.num_actual_tokens, dtype=torch.bool)
    live[6:9] = False
    torch.testing.assert_close(out_f[live], out_t[live], atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize(
    "query_lens", [[40, 1, 17, 3, 96], [1]], ids=["chunked", "one-token"]
)
@pytest.mark.parametrize(
    "state_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"]
)
@torch.inference_mode()
def test_prefill_step_matches_triton(model_dir, state_dtype, query_lens):
    """FlashInfer's prefill updates the state pool in place at the slot
    indices, so it zeroes fresh sequences' slots itself and leaves other slots
    alone. A step whose only prefill row is one token takes the recurrent
    kernel (FlashInfer's prefill needs two tokens)."""
    _require_flashinfer_prefill(state_dtype)
    num_heads = 64
    # Fresh and continued sequences across the 16-token chunk boundary.
    slots = [3, 5, 8, 13, 21][: len(query_lens)]
    has_state = [False, True, True, False, True][: len(query_lens)]
    meta = _non_spec_metadata(query_lens, 0, slots, has_state)
    hidden = _hidden(sum(query_lens))
    flashinfer = _make_layer(
        model_dir, num_heads, prefill="flashinfer", state_dtype=state_dtype
    )
    before = flashinfer[0].kv_cache[1].clone()
    triton = _make_layer(model_dir, num_heads, state_dtype=state_dtype)
    out_t = _forward(*triton, meta, hidden).float()
    out_f = _forward(*flashinfer, meta, hidden).float()

    (conv_t, state_t), (conv_f, state_f) = triton[0].kv_cache, flashinfer[0].kv_cache
    torch.testing.assert_close(conv_f, conv_t, atol=0, rtol=0)
    tol = 3e-2 if state_dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(state_f[slots], state_t[slots], atol=tol, rtol=tol)
    # Different chunked kernels: compare in norm.
    assert (out_f - out_t).norm() < 1e-2 * out_t.norm()
    # Only the sequences' slots (and the null slot 0) may change.
    untouched = [s for s in range(1, NUM_SLOTS) if s not in slots]
    torch.testing.assert_close(state_f[untouched], before[untouched], atol=0, rtol=0)


@torch.inference_mode()
def test_prefill_with_spec_verify_rows(model_dir):
    """With draft-verify rows in the step, FlashInfer's prefill writes the
    prefill rows to their own buffer, scattered by ``non_spec_token_indx``."""
    _require_flashinfer_prefill(torch.float32)
    num_heads, num_spec = 32, 2
    slots = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [11, 11, 11], [14, 14, 14]]
    meta = _spec_metadata(num_spec, [1, 2, 3], [25, 9], slots)
    hidden = _hidden(meta.num_actual_tokens)
    flashinfer = _make_layer(model_dir, num_heads, "triton", "flashinfer", num_spec)
    triton = _make_layer(model_dir, num_heads, "triton", "triton", num_spec)
    out_t = _forward(*triton, meta, hidden).float()
    out_f = _forward(*flashinfer, meta, hidden).float()
    assert (out_f - out_t).norm() < 1e-2 * out_t.norm()
    torch.testing.assert_close(
        flashinfer[0].kv_cache[1][1:], triton[0].kv_cache[1][1:], atol=1e-2, rtol=1e-2
    )


@pytest.mark.parametrize("num_decodes", [0, 3], ids=["prefill", "mixed"])
@torch.inference_mode()
def test_block_table_column_state_indices(model_dir, num_decodes):
    """A step without draft rows in an MTP deployment: the builder's slot
    indices are column 0 of the [N, 1 + num_spec] block table, a strided view.
    It matches the same step with contiguous indices."""
    _require_flashinfer_prefill(torch.float32)
    num_heads, num_spec = 32, 2
    query_lens = [1] * num_decodes + [40, 17, 96]
    slots = [2, 4, 6][:num_decodes] + [9, 12, 15]
    has_state = [True] * num_decodes + [False, True, True]
    hidden = _hidden(sum(query_lens))
    out = {}
    for width in (1, 1 + num_spec):
        meta = _non_spec_metadata(query_lens, num_decodes, slots, has_state, width)
        layer = _make_layer(model_dir, num_heads, "triton", "flashinfer", num_spec)
        out[width] = (_forward(*layer, meta, hidden), layer[0].kv_cache)
    assert meta.non_spec_state_indices_tensor is not None
    assert not meta.non_spec_state_indices_tensor.is_contiguous()
    torch.testing.assert_close(out[3][0], out[1][0], atol=0, rtol=0)
    for strided, contiguous in zip(out[3][1], out[1][1], strict=True):
        torch.testing.assert_close(strided, contiguous, atol=0, rtol=0)


def _refit(layer: Glm5NextLinearAttention, seed: int) -> None:
    """New conv and norm weights through their weight loaders, then the
    post-load hook, as a weight reload runs them."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    shard_shape = (layer.local_projection_size, 1, W)
    for shard_id in range(3):
        weight = 0.5 * torch.randn(shard_shape, device="cuda", generator=g)
        layer.conv1d.weight.weight_loader(layer.conv1d.weight, weight, shard_id)
    norm = 1 + 0.1 * torch.randn(D, device="cuda", generator=g)
    default_weight_loader(layer.o_norm.weight, norm)
    layer.process_weights_after_loading()


@torch.inference_mode()
def test_decode_cuda_graph_replays_new_inputs_and_refit(model_dir):
    """A captured fused decode step replays with new inputs, slots and
    refitted weights: FlashInfer's conv and norm weights are refilled in place,
    at the addresses the graph captured."""
    num_heads, n = 32, 6
    captured = _make_layer(model_dir, num_heads, "auto")
    _require_flashinfer_decode(captured[0])
    eager = _make_layer(model_dir, num_heads, "auto")
    meta = _non_spec_metadata([1] * n, n, [1, 2, 3, 4, 5, 6], [True] * n)
    hidden = _hidden(n, seed=1)
    _forward(*captured, meta, hidden)
    _forward(*eager, meta, hidden)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _forward(*captured, meta, hidden)

    new_hidden = _hidden(n, seed=2)
    hidden.copy_(new_hidden)
    slots = [7, 0, 9, 10, 11, 12]
    assert meta.non_spec_state_indices_tensor is not None
    meta.non_spec_state_indices_tensor.copy_(torch.tensor(slots))
    for layer, _ in (captured, eager):
        _refit(layer, seed=3)
    graph.replay()
    expected = _forward(*eager, meta, new_hidden)
    live = torch.tensor(slots) > 0
    torch.testing.assert_close(out[live], expected[live], atol=0, rtol=0)
    for replayed, reference in zip(captured[0].kv_cache, eager[0].kv_cache):
        torch.testing.assert_close(replayed, reference, atol=0, rtol=0)
