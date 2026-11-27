# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HYV4 Flash ROCm (`amd/`) IndexPool indexer and latent MoE modules against
pure-torch references built from the same weights, on gfx950."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tests.kernels.test_hy_v4_flash_kernels import (
    _ROPE,
    _fwht_quantize_reference,
    _rotate_reference,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype, set_random_seed

DEVICE = "cuda"


def _is_rocm_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


requires_rocm_gfx950 = pytest.mark.skipif(
    not _is_rocm_gfx950(), reason="HYV4 Flash ROCm modules require gfx950"
)


@pytest.fixture(autouse=True)
def _seed():
    set_random_seed(0)


# --------------------------------------------------------------------------
# IndexPool compressed indexer (amd/compressed_indexer.py)
# --------------------------------------------------------------------------

_RATIO, _PAGE, _INDEX_DIM, _ROPE_DIM = 4, 64, 128, 64
_MAX_MODEL_LEN = 512


def _coordinates(logical: torch.Tensor, three_axis: bool) -> torch.Tensor:
    """Text positions, or distinct per-axis MRoPE coordinates of each token."""
    if not three_axis:
        return logical.clone()
    return torch.stack((logical, logical // 2 + 3, 1000 - logical))


def _rms(x, weight, eps):
    x = x.float()
    return x * (x.square().mean(-1, keepdim=True) + eps).rsqrt() * weight.float()


class _IndexerHarness:
    """A `FlashCompressedIndexer` with random weights, bound caches and full
    per-request token streams, stepped through engine-shaped batches."""

    def __init__(self, num_heads, group_topk, three_axis, num_requests=4):
        from vllm.config import CacheConfig
        from vllm.models.hy_v4_flash.amd.compressed_indexer import (
            FlashCompressedIndexer,
        )

        self.budget = group_topk - 1
        self.three_axis = three_axis
        self.config = SimpleNamespace(
            compressed_indexer_compress_ratio=_RATIO,
            compressed_indexer_group_topk=group_topk,
            compressed_indexer_num_attention_heads=num_heads,
            qk_rope_head_dim=_ROPE_DIM,
            rope_parameters=dict(_ROPE),
            rms_norm_eps=1e-5,
            q_lora_rank=48,
            hidden_size=80,
        )
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=_MAX_MODEL_LEN),
            quant_config=None,
            cache_config=CacheConfig(),
        )
        self.buffer = torch.full(
            (256, group_topk * _RATIO), 7, dtype=torch.int32, device=DEVICE
        )
        with set_default_torch_dtype(torch.bfloat16), torch.device(DEVICE):
            self.module = FlashCompressedIndexer(
                vllm_config, self.config, self.buffer, "layer.indexer"
            )
        for name, parameter in self.module.named_parameters():
            if parameter.ndim == 1:
                parameter.data.copy_(1 + 0.1 * torch.randn_like(parameter))
            elif name == "intra_group_ape":
                parameter.data.copy_(0.5 * torch.randn_like(parameter))
            else:
                parameter.data.copy_(
                    torch.randn_like(parameter) * parameter.shape[1] ** -0.5
                )
        pages = _MAX_MODEL_LEN // _RATIO // _PAGE
        self.pool = torch.zeros(
            num_requests * pages + 3, 1, _PAGE, _INDEX_DIM + 4, dtype=torch.uint8
        ).to(DEVICE)
        self.tail = torch.zeros(num_requests + 2, 1, _RATIO, _INDEX_DIM + 3).to(DEVICE)
        self.module.k_cache.bind_kv_cache(self.pool)
        self.module.tail_cache.bind_kv_cache(self.tail)
        self.pool_table = (
            torch.randperm(self.pool.shape[0])[: num_requests * pages]
            .reshape(num_requests, pages)
            .int()
            .to(DEVICE)
        )
        self.tail_table = (
            torch.randperm(self.tail.shape[0])[:num_requests, None].int().to(DEVICE)
        )
        self.hidden = torch.randn(num_requests, _MAX_MODEL_LEN, 80, device=DEVICE)
        self.hidden = self.hidden.bfloat16()
        self.q_c = torch.randn(num_requests, _MAX_MODEL_LEN, 48, device=DEVICE)
        self.q_c = self.q_c.bfloat16()
        self.coordinates = _coordinates(
            torch.arange(_MAX_MODEL_LEN, device=DEVICE), three_axis
        )

    def inputs(self, chunks, rows):
        """Batch tensors for `chunks` of `(request, start, end)` token ranges,
        padded to `rows` with junk rows."""
        hidden = torch.randn(rows, 80, device=DEVICE).bfloat16()
        q_c = torch.randn(rows, 48, device=DEVICE).bfloat16()
        positions = torch.zeros(
            (3, rows) if self.three_axis else (rows,), dtype=torch.int64
        ).to(DEVICE)
        starts, seq_lens, row = [0], [], 0
        for request, start, end in chunks:
            hidden[row : row + end - start] = self.hidden[request, start:end]
            q_c[row : row + end - start] = self.q_c[request, start:end]
            positions[..., row : row + end - start] = self.coordinates[..., start:end]
            row += end - start
            starts.append(row)
            seq_lens.append(end)
        return hidden, q_c, positions, starts, seq_lens

    def metadata(self, starts, seq_lens, num_tokens, max_seq_len):
        from vllm.models.hy_v4_flash.common.indexpool_cache import IndexPoolMetadata

        starts_t = torch.tensor(starts, dtype=torch.int32, device=DEVICE)
        seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
        return {
            prefix: IndexPoolMetadata(
                query_start_loc=starts_t,
                seq_lens=seq_lens_t,
                block_table=table,
                num_actual_tokens=num_tokens,
                max_seq_len=max_seq_len,
            )
            for prefix, table in (
                (self.module.k_cache.prefix, self.pool_table),
                (self.module.tail_cache.prefix, self.tail_table),
            )
        }

    def run(self, chunks, rows):
        from vllm.config import VllmConfig
        from vllm.forward_context import set_forward_context

        hidden, q_c, positions, starts, seq_lens = self.inputs(chunks, rows)
        metadata = self.metadata(starts, seq_lens, starts[-1], max(seq_lens))
        with set_forward_context(metadata, VllmConfig()):
            return self.module(hidden, q_c, positions).clone()

    @torch.no_grad()
    def reference_keys(self, request, num_groups):
        """FP8 pool rows and scales of the first `num_groups` groups."""
        m = self.module
        eps = self.config.rms_norm_eps
        tokens = self.hidden[request, : num_groups * _RATIO]
        keys = F.linear(tokens, m.token_key_projection.weight)
        keys = _rms(keys, m.token_key_norm.weight, eps).bfloat16()
        raw = keys.float().reshape(num_groups, _RATIO, _INDEX_DIM)
        values = raw @ m.key_value_projection.weight.T
        gates = raw @ m.key_gate_projection.weight.T
        weights = (gates + m.intra_group_ape).softmax(1)
        pooled = _rms((values * weights).sum(1), m.key_norm.weight, eps).bfloat16()
        first = self.coordinates[..., : num_groups * _RATIO : _RATIO]
        return _fwht_quantize_reference(_rotate_reference(pooled, first))

    @torch.no_grad()
    def reference_scores(self, request, logical):
        """`[tokens, groups]` scores of the queries at `logical` positions."""
        m = self.module
        heads = self.config.compressed_indexer_num_attention_heads
        hidden = self.hidden[request, logical]
        query = F.linear(self.q_c[request, logical], m.query_projection.weight)
        query = query.reshape(-1, heads, _INDEX_DIM)
        query = _rotate_reference(query, self.coordinates[..., logical])
        quant_q, scale_q = _fwht_quantize_reference(query.reshape(-1, _INDEX_DIM))
        query = (quant_q.float() * scale_q[:, None]).reshape(-1, heads, _INDEX_DIM)
        quant_k, scale_k = self.reference_keys(request, int(logical.max()) // _RATIO)
        keys = quant_k.float() * scale_k[:, None]
        weights = F.linear(hidden, m.head_weight_projection.weight).float()
        weights = weights * (heads**-0.5 * _INDEX_DIM**-0.5)
        return torch.einsum("th,thg->tg", weights, (query @ keys.T).relu())

    def pool_rows(self, request, num_groups):
        groups = torch.arange(num_groups, device=DEVICE)
        pages = self.pool.squeeze(1).flatten(1)[
            self.pool_table[request, groups // _PAGE].long()
        ]
        row = (groups % _PAGE)[:, None]
        quant = pages.gather(1, row * _INDEX_DIM + torch.arange(_INDEX_DIM).to(DEVICE))
        scale = pages.gather(
            1, _PAGE * _INDEX_DIM + row * 4 + torch.arange(4, device=DEVICE)
        )
        return quant.view(torch.float8_e4m3fn), scale.view(torch.float32)[:, 0]


def _assert_indexer_rows(harness, out, chunks):
    """Rows hold the top-`budget` history groups by reference score, expanded
    and ascending, then the open group's causal tail, then -1."""
    budget, row = harness.budget, 0
    for request, start, end in chunks:
        logical = torch.arange(start, end, device=DEVICE)
        scores = harness.reference_scores(request, logical) if end > _RATIO else None
        for i, position in enumerate(range(start, end)):
            upper = position // _RATIO
            kept = min(upper, budget)
            count = kept * _RATIO + position % _RATIO + 1
            got = out[row + i]
            assert bool((got[:count] >= 0).all()) and bool((got[count:] == -1).all())
            history = got[: kept * _RATIO].reshape(kept, _RATIO).long()
            groups = history[:, 0] // _RATIO
            offsets = torch.arange(_RATIO, device=DEVICE)
            assert torch.equal(history, groups[:, None] * _RATIO + offsets)
            assert bool((groups[1:] > groups[:-1]).all())
            tail = torch.arange(upper * _RATIO, position + 1, device=DEVICE)
            assert torch.equal(got[kept * _RATIO : count].long(), tail)
            if upper > budget:
                assert scores is not None
                row_scores = scores[i, :upper]
                chosen = torch.zeros(upper, dtype=torch.bool, device=DEVICE)
                chosen[groups] = True
                # Rescoring in FP32 may only reorder near-ties.
                tolerance = 1e-3 * float(row_scores.abs().max()) + 1e-6
                assert (
                    float(row_scores[chosen].min())
                    >= float(row_scores[~chosen].max()) - tolerance
                ), (request, position)
        row += end - start
    assert bool((out[row:] == -1).all())


def _assert_pool_matches_reference(harness, request, num_groups):
    quant, scale = harness.pool_rows(request, num_groups)
    expected_quant, expected_scale = harness.reference_keys(request, num_groups)
    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
    # A bit-exact match up to rare one-step FP8 rounding flips.
    torch.testing.assert_close(
        quant.float() * scale[:, None],
        expected_quant.float() * expected_scale[:, None],
        rtol=0.125,
        atol=float(expected_scale.max()) * 2**-6,
    )
    mismatched = (quant.view(torch.uint8) != expected_quant.view(torch.uint8)).float()
    assert float(mismatched.mean()) < 0.01


# Three steps of `(request, start, end)` chunks for four requests: a prefill
# then decodes; a short prefill then an extend that completes groups out of the
# tail ring; a long prefill past the group budget then decodes; and a request
# that is idle (seq_len 0) in step one before a prefill and an extend.
_STEPS = [
    [(0, 0, 23), (1, 0, 10), (2, 0, 90), (3, 0, 0)],
    [(0, 23, 24), (1, 10, 37), (2, 90, 91), (3, 0, 5)],
    [(0, 24, 25), (1, 37, 38), (2, 91, 92), (3, 5, 9)],
]


@requires_rocm_gfx950
@pytest.mark.parametrize(
    "num_heads,three_axis,chunked",
    [(16, False, False), (16, True, True), (4, True, False), (4, False, True)],
    ids=["h16-1d", "h16-mrope-chunked", "h4-mrope", "h4-1d-chunked"],
)
def test_compressed_indexer_matches_reference(
    num_heads, three_axis, chunked, dist_init, default_vllm_config, workspace_init
):
    """Multi-step prefill/extend/decode batches select the reference top
    groups and leave reference FP8 keys in the pool, including groups closed
    from keys carried in the tail ring; padded scoring heads (4 -> 16) and
    chunked scoring change nothing."""
    harness = _IndexerHarness(num_heads, group_topk=6, three_axis=three_axis)
    if chunked:
        # Two query rows per scoring launch.
        harness.module.logits_budget = 2 * _PAGE
    for chunks in _STEPS:
        rows = sum(end - start for _, start, end in chunks) + 5
        out = harness.run(chunks, rows)
        _assert_indexer_rows(harness, out, chunks)
    for request, _, end in _STEPS[-1]:
        _assert_pool_matches_reference(harness, request, end // _RATIO)


@requires_rocm_gfx950
def test_compressed_indexer_cuda_graph_replay(
    dist_init, default_vllm_config, workspace_init
):
    """A FULL-graph decode capture (with two padding rows) replays four
    decode steps that close new groups inside the graph, matching the
    reference selection and pool contents."""
    from vllm.config import CUDAGraphMode, VllmConfig
    from vllm.forward_context import set_forward_context

    harness = _IndexerHarness(16, group_topk=6, three_axis=True)
    harness.run([(0, 0, 30), (1, 0, 7), (2, 0, 61), (3, 0, 3)], 104)
    context, rows = [30, 7, 61, 3], 6

    def decode(step):
        return [(r, n + step, n + step + 1) for r, n in enumerate(context)]

    hidden, q_c, positions, starts, seq_lens = harness.inputs(decode(0), rows)
    metadata = harness.metadata(starts, seq_lens, rows, max(seq_lens))
    seq_lens_t = metadata[harness.module.k_cache.prefix].seq_lens

    def forward():
        with set_forward_context(
            metadata, VllmConfig(), cudagraph_runtime_mode=CUDAGraphMode.FULL
        ):
            return harness.module(hidden, q_c, positions)

    forward()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = forward()

    for step in range(4):
        new = harness.inputs(decode(step), rows)
        for static, value in zip((hidden, q_c, positions), new[:3]):
            static.copy_(value)
        seq_lens_t.copy_(torch.tensor(new[4], dtype=torch.int32))
        out.fill_(7)
        graph.replay()
        _assert_indexer_rows(harness, out, decode(step))
    for request, n in enumerate(context):
        _assert_pool_matches_reference(harness, request, (n + 4) // _RATIO)


@requires_rocm_gfx950
def test_compressed_indexer_without_tokens_returns_dead_rows(
    dist_init, default_vllm_config, workspace_init
):
    """No metadata (profiling) and a batch of only empty requests both yield
    all -1 rows and leave the caches untouched."""
    from vllm.config import VllmConfig
    from vllm.forward_context import set_forward_context

    harness = _IndexerHarness(16, group_topk=6, three_axis=False)
    hidden, q_c, positions, _, _ = harness.inputs([], 3)
    with set_forward_context(None, VllmConfig()):
        assert bool((harness.module(hidden, q_c, positions) == -1).all())
    metadata = harness.metadata([0, 0, 0], [0, 0], 0, 0)
    with set_forward_context(metadata, VllmConfig()):
        assert bool((harness.module(hidden, q_c, positions) == -1).all())
    assert not harness.pool.any() and not harness.tail.any()


# --------------------------------------------------------------------------
# Latent MoE (amd/moe.py)
# --------------------------------------------------------------------------


_AITER_FP8_SWIGLU_REASON = (
    "AITER < 0.1.30 FP8 block-scale fused MoE (AiterFp8BlockExperts) ignores "
    "swiglu_limit, so HY4 FP8 routed experts run unclamped SwiGLU under "
    "VLLM_ROCM_USE_AITER=1; fixed by ROCm/aiter#7019"
)


def _aiter_drops_fp8_swiglu_limit() -> bool:
    from importlib.metadata import version

    from packaging.version import Version

    return Version(version("amd-aiter")).release < (0, 1, 30)


@pytest.fixture(params=[False, True], ids=["triton", "aiter"])
def aiter_env(request, monkeypatch):
    from vllm._aiter_ops import is_aiter_found, rocm_aiter_ops

    if request.param and not is_aiter_found():
        pytest.skip("AITER is not installed")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1" if request.param else "0")
    rocm_aiter_ops.refresh_env_variables()
    yield request.param
    monkeypatch.undo()
    rocm_aiter_ops.refresh_env_variables()


@pytest.fixture
def moe_vllm_config(tmp_path):
    """The current `VllmConfig` of a tiny HY4 Flash checkpoint whose SwiGLU
    limit is small enough that the clamp is active for most tokens."""
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.models.hy_v4_flash.configs import HYV4FlashConfig

    config = HYV4FlashConfig(
        hidden_size=256,
        routed_expert_hidden_size=128,
        n_routed_experts=16,
        num_experts_per_tok=4,
        moe_intermediate_size=128,
        n_shared_experts=1,
        routed_scaling_factor=3.998,
        norm_topk_prob=True,
        swiglu_limit=1.0,
        num_hidden_layers=2,
    )
    config.architectures = ["HYV4FlashForCausalLM"]
    config.save_pretrained(tmp_path)
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path), dtype="bfloat16", skip_tokenizer_init=True
        )
    )
    with set_current_vllm_config(vllm_config):
        yield vllm_config


def _block_fp8(weight: torch.Tensor):
    """`[..., N, K]` -> FP8 with `[..., N/128, K/128]` scales, dequantized."""
    *lead, n, k = weight.shape
    blocks = weight.float().reshape(*lead, n // 128, 128, k // 128, 128)
    scale = blocks.abs().amax((-3, -1)) / 448.0
    quant = (blocks / scale[..., None, :, None]).to(torch.float8_e4m3fn)
    dequant = quant.float() * scale[..., None, :, None]
    return quant.reshape(weight.shape), scale, dequant.reshape(weight.shape)


def _load_random(moe, name, std, shift=0.0):
    """Fill one parameter (block-FP8 quantized when the layer is FP8) and
    return the effective high-precision weight."""
    module_name, _, leaf = name.rpartition(".")
    module = moe.get_submodule(module_name) if module_name else moe
    parameter = getattr(module, leaf)
    value = torch.randn(parameter.shape, device=DEVICE) * std + shift
    if parameter.dtype != torch.float8_e4m3fn:
        parameter.data.copy_(value)
        return parameter.data.float().clone()
    quant, scale, dequant = _block_fp8(value)
    parameter.data.copy_(quant)
    getattr(module, leaf.replace("weight", "weight_scale_inv")).data.copy_(scale)
    return dequant


def _fp8_activations(x: torch.Tensor) -> torch.Tensor:
    """Dynamic per-token, 128-wide group FP8 activations, dequantized."""
    groups = x.float().unflatten(-1, (-1, 128))
    scale = groups.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448.0
    return ((groups / scale).to(torch.float8_e4m3fn).float() * scale).flatten(-2)


def _moe_reference(weights, x, config, fp8):
    """Biased sigmoid top-k routing (bias picks experts, unbiased scores
    weight them), renormalized and scaled, clamped SwiGLU experts in the
    latent space, plain SwiGLU shared expert, then the combine RMSNorm."""
    act = _fp8_activations if fp8 else torch.Tensor.float
    scores = (x.float() @ weights["gate"].T).sigmoid()
    chosen = (scores + weights["bias"]).topk(config.num_experts_per_tok, -1).indices
    route = scores.gather(1, chosen)
    route = route / route.sum(-1, keepdim=True) * config.routed_scaling_factor
    latent = act((act(x) @ weights["down"].T).bfloat16())
    limit = config.swiglu_limit
    routed = torch.zeros_like(latent)
    for slot in range(chosen.shape[1]):
        w13, w2 = weights["w13"][chosen[:, slot]], weights["w2"][chosen[:, slot]]
        gate, up = torch.einsum("th,tih->ti", latent, w13).chunk(2, -1)
        hidden = F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)
        hidden = act(hidden.bfloat16())
        routed += route[:, slot, None] * torch.einsum("tk,tnk->tn", hidden, w2)
    routed = act(routed.bfloat16()) @ weights["up"].T
    gate, up = (act(x) @ weights["shared_gate_up"].T).chunk(2, -1)
    shared = act((F.silu(gate) * up).bfloat16()) @ weights["shared_down"].T
    return _rms(routed + shared, weights["norm"], config.rms_norm_eps)


@requires_rocm_gfx950
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8-block"])
@pytest.mark.parametrize("num_tokens", [1, 7, 130])
def test_latent_moe_matches_reference(
    num_tokens,
    fp8,
    aiter_env,
    dist_init,
    moe_vllm_config,
    workspace_init,
    request,
):
    """`HYV4FlashMoE` routes in the full width, runs clamped-SwiGLU experts in
    the latent width between the down/up projections, adds the shared expert
    and applies the sandwich norm, matching an eager reference for BF16 and
    block-FP8 weights on the Triton and AITER expert backends."""
    if aiter_env and fp8 and _aiter_drops_fp8_swiglu_limit():
        request.applymarker(
            pytest.mark.xfail(strict=True, reason=_AITER_FP8_SWIGLU_REASON)
        )
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizeMethodBase,
    )
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.models.hy_v4.amd.model import _normalize_hyv4_config
    from vllm.models.hy_v4_flash.amd.moe import HYV4FlashMoE

    config = _normalize_hyv4_config(moe_vllm_config.model_config.hf_text_config)
    quant_config = Fp8Config(weight_block_size=[128, 128]) if fp8 else None
    with set_default_torch_dtype(torch.bfloat16), torch.device(DEVICE):
        moe = HYV4FlashMoE(config, quant_config, "model.layers.1.mlp")
    weights = {
        key: _load_random(moe, name, std, *shift)
        for key, name, std, *shift in (
            ("gate", "gate.weight", 0.5),
            ("bias", "expert_bias", 0.3),
            ("down", "routed_expert_down_proj.weight", 256**-0.5),
            ("up", "routed_expert_up_proj.weight", 128**-0.5),
            ("w13", "experts.routed_experts.w13_weight", 128**-0.5),
            ("w2", "experts.routed_experts.w2_weight", 128**-0.5),
            ("shared_gate_up", "shared_experts.gate_up_proj.weight", 256**-0.5),
            ("shared_down", "shared_experts.down_proj.weight", 128**-0.5),
            ("norm", "combine_output_norm.weight", 0.1, 1.0),
        )
    }
    # As the model loader does: once per module owning a quant method.
    for module in moe.modules():
        method = getattr(module, "quant_method", None)
        if isinstance(method, QuantizeMethodBase):
            method.process_weights_after_loading(module)
    x = torch.randn(num_tokens, config.hidden_size, device=DEVICE).bfloat16()

    with set_forward_context(None, moe_vllm_config):
        out = moe(x)

    expected = _moe_reference(weights, x, config, fp8)
    if not fp8:
        torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=3e-2)
        return
    # The FP8 GEMMs round their activations differently from the eager
    # emulation, so bound the relative error; dropping the clamp costs ~50%.
    error = (out.float() - expected).norm() / expected.norm()
    assert float(error) < 0.05, float(error)
