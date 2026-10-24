# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math
import os
import tempfile

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import (
    CompilationConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.rotary_embedding.packed_qk_rope import (
    packed_qk_rope_,
)
from vllm.model_executor.models.kimi_k25_vit import (
    KimiK25MultiModalProjector,
    KimiVisionPatchMergerProjector,
    MoonVision3dPatchEmbed,
    MoonViT3dPretrainedModel,
    MoonViTEncoderLayer,
)
from vllm.models.kimi_k3.amd.model import KimiK3ForConditionalGeneration
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.kimi_k3 import KimiK3Config, KimiK3VisionConfig
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="Kimi-K3 Vision Tower compile tests require ROCm",
)


@pytest.fixture(scope="module")
def should_do_global_cleanup_after_test():
    """The module-scoped distributed state and captured graphs must outlive
    each test, so skip the per-test global cleanup from tests/conftest.py."""
    return False


@pytest.fixture(scope="module")
def vllm_compile_config():
    """VllmConfig with compile_mm_encoder enabled and initialized dist env."""
    cfg = VllmConfig(compilation_config=CompilationConfig(compile_mm_encoder=True))
    fd, temp_file = tempfile.mkstemp()
    os.close(fd)
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with set_current_vllm_config(cfg):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{temp_file}",
            local_rank=0,
            backend="nccl",
        )
        initialize_model_parallel(1, 1)
        yield cfg
        destroy_model_parallel()
        destroy_distributed_environment()
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        torch.set_default_dtype(prev_dtype)


def _init_weights(module: nn.Module):
    """Initialize empty parameter tensors to avoid NaN in random tests."""
    for p in module.parameters():
        if p.data.numel() > 0:
            if p.data.ndim >= 2:
                p.data.normal_(0.0, 0.02)
            else:
                p.data.fill_(0.01)


def test_packed_qk_rope_custom_op_compile():
    """Verify packed_qk_rope custom op compiles cleanly without Dynamo breaks."""
    seqlen = 64
    nheads = 16
    headdim = 64
    rotary_dim = headdim

    xqkv_eager = torch.randn(
        seqlen, 3, nheads, headdim, device="cuda", dtype=torch.bfloat16
    )
    xqkv_comp = xqkv_eager.clone()

    angles = torch.randn(seqlen, rotary_dim // 2, device="cuda", dtype=torch.float32)
    freqs = torch.polar(torch.ones_like(angles), angles).contiguous()

    # Eager execution
    packed_qk_rope_(xqkv_eager, freqs)

    # Compiled execution
    @torch.compile(dynamic=True)
    def fn(x, f):
        packed_qk_rope_(x, f)
        return x

    res_comp = fn(xqkv_comp, freqs)
    torch.testing.assert_close(res_comp, xqkv_eager, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("seqlen", [128, 256])
def test_moonvit_encoder_layer_compile_parity(vllm_compile_config, seqlen: int):
    """Verify MoonViTEncoderLayer forward pass compiles and matches eager."""
    v_cfg = KimiK3VisionConfig()
    hidden_dim = v_cfg.hidden_size
    num_heads = v_cfg.num_attention_heads
    mlp_dim = v_cfg.intermediate_size
    device = torch.device("cuda")

    torch.manual_seed(42)
    layer = (
        MoonViTEncoderLayer(
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            attn_bias=v_cfg.attn_bias,
            linear_bias=v_cfg.linear_bias,
            qkv_hidden_size=v_cfg.qkv_hidden_size,
            norm_type=v_cfg.norm_type,
            prefix="test_encoder_layer",
        )
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )
    _init_weights(layer)

    x = torch.randn(seqlen, hidden_dim, device=device, dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, seqlen], device=device, dtype=torch.int32)
    head_dim = v_cfg.qkv_hidden_size // num_heads
    angles = torch.randn(seqlen, head_dim // 2, device=device, dtype=torch.float32)
    freqs = torch.polar(torch.ones_like(angles), angles).contiguous()
    max_seqlen = torch.tensor(seqlen, dtype=torch.int32)

    with torch.inference_mode():
        eager_out = layer(x, cu_seqlens, freqs, max_seqlen=max_seqlen)

    compiled_layer = torch.compile(layer.forward, mode="default", dynamic=True)
    with torch.inference_mode():
        comp_out = compiled_layer(x, cu_seqlens, freqs, max_seqlen=max_seqlen)

    cos_sim = F.cosine_similarity(
        eager_out.float().flatten(1), comp_out.float().flatten(1), dim=-1
    ).mean()
    assert cos_sim.item() > 0.999, f"Cosine similarity {cos_sim.item()} <= 0.999"


@pytest.mark.parametrize("num_tokens", [64, 128])
def test_kimik25_projector_compile_parity(vllm_compile_config, num_tokens: int):
    """Verify KimiK25MultiModalProjector compiles and matches eager."""
    v_cfg = KimiK3VisionConfig()
    out_dim = 512
    device = torch.device("cuda")

    torch.manual_seed(42)
    proj = (
        KimiK25MultiModalProjector(
            config=v_cfg,
            out_hidden_size=out_dim,
            use_data_parallel=False,
            prefix="test_proj",
        )
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )
    _init_weights(proj)

    features = torch.randn(
        num_tokens, 4, v_cfg.hidden_size, device=device, dtype=torch.bfloat16
    )

    with torch.inference_mode():
        eager_out = proj(features)

    compiled_proj = torch.compile(proj.forward, mode="default", dynamic=True)
    with torch.inference_mode():
        comp_out = compiled_proj(features)

    torch.testing.assert_close(comp_out, eager_out, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("num_patches", [144, 288])
def test_moonvision3d_patchembed_compile_parity(vllm_compile_config, num_patches: int):
    """Verify MoonVision3dPatchEmbed compiles into 1 graph without breaks."""
    v_cfg = KimiK3VisionConfig()
    device = torch.device("cuda")

    torch.manual_seed(42)
    pe = (
        MoonVision3dPatchEmbed(
            out_dim=v_cfg.hidden_size,
            patch_size=v_cfg.patch_size,
            pos_emb_height=14,
            pos_emb_width=14,
            pos_emb_time=4,
            patch_embed_proj_bias=v_cfg.patch_embed_proj_bias,
        )
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )
    _init_weights(pe)

    x = torch.randn(
        num_patches,
        3,
        v_cfg.patch_size,
        v_cfg.patch_size,
        device=device,
        dtype=torch.bfloat16,
    )
    pos_embeds = torch.randn(
        num_patches, v_cfg.hidden_size, device=device, dtype=torch.bfloat16
    )

    with torch.inference_mode():
        eager_out = pe(x, pos_embeds=pos_embeds)

    compiled_pe = torch.compile(pe.forward, mode="default", dynamic=True)
    with torch.inference_mode():
        comp_out = compiled_pe(x, pos_embeds=pos_embeds)

    torch.testing.assert_close(comp_out, eager_out, atol=1e-3, rtol=1e-3)

    # Verify 0 TorchDynamo graph breaks
    explanation = torch._dynamo.explain(pe.forward)(x, pos_embeds=pos_embeds)
    assert explanation.graph_break_count == 0, (
        f"Graph breaks found: {explanation.break_reasons}"
    )
    assert len(explanation.graphs) == 1, (
        f"Expected 1 graph, got {len(explanation.graphs)}"
    )


@pytest.mark.parametrize("num_tokens", [36, 72])
def test_kimi_vision_patch_merger_projector_compile_parity(
    vllm_compile_config, num_tokens: int
):
    """Verify KimiVisionPatchMergerProjector compiles into 1 graph."""
    v_cfg = KimiK3VisionConfig()
    out_dim = 512
    device = torch.device("cuda")

    torch.manual_seed(42)
    proj = (
        KimiK25MultiModalProjector(
            config=v_cfg,
            out_hidden_size=out_dim,
            use_data_parallel=False,
            prefix="test_proj_merger",
        )
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )
    _init_weights(proj)

    merger_and_proj = (
        KimiVisionPatchMergerProjector(proj)
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )

    num_patches = num_tokens * 4
    hidden_states = torch.randn(
        num_patches, v_cfg.hidden_size, device=device, dtype=torch.bfloat16
    )
    merge_gather_idx = torch.randint(
        0, num_patches, (num_tokens, 4), device=device, dtype=torch.int64
    )

    with torch.inference_mode():
        eager_out = merger_and_proj(hidden_states, merge_gather_idx)

    compiled_merger_proj = torch.compile(
        merger_and_proj.forward, mode="default", dynamic=True
    )
    with torch.inference_mode():
        comp_out = compiled_merger_proj(hidden_states, merge_gather_idx)

    torch.testing.assert_close(comp_out, eager_out, atol=1e-3, rtol=1e-3)

    # Verify 0 TorchDynamo graph breaks
    explanation = torch._dynamo.explain(merger_and_proj.forward)(
        hidden_states, merge_gather_idx
    )
    assert explanation.graph_break_count == 0, (
        f"Graph breaks found: {explanation.break_reasons}"
    )
    assert len(explanation.graphs) == 1, (
        f"Expected 1 graph, got {len(explanation.graphs)}"
    )


def test_kimi_k3_vision_tower_full_pipeline_parity(vllm_compile_config):
    """Verify that fused patchmerger + projector matches legacy eager path bitwise."""
    v_cfg = KimiK3VisionConfig()
    out_dim = 512
    device = torch.device("cuda")

    torch.manual_seed(42)
    vt = MoonViT3dPretrainedModel(v_cfg).to(device=device, dtype=torch.bfloat16).eval()
    _init_weights(vt)
    proj = (
        KimiK25MultiModalProjector(
            config=v_cfg,
            out_hidden_size=out_dim,
            use_data_parallel=False,
            prefix="test_proj_full",
        )
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )
    _init_weights(proj)
    merger_proj = (
        KimiVisionPatchMergerProjector(proj)
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )

    grid_thws = [[1, 28, 28], [1, 14, 28]]
    total_patches = sum(t * h * w for t, h, w in grid_thws)
    pixel_values = torch.randn(
        total_patches,
        3,
        v_cfg.patch_size,
        v_cfg.patch_size,
        device=device,
        dtype=torch.bfloat16,
    )

    with torch.inference_mode():
        # 1. Legacy eager forward (patch embed -> encoder -> tpool_patch_merger -> proj)
        legacy_vt_out = vt(pixel_values, grid_thws)
        legacy_proj_out = proj(torch.cat(legacy_vt_out)).reshape(-1, out_dim)

        # 2. Optimized pipeline (metadata -> patch embed -> encoder -> merger_proj)
        meta = vt.prepare_encoder_cudagraph_metadata(
            grid_thws, max_batch_size=len(grid_thws), device=device
        )
        gather_idx = meta.pop("merge_gather_idx")
        comp_vt_out = vt(pixel_values, None, encoder_metadata=meta)
        comp_proj_out = merger_proj(comp_vt_out, gather_idx).reshape(-1, out_dim)

    torch.testing.assert_close(comp_proj_out, legacy_proj_out, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Encoder CUDA graph tests
#
# KimiK3ForConditionalGeneration cannot be built from a tiny config (its
# KimiLinear text layers need a full MLA/KDA config), so the harnesses below
# subclass it and only replace __init__: every method under test is the real
# one, driven by a real ModelConfig, VllmConfig and EncoderCudaGraphManager.
# ---------------------------------------------------------------------------

_TINY_K3_CONFIG = {
    "architectures": ["KimiK3ForConditionalGeneration"],
    "model_type": "kimi_k3",
    "text_config": {
        "model_type": "kimi_linear",
        "hidden_size": 512,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "intermediate_size": 1024,
        "vocab_size": 1024,
    },
    "vision_config": {"num_hidden_layers": 1},
    "max_position_embeddings": 4096,
}
_BUDGETS = [64, 128]
_MAX_VISION_ITEMS = 2


def _model_config(path, max_model_len: int = 4096) -> ModelConfig:
    return ModelConfig(
        model=str(path),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=max_model_len,
        limit_mm_per_prompt={"image": 4, "video": 0},
        trust_remote_code=True,
    )


def _vllm_config(
    model_config: ModelConfig,
    *,
    compile_mm_encoder: bool = True,
    budgets: list[int] | None = None,
    max_num_batched_tokens: int = 2048,
) -> VllmConfig:
    return VllmConfig(
        model_config=model_config,
        scheduler_config=SchedulerConfig.default_factory(
            max_model_len=model_config.max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        compilation_config=CompilationConfig(
            compile_mm_encoder=compile_mm_encoder,
            cudagraph_mm_encoder=True,
            encoder_cudagraph_token_budgets=budgets or [],
            encoder_cudagraph_max_vision_items_per_batch=_MAX_VISION_ITEMS,
        ),
    )


class _ConfigOnlyHarness(KimiK3ForConditionalGeneration):
    """Real encoder-cudagraph methods on a config, without any weights."""

    def __init__(self, config, model_config=None) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.model_config = model_config
        self.hidden_size = config.text_config.hidden_size


class _EncoderHarness(_ConfigOnlyHarness):
    """Adds the real vision tower and projectors with dummy weights."""

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__(model_config.hf_config, model_config)
        v_cfg = self.config.vision_config
        self.device = torch.device("cuda:0")
        self.vision_tower = MoonViT3dPretrainedModel(v_cfg)
        self.mm_projector = KimiK25MultiModalProjector(
            config=v_cfg,
            out_hidden_size=self.hidden_size,
            use_data_parallel=False,
            prefix="mm_projector",
        )
        self.merger_projector = KimiVisionPatchMergerProjector(self.mm_projector)
        self.to(device=self.device, dtype=torch.bfloat16)


def _build_encoder(path, compile_mm_encoder: bool):
    """Real model plus a manager that has already captured its graphs."""
    cfg = _vllm_config(
        _model_config(path),
        compile_mm_encoder=compile_mm_encoder,
        budgets=_BUDGETS,
    )
    with set_current_vllm_config(cfg):
        model = _EncoderHarness(cfg.model_config)
        _init_weights(model)
        model.eval()
        mgr = EncoderCudaGraphManager(cfg, model.device, torch.bfloat16, model)
        mgr.capture(torch.cuda.graph_pool_handle())
    return model, mgr


@pytest.fixture(scope="module")
def tiny_k3_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny_kimi_k3")
    (path / "config.json").write_text(json.dumps(_TINY_K3_CONFIG))
    return path


@pytest.fixture(scope="module")
def encoder_stack(vllm_compile_config, tiny_k3_dir):
    return _build_encoder(tiny_k3_dir, compile_mm_encoder=True)


@pytest.fixture(scope="module")
def config_only_model(tiny_k3_dir):
    model_config = _model_config(tiny_k3_dir)
    return _ConfigOnlyHarness(model_config.hf_config, model_config)


def _mm_kwargs(model, grids: list[list[int]], seed: int = 0) -> dict:
    patch = model.config.vision_config.patch_size
    generator = torch.Generator(device="cuda").manual_seed(seed)
    num_patches = sum(t * h * w for t, h, w in grids)
    pixel_values = torch.randn(
        num_patches,
        3,
        patch,
        patch,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    return {"pixel_values": pixel_values, "grid_thws": grids}


def _execute(mgr, mm_kwargs) -> list[torch.Tensor]:
    with torch.inference_mode():
        return mgr.execute(mm_kwargs)


def _eager(model, mm_kwargs, tokens: list[int]) -> list[torch.Tensor]:
    with torch.inference_mode():
        return list(model.encoder_eager_forward(mm_kwargs).split(tokens))


def _legacy(model, mm_kwargs) -> torch.Tensor:
    """Reference that bypasses the encoder-cudagraph helpers entirely."""
    with torch.inference_mode():
        features = model.vision_tower(mm_kwargs["pixel_values"], mm_kwargs["grid_thws"])
        out = model.mm_projector(torch.cat(features))
    return out.reshape(-1, out.shape[-1])


def _graph_stats(mgr) -> tuple[int, int]:
    stats = mgr.get_cumulative_stats()
    return stats["graph_hits"], stats["graph_misses"]


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Allow 2 bf16 ulps at the output scale (graph replay runs padded GEMMs).

    The outputs must be large enough for that bound to still catch real errors.
    """
    assert actual.shape == expected.shape
    bound = 2 * expected.abs().max().item() * 2**-7
    assert expected.float().std().item() > 10 * bound, "output too small to test"
    diff = (actual.float() - expected.float()).abs().max().item()
    assert diff <= bound, f"max abs diff {diff} > {bound}"


def _out_tokens(model, grid: list[int]) -> int:
    kh, kw = model.config.vision_config.merge_kernel_size
    return (grid[1] // kh) * (grid[2] // kw)


def test_encoder_cudagraph_item_specs(config_only_model):
    """Input patches are t*h*w and output tokens follow the merge kernel."""
    grids = [[1, 28, 28], [1, 14, 28], [2, 14, 14]]
    specs = config_only_model.get_encoder_cudagraph_item_specs({"grid_thws": grids})
    assert [s.input_size for s in specs] == [784, 392, 392]
    assert [s.output_tokens for s in specs] == [196, 98, 49]


def test_encoder_cudagraph_item_specs_asymmetric_merge_kernel():
    """A (2, 4) kernel on a [1, 14, 28] grid gives 7 * 7 tokens, not 3 * 14."""
    vision_config = KimiK3VisionConfig(merge_kernel_size=(2, 4))
    model = _ConfigOnlyHarness(KimiK3Config(vision_config=vision_config))
    specs = model.get_encoder_cudagraph_item_specs({"grid_thws": [[1, 14, 28]]})
    assert specs[0].output_tokens == 49


def test_select_encoder_cudagraph_items(config_only_model):
    """Selection keeps the requested order and accepts list/tensor inputs."""
    grids = [[1, 14, 14], [1, 14, 28], [1, 28, 28]]
    mm_kwargs = _mm_kwargs(config_only_model, grids)
    p0, p1, p2 = mm_kwargs["pixel_values"].split([196, 392, 784])

    selected = config_only_model.select_encoder_cudagraph_items(mm_kwargs, [2, 0])
    assert torch.equal(selected["pixel_values"], torch.cat([p2, p0]))
    assert selected["grid_thws"].tolist() == [grids[2], grids[0]]

    empty = config_only_model.select_encoder_cudagraph_items(mm_kwargs, [])
    assert empty["pixel_values"].shape[0] == 0
    assert len(empty["grid_thws"]) == 0

    as_tensors = {
        "pixel_values": [p0, p1, p2],
        "grid_thws": torch.tensor(grids, device="cuda"),
    }
    selected = config_only_model.select_encoder_cudagraph_items(as_tensors, [1])
    assert torch.equal(selected["pixel_values"], p1)
    assert selected["grid_thws"].tolist() == [grids[1]]


@pytest.mark.parametrize(
    "max_num_batched_tokens, max_model_len, expected_max",
    [
        (8192, 4096, 1040),  # capped: graphs give no speedup above ~1k tokens
        (512, 4096, 512),  # follows the scheduler limit
        (8192, 256, 256),  # follows the model length
    ],
)
def test_encoder_cudagraph_budget_range(
    tiny_k3_dir, max_num_batched_tokens, max_model_len, expected_max
):
    model_config = _model_config(tiny_k3_dir, max_model_len)
    model = _ConfigOnlyHarness(model_config.hf_config, model_config)
    cfg = _vllm_config(model_config, max_num_batched_tokens=max_num_batched_tokens)
    assert model.get_encoder_cudagraph_budget_range(cfg) == (64, expected_max)


def test_manager_budgets_cap_auto_and_keep_explicit(encoder_stack, tiny_k3_dir):
    model, _ = encoder_stack
    model_config = _model_config(tiny_k3_dir)

    auto = _vllm_config(model_config, max_num_batched_tokens=8192)
    auto_mgr = EncoderCudaGraphManager(auto, model.device, torch.bfloat16, model)
    assert max(auto_mgr.token_budgets) == 1040

    explicit = _vllm_config(
        model_config, budgets=[64, 128, 2048], max_num_batched_tokens=8192
    )
    explicit_mgr = EncoderCudaGraphManager(
        explicit, model.device, torch.bfloat16, model
    )
    assert explicit_mgr.token_budgets == [64, 128, 2048]


@pytest.mark.parametrize("max_batch_size", [1, 2])
@pytest.mark.parametrize("token_budget", [64, 128, 768, 1040])
def test_capture_inputs_cover_the_budget(encoder_stack, token_budget, max_batch_size):
    """Dummy grids must produce at least token_budget outputs, and not much more."""
    model, _ = encoder_stack
    kh, kw = model.config.vision_config.merge_kernel_size
    capture = model.prepare_encoder_cudagraph_capture_inputs(
        token_budget=token_budget,
        max_batch_size=max_batch_size,
        max_frames_per_batch=0,
        device=model.device,
        dtype=torch.bfloat16,
    )
    config = model.get_encoder_cudagraph_config()
    assert set(capture.values) == set(config.buffer_keys)

    pixel_values = capture.values["pixel_values"]
    assert pixel_values.dtype == torch.bfloat16
    assert pixel_values.device == model.device
    output_tokens = pixel_values.shape[0] // (kh * kw)
    per_item = math.ceil(token_budget / max_batch_size)
    assert (
        token_budget
        <= output_tokens
        <= max_batch_size * (per_item + math.ceil(math.sqrt(per_item)))
    )
    assert capture.values["max_seqlen"].item() >= token_budget * kh * kw


def test_capture_inputs_reject_budget_beyond_rope_capacity(encoder_stack):
    model, _ = encoder_stack
    with pytest.raises(ValueError, match="exceeds K3 RoPE capacity"):
        model.prepare_encoder_cudagraph_capture_inputs(
            token_budget=10_000_000,
            max_batch_size=1,
            max_frames_per_batch=0,
            device=model.device,
            dtype=torch.bfloat16,
        )


def test_replay_buffers_match_vision_tower_metadata(encoder_stack):
    model, _ = encoder_stack
    grids = [[1, 14, 14], [1, 14, 28]]
    mm_kwargs = _mm_kwargs(model, grids)
    replay = model.prepare_encoder_cudagraph_replay_buffers(
        mm_kwargs, max_batch_size=4, max_frames_per_batch=0
    )
    expected = model.vision_tower.prepare_encoder_cudagraph_metadata(
        grids, max_batch_size=4, device=model.device
    )
    config = model.get_encoder_cudagraph_config()
    assert set(replay.values) == set(config.buffer_keys)
    assert torch.equal(replay.values["pixel_values"], mm_kwargs["pixel_values"])
    for key, value in expected.items():
        if value is None:
            assert replay.values[key] is None, key
        else:
            assert torch.equal(replay.values[key], value), key


@pytest.mark.parametrize("grid", [[1, 16, 16], [1, 14, 14]], ids=["exact", "under"])
def test_single_image_replay_matches_eager(encoder_stack, grid):
    """64 tokens fill the 64 budget; 49 tokens replay it padded."""
    model, mgr = encoder_stack
    mm_kwargs = _mm_kwargs(model, [grid])
    tokens = _out_tokens(model, grid)
    hits, misses = _graph_stats(mgr)

    (out,) = _execute(mgr, mm_kwargs)

    assert _graph_stats(mgr) == (hits + 1, misses)
    _assert_close(out, _eager(model, mm_kwargs, [tokens])[0])
    _assert_close(out, _legacy(model, mm_kwargs))


def test_two_images_share_one_replay(encoder_stack):
    model, mgr = encoder_stack
    grids = [[1, 14, 14], [1, 14, 28]]
    mm_kwargs = _mm_kwargs(model, grids)
    tokens = [_out_tokens(model, g) for g in grids]
    hits, misses = _graph_stats(mgr)

    outs = _execute(mgr, mm_kwargs)

    assert _graph_stats(mgr) == (hits + 2, misses)
    assert [o.shape[0] for o in outs] == tokens
    for out, expected in zip(outs, _eager(model, mm_kwargs, tokens)):
        _assert_close(out, expected)


def test_replay_does_not_reuse_stale_buffers(encoder_stack):
    """Different token counts in the same budget graph, then the first again."""
    model, mgr = encoder_stack
    first = _mm_kwargs(model, [[1, 14, 28]], seed=1)  # 98 tokens
    second = _mm_kwargs(model, [[1, 16, 18]], seed=2)  # 72 tokens
    hits, misses = _graph_stats(mgr)

    out_first = _execute(mgr, first)[0]
    out_second = _execute(mgr, second)[0]
    out_first_again = _execute(mgr, first)[0]

    assert _graph_stats(mgr) == (hits + 3, misses)
    assert torch.equal(out_first, out_first_again)
    _assert_close(out_first, _eager(model, first, [98])[0])
    _assert_close(out_second, _eager(model, second, [72])[0])


def test_oversize_item_falls_back_to_eager(encoder_stack):
    """196 tokens exceed the largest budget (128), alone and next to a graphed item."""
    model, mgr = encoder_stack
    big = _mm_kwargs(model, [[1, 28, 28]], seed=3)
    hits, misses = _graph_stats(mgr)

    (out_big,) = _execute(mgr, big)

    assert _graph_stats(mgr) == (hits, misses + 1)
    assert torch.equal(out_big, _eager(model, big, [196])[0])

    small = _mm_kwargs(model, [[1, 14, 14]], seed=4)
    mixed = {
        "pixel_values": torch.cat([small["pixel_values"], big["pixel_values"]]),
        "grid_thws": small["grid_thws"] + big["grid_thws"],
    }
    hits, misses = _graph_stats(mgr)

    out_small, out_big_again = _execute(mgr, mixed)

    assert _graph_stats(mgr) == (hits + 1, misses + 1)
    _assert_close(out_small, _eager(model, small, [49])[0])
    assert torch.equal(out_big_again, out_big)


def test_items_exceeding_the_budget_together_are_split(encoder_stack):
    """Three 98-token items never fit one 128-token replay, so each replays alone."""
    model, mgr = encoder_stack
    grids = [[1, 14, 28]] * 3
    mm_kwargs = _mm_kwargs(model, grids, seed=5)
    hits, misses = _graph_stats(mgr)

    outs = _execute(mgr, mm_kwargs)

    assert _graph_stats(mgr) == (hits + 3, misses)
    for out, expected in zip(outs, _eager(model, mm_kwargs, [98] * 3)):
        _assert_close(out, expected)


def test_replay_without_compile(tiny_k3_dir, vllm_compile_config):
    """Graphs capture and replay correctly when the encoder is not compiled."""
    model, mgr = _build_encoder(tiny_k3_dir, compile_mm_encoder=False)
    mm_kwargs = _mm_kwargs(model, [[1, 16, 16]])
    hits, misses = _graph_stats(mgr)

    (out,) = _execute(mgr, mm_kwargs)

    assert _graph_stats(mgr) == (hits + 1, misses)
    _assert_close(out, _eager(model, mm_kwargs, [64])[0])


def test_cudagraph_forward_leaves_its_inputs_untouched(encoder_stack):
    model, _ = encoder_stack
    capture = model.prepare_encoder_cudagraph_capture_inputs(
        token_budget=64,
        max_batch_size=1,
        max_frames_per_batch=0,
        device=model.device,
        dtype=torch.bfloat16,
    )
    keys = set(capture.values)

    with torch.inference_mode():
        model.encoder_cudagraph_forward(capture.values)

    assert set(capture.values) == keys
