# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FP8 scaling (dynamic and static) in MMEncoderAttention."""

import contextlib
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.model_executor.layers.attention.mm_encoder_attention import (
    _FP8_AMAX_HISTORY_LEN,
    _FP8_MAX,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    is_flashinfer_cudnn_fp8_prefill_attn_supported,
)

LAYER_0 = "visual.blocks.0.attn.attn"
LAYER_1 = "visual.blocks.1.attn.attn"
NUM_HEADS = 16
HEAD_DIM = 72


def _is_mi3xx() -> bool:
    if not current_platform.is_rocm():
        return False

    from vllm.platforms.rocm import on_mi3xx

    return on_mi3xx()


@contextlib.contextmanager
def _build_attention(mm_config, attn_backend=None):
    """Yield an MMEncoderAttention with the given multimodal config.

    The VllmConfig context stays active while the test runs so that
    ``get_multimodal_config()`` calls during the forward path resolve. Also
    invokes ``process_weights_after_loading`` to simulate the model loader's
    auto-scan. Yields ``None`` if FlashInfer cuDNN is not available.
    """
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.attention.mm_encoder_attention import (
        MMEncoderAttention,
    )
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    if attn_backend is None:
        attn_backend = AttentionBackendEnum.FLASHINFER

    if (
        attn_backend == AttentionBackendEnum.FLASHINFER
        and not is_flashinfer_cudnn_fp8_prefill_attn_supported()
    ):
        yield None
        return

    vllm_config = VllmConfig()
    vllm_config.model_config = MagicMock(spec=ModelConfig, multimodal_config=mm_config)

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.model_executor.layers.attention.mm_encoder_attention"
            ".get_vit_attn_backend",
            return_value=attn_backend,
        ),
    ):
        attn = MMEncoderAttention(
            num_heads=NUM_HEADS,
            head_size=HEAD_DIM,
            prefix=LAYER_0,
        )
        attn.process_weights_after_loading()
        yield attn


@pytest.fixture
def _make_attention():
    """Create an MMEncoderAttention with dynamic FP8 scaling."""
    from vllm.config.multimodal import MultiModalConfig

    with _build_attention(MultiModalConfig(mm_encoder_attn_dtype="fp8")) as attn:
        yield attn


@pytest.fixture
def _make_static_attention(tmp_path):
    """Create an MMEncoderAttention with static FP8 scales from a file."""
    from vllm.config.multimodal import MultiModalConfig

    scale_file = tmp_path / "scales.json"
    scale_file.write_text(
        json.dumps(
            {
                LAYER_0: {"q": 224.0, "k": 198.0, "v": 210.0},
                LAYER_1: {"q": 100.0, "k": 110.0, "v": 120.0},
            }
        )
    )
    with _build_attention(
        MultiModalConfig(
            mm_encoder_attn_dtype="fp8",
            mm_encoder_fp8_scale_path=str(scale_file),
        )
    ) as attn:
        yield attn


def test_dynamic_scaling_updates_scales(_make_attention) -> None:
    """Verify that _record_amax_and_update_scales updates scale buffers."""
    attn = _make_attention
    if attn is None or not attn.fp8_enabled:
        pytest.skip("FP8 attention not available (FlashInfer backend required)")

    attn = attn.to("cuda")

    S, H, D = 32, NUM_HEADS, HEAD_DIM
    q = torch.full((S, H, D), 2.0, device="cuda", dtype=torch.bfloat16)
    k = torch.full((S, H, D), 3.0, device="cuda", dtype=torch.bfloat16)
    v = torch.full((S, H, D), 4.0, device="cuda", dtype=torch.bfloat16)

    attn._record_amax_and_update_scales(q, k, v)

    expected_q_scale = 2.0 / _FP8_MAX
    expected_k_scale = 3.0 / _FP8_MAX
    expected_v_scale = 4.0 / _FP8_MAX

    torch.testing.assert_close(attn._fp8_q_scale.item(), expected_q_scale)
    torch.testing.assert_close(attn._fp8_k_scale.item(), expected_k_scale)
    torch.testing.assert_close(attn._fp8_v_scale.item(), expected_v_scale)


def test_circular_buffer_wraps(_make_attention) -> None:
    """Verify the amax circular buffer wraps at HISTORY_LEN."""
    attn = _make_attention
    if attn is None or not attn.fp8_enabled:
        pytest.skip("FP8 attention not available (FlashInfer backend required)")

    attn = attn.to("cuda")
    S, H, D = 16, NUM_HEADS, HEAD_DIM

    for i in range(_FP8_AMAX_HISTORY_LEN + 2):
        mag = float(i + 1)
        q = torch.full((S, H, D), mag, device="cuda", dtype=torch.bfloat16)
        k = torch.full((S, H, D), mag, device="cuda", dtype=torch.bfloat16)
        v = torch.full((S, H, D), mag, device="cuda", dtype=torch.bfloat16)
        attn._record_amax_and_update_scales(q, k, v)

    assert attn._fp8_amax_pos == 2

    expected_max = float(_FP8_AMAX_HISTORY_LEN + 2)
    expected_scale = expected_max / _FP8_MAX
    torch.testing.assert_close(attn._fp8_q_scale.item(), expected_scale)


def test_static_scales_loaded(_make_static_attention) -> None:
    """Verify static scales are loaded from the JSON file."""
    attn = _make_static_attention
    if attn is None or not attn.fp8_enabled:
        pytest.skip("FP8 attention not available (FlashInfer backend required)")

    assert attn.fp8_enabled
    assert not attn._fp8_dynamic_scale

    # Layer 0 scales (the layer this attention was created with).
    assert attn._fp8_q_scale.item() == 224.0
    assert attn._fp8_k_scale.item() == 198.0
    assert attn._fp8_v_scale.item() == 210.0

    assert not attn.skip_scale_q
    assert not attn.skip_scale_k
    assert not attn.skip_scale_v

    # No amax history buffers for static scaling.
    assert not hasattr(attn, "_fp8_q_amax")


@pytest.mark.skipif(
    not _is_mi3xx(),
    reason="AITER FP8 attention requires MI300/MI350",
)
def test_aiter_static_scales_loaded(tmp_path) -> None:
    """Verify AITER reuses the existing static FP8 scale loading path."""
    from vllm.config.multimodal import MultiModalConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    scale_file = tmp_path / "aiter_scales.json"
    scale_file.write_text(json.dumps({LAYER_0: {"q": 224.0, "k": 198.0, "v": 210.0}}))
    mm_config = MultiModalConfig(
        mm_encoder_attn_dtype="fp8",
        mm_encoder_fp8_scale_path=str(scale_file),
    )

    with _build_attention(mm_config, AttentionBackendEnum.ROCM_AITER_FA) as attn:
        assert attn is not None
        assert attn.fp8_enabled
        assert not attn._fp8_dynamic_scale
        assert attn._fp8_q_scale.item() == 224.0
        assert attn._fp8_k_scale.item() == 198.0
        assert attn._fp8_v_scale.item() == 210.0


def test_static_scales_missing_layer(tmp_path) -> None:
    """Verify error when requested layer is not in the scale file."""
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.config.multimodal import MultiModalConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    if not is_flashinfer_cudnn_fp8_prefill_attn_supported():
        pytest.skip("FlashInfer cuDNN not available")

    scale_file = tmp_path / "wrong_layer.json"
    scale_file.write_text(
        json.dumps({"visual.blocks.99.attn": {"q": 1.0, "k": 1.0, "v": 1.0}})
    )
    mm_config = MultiModalConfig(
        mm_encoder_attn_dtype="fp8",
        mm_encoder_fp8_scale_path=str(scale_file),
    )
    vllm_config = VllmConfig()
    vllm_config.model_config = MagicMock(spec=ModelConfig, multimodal_config=mm_config)

    from vllm.model_executor.layers.attention.mm_encoder_attention import (
        MMEncoderAttention,
    )

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.model_executor.layers.attention.mm_encoder_attention"
            ".get_vit_attn_backend",
            return_value=AttentionBackendEnum.FLASHINFER,
        ),
    ):
        attn = MMEncoderAttention(
            num_heads=NUM_HEADS,
            head_size=HEAD_DIM,
            prefix=LAYER_0,
        )
        with pytest.raises(ValueError, match="scales not found for layer"):
            attn.process_weights_after_loading()


def test_dynamic_scales_auto_save(tmp_path) -> None:
    """Verify scales are saved to disk after the amax buffer fills."""
    import vllm.model_executor.layers.attention.mm_encoder_attention as _mod
    from vllm.config.multimodal import MultiModalConfig

    if not is_flashinfer_cudnn_fp8_prefill_attn_supported():
        pytest.skip("FlashInfer cuDNN not available")

    # Reset module-level state between runs (other tests may have left
    # state behind after triggering a save).
    _mod._fp8_scale_save_path = None
    _mod._fp8_saved_scale_refs.clear()

    save_file = tmp_path / "auto_scales.json"
    with _build_attention(
        MultiModalConfig(
            mm_encoder_attn_dtype="fp8",
            mm_encoder_fp8_scale_save_path=str(save_file),
        )
    ) as attn:
        if attn is None or not attn.fp8_enabled:
            pytest.skip("FP8 attention not available")

        attn = attn.to("cuda")
        S, H, D = 16, NUM_HEADS, HEAD_DIM

        # Run exactly _FP8_AMAX_HISTORY_LEN forward passes.
        for i in range(_FP8_AMAX_HISTORY_LEN):
            mag = float(i + 1)
            q = torch.full((S, H, D), mag, device="cuda", dtype=torch.bfloat16)
            k = torch.full((S, H, D), mag * 0.5, device="cuda", dtype=torch.bfloat16)
            v = torch.full((S, H, D), mag * 0.3, device="cuda", dtype=torch.bfloat16)
            attn._record_amax_and_update_scales(q, k, v)

    # File should have been written on the 16th call (buffer wrap).
    assert save_file.is_file(), "Scale file was not saved"
    scales = json.loads(save_file.read_text())
    assert LAYER_0 in scales
    assert set(scales[LAYER_0].keys()) == {"q", "k", "v"}
    for val in scales[LAYER_0].values():
        assert isinstance(val, float) and val > 0

    # Path is cleared after the one-shot save fires.
    assert _mod._fp8_scale_save_path is None


def _save_scales_state(monkeypatch: pytest.MonkeyPatch, path: Path):
    """Arm the module's scale auto-save for ``path`` (margin 2) and return the
    module plus a list of the renames it makes; restored after the test."""
    import vllm.model_executor.layers.attention.mm_encoder_attention as mod

    monkeypatch.setattr(mod, "_fp8_scale_save_path", str(path))
    monkeypatch.setattr(mod, "_fp8_scale_save_margin", 2.0)
    monkeypatch.setattr(mod, "_fp8_saved_scale_refs", {})
    monkeypatch.setattr(mod, "_fp8_scale_save_every_wrap", False)
    renames: list[tuple[str, str, dict]] = []
    replace = os.replace

    def checked_replace(src, dst):
        # The temporary file must already hold the complete JSON.
        renames.append((str(src), str(dst), json.loads(Path(src).read_text())))
        replace(src, dst)

    monkeypatch.setattr(mod.os, "replace", checked_replace)
    return mod, renames


def test_scale_file_saved_atomically(tmp_path, monkeypatch) -> None:
    """The first amax wrap publishes the scale file with one rename (readers,
    e.g. other data-parallel encoder ranks, never see a partial file), leaves
    no temporary file behind and disarms the one-shot save."""
    save_file = tmp_path / "scales.json"
    save_file.write_text('{"torn": ')  # left by an interrupted writer
    mod, renames = _save_scales_state(monkeypatch, save_file)
    t = torch.tensor

    mod._maybe_save_fp8_scales(LAYER_0, t(1.0), t(2.0), t(3.0), False)
    assert renames == []
    mod._maybe_save_fp8_scales(LAYER_1, t(4.0), t(5.0), t(6.0), True)

    expected = {
        LAYER_0: {"q": 2.0, "k": 4.0, "v": 6.0},
        LAYER_1: {"q": 8.0, "k": 10.0, "v": 12.0},
    }
    assert len(renames) == 1
    src, dst, content = renames[0]
    assert dst == str(save_file) and src != dst
    assert content == expected
    assert json.loads(save_file.read_text()) == expected
    assert [p.name for p in tmp_path.iterdir()] == ["scales.json"]
    assert mod._fp8_scale_save_path is None
    mod._maybe_save_fp8_scales(LAYER_0, t(9.0), t(9.0), t(9.0), True)
    assert len(renames) == 1


def test_scale_file_rewritten_every_wrap(tmp_path, monkeypatch) -> None:
    """With mm_encoder_fp8_scale_save_every_wrap every wrap rewrites the
    file, which ends with the scales of the last amax window."""
    save_file = tmp_path / "scales.json"
    mod, renames = _save_scales_state(monkeypatch, save_file)
    monkeypatch.setattr(mod, "_fp8_scale_save_every_wrap", True)
    t = torch.tensor

    mod._maybe_save_fp8_scales(LAYER_0, t(1.0), t(1.0), t(1.0), True)
    mod._maybe_save_fp8_scales(LAYER_0, t(3.0), t(4.0), t(5.0), True)

    assert len(renames) == 2
    assert json.loads(save_file.read_text()) == {
        LAYER_0: {"q": 6.0, "k": 8.0, "v": 10.0}
    }
    assert [p.name for p in tmp_path.iterdir()] == ["scales.json"]
    assert mod._fp8_scale_save_path == str(save_file)


def test_scale_save_every_wrap_requires_save_path(tmp_path) -> None:
    from vllm.config.multimodal import MultiModalConfig

    with pytest.raises(ValueError, match="requires 'mm_encoder_fp8_scale_save_path'"):
        MultiModalConfig(
            mm_encoder_attn_dtype="fp8", mm_encoder_fp8_scale_save_every_wrap=True
        )
    cfg = MultiModalConfig(
        mm_encoder_attn_dtype="fp8",
        mm_encoder_fp8_scale_save_path=str(tmp_path / "scales.json"),
        mm_encoder_fp8_scale_save_every_wrap=True,
    )
    assert cfg.mm_encoder_fp8_scale_save_every_wrap


@pytest.mark.parametrize("every_wrap", [False, True])
def test_scale_file_written_once_per_wrap(tmp_path, monkeypatch, every_wrap) -> None:
    """Layers wrap in the same forward; the file is written once per wrap, by
    the last layer, with every layer's complete window (not once per layer)."""
    save_file = tmp_path / "scales.json"
    mod, renames = _save_scales_state(monkeypatch, save_file)
    monkeypatch.setattr(mod, "_fp8_scale_save_every_wrap", every_wrap)
    num_layers, num_forwards = 8, 2 * _FP8_AMAX_HISTORY_LEN
    layers = [f"visual.blocks.{i}.attn.attn" for i in range(num_layers)]
    for step in range(num_forwards):
        wrapped = step % _FP8_AMAX_HISTORY_LEN == _FP8_AMAX_HISTORY_LEN - 1
        for i, name in enumerate(layers):
            s = torch.tensor(float(step * num_layers + i))
            mod._maybe_save_fp8_scales(name, s, s, s, wrapped)

    assert len(renames) == (2 if every_wrap else 1)
    last_step = _FP8_AMAX_HISTORY_LEN - 1 if not every_wrap else num_forwards - 1
    assert renames[-1][2] == {
        name: {k: 2.0 * (last_step * num_layers + i) for k in "qkv"}
        for i, name in enumerate(layers)
    }


@pytest.mark.parametrize("first_rank", [True, False])
def test_scale_save_armed_on_first_rank_only(tmp_path, monkeypatch, first_rank):
    """Only the global first rank writes the scale file, so the result does
    not depend on which rank of a data-parallel encoder writes last."""
    import vllm.model_executor.layers.attention.mm_encoder_attention as mod
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.config.multimodal import MultiModalConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    monkeypatch.setattr(mod, "_fp8_scale_save_path", None)
    monkeypatch.setattr(mod, "_fp8_scale_save_every_wrap", False)
    monkeypatch.setattr(mod, "is_global_first_rank", lambda: first_rank)
    monkeypatch.setattr(
        mod, "is_flashinfer_cudnn_fp8_prefill_attn_supported", lambda: True
    )
    monkeypatch.setattr(mod, "_get_flashinfer_workspace_buffer", lambda: None)
    monkeypatch.setattr(
        mod, "get_vit_attn_backend", lambda **_: AttentionBackendEnum.FLASHINFER
    )
    save_file = tmp_path / "scales.json"
    vllm_config = VllmConfig()
    vllm_config.model_config = MagicMock(
        spec=ModelConfig,
        multimodal_config=MultiModalConfig(
            mm_encoder_attn_dtype="fp8",
            mm_encoder_fp8_scale_save_path=str(save_file),
            mm_encoder_fp8_scale_save_every_wrap=True,
        ),
    )
    with set_current_vllm_config(vllm_config):
        mod.MMEncoderAttention(num_heads=NUM_HEADS, head_size=HEAD_DIM, prefix=LAYER_0)

    if first_rank:
        assert mod._fp8_scale_save_path == str(save_file)
        assert mod._fp8_scale_save_every_wrap
    else:
        assert mod._fp8_scale_save_path is None
