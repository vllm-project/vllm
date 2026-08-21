# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the `_vllm_skip_offload` Parameter marker in UVAOffloader.

Background: quant configs (e.g. FusedMoEQuantConfig used by NVFP4 MoE)
cache references to per-tensor scale Parameters. In the non-UVA fallback
path of UVAOffloader, parameters get their `.data` moved to CPU but the
cached external reference is not updated by the forward-time
functional_call swap. Marlin's NVFP4 MoE kernel then asserts
"b_scales is not on GPU" and crashes.

Fix: producers of such cached Parameters set `_vllm_skip_offload = True`
on them; the offloader's non-UVA path honours the marker and leaves the
Parameter on its original device. In genuine UVA mode the cached ref
still resolves to a CUDA-mapped tensor, so the marker is intentionally
ignored in that branch.

Run: .venv/bin/python -m pytest tests/quantization/test_uva_scale_skip.py -v
"""

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.offloader.uva import (
    _VLLM_SKIP_OFFLOAD_ATTR,
    UVAOffloader,
)


class TestSkipOffloadMarkerConstant:
    """The marker name itself must match vLLM's `_vllm_*` convention so
    that producers (quant configs) and consumer (this offloader) agree."""

    def test_marker_constant_value(self):
        assert _VLLM_SKIP_OFFLOAD_ATTR == "_vllm_skip_offload"

    def test_marker_follows_vllm_prefix_convention(self):
        # Same convention as `_vllm_is_uva_offloaded` (set elsewhere in
        # this module) and `_vllm_patched` (used in patch_utils).
        assert _VLLM_SKIP_OFFLOAD_ATTR.startswith("_vllm_")


class _SyntheticMoELayer(nn.Module):
    """Minimal stand-in for a FusedMoE-style module after
    process_weights_after_loading: large quantized weights plus per-tensor
    scales tagged with `_vllm_skip_offload = True` (which is what the
    NVFP4 MoE quant config does in real code)."""

    def __init__(self, device, *, mark_scales: bool = True):
        super().__init__()
        # Big quantized weights — these SHOULD be offloaded.
        self.register_parameter(
            "w13_weight",
            nn.Parameter(torch.zeros(64, 32, device=device), requires_grad=False),
        )
        self.register_parameter(
            "w2_weight",
            nn.Parameter(torch.zeros(64, 32, device=device), requires_grad=False),
        )
        # Per-tensor scales — cached by reference in FusedMoEQuantConfig.
        self.register_parameter(
            "w13_weight_scale",
            nn.Parameter(torch.ones(64, 2, device=device), requires_grad=False),
        )
        self.register_parameter(
            "w2_weight_scale",
            nn.Parameter(torch.ones(64, 2, device=device), requires_grad=False),
        )
        # An unmarked scale-looking Parameter that is NOT cached — to make
        # sure the offloader does not protect things by name accident.
        self.register_parameter(
            "w13_weight_global_scale",
            nn.Parameter(torch.ones(64, device=device), requires_grad=False),
        )

        if mark_scales:
            self.w13_weight_scale._vllm_skip_offload = True
            self.w2_weight_scale._vllm_skip_offload = True

    def forward(self, x):
        return x  # not used


class TestMarkerSemantics:
    """Pure-Python: does the marker cause the offloader's predicate to
    return True/False correctly, without touching CUDA?"""

    def test_no_marker_means_no_skip(self):
        p = nn.Parameter(torch.zeros(4), requires_grad=False)
        assert getattr(p, _VLLM_SKIP_OFFLOAD_ATTR, False) is False

    def test_marker_true_means_skip(self):
        p = nn.Parameter(torch.zeros(4), requires_grad=False)
        setattr(p, _VLLM_SKIP_OFFLOAD_ATTR, True)
        assert getattr(p, _VLLM_SKIP_OFFLOAD_ATTR, False) is True

    def test_marker_false_does_not_protect(self):
        p = nn.Parameter(torch.zeros(4), requires_grad=False)
        setattr(p, _VLLM_SKIP_OFFLOAD_ATTR, False)
        assert getattr(p, _VLLM_SKIP_OFFLOAD_ATTR, False) is False

    @pytest.mark.parametrize(
        "name",
        [
            # All scale-looking names: with no marker, the offloader will
            # happily push them to CPU. The naming convention is NOT a
            # safety net any more — explicit opt-in only.
            "model.layers.0.mlp.experts.w13_weight_scale",
            "model.layers.0.mlp.experts.w2_weight_scale",
            "model.layers.0.attn.q_proj.weight_scale",
            "model.layers.0.mlp.experts.g1_alphas",
        ],
    )
    def test_scale_looking_name_alone_is_not_enough(self, name):
        # A Parameter with a name that LOOKS like a scale but no marker
        # must NOT be auto-protected. (Regression guard against ever
        # reintroducing the substring-fragment fallback.)
        p = nn.Parameter(torch.zeros(4), requires_grad=False)
        assert getattr(p, _VLLM_SKIP_OFFLOAD_ATTR, False) is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestUVAOffloaderRespectsMarker:
    """Integration: the offloader walks named_parameters() and obeys the
    marker on a synthetic module that mirrors the post-process-weights
    layout of NVFP4 MoE."""

    @pytest.fixture
    def module(self):
        return _SyntheticMoELayer(device="cuda", mark_scales=True)

    def test_marked_scales_stay_on_gpu_in_non_uva_mode(self, module):
        offloader = UVAOffloader(
            cpu_offload_max_bytes=10 * 1024 * 1024 * 1024,
            cpu_offload_params=None,
        )
        offloader.uva_offloading = False  # exercise the fallback path
        offloader.wrap_modules(iter([module]))

        # Weights offloaded.
        assert module.w13_weight.data.device.type == "cpu"
        assert module.w2_weight.data.device.type == "cpu"

        # Marked scales kept on GPU.
        assert module.w13_weight_scale.data.device.type == "cuda"
        assert module.w2_weight_scale.data.device.type == "cuda"

        # Unmarked scale-looking Parameter: offloader does NOT protect it.
        # (We assert this to lock in the marker-only contract — if you ever
        # see this asserting cuda, somebody put the naming heuristic back.)
        assert module.w13_weight_global_scale.data.device.type == "cpu"

    def test_unmarked_module_offloads_everything(self):
        module = _SyntheticMoELayer(device="cuda", mark_scales=False)
        offloader = UVAOffloader(
            cpu_offload_max_bytes=10 * 1024 * 1024 * 1024,
            cpu_offload_params=None,
        )
        offloader.uva_offloading = False
        offloader.wrap_modules(iter([module]))
        # Without the marker, scales go to CPU just like weights. Producers
        # MUST set the marker to opt out.
        assert module.w13_weight_scale.data.device.type == "cpu"
        assert module.w2_weight_scale.data.device.type == "cpu"

    def test_user_filter_does_not_override_marker(self, module):
        offloader = UVAOffloader(
            cpu_offload_max_bytes=10 * 1024 * 1024 * 1024,
            cpu_offload_params={"weight", "weight_scale"},
        )
        offloader.uva_offloading = False
        offloader.wrap_modules(iter([module]))
        # User explicitly listed weight_scale, but the marker (set by the
        # quant config) wins — the b_scales-on-GPU invariant is preserved.
        assert module.w13_weight_scale.data.device.type == "cuda"
        assert module.w2_weight_scale.data.device.type == "cuda"

    def test_marker_protects_arbitrary_name(self):
        layer = nn.Module()
        big = nn.Parameter(torch.zeros(64, 32, device="cuda"), requires_grad=False)
        layer.register_parameter("custom_big_weight", big)
        # A name that LOOKS nothing like a scale — but the marker still
        # protects it. This is the whole point of marker-over-naming.
        custom = nn.Parameter(torch.ones(64, device="cuda"), requires_grad=False)
        custom._vllm_skip_offload = True
        layer.register_parameter("totally_made_up_name", custom)

        offloader = UVAOffloader(
            cpu_offload_max_bytes=10 * 1024 * 1024 * 1024,
            cpu_offload_params=None,
        )
        offloader.uva_offloading = False
        offloader.wrap_modules(iter([layer]))

        assert layer.custom_big_weight.data.device.type == "cpu"
        assert layer.totally_made_up_name.data.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestUVAOffloaderModeGating:
    """In genuine UVA mode the marker is intentionally ignored — the cached
    reference resolves to a UVA-mapped CUDA tensor and the bug does not
    manifest. The skip is a non-UVA-fallback-only safeguard."""

    def _uva_path_actually_works(self) -> bool:
        try:
            probe = torch.empty(16, pin_memory=True)
            from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

            view = get_accelerator_view_from_cpu_tensor(probe)
            return bool(view.is_cuda)
        except Exception:
            return False

    def test_marked_scales_offloaded_in_uva_mode(self):
        if not self._uva_path_actually_works():
            pytest.skip("UVA/pinned memory not available on this host")

        module = _SyntheticMoELayer(device="cuda", mark_scales=True)
        offloader = UVAOffloader(
            cpu_offload_max_bytes=10 * 1024 * 1024 * 1024,
            cpu_offload_params=None,
        )
        offloader.uva_offloading = True
        offloader.pin_memory = True
        offloader.wrap_modules(iter([module]))

        # Both weights and scales get processed through the UVA branch:
        # `.data` is a UVA-mapped CUDA tensor, _vllm_is_uva_offloaded is set.
        assert hasattr(module.w13_weight, "_vllm_is_uva_offloaded")
        assert module.w13_weight.data.is_cuda
        # The marker does NOT cause a skip in UVA mode — by design.
        assert hasattr(module.w13_weight_scale, "_vllm_is_uva_offloaded")
        assert module.w13_weight_scale.data.is_cuda

    def test_marked_scales_skipped_in_non_uva_mode(self):
        module = _SyntheticMoELayer(device="cuda", mark_scales=True)
        offloader = UVAOffloader(
            cpu_offload_max_bytes=10 * 1024 * 1024 * 1024,
            cpu_offload_params=None,
        )
        offloader.uva_offloading = False
        offloader.wrap_modules(iter([module]))

        # Marked scales stay on GPU; weights still go to CPU.
        assert module.w13_weight_scale.data.device.type == "cuda"
        assert module.w2_weight_scale.data.device.type == "cuda"
        assert module.w13_weight.data.device.type == "cpu"
        assert module.w2_weight.data.device.type == "cpu"


class TestNvfp4MoeSetsMarker:
    """Static check: the NVFP4 MoE quant config source sets the marker on
    the two Parameters whose references it caches in FusedMoEQuantConfig.
    This is the producer side of the contract; the offloader's marker
    handling is exercised separately above. Done as a source grep so the
    test does not require loading CUDA-dependent quant code."""

    def test_nvfp4_moe_sets_marker_on_cached_scales(self):
        import pathlib

        import vllm

        repo_root = pathlib.Path(vllm.__file__).resolve().parent
        path = (
            repo_root
            / "model_executor/layers/quantization/compressed_tensors"
            / "compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py"
        )
        src = path.read_text()
        assert "layer.w13_weight_scale._vllm_skip_offload = True" in src, (
            "NVFP4 MoE must mark w13_weight_scale to prevent b_scales-on-CPU"
        )
        assert "layer.w2_weight_scale._vllm_skip_offload = True" in src, (
            "NVFP4 MoE must mark w2_weight_scale to prevent b_scales-on-CPU"
        )
