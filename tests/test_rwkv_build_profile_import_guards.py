# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.build_profile import BuildProfileMetadata
from vllm.model_executor.layers.quantization.utils import w8a8_utils


def test_rwkv_profile_skips_unbuilt_stable_cutlass_probes(monkeypatch) -> None:
    metadata = BuildProfileMetadata(
        profile="rwkv",
        configured_targets=("_rapid_sampling", "rwkv7_ops"),
        external_projects=(),
        unrestricted=False,
    )
    monkeypatch.setattr(w8a8_utils, "get_build_profile_metadata", lambda: metadata)
    monkeypatch.setattr(w8a8_utils.current_platform, "is_cuda", lambda: True)

    def unexpected_probe(*args, **kwargs):
        raise AssertionError("reduced artifact must not call full CUTLASS ops")

    monkeypatch.setattr(
        w8a8_utils.ops, "cutlass_scaled_mm_supports_fp8", unexpected_probe
    )
    monkeypatch.setattr(
        w8a8_utils.ops, "cutlass_scaled_mm_supports_block_fp8", unexpected_probe
    )
    monkeypatch.setattr(
        w8a8_utils.ops, "cutlass_group_gemm_supported", unexpected_probe
    )

    assert not w8a8_utils.cutlass_fp8_supported()
    assert not w8a8_utils.cutlass_block_fp8_supported()
    assert not w8a8_utils.cutlass_group_gemm_supported()
