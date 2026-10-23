# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Every shipped K1 / K2 build compiles for gfx950 (GPU-less: FlyDSL's
compile-only mode) through the runtime's own entry point
(``runtime.layers.compile_build``) and passes the ISA gate of a persistent,
all-resident kernel (design 5.13, DESIGN_BC 5.9): no private segment (a
spilling kernel can lose residency and deadlock), at most 256 VGPRs (2 waves a
SIMD), at most a CU's 160 KiB of LDS, and no readfirstlane waterfall (every
mailbox and weight load has a wave-uniform buffer resource). A K1 build also
holds one CTA a CU (more than 128 VGPRs or more than half a CU's LDS)."""

import glob
import os

import pytest
import regex as re
import torch

pytest.importorskip("flydsl")
pytest.importorskip("aiter")

from vllm.models.muse_spark.amd.specialized_decode.mono import spec  # noqa: E402
from vllm.models.muse_spark.amd.specialized_decode.mono.common.sync import (  # noqa: E402
    WAIT_TICKS,
)


def _compiles_for_gfx950() -> bool:
    """A GPU-less compile must target gfx950: FlyDSL falls back to gfx942
    without a device and then fails to select gfx950's permlane swaps."""
    if (os.environ.get("FLYDSL_GPU_ARCH") or os.environ.get("ARCH")) == "gfx950":
        return True
    if not torch.cuda.is_available():
        return False
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


_W = range(1, spec.MAX_ROWS + 1)
# every build the runtime serves (Phase A, TEP8's expert-parallel experts, B1's
# K1, B2's gated and C's split cores, at every width and attention kind) and the
# stalled-rank fault build test_numerics launches
BUILDS = [
    *(spec.K2Build(tokens=s) for s in _W),
    *(spec.K2Build(tokens=s, moe="ep") for s in _W),
    spec.K2Build(
        tokens=8,
        fault_rank=3,
        fault_epoch=2,
        fault_slot=spec.k2_slot(1),
        fault_ticks=WAIT_TICKS * 3 // 2,
    ),
    *(spec.K2Build(tokens=s, core="gated") for s in _W),
    *(
        spec.K2Build(tokens=s, core="split", window=w)
        for s in _W
        for w in (False, True)
    ),
    *(spec.K1Build(tokens=s, rope=r) for s in _W for r in (False, True)),
]

# a 64-bit compare of a VGPR buffer resource against its first lane's: a
# readfirstlane waterfall around a load whose base is not wave-uniform
_WATERFALL = re.compile(r"v_cmp_eq_u64_e\d+\s+\S+,\s*s\[\d+:\d+\],\s*v\[")


def _resource(txt: str, name: str) -> int:
    return int(re.search(rf"\.amdhsa_{name}\s+(\d+)", txt).group(1))


@pytest.mark.skipif(
    not _compiles_for_gfx950(),
    reason="GPU-less compile needs FLYDSL_GPU_ARCH=gfx950 (or a gfx950 GPU)",
)
@pytest.mark.parametrize("key", BUILDS, ids=lambda k: k.kernel_name())
def test_build_passes_the_isa_gate(key, tmp_path, monkeypatch):
    from vllm.models.muse_spark.amd.specialized_decode.mono import k1, layer
    from vllm.models.muse_spark.amd.specialized_decode.runtime.layers import (
        compile_build,
    )

    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(tmp_path))
    (k1 if isinstance(key, spec.K1Build) else layer)._BUILDS.pop(key, None)
    try:
        compile_build(key)
    except RuntimeError as e:  # the load, past the compile, needs a device
        if "hipErrorNoDevice" not in str(e) and "no ROCm" not in str(e):
            raise
    name = key.kernel_name()
    for path in glob.glob(f"{tmp_path}/**/*final_isa.s", recursive=True):
        with open(path) as f:
            txt = f.read()
        if re.search(rf"\.amdhsa_kernel {re.escape(name)}\b", txt):
            break
    else:
        raise AssertionError(f"no final ISA of {name} under {tmp_path}")
    vgpr = _resource(txt, "next_free_vgpr")
    lds = _resource(txt, "group_segment_fixed_size")
    assert _resource(txt, "private_segment_fixed_size") == 0, name
    assert vgpr <= spec.VGPR_MAX, (name, vgpr)
    assert lds <= spec.LDS_BYTES, (name, lds)
    if isinstance(key, spec.K1Build):
        assert vgpr > 128 or lds > spec.LDS_BYTES // 2, (name, vgpr, lds)
    loops = _WATERFALL.findall(txt)
    assert not loops, f"{name}: {len(loops)} waterfall compares, e.g. {loops[0]}"
