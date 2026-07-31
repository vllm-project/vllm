# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-free routing tests for the batch-invariant custom-allreduce path.

Validates, without any GPU, torch.distributed group, or engine build, that:
1. ``VLLM_BATCH_INVARIANT=1`` no longer forces ``disable_custom_all_reduce``
   (config layer);
2. the CUDA dispatch in ``csrc/custom_all_reduce.cuh`` isolates the 1-stage
   kernel under invariance and makes the 2-stage kernel unreachable — proven
   by EXECUTING the actual dispatch block, extracted verbatim from the header
   and compiled as host C++ with a stub kernel-launch macro;
3. the communicator drops non-audited allreduce backends (FlashInfer / AITER
   / QuickReduce) under invariance (source contract).

The host-compiled simulation exists so that a routing regression fails in
plain CPU CI, long before a multi-GPU determinism run would catch it.
"""

import pathlib
import shutil
import subprocess
import sys
import tempfile

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
CUH = REPO_ROOT / "csrc" / "custom_all_reduce.cuh"
COMMUNICATOR = (
    REPO_ROOT / "vllm" / "distributed" / "device_communicators" / "cuda_communicator.py"
)
BATCH_INVARIANT_PY = (
    REPO_ROOT / "vllm" / "model_executor" / "layers" / "batch_invariant.py"
)

# --------------------------------------------------------------------------- #
# 1. Config layer: invariance must not blanket-disable custom allreduce
# --------------------------------------------------------------------------- #


def test_parallel_config_not_forced_disabled(monkeypatch):
    torch = pytest.importorskip("torch")  # noqa: F841 (vllm imports need it)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    from vllm.config.parallel import ParallelConfig

    config = ParallelConfig()
    assert config.disable_custom_all_reduce is False, (
        "VLLM_BATCH_INVARIANT=1 must not blanket-disable custom allreduce; "
        "the 1-stage kernel is order-fixed and pinned by the dispatcher."
    )


# --------------------------------------------------------------------------- #
# 2. Kernel dispatch: execute the real decision block as host C++
# --------------------------------------------------------------------------- #


def _extract(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[start:end]


HARNESS_TEMPLATE = r"""
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace vllm {
inline bool vllm_is_batch_invariant() {
  const char* val = std::getenv("VLLM_BATCH_INVARIANT");
  return (val && std::atoi(val) != 0);
}
}  // namespace vllm

static const char* chosen = "none";
#define KL(ngpus, name) chosen = #name;

int main(int argc, char** argv) {
  // scenario inputs: world_size fully_connected bytes
  int world_size_ = std::atoi(argv[1]);
  bool fully_connected_ = std::atoi(argv[2]) != 0;
  long bytes = std::atol(argv[3]);
  try {
    // ---- begin verbatim extraction from csrc/custom_all_reduce.cuh ----
{ENV_AND_GATE_BLOCK}
{REDUCE_CASE_BLOCK}
    switch (world_size_) {
      REDUCE_CASE(2)
      REDUCE_CASE(4)
      REDUCE_CASE(6)
      REDUCE_CASE(8)
      default:
        throw std::runtime_error("unsupported world size");
    }
    // ---- end verbatim extraction ----
  } catch (const std::runtime_error& e) {
    std::printf("THROW\n");
    return 0;
  }
  std::printf("%s\n", chosen);
  return 0;
}
"""

# (world_size, fully_connected, bytes, VLLM_BATCH_INVARIANT, ALGO) -> expected
DISPATCH_SCENARIOS = [
    # Invariance forces 1-stage everywhere the 2-stage would otherwise win:
    ((8, 1, 1 << 20), "1", "", "cross_device_reduce_1stage"),
    ((4, 1, 1 << 20), "1", "", "cross_device_reduce_1stage"),
    ((6, 1, 1 << 20), "1", "", "cross_device_reduce_1stage"),
    ((2, 1, 4 << 10), "1", "", "cross_device_reduce_1stage"),
    # Invariance + explicit 2-stage request must hard-error:
    ((8, 1, 1 << 20), "1", "2stage", "THROW"),
    # Baseline (no invariance): existing heuristics must be untouched:
    ((8, 1, 1 << 20), "0", "", "cross_device_reduce_2stage"),
    ((8, 1, 100 << 10), "0", "", "cross_device_reduce_1stage"),
    ((4, 1, 256 << 10), "0", "", "cross_device_reduce_1stage"),
    ((4, 1, 1 << 20), "0", "", "cross_device_reduce_2stage"),
    ((2, 1, 1 << 20), "0", "", "cross_device_reduce_1stage"),
    ((8, 1, 1 << 20), "0", "2stage", "cross_device_reduce_2stage"),
    ((8, 1, 1 << 20), "0", "1stage", "cross_device_reduce_1stage"),
]


@pytest.mark.skipif(shutil.which("c++") is None, reason="no C++ compiler")
def test_dispatch_isolates_1stage_under_invariance(tmp_path):
    text = CUH.read_text()
    env_and_gate = _extract(
        text, "    // Check environment variable once", "#define KL"
    )
    reduce_case = _extract(text, "#define REDUCE_CASE", "    switch (world_size_)")

    src = HARNESS_TEMPLATE.replace("{ENV_AND_GATE_BLOCK}", env_and_gate).replace(
        "{REDUCE_CASE_BLOCK}", reduce_case
    )
    cpp = tmp_path / "dispatch_sim.cpp"
    binary = tmp_path / "dispatch_sim"
    cpp.write_text(src)
    subprocess.run(["c++", "-std=c++17", "-fsyntax-only", str(cpp)], check=True)
    subprocess.run(["c++", "-std=c++17", "-o", str(binary), str(cpp)], check=True)

    for (ws, fc, nbytes), invariant, algo, expected in DISPATCH_SCENARIOS:
        # vllm_is_batch_invariant() caches per process upstream; the sim
        # re-reads env, and each scenario is its own process anyway.
        env = {
            "VLLM_BATCH_INVARIANT": invariant,
            "PATH": "/usr/bin:/bin",
        }
        if algo:
            env["VLLM_CUSTOM_ALLREDUCE_ALGO"] = algo
        out = subprocess.run(
            [str(binary), str(ws), str(fc), str(nbytes)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert out == expected, (
            f"dispatch(world_size={ws}, fully_connected={fc}, "
            f"bytes={nbytes}, VLLM_BATCH_INVARIANT={invariant}, "
            f"ALGO={algo!r}) chose {out!r}, expected {expected!r}"
        )


# --------------------------------------------------------------------------- #
# 3. Communicator source contracts (no vllm import required)
# --------------------------------------------------------------------------- #


def test_cuh_gate_is_present_and_ordered():
    text = CUH.read_text()
    assert '#include "core/batch_invariant.hpp"' in text
    gate = text.index("vllm_is_batch_invariant()")
    dispatch = text.index("#define REDUCE_CASE")
    assert gate < dispatch, "invariance gate must precede kernel dispatch"
    gate_block = _extract(text, "if (vllm::vllm_is_batch_invariant())", "#define KL")
    assert "force_1stage = true" in gate_block
    assert "force_2stage" in gate_block and "throw" in gate_block


def test_communicator_drops_unaudited_backends_under_invariance():
    text = COMMUNICATOR.read_text()
    gate = _extract(
        text, "if envs.VLLM_BATCH_INVARIANT:", "self.use_custom_allreduce ="
    )
    assert "use_flashinfer_allreduce = False" in gate
    assert "use_aiter_allreduce = False" in gate
    qr_block = _extract(
        text, "if (\n            use_custom_allreduce", "QuickAllReduce("
    )
    assert "not envs.VLLM_BATCH_INVARIANT" in qr_block


def test_override_envs_pins_flashinfer_allreduce_off():
    text = BATCH_INVARIANT_PY.read_text()
    block = _extract(
        text, "def override_envs_for_invariance", "def init_batch_invariance"
    )
    assert 'os.environ["VLLM_ALLREDUCE_USE_FLASHINFER"] = "0"' in block


if __name__ == "__main__":
    # Minimal runner for environments without pytest (e.g. dev laptops):
    # executes the tests that need no vllm/torch install.
    import traceback

    class _TmpPath:
        def __enter__(self):
            self._d = tempfile.mkdtemp(prefix="dispatch_sim_")
            return pathlib.Path(self._d)

        def __exit__(self, *a):
            shutil.rmtree(self._d, ignore_errors=True)

    failures = 0
    for fn in (
        test_cuh_gate_is_present_and_ordered,
        test_communicator_drops_unaudited_backends_under_invariance,
        test_override_envs_pins_flashinfer_allreduce_off,
    ):
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    try:
        with _TmpPath() as tmp:
            test_dispatch_isolates_1stage_under_invariance(tmp)
        print(
            "PASS test_dispatch_isolates_1stage_under_invariance "
            f"({len(DISPATCH_SCENARIOS)} scenarios)"
        )
    except Exception:
        failures += 1
        print("FAIL test_dispatch_isolates_1stage_under_invariance")
        traceback.print_exc()
    sys.exit(1 if failures else 0)
