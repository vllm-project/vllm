# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Completeness check for the hardware taxonomy.

Every hardware judgement routes through ci_analyzer.hardware; this test pins
the table against the devices present in the job YAML at HEAD, so a new device
lands here loudly instead of silently having no family.
"""

from ci_analyzer import hardware
from ci_analyzer.curated import INFRA_DEVICES
from ci_analyzer.jobs.buildkite import load_pipeline_configs, load_steps
from ci_analyzer.jobs.model import LoadReport


def test_every_device_at_head_has_a_family(repo):
    report = LoadReport()
    devices = set()
    for config in load_pipeline_configs(repo):
        for step in load_steps(repo, config, report):
            if step.device:
                devices.add(step.device)
    unmapped = {
        d
        for d in devices
        if hardware.family_of_device(d) is None and d not in INFRA_DEVICES
    }
    assert not unmapped, (
        f"devices with no family: {sorted(unmapped)}; add them to "
        "hardware.py so every consumer (tagging, scoping, replica) agrees"
    )


def test_exclusive_namespaces_never_claim_gpu_worker():
    """Regression: cpu_worker.py imports gpu_worker.Worker, so the gpu
    worker namespace must never be hardware-exclusive."""
    assert hardware.exclusive_family_of_path("vllm/v1/worker/gpu_worker.py") is None
    assert (
        hardware.exclusive_family_of_path("vllm/v1/worker/gpu/model_runner.py") is None
    )
    assert hardware.exclusive_family_of_path("vllm/v1/kv_offload/cpu/common.py") is None
    assert hardware.exclusive_family_of_path("csrc/cpu/cpu_attn.cpp") == "cpu"
    assert hardware.exclusive_family_of_path("csrc/rocm/attention.cu") == "amd"


def test_rocm_basename_is_amd_exclusive():
    """A rocm-named file outside csrc/rocm/ is amd-exclusive by basename, but the
    additive aiter token alone is not (exclusion keys on namespace/rocm-name)."""
    assert (
        hardware.exclusive_family_of_path("vllm/attention/ops/rocm_aiter_mla.py")
        == "amd"
    )
    assert hardware.exclusive_family_of_path("vllm/attention/ops/aiter_mla.py") is None


def test_family_of_device_spot_checks():
    assert hardware.family_of_device("h200_35gb") == "cuda"
    assert hardware.family_of_device("b200-k8s") == "cuda"
    assert hardware.family_of_device("mi300_4") == "amd"
    assert hardware.family_of_device("amd_cpu") == "cpu"
    assert hardware.family_of_device("intel_gpu") == "xpu"
    assert hardware.family_of_device("cpu-small") is None


def test_family_of_filename():
    """Data-file device tags: device_name= fields, bare platform names, and
    the digit-guard that keeps 'mixtral' from matching the 'mi' amd prefix."""
    f = hardware.family_of_filename
    assert f("device_name=AMD_Instinct_MI325X,cache_dtype=float16.json") == "amd"
    assert f("nvidia_b200.json") == "cuda"
    assert f("NVIDIA_H200.json") == "cuda"
    assert f("NVIDIA_GB200.json") == "cuda"  # b200 substring of gb200
    assert f("zzz_probe.json") is None
    assert f("mixtral_moe.json") is None  # 'mi' prefix, no digit -> no match


def test_device_prefix_of_filename():
    """The finer device prefix a config filename names, for exact-device
    scoping (vs family_of_filename's family). None -> no device token."""
    p = hardware.device_prefix_of_filename
    assert p("E=8,N=3584,device_name=NVIDIA_H200.json") == "h200"
    assert p("device_name=NVIDIA_B200.json") == "b200"
    assert p("device_name=AMD_Instinct_MI300X.json") == "mi"
    assert p("E=8,N=3584.json") is None  # no device -> fall back to family


def test_device_scoped_out():
    """A step is scoped out of a device-named file only when its device is a
    KNOWN different device; unknown/None devices are kept (conservative)."""
    from types import SimpleNamespace as St

    out = hardware.device_scoped_out
    h200 = St(device="h200_35gb", mirror_hw=None)
    b200 = St(device="b200-k8s", mirror_hw=None)
    mi = St(device="mi300_1", mirror_hw=None)
    unknown = St(device=None, mirror_hw=None)
    amd_mirror = St(device="h200_35gb", mirror_hw="amd")
    # file scoped to h200
    assert not out(h200, "h200")  # same device kept
    assert out(b200, "h200")  # same family, different prefix -> dropped
    assert out(mi, "h200")  # cross-family -> dropped
    assert out(amd_mirror, "h200")  # runs on amd hardware -> dropped
    assert not out(unknown, "h200")  # unknown queue kept
    # file scoped to mi (amd): the amd-mirror step is kept
    assert not out(mi, "mi")
    assert not out(amd_mirror, "mi")
    assert out(h200, "mi")


def test_exclusivity_disable_is_derived_invariant(state):
    """Soundness net for subtractive exclusion, re-derived independently from the
    live graph (no snapshot): a file is scoped to its family only when nothing
    outside that family imports it at module level, else exclusion fails open.
    Catches exclusivity_violations regressions without pinning a drifting file
    list."""
    from ci_analyzer.curated import EXCLUSIVE_IMPORT_EXCEPTIONS
    from ci_analyzer.hardware import exclusive_family_of_path

    pr = state.full.plain_reverse

    def has_cross_family_importer(f):
        family = exclusive_family_of_path(f)
        for importer in pr.get(f, ()):
            if EXCLUSIVE_IMPORT_EXCEPTIONS.get((importer, f)):
                continue
            if exclusive_family_of_path(importer) == family:
                continue
            return True
        return False

    expected = {
        f
        for f in state.full.index.file_to_module
        if exclusive_family_of_path(f) is not None and has_cross_family_importer(f)
    }
    disabled = set(state.exclusive_disabled)
    assert disabled == expected, {
        "wrongly scoped (under-selection)": sorted(expected - disabled),
        "stale disable (no live cross importer)": sorted(disabled - expected),
    }
    for (importer, member), _guard in EXCLUSIVE_IMPORT_EXCEPTIONS.items():
        assert importer in pr.get(member, set()), (
            f"dead exception entry: {importer} no longer imports {member}"
        )


def test_exclusive_import_exception_guards_still_present(repo):
    """Each cited guard call must still exist in the importer's source; if upstream
    makes the cross-family import unconditional, subtractive exclusion silently
    under-selects. Machine-check the guard text rather than trust a comment."""
    import regex as re
    from ci_analyzer.curated import EXCLUSIVE_IMPORT_EXCEPTIONS

    for (importer, _member), citation in EXCLUSIVE_IMPORT_EXCEPTIONS.items():
        quoted = re.search(r"`([^`]*)`", citation)
        assert quoted, f"citation missing a backticked guard: {citation}"
        call = re.search(r"[A-Za-z_][\w.]*\(", quoted.group(1))
        assert call, f"no guard call in citation: {citation}"
        src = (repo / importer).read_text()
        assert call.group(0) in src, (
            f"guard {call.group(0)!r} gone from {importer}; exclusion may be unsound"
        )


def test_no_tpu_worker_namespace_prefix(repo):
    """The live counterexample that forbids a vllm/v1/worker/tpu prefix:
    the shared LoRA path imports tpu_input_batch at module level."""
    from ci_analyzer.hardware import exclusive_family_of_path

    src = (repo / "vllm/v1/worker/lora_model_runner_mixin.py").read_text()
    assert "tpu_input_batch" in src, (
        "counterexample gone: a worker/tpu prefix may now be sound; "
        "re-probe before adding one"
    )
    assert exclusive_family_of_path("vllm/v1/worker/tpu_input_batch.py") is None


def test_basename_token_scoped_to_source_extensions():
    """The rocm-basename heuristic is a Python/shell source convention: a foreign
    workspace (rust/) names files freely, so it is excluded (the allowlist fails
    safe)."""
    assert hardware.exclusive_family_of_path("rust/src/rocm_support.rs") is None
    sh = "tools/install_torchcodec_rocm.sh"
    assert hardware.exclusive_family_of_path(sh) == "amd"


def test_mirror_runs_on_its_own_hardware_whatever_device_it_lists():
    """A mirror block that omits `device:` inherits the parent's, so judging it
    by device would exclude an AMD mirror from AMD-exclusive files. Every mirror
    sets one today; this pins the polarity for the first that does not."""
    from types import SimpleNamespace as St

    excluded = hardware.device_excluded_for_path
    rocm = "tests/kernels/test_rocm_thing.py"
    inherited = St(device="h100", mirror_hw="amd")
    cuda = St(device="h100", mirror_hw=None)
    assert not excluded(rocm, inherited.device, inherited)
    assert excluded(rocm, cuda.device, cuda)
    # and the mirror IS excluded from a family that is not its own
    assert excluded("vllm/v1/worker/xpu_worker.py", inherited.device, inherited)
