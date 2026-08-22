# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Completeness check for the hardware taxonomy.

Every hardware judgement routes through ci_selector.codemap.hardware; this test pins
the table against the devices present in the job YAML at HEAD, so a new device
lands here loudly instead of silently having no family.
"""

import pytest
from ci_selector.codemap import hardware
from ci_selector.codemap.pipeline.buildkite import load_pipeline_configs, load_steps
from ci_selector.codemap.pipeline.step import LoadReport
from ci_selector.handwritten import INFRA_DEVICES
from helpers import HW, drift_message


@pytest.mark.drift
def test_every_device_at_head_has_a_family(vllm_repo):
    report = LoadReport()
    devices = set()
    for config in load_pipeline_configs(vllm_repo):
        for step in load_steps(vllm_repo, config, report):
            if step.device:
                devices.add(step.device)
    unmapped = {
        d
        for d in devices
        if hardware.family_of_device(d) is None and d not in INFRA_DEVICES
    }
    assert not unmapped, drift_message(
        f"Job yaml names devices with no hardware family: {sorted(unmapped)}",
        "Family tagging, exclusive-namespace scoping and the zero-jobs rule all "
        "key off the family, so an unmapped device silently opts out of all "
        "three.",
        f"real test hardware: add it to FAMILY_DEVICE_PREFIXES or "
        f"FAMILY_DEVICE_EXACT in {HW}",
        f"a build runner rather than test hardware: add it to INFRA_DEVICES in {HW}",
    )


@pytest.mark.drift
def test_exclusive_namespaces_still_match_files_at_head(vllm_repo):
    """Anti-vacuity for the subtractive table.

    `test_exclusivity_disable_is_derived_invariant` checks the rule is applied
    soundly, but it re-derives from the same live graph, so a namespace that
    moved upstream reads as "nothing there" on both sides and passes. Only this
    test notices that an entry stopped matching anything.
    """
    from ci_selector.handwritten import EXCLUSIVE_NAMESPACES

    dead = []
    for prefixes, exact, family in EXCLUSIVE_NAMESPACES:
        dead += [(p, family) for p in prefixes if not any(vllm_repo.glob(p + "*"))]
        dead += [(e, family) for e in exact if not (vllm_repo / e).is_file()]
    assert not dead, drift_message(
        "EXCLUSIVE_NAMESPACES entries match no file at HEAD: "
        + ", ".join(f"{p} ({fam})" for p, fam in dead),
        "These entries are what scope a file to one hardware family. An entry "
        "matching nothing scopes nothing, so those files go back to running on "
        "every platform. Over-selection, but silent, and it hides the day the "
        "namespace genuinely stops being single-platform.",
        f"the path moved or was renamed in vLLM: update EXCLUSIVE_NAMESPACES in {HW}",
        f"the namespace is gone for good: delete the entry from {HW}",
    )


@pytest.mark.drift
def test_path_token_families_cover_every_device_family():
    """Cross-check the unguarded table against the guarded one.

    `FAMILY_DEVICE_*` is pinned to the live job yaml, so a new family lands
    there loudly. Nothing pins `PATH_TOKEN_FAMILIES`, so the same family can go
    missing here in silence and every path carrying its name is left untagged.
    cuda is the deliberate exception: it is the default and needs no token.
    """
    from ci_selector.handwritten import (
        FAMILY_DEVICE_EXACT,
        FAMILY_DEVICE_PREFIXES,
        PATH_TOKEN_FAMILIES,
    )

    device_families = set(FAMILY_DEVICE_PREFIXES) | set(FAMILY_DEVICE_EXACT)
    token_families = {family for _tokens, family in PATH_TOKEN_FAMILIES}
    assert device_families - token_families == {"cuda"}, drift_message(
        "A device family has no path tokens: "
        f"{sorted(device_families - token_families - {'cuda'})}",
        "Path tagging is how a file reaches hardware when no device names it. "
        "A family with no tokens tags nothing, so its files read as "
        "platform-neutral and stop being scoped to it.",
        f"add a token row for the family to PATH_TOKEN_FAMILIES in {HW}",
    )
    assert not token_families - device_families, drift_message(
        "PATH_TOKEN_FAMILIES names a family no device maps to: "
        f"{sorted(token_families - device_families)}",
        "The family can never match a step, so the tokens tag files that "
        "nothing then selects on.",
        f"the family was renamed: fix the row in PATH_TOKEN_FAMILIES in {HW}",
        f"the hardware is gone: delete the row from {HW}",
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
    from ci_selector.codemap.hardware import exclusive_family_of_path
    from ci_selector.handwritten import EXCLUSIVE_IMPORT_EXCEPTIONS

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


@pytest.mark.drift
def test_exclusive_import_exception_guards_still_present(vllm_repo):
    """Each cited guard call must still exist in the importer's source; if upstream
    makes the cross-family import unconditional, subtractive exclusion silently
    under-selects. Machine-check the guard text rather than trust a comment."""
    import regex as re
    from ci_selector.handwritten import EXCLUSIVE_IMPORT_EXCEPTIONS

    cost = (
        "The exception says this cross-family import is safe because a runtime "
        "check guards it. With the guard gone the import runs everywhere, the "
        "file is no longer single-platform, and we still subtract its jobs. "
        "That is under-selection: a real failure never gets a job to fail in."
    )
    for (importer, member), citation in EXCLUSIVE_IMPORT_EXCEPTIONS.items():
        quoted = re.search(r"`([^`]*)`", citation)
        assert quoted, drift_message(
            f"The EXCLUSIVE_IMPORT_EXCEPTIONS entry for {importer} cites no "
            f"guard in backticks: {citation}",
            cost,
            f"quote the guarding condition in backticks in {HW}, so this test "
            "can machine-check it",
        )
        call = re.search(r"[A-Za-z_][\w.]*\(", quoted.group(1))
        assert call, drift_message(
            f"The guard cited for {importer} contains no call to look for: {citation}",
            cost,
            f"cite the actual runtime check, not a comment, in {HW}",
        )
        src = (vllm_repo / importer).read_text()
        assert call.group(0) in src, drift_message(
            f"{importer} no longer contains {call.group(0)!r}, the guard that "
            f"made its import of {member} safe.",
            cost,
            f"the guard moved or was renamed: update the citation in {HW}",
            f"the import is now unconditional: delete the entry from {HW} and "
            "the exclusion disables itself",
        )


def test_no_tpu_worker_namespace_prefix(vllm_repo):
    """The live counterexample that forbids a vllm/v1/worker/tpu prefix:
    the shared LoRA path imports tpu_input_batch at module level."""
    from ci_selector.codemap.hardware import exclusive_family_of_path

    src = (vllm_repo / "vllm/v1/worker/lora_model_runner_mixin.py").read_text()
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
    sets one today; this pins the direction for the first that does not."""
    from types import SimpleNamespace as St

    excluded = hardware.device_excluded_for_path
    rocm = "tests/kernels/test_rocm_thing.py"
    inherited = St(device="h100", mirror_hw="amd")
    cuda = St(device="h100", mirror_hw=None)
    assert not excluded(rocm, inherited.device, inherited)
    assert excluded(rocm, cuda.device, cuda)
    # and the mirror IS excluded from a family that is not its own
    assert excluded("vllm/v1/worker/xpu_worker.py", inherited.device, inherited)
