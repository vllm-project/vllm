# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Completeness check for the hardware taxonomy.

Every hardware judgement routes through ci_selector.codemap.hardware; this test pins
the table against the devices present in the job YAML at HEAD, so a new device
lands here loudly instead of silently having no family.
"""

import subprocess

import pytest
from ci_selector.codemap import hardware
from ci_selector.codemap.pipeline.buildkite import load_pipeline_configs, load_steps
from ci_selector.codemap.pipeline.step import LoadReport
from ci_selector.codemap.state import _gpu_name_aliases
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
    assert len(devices) >= 10, drift_message(
        f"Only {len(devices)} devices were read out of the job yaml.",
        "Family routing keys off these. Reading none maps none, and an empty "
        "set satisfies the check below exactly like a fully mapped one.",
        "the yaml moved or changed shape: check load_pipeline_configs and "
        "load_steps against .buildkite/ at HEAD",
    )
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
    """Data-file device tags: platform tokens first, then the vendor named in
    the device_name= field or in a vendor-named stem."""
    f = hardware.family_of_filename
    assert f("device_name=AMD_Instinct_MI325X,cache_dtype=float16.json") == "amd"
    assert (
        f(
            "nvidia_b200.json",
            "vllm/kernels/helion/configs/silu_mul_fp8/nvidia_b200.json",
        )
        == "cuda"
    )
    # Only under the helion tree: a vendor-named asset anywhere else is not a
    # device-named config, or it scopes out of every queue of that vendor.
    assert f("nvidia_b200.json", "vllm/notes/nvidia_b200.json") is None
    # A bare device-named file outside the helion tree stays unscoped: no
    # family, no narrowing, over-select. Only the field carries authority.
    assert f("NVIDIA_H200.json") is None
    assert f("E=8,N=1792,device_name=NVIDIA_GB200.json") == "cuda"
    assert f("zzz_probe.json") is None
    assert f("mixtral_moe.json") is None  # names no vendor


def test_device_name_of_filename():
    """The device a data filename names, read as the loaders read it: the
    `device_name=` field, else a vendor-named stem, else nothing."""
    d = hardware.device_name_of_filename
    assert d("E=8,N=3584,device_name=NVIDIA_H200.json") == "NVIDIA_H200"
    assert d("E=256,device_name=NVIDIA_H20-3e,dtype=fp8.json") == "NVIDIA_H20-3e"
    assert (
        d(
            "nvidia_h100.json",
            "vllm/kernels/helion/configs/silu_mul_fp8/nvidia_h100.json",
        )
        == "nvidia_h100"
    )
    assert d("nvidia_h100.json", "vllm/notes/nvidia_h100.json") is None
    assert d("E=8,N=3584.json") is None
    # A stem naming no vendor is not a device: this one used to resolve to the
    # amd `mi` prefix and scope a chat template to AMD.
    assert d("template_minicpmv45.jinja") is None


def test_device_scoped_out():
    """A step is scoped out of a device-named file only when its queue reports a
    known different device. Every unknown keeps the step."""
    from types import SimpleNamespace as St

    out = hardware.device_scoped_out
    aliases = {"nvidia_h100_80gb_hbm3": "nvidia_h100"}
    h200 = St(device="h200_35gb", mirror_hw=None)
    b200 = St(device="b200-k8s", mirror_hw=None)
    h100 = St(device="h100", mirror_hw=None)
    mi = St(device="mi300_1", mirror_hw=None)
    mi355 = St(device="mi355_1", mirror_hw=None)
    nodevice = St(device=None, mirror_hw=None)
    amd_mirror = St(device="h200_35gb", mirror_hw="amd")

    assert not out(h200, "NVIDIA_H200", aliases)
    assert out(b200, "NVIDIA_H200", aliases)  # same family, other device
    assert out(mi, "NVIDIA_H200", aliases)  # cross-family
    assert out(amd_mirror, "NVIDIA_H200", aliases)  # runs on amd whatever it lists
    assert not out(nodevice, "NVIDIA_H200", aliases)  # no device -> kept

    # The collisions a prefix match cannot express.
    assert out(h200, "NVIDIA_H20", aliases)  # "h200_35gb".startswith("h20")
    assert out(St(device="l4", mirror_hw=None), "NVIDIA_L40S", aliases)  # l4/l40s

    # A queue with no alias row keeps the step rather than guessing.
    assert not out(mi355, "AMD_Instinct_MI355X", aliases)

    # Helion configs are filed under the canonicalized name, so the h100 queue
    # matches one.
    assert not out(h100, "nvidia_h100", aliases)
    assert out(h100, "nvidia_b200", aliases)
    # An amd file: the mi300 queue and every amd mirror keep it, cuda drops it.
    assert not out(mi, "AMD_Instinct_MI300X", aliases)
    assert not out(amd_mirror, "AMD_Instinct_MI300X", aliases)
    assert out(h200, "AMD_Instinct_MI300X", aliases)


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
        source = vllm_repo / importer
        assert source.is_file(), drift_message(
            f"{importer} no longer exists, so the exception it carries for "
            f"{member} vouches for nothing.",
            cost,
            f"the file moved: update the key in EXCLUSIVE_IMPORT_EXCEPTIONS in {HW}",
            f"the file is gone for good: delete the entry from {HW}",
        )
        src = source.read_text()
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


@pytest.mark.drift
def test_device_tables_still_match_devices_at_head(vllm_repo):
    """Dead-entry side of the device tables, which every sibling list has.

    `test_every_device_at_head_has_a_family` only asks that each live device
    finds a row. A row matching no device is the other direction: harmless on
    its own, but it is how a table drifts into naming hardware CI retired.

    Exact names only. `FAMILY_DEVICE_PREFIXES` is a rule for whatever queue
    names a family may use, not a claim that one exists today, so an unmatched
    prefix is not dead: `tpu` matches nothing at HEAD and should still be there
    the day a tpu queue comes back.
    """
    from ci_selector.handwritten import FAMILY_DEVICE_EXACT, INFRA_DEVICES

    report = LoadReport()
    devices = {
        step.device
        for config in load_pipeline_configs(vllm_repo)
        for step in load_steps(vllm_repo, config, report)
        if step.device
    }
    dead = [f"INFRA_DEVICES: {d}" for d in sorted(INFRA_DEVICES) if d not in devices]
    dead += [
        f"FAMILY_DEVICE_EXACT[{family}]: {d}"
        for family, exact in sorted(FAMILY_DEVICE_EXACT.items())
        for d in sorted(exact)
        if d not in devices
    ]
    assert not dead, drift_message(
        f"Device table entries match no device in the job yaml: {dead}",
        "Each entry exists to tag a real queue. One that tags nothing is "
        "carrying a name CI stopped using, and it hides the day that name "
        "comes back meaning different hardware.",
        f"the device was renamed: update the entry in {HW}",
        "the hardware is retired FOR GOOD: delete the entry from "
        + HW
        + ". A queue that is only paused should keep its row, or it comes back "
        "with no family and nothing says so",
    )


@pytest.mark.drift
def test_every_device_name_in_the_tree_resolves(vllm_repo):
    """Every device a data filename names must resolve to a family. The guards
    above walk the other way, queue -> family; nothing walked this direction."""
    names = subprocess.check_output(
        ["git", "-C", str(vllm_repo), "ls-files", "vllm/"], text=True
    ).split()
    devices = {
        d
        for n in names
        if (d := hardware.device_name_of_filename(n.rsplit("/", 1)[-1], n))
    }
    assert len(devices) >= 25, drift_message(
        f"only {len(devices)} device names were read out of the tree",
        "The resolver is the thing under test here. Reading none would satisfy "
        "the check below exactly like reading them all and placing them.",
        "the tuning files moved or changed shape: check "
        "hardware.device_name_of_filename against vllm/ at HEAD",
    )
    unplaced = sorted(d for d in devices if hardware.family_of_device_name(d) is None)
    assert not unplaced, drift_message(
        f"Tuning files name devices with no family: {unplaced}",
        "A device-named data file with no family gets no device scope, so it "
        "falls back to its package's whole reverse closure. Inside the import "
        "cycle that is every test in the repo, for one JSON edit.",
        f"a new vendor: add its token to DEVICE_NAME_FAMILIES in {HW}",
    )


@pytest.mark.drift
def test_queue_alias_rows_still_describe_real_hardware(vllm_repo):
    """Shape guard on the alias table. The mapping itself is a fact about real
    machines that no test can check, so check both ends: the queue must be live
    in the yaml, and the device name must be one some tuning file carries."""
    from ci_selector.handwritten import QUEUE_DEVICE_NAMES

    report = LoadReport()
    live = {
        step.device
        for config in load_pipeline_configs(vllm_repo)
        for step in load_steps(vllm_repo, config, report)
        if step.device
    }
    # A value pasted onto the wrong key satisfies both legs below: the queue is
    # live and the device is filed, just not by each other.
    crossed = sorted(
        f"{q} -> {dev}"
        for q, dev in QUEUE_DEVICE_NAMES.items()
        if hardware.family_of_device(q) != hardware.family_of_device_name(dev)
    )
    assert not crossed, drift_message(
        f"Alias rows pair a queue with another family's device: {crossed}",
        "The row then scopes every file of the queue's own family out of it, "
        "and each half looks correct on its own.",
        f"fix the pairing in {HW}",
    )

    dead_queues = sorted(q for q in QUEUE_DEVICE_NAMES if q not in live)
    assert not dead_queues, drift_message(
        f"Alias rows name queues that no longer exist: {dead_queues}",
        "The row scopes nothing, and it hides the day that queue name returns "
        "pointing at different hardware.",
        f"the queue was renamed or retired: update or delete the row in {HW}",
    )

    names = subprocess.check_output(
        ["git", "-C", str(vllm_repo), "ls-files", "vllm/"], text=True
    ).split()
    filed = {
        d
        for n in names
        if (d := hardware.device_name_of_filename(n.rsplit("/", 1)[-1], n))
    }
    aliases = _gpu_name_aliases(vllm_repo)
    # The table has to have loaded, or the canonical leg below decides nothing
    # and this reads green while testing a single comparison.
    assert aliases.get("nvidia_h100_80gb_hbm3") == "nvidia_h100", drift_message(
        "vLLM's _GPU_NAME_ALIASES no longer canonicalizes the H100 SXM name",
        "The helion configs are filed under the canonical name, so without the "
        "table they scope out of the only queue that can load them.",
        "the table moved or was renamed: check _gpu_name_aliases in state.py "
        "against vllm/kernels/helion/utils.py",
    )
    # l4 and gh200 have no tuning files of their own -- the row is what scopes
    # every other device's files out of them. Asserted as a positive control
    # rather than skipped: the day such a file lands, that row starts scoping
    # it out of its own queue and this has to say so.
    expected_unfiled = {"gh200", "l4"}
    for q in sorted(expected_unfiled):
        dev = QUEUE_DEVICE_NAMES[q]
        assert dev not in filed and (
            hardware.canonicalize_gpu_name(dev, aliases) not in filed
        ), drift_message(
            f"A tuning file now names {dev}, the device the {q} row exempts",
            "That row was written on the premise nothing is filed under it. A "
            "file named for it is now scoped out of its own queue.",
            f"re-verify the {q} row against a CI job log, then drop it from "
            "expected_unfiled in this test",
        )
    unfiled = sorted(
        q
        for q, dev in QUEUE_DEVICE_NAMES.items()
        if q not in expected_unfiled
        and dev not in filed
        and hardware.canonicalize_gpu_name(dev, aliases) not in filed
    )
    assert not unfiled, drift_message(
        f"Alias rows name devices no tuning file carries: {unfiled}",
        "Either the row misspells what the hardware reports, in which case it "
        "scopes that device's own files out of its own queue and a test is "
        "lost, or the device genuinely has no tuning files and the row only "
        "scopes others out, which is fine but should be deliberate.",
        f"check the spelling against a real CI job log, then fix the row in {HW}",
        "the device has no tuning files on purpose: add it to the expected "
        "list in this test",
    )


@pytest.mark.drift
def test_stem_named_configs_still_resolve_to_a_device(vllm_repo):
    """The helion tree names its configs for the device instead of carrying a
    `device_name=` field, because its shape keys live inside the file rather
    than in the filename. `device_name_of_filename` reads the stem only under
    that one path, so if the tree moves those files silently lose their scope
    and gain back their package's whole reverse closure.
    """
    from ci_selector.codemap.hardware import _STEM_NAMED_DIR

    configs = [
        n
        for n in subprocess.check_output(
            ["git", "-C", str(vllm_repo), "ls-files", _STEM_NAMED_DIR], text=True
        ).split()
        if n.endswith(".json")
    ]
    assert len(configs) >= 10, drift_message(
        f"{_STEM_NAMED_DIR} holds {len(configs)} json configs, expected at least 10",
        "That tree is the only place a filename stem counts as a device. Empty, "
        "the rule is dead and any config left behind runs the whole reverse "
        "closure of its package for a one-file edit.",
        "the helion configs moved: update _STEM_NAMED_DIR in "
        "ci_selector/codemap/hardware.py",
    )
    unscoped = sorted(
        n.removeprefix(_STEM_NAMED_DIR)
        for n in configs
        if hardware.device_name_of_filename(n.rsplit("/", 1)[-1], n) is None
    )
    assert not unscoped, drift_message(
        f"Stem-named configs no longer resolve to a device: {unscoped[:5]}",
        "Each one loses its device scope and gains back every step its package "
        "reaches, which inside the import cycle is most of the pipeline.",
        "the naming convention changed: check canonicalize_gpu_name and "
        "DEVICE_NAME_FAMILIES against vllm/kernels/helion/",
    )
