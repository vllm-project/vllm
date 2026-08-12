# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hardware taxonomy: the single home for device and path family facts.

Every hardware judgement routes through this table: additive tagging (a
rocm-named file selects AMD jobs), subtractive scoping (a CPU-exclusive file
cannot affect GPU jobs), the generator replica's AMD-native rule, and the
completeness test that pins the table against the devices present in the job YAML.

Two hard-won rules live here as structure, not comments:
- Path families match by curated NAMESPACES for subtraction, never generic
  tokens: kv_offload/cpu/common.py legitimately runs inside CUDA jobs.
- NO exclusive namespace for vllm/v1/worker/gpu*: CPU and XPU workers subclass
  the gpu worker via module-level imports, so gpu-namespace changes genuinely
  affect CPU jobs.
"""

from __future__ import annotations

import regex as re

from .curated import (
    BASENAME_TOKEN_EXTENSIONS as _BASENAME_TOKEN_EXTENSIONS,
)
from .curated import (
    EXCLUSIVE_IMPORT_EXCEPTIONS as _EXCLUSIVE_IMPORT_EXCEPTIONS,
)
from .curated import (
    EXCLUSIVE_NAMESPACES as _EXCLUSIVE_NAMESPACES,
)
from .curated import (
    FAMILY_DEVICE_EXACT as _FAMILY_DEVICE_EXACT,
)
from .curated import (
    FAMILY_DEVICE_PREFIXES as _FAMILY_DEVICE_PREFIXES,
)
from .curated import (
    PATH_TOKEN_FAMILIES as _PATH_TOKEN_FAMILIES,
)


def family_of_device(device: str | None) -> str | None:
    if not device:
        return None
    for family, prefixes in _FAMILY_DEVICE_PREFIXES.items():
        if device.startswith(prefixes):
            return family
    for family, exact in _FAMILY_DEVICE_EXACT.items():
        if device in exact:
            return family
    return None


def family_of_path(path: str) -> str | None:
    tokens = set(re.split(r"[/_.]", path.lower()))
    for token_set, family in _PATH_TOKEN_FAMILIES:
        if tokens & token_set:
            return family
    return None


_FILENAME_TOKEN_RE = re.compile(r"[,_.=\-]+")


def family_of_filename(name: str) -> str | None:
    """Device family from a data FILENAME (tuning/config jsons): handles
    `device_name=<value>` fields and bare platform-named files
    (`nvidia_b200.json`). ADDITIVE tagging only, never subtractive -- unlike
    family_of_path this splits on `=,` too, so `device_name=amd_...` yields the
    token `amd`, and matches a device prefix only when the token carries a
    digit (mi325x/h200/b200 yes, mixtral/min no) or a length>=4 prefix appears
    as a substring (b200 in gb200)."""
    tokens = [t for t in _FILENAME_TOKEN_RE.split(name.lower()) if t]
    token_set = set(tokens)
    for family_tokens, family in _PATH_TOKEN_FAMILIES:
        if token_set & family_tokens:
            return family
    for family, prefixes in _FAMILY_DEVICE_PREFIXES.items():
        for token in tokens:
            has_digit = any(c.isdigit() for c in token)
            for prefix in prefixes:
                if has_digit and token.startswith(prefix):
                    return family
                if len(prefix) >= 4 and prefix in token:
                    return family
    return None


def device_prefix_of_filename(name: str) -> str | None:
    """The SPECIFIC device prefix a tuning/config filename names (h200, b200,
    mi, intel_gpu), finer than family_of_filename. A file named for device X is
    unreadable on a runner with a different known device (the loader matches
    get_device_name_as_file_name() to the runtime GPU), so this scopes the
    family floor + owning-closure routing to the exact device. None when no
    device token is present -> fall back to the whole family (no scoping)."""
    tokens = [t for t in _FILENAME_TOKEN_RE.split(name.lower()) if t]
    for _family, prefixes in _FAMILY_DEVICE_PREFIXES.items():
        for token in tokens:
            has_digit = any(c.isdigit() for c in token)
            for prefix in prefixes:
                if has_digit and token.startswith(prefix):
                    return prefix
                if len(prefix) >= 4 and prefix in token:
                    return prefix
    for _family, exact in _FAMILY_DEVICE_EXACT.items():
        for device in exact:
            if device in tokens:
                return device
    return None


def device_scoped_out(step, file_prefix: str) -> bool:
    """A step that provably never loads a file scoped to `file_prefix`: its
    device is a KNOWN device other than that prefix. Unknown-device steps
    (device=None or unmapped) are kept -- conservative. An amd-mirror step runs
    on amd hardware whatever its listed device."""
    file_family = family_of_device(file_prefix)
    if step.mirror_hw == "amd":
        return file_family != "amd"
    device = step.device
    if not device:
        return False
    fam = family_of_device(device)
    if fam is None:
        return False
    if fam != file_family:
        return True
    return not device.startswith(file_prefix)


def exclusive_family_of_path(path: str) -> str | None:
    """The single family allowed to run `path`, or None if unrestricted."""
    for prefixes, exact, family in _EXCLUSIVE_NAMESPACES:
        if path.startswith(prefixes) or path in exact:
            return family
    if path.endswith(_BASENAME_TOKEN_EXTENSIONS) and "rocm" in path.rsplit("/", 1)[-1]:
        return "amd"
    return None


def device_excluded_for_path(path: str, device: str | None, step=None) -> bool:
    """True when `device` provably cannot execute `path`.

    A mirror runs on ITS hardware whatever device it lists, matching
    device_scoped_out. Every mirror block sets an explicit device today, so
    this only guards the first one that omits it -- for which the inherited
    parent device would otherwise exclude the mirror from its own family."""
    allowed = exclusive_family_of_path(path)
    if allowed is None:
        return False
    if step is not None and step.mirror_hw:
        return step.mirror_hw != allowed
    if device is None:
        return False
    family = family_of_device(device)
    return family is not None and family != allowed


def step_in_family(step, family: str) -> bool:
    if family == "amd" and step.mirror_hw == "amd":
        return True
    return family_of_device(step.device) == family


def exclusivity_violations(
    plain_reverse: dict[str, set[str]], files
) -> dict[str, set[str]]:
    """The derived invariant behind every subtractive exclusion: an
    exclusive-namespace member (incl. rocm-basename files) must have no
    module-level importer outside its family, minus the hand-verified
    runtime-guarded exceptions. A violating member's exclusion is disabled
    at selection time (fail-open); the curated-namespace half is also
    pinned clean by an oracle test."""
    out: dict[str, set[str]] = {}
    for f in files:
        family = exclusive_family_of_path(f)
        if family is None:
            continue
        for importer in plain_reverse.get(f, ()):
            if _EXCLUSIVE_IMPORT_EXCEPTIONS.get((importer, f)):
                continue
            # Excuse only a family-exclusive importer, not additive token
            # tagging (family_of_path): a rocm-named test imports on cuda too.
            if exclusive_family_of_path(importer) == family:
                continue
            out.setdefault(f, set()).add(importer)
    return out
