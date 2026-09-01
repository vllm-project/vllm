# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hardware taxonomy: the one home for device and path family facts.

Every hardware call routes through here: adding jobs (a rocm-named file selects
AMD), removing them (a CPU-only file cannot affect GPU jobs), the generator
replica's AMD rule, and the test pinning the table against the job yaml.

Two rules are built into the structure rather than left as comments:
- Subtraction matches curated namespaces, never generic tokens. A file with
  `cpu` in its path can still run inside a CUDA job.
- There is no exclusive namespace for the gpu worker. CPU and XPU workers
  subclass it at module level, so a gpu change really does reach CPU jobs.
"""

from __future__ import annotations

import regex as re

from ..handwritten import (
    EXCLUSIVE_IMPORT_EXCEPTIONS as _EXCLUSIVE_IMPORT_EXCEPTIONS,
)
from ..handwritten import (
    EXCLUSIVE_NAMESPACES as _EXCLUSIVE_NAMESPACES,
)
from ..handwritten import (
    FAMILY_DEVICE_EXACT as _FAMILY_DEVICE_EXACT,
)
from ..handwritten import (
    FAMILY_DEVICE_PREFIXES as _FAMILY_DEVICE_PREFIXES,
)
from ..handwritten import (
    PATH_TOKEN_FAMILIES as _PATH_TOKEN_FAMILIES,
)
from ..handwritten import (
    REQUIREMENTS_EXTRA_TOKEN_FAMILIES as _REQUIREMENTS_EXTRA_TOKEN_FAMILIES,
)

BASENAME_TOKEN_EXTENSIONS: tuple[str, ...] = (".py", ".sh")
AMD_BASENAME_TOKEN = "rocm"


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


def requirements_family_of_path(path: str) -> str | None:
    """Family for a requirements file, where the filename names the device.
    The extra tokens (cuda) hold only under requirements/: globally they would
    misfire on vllm/ paths and on the generic Dockerfile."""
    if not path.startswith("requirements/"):
        return None
    family = family_of_path(path)
    if family:
        return family
    tokens = set(re.split(r"[/_.]", path.lower()))
    for token_set, extra in _REQUIREMENTS_EXTRA_TOKEN_FAMILIES:
        if tokens & token_set:
            return extra
    return None


_FILENAME_TOKEN_RE = re.compile(r"[,_.=\-]+")


def family_of_filename(name: str) -> str | None:
    """Device family from a data filename, handling both `device_name=<value>`
    fields and plainly named files. Only ever adds jobs, never removes them.
    Splits on more separators than family_of_path, and matches a device prefix
    only when the token carries a digit or the prefix is long enough to be
    unambiguous as a substring."""
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
    """The exact device a filename names, finer than family_of_filename. The
    loader matches the filename to the runtime GPU, so a file named for one
    device is unreadable on another and its routing scopes to that device.
    None means no device token, so the whole family stands."""
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
    """A step that can never load a file scoped to `file_prefix`, because its
    device is a known one that is not that prefix. Unknown-device steps are
    kept. An amd mirror runs on amd whatever device it lists."""
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
    if (
        path.endswith(BASENAME_TOKEN_EXTENSIONS)
        and AMD_BASENAME_TOKEN in path.rsplit("/", 1)[-1]
    ):
        return "amd"
    return None


def device_excluded_for_path(path: str, device: str | None, step=None) -> bool:
    """True when `device` cannot possibly run `path`.

    A mirror runs on its own hardware whatever device it lists. Every mirror
    sets one today, so this only guards the first that does not, whose
    inherited parent device would otherwise exclude it from its own family."""
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
    """The rule every hardware exclusion rests on: a file exclusive to one
    family must have no module-level importer outside it, minus the checked
    runtime-guarded exceptions. A file that breaks this loses its exclusion at
    selection time and fails open."""
    out: dict[str, set[str]] = {}
    for f in files:
        family = exclusive_family_of_path(f)
        if family is None:
            continue
        for importer in plain_reverse.get(f, ()):
            if _EXCLUSIVE_IMPORT_EXCEPTIONS.get((importer, f)):
                continue
            # Only a family-exclusive importer is excused, not one merely
            # carrying the token: a rocm-named test still imports on cuda.
            if exclusive_family_of_path(importer) == family:
                continue
            out.setdefault(f, set()).add(importer)
    return out
