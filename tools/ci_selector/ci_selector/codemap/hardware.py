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
    DEVICE_NAME_FAMILIES as _DEVICE_NAME_FAMILIES,
)
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
    QUEUE_DEVICE_NAMES as _QUEUE_DEVICE_NAMES,
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


_DEVICE_NAME_FIELD = re.compile(r"device_name=([^,]+)")
# The one tree whose configs are named for the device instead of carrying a
# `device_name=` field, keyed on vLLM's canonicalized GPU name.
_STEM_NAMED_DIR = "vllm/kernels/helion/configs/"


def family_of_device_name(device_name: str) -> str | None:
    """Family from a reported device name, by vendor token. A model number
    alone does not name a family."""
    low = device_name.lower()
    for token, family in _DEVICE_NAME_FAMILIES:
        if low.startswith(token):
            return family
    return None


def device_name_of_filename(name: str, path: str = "") -> str | None:
    """The device a data filename names, read the way the loaders read it.

    `device_name=<value>` is the field the tuning loader composes and opens. A
    file without one falls back to its stem, the key the helion loader uses --
    but only under the directory that convention belongs to, or any
    vendor-named asset in the tree would look device-scoped.
    """
    field = _DEVICE_NAME_FIELD.search(name)
    if field:
        return field.group(1).removesuffix(".json") or None
    if not path.startswith(_STEM_NAMED_DIR):
        return None
    stem = name.removesuffix(".json")
    return stem if family_of_device_name(stem) else None


def family_of_filename(name: str, path: str = "") -> str | None:
    """Device family from a data filename. Only ever adds jobs, never removes
    them. Splits on more separators than family_of_path."""
    tokens = {t for t in _FILENAME_TOKEN_RE.split(name.lower()) if t}
    for family_tokens, family in _PATH_TOKEN_FAMILIES:
        if tokens & family_tokens:
            return family
    device = device_name_of_filename(name, path)
    return family_of_device_name(device) if device else None


def device_scoped_out(step, file_device: str, aliases: dict[str, str]) -> bool:
    """A step that can never load a file named for `file_device`.

    Fails open: anything unknown -- no device, no family, no alias row -- keeps
    the step, so only a positive mismatch ever subtracts. An amd mirror runs on
    amd whatever device it lists.
    """
    file_family = family_of_device_name(file_device)
    if file_family is None:
        # A vendor we cannot place. Nothing below can reason about it, and the
        # comparisons would read the None as a mismatch and subtract.
        return False
    device = step.device or ""
    listed = family_of_device(device)
    fam = "amd" if step.mirror_hw == "amd" else listed
    if fam is None:
        return False
    if fam != file_family:
        return True
    if step.mirror_hw == "amd" and listed != "amd":
        # A mirror listing its cuda parent's queue: we know the family it runs
        # on, not which part, so the family check above is as far as we get.
        return False
    queue_device = _QUEUE_DEVICE_NAMES.get(device)
    if queue_device is None:
        return False
    if file_device == queue_device:
        return False
    if not aliases:
        # The table could not be read. Costing a narrowing beats scoping a
        # helion config out of the queue that is the only one able to load it.
        return False
    return file_device != canonicalize_gpu_name(queue_device, aliases)


def canonicalize_gpu_name(name: str, aliases: dict[str, str]) -> str:
    """Mirrors `canonicalize_gpu_name` in vllm/kernels/helion/utils.py: a
    reported GPU name to the stem its helion configs are filed under."""
    canon = re.sub(r"[\s/-]+", "_", name.lower())
    return aliases.get(canon, canon)


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
