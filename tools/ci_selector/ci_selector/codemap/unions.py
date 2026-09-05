# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The two passes that run after a rule has answered, not rules themselves.

A rule decides what a path IS. These add what every path owes regardless: the
steps that declare it, and the steps running on an image it is built into.
Running them in one place is what stops a new rule under-selecting by
forgetting either. Both are idempotent, both skip run-all, and each keeps its own exempt
set.
"""

from __future__ import annotations

from . import build_map, hardware
from .claim import Claim
from .state import RepoState, _graph_known
from .step_refs import _source_dep_steps, _source_dep_steps_ungated

# Only the rules that are certain nothing runs are exempt: a doc cannot break a
# build, and dead hardware runs nothing. release-ci is NOT here, because a
# release script can also carry a live test, so it must still pick up its
# genuine declarers.
_DEP_UNION_EXEMPT = frozenset({"no-code", "no-hardware"})


def _apply_declarer_union(state: RepoState, path: str, claim: Claim) -> Claim:
    """Add the steps that declare `path`, on top of whatever rule fired. Doing
    it in one place makes it impossible for a classifier to under-select by
    forgetting declarers, so every claim passes through here, including the
    ones built outside `_classify`. Idempotent. Skipped for run-all, which is
    already everything, and for the exempt rules."""
    if claim.run_all or claim.rule in _DEP_UNION_EXEMPT:
        return claim
    # On a file the graph knows, the graph is the better answer, so drop
    # declarers that matched only through a catch-all prefix. Graph-blind files
    # keep the full union.
    #
    # An unmodeled dynamic import used to put the catch-all deps back, on the
    # grounds that missing edges undercut the graph. Removed: it leaned on the
    # very yaml this project exists to delete, and catch-all declarers are a
    # small slice of the pipeline, so they could not have covered an unknown
    # edge anyway. What is left is the warning, the fail-open on the site file,
    # and the check going red.
    if claim.rule == "release-ci":
        # This claim says nothing runs, and a declarer is the evidence that
        # the file is still tested, so the switch must not silence it.
        declarers = _source_dep_steps_ungated(state, path)
        omitted = 0
    elif _graph_known(state, path):
        declarers = _source_dep_steps(state, path, specific_only=True)
        omitted = len(_source_dep_steps(state, path) - declarers)
    else:
        declarers = _source_dep_steps(state, path)
        omitted = 0
    added = declarers - claim.step_ids
    if added:
        claim.step_ids |= added
        claim.detail += f"; +{len(added)} steps declare it as a source dep"
    if omitted:
        claim.detail += f"; {omitted} catch-all-only declarers omitted"
    return claim


# Every "nothing to run" rule, plus the one rule that already read the build
# graph and scoped its own answer. The union must not revive a path a rule
# established runs nothing (retired yamls, release-only scripts, inert CI
# trees and inert files all get copied into images), and must not undo the
# rust rule's scoping, which deliberately dropped the borrowed images. A
# separate set from _DEP_UNION_EXEMPT on purpose: "inert" stays out of that
# one, since a declarer is the evidence that disproves the veto.
_IMAGE_UNION_EXEMPT = frozenset(
    {"no-code", "no-hardware", "legacy-ci", "inert-ci", "inert", "release-ci", "rust"}
)


def _apply_image_input_union(state: RepoState, path: str, claim: Claim) -> Claim:
    """Add the steps that run on an image this file is built into.

    A pass and not a rule, and that distinction is the design. Claiming these
    paths would run before every rule below and would override hardware scoping
    and the `no-code` answer, both of which are right today. Adding leaves
    those decisions alone and puts what the build DAG knows on top.

    This carries the whole build layer now that `run_all_patterns` is unread:
    the C++, cmake and shared requirements files reach the AMD and Intel
    pipelines through here and nowhere else.

    Added non-droppably: a coverage row says which functions a step ran and
    nothing about which image it ran on. For csrc files a later pass can still
    mark these steps droppable, on wrapper names a row can speak to.
    """
    if claim.run_all or claim.rule in _IMAGE_UNION_EXEMPT or claim.image_union_exempt:
        return claim
    steps = state.artifacts.steps_for_input(path) & state.auto_step_ids
    # A family-exclusive file cannot affect another family's jobs even though
    # its tree is copied into that family's image. Scope rather than skip, so a
    # CPU-only source still reaches the CPU suites. Without this the union
    # brings back the very over-selection the exclusive-family rule exists for.
    family = hardware.exclusive_family_of_path(path)
    if family and path not in state.exclusive_disabled:
        steps &= state.family_steps(family)
    # A mapped file only reaches the images whose builds compile it. Keep this
    # before `added` is computed. An unmapped path keeps the full set, and so
    # does everything if the device list has gaps, since family_steps() is then
    # incomplete and cannot be trusted to subtract.
    fams = state.build_map.families.get(path)
    if fams and not state.preflight.unmapped_devices and build_map.mode() == "on":
        steps &= _build_map_allowed(state, fams)
    added = steps - claim.step_ids
    if added:
        claim.step_ids |= added
        claim.detail += f"; +{len(added)} steps run on an image this file is built into"
    return claim


def _build_map_allowed(state: RepoState, fams: frozenset[str]) -> set[str]:
    """The steps these families may keep.

    "cuda" is the remainder that carries no device token, which is where the
    main image's GPU suites live; reading it as a token family would drop them
    all. "other" is every token family except amd and cpu.
    """
    per, union, nonfamily = state.family_partition()
    allowed: set[str] = set()
    if build_map.CUDA in fams:
        allowed |= nonfamily
    if build_map.AMD in fams:
        allowed |= per.get("amd", frozenset())
    if build_map.CPU in fams:
        allowed |= per.get("cpu", frozenset())
    if build_map.OTHER in fams:
        allowed |= union - per.get("amd", frozenset()) - per.get("cpu", frozenset())
    return allowed
