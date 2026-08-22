# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The build-artifact DAG, and the drift guards that keep it honest.

Every assertion here is against the live checkout, so each one carries a
detection floor: the derivation collapsing to nothing must fail loudly rather
than satisfy an emptiness check.
"""

import pytest
import regex as re
from ci_selector.codemap.classify import select
from ci_selector.codemap.pipeline.images import build_artifact_graph
from helpers import drift_message

# `- key` under a depends_on block, and the inline `depends_on: key` form.
_DEP_BLOCK = re.compile(r"^\s*depends_on:\s*$")
_DEP_INLINE = re.compile(r"^\s*depends_on:\s*\[?\s*([\w.-]+)")
_LIST_ITEM = re.compile(r"^\s*-\s*([\w.-]+)\s*$")


def _oracle_dep_keys(vllm_repo, pipelines):
    """Every key named in a depends_on, re-read from the raw yaml.

    Independent of the parser under test: a line scanner over the job files,
    not `Step.depends_on`. Catches the parser silently dropping the field.
    """
    keys = set()
    for pdata in pipelines:
        for job_dir in pdata.config.job_dirs:
            for path in sorted((vllm_repo / job_dir).rglob("*.yaml")):
                in_block = False
                for raw in path.read_text().splitlines():
                    line = raw.split("#", 1)[0]
                    if _DEP_BLOCK.match(line):
                        in_block = True
                        continue
                    if in_block:
                        item = _LIST_ITEM.match(line)
                        if item:
                            keys.add(item.group(1))
                            continue
                        in_block = False
                    inline = _DEP_INLINE.match(line)
                    if inline:
                        keys.add(inline.group(1))
    return keys


def test_producers_match_the_depends_on_oracle(state, vllm_repo):
    """Drift oracle: the producer set is exactly the keys some step depends on.

    Producers are derived structurally rather than by an `image-build*` key
    prefix, which is what lets `arm64-image-build` and the two AMD base-image
    steps be found at all.
    """
    graph = build_artifact_graph(vllm_repo, state.pipelines)
    oracle = _oracle_dep_keys(vllm_repo, state.pipelines)
    assert len(oracle) >= 4, "depends_on oracle collapsed; update the scanner"

    published = {
        s.buildkite_key
        for p in state.pipelines
        for s in p.steps
        if s.step_id in graph.dependents
    }
    unseen = published - oracle
    assert not unseen, f"derived a producer the raw yaml never depends on: {unseen}"


@pytest.mark.drift
def test_every_producer_resolves_a_dockerfile_at_head(state, vllm_repo):
    """A producer whose command text names no Dockerfile cannot be routed to,
    so the rule stands down for its image. That is safe but silent, and at
    HEAD it should never happen: every image build reaches its Dockerfile
    either directly or through one bake-file hop."""
    graph = build_artifact_graph(vllm_repo, state.pipelines)
    assert graph.dependents, drift_message(
        "No image producers were derived at all, so this guard is measuring nothing.",
        "The image-input rule is what connects a Dockerfile edit to the jobs "
        "built from it. With no producers it covers nothing, silently.",
        "the depends_on shape changed in the job yaml: teach the parser in "
        "ci_selector/codemap/pipeline/images.py",
    )
    assert not graph.unresolved, drift_message(
        f"{len(graph.unresolved)} image builds name no Dockerfile: "
        f"{sorted(graph.unresolved)}",
        "A producer we cannot trace to a Dockerfile gets no image-input "
        "coverage, so editing what it is built from selects nothing for it. "
        "Safe on its own, but it is how a build silently leaves the map.",
        "the build command changed shape: teach "
        "ci_selector/codemap/pipeline/images.py to read it",
        "the build moved to a bake file we do not follow: add the hop there",
    )


@pytest.mark.drift
def test_every_built_dockerfile_has_a_producer_at_head(state, vllm_repo):
    """The reverse direction, so a Dockerfile joining the build surfaces.

    Scoped to Dockerfiles some producer actually builds: `docker/` also holds
    files no CI step consumes (the gfx1250 variants), and those are correctly
    absent rather than a drift signal.
    """
    graph = build_artifact_graph(vllm_repo, state.pipelines)
    built = {f for files in graph.defined_by.values() for f in files}
    assert len(built) >= 5, drift_message(
        f"Only {len(built)} built Dockerfiles were found, so the scanner has "
        "collapsed and this guard cannot see anything.",
        "An empty set passes the per-file loop below trivially, which is "
        "exactly what a broken scanner looks like from the outside.",
        "the Dockerfile scan in ci_selector/codemap/pipeline/images.py stopped "
        "matching: check it against docker/ at HEAD",
    )
    for path in sorted(built):
        assert graph.producers_of.get(path), drift_message(
            f"{path} is built by no producer.",
            "Nothing connects edits to this Dockerfile to the jobs that build "
            "it, so those jobs stop being selected on it.",
            "a new image build was added: teach "
            "ci_selector/codemap/pipeline/images.py to read its command",
        )
        assert (vllm_repo / path).is_file(), drift_message(
            f"A producer builds {path}, which does not exist at HEAD.",
            "The producer's image inputs resolve to nothing, so the step loses "
            "its image-input coverage entirely.",
            "the Dockerfile was renamed: the producer's command still names the "
            "old path, so fix it upstream in the job yaml",
        )


def test_self_building_steps_are_derived_and_scoped(state, vllm_repo):
    """A step that depends on nothing may still consume an image: it builds
    one in-step.

    Every CPU and Arm suite in vllm_ci and vllm_rocm_ci does, compiling
    `docker/Dockerfile.cpu` itself, and reading the DAG through `depends_on`
    alone therefore missed them -- a `csrc/` change reached 4 of 15 AMD steps.
    Keyed by the Dockerfile the step actually names, so the fix does not
    degrade into "a step with no edges might consume anything": an xpu image
    edit must still leave the CPU suites alone.
    """
    graph = build_artifact_graph(vllm_repo, state.pipelines)
    assert graph.self_builders, (
        "no self-building steps derived; the AMD and Arm suites lose their "
        "only route to a build-input change"
    )
    cpu = graph.self_builders.get("docker/Dockerfile.cpu", set())
    assert {s.split(":", 1)[0] for s in cpu} >= {"vllm_ci", "vllm_rocm_ci"}, (
        f"the CPU suites stopped resolving their own image: {sorted(cpu)[:3]}"
    )
    for step in cpu:
        assert step not in graph.consumers_of_image("docker/Dockerfile.xpu"), (
            f"{step} builds the cpu image but was routed to the xpu one"
        )


def test_image_definition_routes_to_that_image_only(state, vllm_repo):
    """The precision the pattern language cannot express.

    `run_all_patterns` needs `docker/Dockerfile` plus an exclude of
    `docker/Dockerfile.` to keep the variants out. Derived, each variant goes
    to the steps that depend on its own producer, so the xpu image must not
    drag in the steps that run on the CUDA image.
    """
    graph = build_artifact_graph(vllm_repo, state.pipelines)
    xpu = graph.steps_for("docker/Dockerfile.xpu")
    cuda = graph.steps_for("docker/Dockerfile")
    assert xpu and cuda, "expected both images to be derived at HEAD"
    assert not (xpu & cuda), "an image's consumers leaked into another's"

    sel = select(state, ["docker/Dockerfile.xpu"])
    assert not sel.run_all, "an image definition must not fall through to run-all"
    rules = {r for rs in sel.selected_rules.values() for r in rs}
    assert "image-input" in rules


def test_image_input_steps_are_not_droppable(state, vllm_repo):
    """Coverage rows record which functions a step ran; they say nothing about
    which image it runs on. So no evidence can overturn this routing, and the
    coverage record must never be handed a path it could weigh against it."""
    from ci_selector.codemap.classify import _classify_image_input

    graph = build_artifact_graph(vllm_repo, state.pipelines)
    built = sorted({f for files in graph.defined_by.values() for f in files})
    assert built, "no image-definition files at HEAD"
    for path in built:
        claim = _classify_image_input(state, path)
        if claim is None:
            continue
        assert not claim.droppable_step_ids, path
        assert not claim.droppable_test_files, path


def test_image_inputs_include_the_compiled_trees_only(state, vllm_repo):
    """The carve-out that makes input routing usable at all.

    Every non-blanket COPY source expands to 56% of the vllm_repo, so a directory
    copy is taken only where the import graph cannot route the contents. The
    split is bimodal at HEAD, so this asserts the two ends rather than a
    threshold: compiled trees in, Python trees out.
    """
    graph = build_artifact_graph(vllm_repo, state.pipelines)
    dirs = {k for k in state.artifacts.inputs_of if k.endswith("/")}
    assert graph.defined_by, "no producers; the fixture collapsed"
    assert dirs, "no directory inputs derived; csrc/ would lose its only route"

    for tree in ("csrc/", "cmake/", "rust/"):
        assert tree in dirs, f"{tree} is compiled into the image and must route"
    for tree in ("tests/", "examples/", "benchmarks/", "vllm/v1/"):
        assert tree not in dirs, (
            f"{tree} is already routed by the import graph; taking it as an "
            "image input gives most of the vllm_repo the full image closure"
        )


def test_image_input_union_does_not_resurrect_a_zero_claim(state):
    """A rule that positively established "nothing to run" must win.

    A retired yaml, a release-only script and an inert CI tree are all copied
    into the image, and none of them can change what a test does. Unioning the
    image closure onto them would undo three rules at once.
    """
    from ci_selector.codemap.classify import _classify

    for path in (".buildkite/test-amd.yaml", ".buildkite/scripts/build-macos-wheel.sh"):
        claim = _classify(state, path, None)
        assert not claim.run_all, path
        assert "run on an image" not in claim.detail, (
            f"{path}: the image union overrode a zero-jobs claim"
        )


def test_image_input_union_is_scoped_to_the_files_own_family(state):
    """`csrc/` is copied into the CUDA image, but `csrc/cpu/` cannot affect a
    CUDA job. Without scoping, the union re-creates the bare-complement bug the
    exclusive-family rule exists to avoid, because a device=None GPU suite is
    not removed by the apply-time hardware filter."""
    from ci_selector.codemap.classify import select

    sel = select(state, ["csrc/cpu/cpu_attn.cpp"])
    assert "vllm_ci:CPU-Kernel Tests" in sel.selected, "cpu family lost its route"
    assert "vllm_ci:distributed-comm-ops" not in sel.selected, (
        "a device=None GPU suite leaked in through the image union"
    )
