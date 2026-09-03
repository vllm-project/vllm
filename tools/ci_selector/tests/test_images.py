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
from ci_selector.codemap.pipeline.images import (
    _DOCKERFILE_TOKEN,
    _resolve,
    build_artifact_graph,
)
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


def _statically_named(token: str) -> bool:
    """Whether a Dockerfile token is a path we can check at all.

    Commands also build interpolated paths (`${REPO}/Dockerfile.rocm`), which
    arrive as a fragment and can never resolve. Demanding those resolve would
    fail at HEAD, so only whole repo-relative paths are checked.

    Strip the shell punctuation first, exactly as `_resolve` does. Testing the
    raw token rejected `-docker/Dockerfile.rocm_base` (the `-` comes from a
    `${VAR:-default}`), which resolves fine and is the ONLY token two producers
    have, so a rename of that Dockerfile passed this guard green.
    """
    candidate = token.lstrip("-$:{").rstrip("}\"'")
    return (
        "/" in candidate and not candidate.startswith("/") and candidate[:1].isalnum()
    )


@pytest.mark.drift
def test_every_dockerfile_a_build_names_still_exists(state, vllm_repo):
    """The reverse direction: a Dockerfile leaving the build surfaces.

    A producer that names several keeps routing on the ones that survive, so a
    rename is invisible to `unresolved`, which only fires when a producer
    resolves none at all. Scoped to whole repo-relative paths: `docker/` also
    holds files no CI step consumes (the gfx1250 variants), and interpolated
    paths arrive as fragments that can never resolve.
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
    # Read back from the commands, not from the graph. `built` is the union of
    # defined_by, so asking whether its members have a producer inverts a dict
    # against itself, and _resolve already applied is_file() to every one.
    named = {}
    for pdata in state.pipelines:
        for producer in graph.defined_by:
            haystack = getattr(pdata.targets.get(producer), "haystack", "") or ""
            for token in _DOCKERFILE_TOKEN.findall(haystack):
                if _statically_named(token):
                    named.setdefault(producer, set()).add(token)
    unchecked = sorted(set(graph.defined_by) - set(named))
    assert not unchecked, drift_message(
        f"These image producers name no checkable Dockerfile path: {unchecked}",
        "A producer contributing no token is invisible to the check below, so "
        "a Dockerfile only it builds can be renamed away in silence. That is "
        "how this guard was blind to docker/Dockerfile.rocm_base.",
        "the command interpolates the whole path: teach _statically_named the "
        "new shape, or resolve the variable here",
    )
    checked = sum(len(t) for t in named.values())
    assert checked >= 10, drift_message(
        f"Only {checked} Dockerfile paths could be read back out of the build "
        "commands, so this guard is checking almost nothing.",
        "It is the only thing that notices a renamed Dockerfile while the "
        "producer still builds another one, which is silent everywhere else.",
        "the commands changed shape: check _DOCKERFILE_TOKEN in "
        "ci_selector/codemap/pipeline/images.py against .buildkite/ at HEAD",
    )
    stale = {
        producer: sorted(t for t in tokens if _resolve(vllm_repo, t) is None)
        for producer, tokens in named.items()
    }
    stale = {k: v for k, v in stale.items() if v}
    assert not stale, drift_message(
        f"Build commands name Dockerfiles that do not exist at HEAD: {stale}",
        "A producer naming several Dockerfiles keeps routing on the ones that "
        "survive, so the renamed file reaches no step and nothing goes red.",
        "the Dockerfile was renamed: fix the path in the job yaml upstream",
        "the Dockerfile is gone: drop it from the build command upstream",
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


_FROM = re.compile(r"^\s*FROM\s+\S+(?:\s+AS\s+(\S+))?", re.I)
_COPY = re.compile(r"^\s*(?:COPY|ADD)\s+(.*)$", re.I)
_COPY_FROM = re.compile(r"--from=(\S+)")


def _oracle_build_stage_files(vllm_repo, dockerfiles):
    """Exact-file COPY sources landing in a stage some later `--from=` consumes.

    Independent of the parser under test: a line scanner over the Dockerfiles,
    not `copy_inputs`. Nothing in the tool is stage-aware, so a build input is
    indistinguishable from runtime payload anywhere else.
    """
    build: set[str] = set()
    for rel in dockerfiles:
        text = (vllm_repo / rel).read_text() if (vllm_repo / rel).is_file() else ""
        stage, per_stage, consumed = None, {}, set()
        for raw in text.replace("\\\n", " ").splitlines():
            line = raw.split("#", 1)[0].rstrip()
            begins = _FROM.match(line)
            if begins:
                stage = begins.group(1)
                continue
            copies = _COPY.match(line)
            if not copies:
                continue
            body = copies.group(1)
            staged = _COPY_FROM.search(body)
            if staged:
                consumed.add(staged.group(1))
                continue
            tokens = [t for t in body.split() if not t.startswith("--")]
            for src in tokens[:-1]:
                path = src.lstrip("./").rstrip("/")
                if path and (vllm_repo / path).is_file():
                    per_stage.setdefault(stage, set()).add(path)
        for name, files in per_stage.items():
            if name in consumed:
                build |= files
    return build


@pytest.mark.drift
def test_a_build_stage_input_keeps_its_image_union(state, vllm_repo):
    """A file the wheel is compiled from must keep its image routing.

    `vllm/envs.py` is the case worth naming: setup.py path-loads it for the
    build variables, and `docker/Dockerfile` copies it into the compile stage
    beside CMakeLists.txt, cmake/ and csrc/. A path-load is not an import and
    setup.py is not a graph node, so the import graph structurally cannot see
    that edge; the image rule is the only place it is encoded.
    """
    built = {f for files in state.artifacts.defined_by.values() for f in files}
    assert built, "no image-definition files at HEAD"
    build_stage = _oracle_build_stage_files(vllm_repo, built)
    assert len(build_stage) >= 5 and "vllm/envs.py" in build_stage, drift_message(
        f"the build-stage scanner found {len(build_stage)} files and "
        f"{'kept' if 'vllm/envs.py' in build_stage else 'lost'} vllm/envs.py",
        "the guard below stops covering compiled inputs",
        "check whether docker/Dockerfile still uses `COPY --from=<stage>`",
    )
    missing = sorted(f for f in build_stage if f not in state.artifacts.inputs_of)
    assert not missing, (
        f"{len(missing)} files the wheel is compiled from lost their image "
        f"routing: {missing[:3]}"
    )

    # Stage-ness alone does not produce envs.py's breadth; the borrowed
    # whole-context images do. Its literal copiers are Dockerfile and
    # Dockerfile.rocm, so anything beyond them is borrowed.
    envs = state.artifacts.inputs_of["vllm/envs.py"]
    literal = {"docker/Dockerfile", "docker/Dockerfile.rocm"}
    assert envs > literal, (
        f"vllm/envs.py stopped borrowing the shared images (got {sorted(envs)}); "
        "its union collapses to the two Dockerfiles that name it"
    )

    # Control, so this is not "everything is wide": a runtime-stage COPY is not
    # a compiled input and must not be dragged in by the assertions above.
    assert "vllm/collect_env.py" not in build_stage, (
        "collect_env.py is a runtime-stage COPY (docker/Dockerfile:852); the "
        "scanner has stopped distinguishing build stages from payload"
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
    assert "vllm_ci:cpu-kernel-tests" in sel.selected, "cpu family lost its route"
    assert "vllm_ci:distributed-comm-ops" not in sel.selected, (
        "a device=None GPU suite leaked in through the image union"
    )
