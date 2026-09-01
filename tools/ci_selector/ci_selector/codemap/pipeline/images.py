# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The build-artifact DAG: which steps produce container images, what each
image is built from, and which steps consume it.

Nothing is listed by hand. A step with a `depends_on` names the producer it
needs, and a step with none may still build the image itself, which is how the
CPU and Arm suites work. Reading only the first left most of the AMD pipeline
unselected on a `csrc/` change.

Image inputs are read too, but a build stage copies whole trees in, so the COPY
list says what is inside the image, not what makes it rebuild. A directory copy
is only used when the import graph cannot route its contents anyway. Inputs are
added as a union and not a claim, so they cannot override hardware scoping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import regex as re

# Above this the graph routes the tree well enough that the copy adds only
# noise. Sits in a wide gap between the two kinds of tree.
GRAPH_BLIND_CEILING = 0.25

# A Dockerfile path anywhere in a build command, including a shell default like
# `${VAR:-docker/Dockerfile.cpu}`. Anchored on the basename, since no build step
# names its Dockerfile directly.
_DOCKERFILE_TOKEN = re.compile(r"[\w./-]*Dockerfile[\w.-]*")
_HCL_TOKEN = re.compile(r"[\w./-]+\.hcl")
_HCL_DOCKERFILE = re.compile(r'dockerfile\s*=\s*"([^"]+)"')


@dataclass
class ArtifactGraph:
    """Producers and their consumers, per pipeline-qualified step id."""

    # producer step_id -> every step_id that transitively depends on it
    dependents: dict[str, set[str]] = field(default_factory=dict)
    # producer step_id -> the image-definition files it builds from
    defined_by: dict[str, set[str]] = field(default_factory=dict)
    # image-definition file -> the producer step_ids that read it
    producers_of: dict[str, set[str]] = field(default_factory=dict)
    # producers naming no Dockerfile: we stand down rather than guess
    unresolved: set[str] = field(default_factory=set)
    # image INPUT -> the image-definition files it feeds. Exact paths, plus
    # directory prefixes kept with a trailing "/".
    inputs_of: dict[str, set[str]] = field(default_factory=dict)
    # Same keys, but only images that COPY the source by name. `inputs_of`
    # also folds in the borrowed whole-context images and cannot be unmixed
    # afterwards; the rust rule needs "built in on purpose", not "contains it".
    explicit_inputs_of: dict[str, set[str]] = field(default_factory=dict)
    # image-definition file -> steps that build it in-step instead of depending
    # on a producer. Keyed by Dockerfile, so an xpu edit misses a CPU suite.
    self_builders: dict[str, set[str]] = field(default_factory=dict)

    def consumers_of_image(self, dockerfile: str) -> set[str]:
        """The producers that build the image, everything downstream of them,
        and the steps that build it for themselves."""
        out = set(self.self_builders.get(dockerfile, ()))
        for pid in self.producers_of.get(dockerfile, ()):
            out.add(pid)
            out |= self.dependents.get(pid, set())
        return out

    def steps_for(self, path: str) -> set[str]:
        """`path` is an image definition: every step that runs on that image."""
        return self.consumers_of_image(path)

    def steps_for_input(self, path: str) -> set[str]:
        """Producers whose image `path` feeds, plus everything downstream.
        Exact match first, then directory prefixes, so `csrc/foo.cu` resolves
        through the `csrc/` copy without listing every file."""
        out: set[str] = set()
        for df in self._images_for_input(path, self.inputs_of):
            out |= self.consumers_of_image(df)
        return out

    def explicit_images_of(self, path: str) -> set[str]:
        """Images that COPY `path` by name, exact or by directory. Excludes the
        whole-context images that only borrow it."""
        return self._images_for_input(path, self.explicit_inputs_of)

    @staticmethod
    def _images_for_input(path: str, table: dict[str, set[str]]) -> set[str]:
        images = set(table.get(path, ()))
        for prefix, dfs in table.items():
            if prefix.endswith("/") and path.startswith(prefix):
                images |= dfs
        return images


def _resolve(repo: Path, token: str) -> str | None:
    """A repo-relative path if the token names a real file. Shell defaults
    arrive with punctuation glued on, so strip that first."""
    candidate = token.lstrip("-$:{").rstrip("}\"'")
    return candidate if candidate and (repo / candidate).is_file() else None


def _definition_files(repo: Path, haystack: str) -> set[str]:
    """Image-definition files named in a producer's command text or a script it
    runs. Bake files are followed one hop into their `dockerfile =` targets."""
    found: set[str] = set()
    for token in _DOCKERFILE_TOKEN.findall(haystack):
        path = _resolve(repo, token)
        if path:
            found.add(path)
    for token in _HCL_TOKEN.findall(haystack):
        hcl = _resolve(repo, token)
        if not hcl:
            continue
        found.add(hcl)
        try:
            text = (repo / hcl).read_text()
        except OSError:
            continue
        for inner in _HCL_DOCKERFILE.findall(text):
            path = _resolve(repo, inner)
            if path:
                found.add(path)
    return found


def build_artifact_graph(repo: Path, pipelines) -> ArtifactGraph:
    """Derive the graph from step metadata alone.

    A step is a producer exactly when another step names its published key in
    `depends_on`, so a producer not named `image-build-*` is caught too.
    Resolution is per pipeline, because one job_dir feeds more than one config
    and the same key appears in each. `buildkite_key` is the published spelling
    and the only one `depends_on` can name.
    """
    graph = ArtifactGraph()
    for pdata in pipelines:
        # Depends on nothing but names a Dockerfile: it builds that in-step.
        for step in pdata.steps:
            if step.depends_on:
                continue
            targets = pdata.targets.get(step.step_id)
            for path in _definition_files(repo, getattr(targets, "haystack", "") or ""):
                graph.self_builders.setdefault(path, set()).add(step.step_id)
        by_key = {s.buildkite_key: s.step_id for s in pdata.steps if s.buildkite_key}
        needs = {
            s.step_id: {by_key[d] for d in (s.depends_on or []) if d in by_key}
            for s in pdata.steps
        }
        for producer in {pid for deps in needs.values() for pid in deps}:
            graph.dependents[producer] = _downstream(producer, needs)
            targets = pdata.targets.get(producer)
            files = _definition_files(repo, getattr(targets, "haystack", "") or "")
            if not files:
                graph.unresolved.add(producer)
                continue
            graph.defined_by[producer] = files
            for path in files:
                graph.producers_of.setdefault(path, set()).add(producer)
    return graph


def add_image_inputs(
    repo: Path,
    graph: ArtifactGraph,
    files: dict[str, set[str]],
    dirs: dict[str, set[str]],
    blanket: set[str],
    is_graph_known,
    family_of,
) -> None:
    """Populate `inputs_of`: repo path -> the images it is built into.

    An image built with a whole-context `COPY . .` has no readable input list,
    and the main images are all built that way. It borrows from the images that
    can be read, but only the shared ones: a source counts when the images
    copying it span more than one hardware family, since a file every platform
    needs is a file the unreadable one needs too.

    Shared-only was measured. Borrowing everything turns a step-definition edit
    into an image rebuild, because one Dockerfile copies `.buildkite` in as test
    payload. Borrowing nothing leaves the xpu image with no inputs at all.
    """

    def images_for(dockerfiles) -> set[str]:
        # The generic Dockerfile has no hardware token and is its own family,
        # so a cuda-only requirements file stays narrow and a shared one does
        # not.
        shared = len({family_of(df) for df in dockerfiles}) > 1
        return set(dockerfiles) | (blanket if shared else set())

    for src, dockerfiles in files.items():
        graph.explicit_inputs_of.setdefault(src, set()).update(dockerfiles)
        graph.inputs_of.setdefault(src, set()).update(images_for(dockerfiles))

    for src, dockerfiles in dirs.items():
        inside = [
            f.relative_to(repo).as_posix()
            for f in (repo / src).rglob("*")
            if f.is_file() and ".git" not in f.parts
        ]
        if not inside:
            continue
        known = sum(1 for f in inside if is_graph_known(f))
        if known / len(inside) > GRAPH_BLIND_CEILING:
            continue  # the graph already routes this tree; leave it alone
        graph.explicit_inputs_of.setdefault(src + "/", set()).update(dockerfiles)
        graph.inputs_of.setdefault(src + "/", set()).update(images_for(dockerfiles))


def _downstream(producer: str, needs: dict[str, set[str]]) -> set[str]:
    """Transitive dependents of `producer`. Never revisits a step, so a cycle
    cannot hang this."""
    seen: set[str] = set()
    frontier = [producer]
    while frontier:
        current = frontier.pop()
        for step_id, deps in needs.items():
            if current in deps and step_id not in seen:
                seen.add(step_id)
                frontier.append(step_id)
    return seen
