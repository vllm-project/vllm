# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ROCm image keys in .buildkite/scripts/rocm/build.sh must cover every
build-context file the keyed Dockerfile parts read, or editing that file would
ship a stale image."""

import os
import subprocess
import time
from pathlib import Path

import pytest
import regex as re

ROOT = Path(__file__).parents[2]
MARKER = "# ==== per-commit stages: nothing below is part of the runtime key ===="


def context_sources(dockerfile: str) -> set[str]:
    text = dockerfile.replace("\\\n", " ")
    srcs = set()
    for args in re.findall(r"^\s*(?:COPY|ADD)\s+(.*)$", text, re.M):
        words = [w for w in args.split() if not w.startswith("--")]
        if "--from=" not in args and not words[0].startswith("http"):
            srcs.update(words[:-1])
    for opts in re.findall(r"--mount=type=bind,(\S*)", text):
        kv = dict(o.partition("=")[::2] for o in opts.split(","))
        if "from" not in kv:
            srcs.add(kv.get("source", kv.get("src", ".")))
    return srcs


def key_paths(name: str, stack: str = "") -> set[str]:
    script = (ROOT / ".buildkite/scripts/rocm/build.sh").read_text()
    # keys_at() computes both keys, indented in its body.
    block_match = re.search(rf"^\s*{name}=\$\(.*?sha256sum", script, re.M | re.S)
    assert block_match is not None, f"{name}= block not found in build.sh"
    pattern = r"\b(?:requirements|tools|tests)/[\w./-]+|\brust-toolchain\.toml\b"
    paths = set(re.findall(pattern, block_match.group(0)))
    # therock_wheels.py is injected into Ks via a helper variable (rocm stack only)
    if name == "Ks" and stack == "rocm":
        m = re.search(r"_[Tt][Rr][Ww]=.*?tools/[\w./-]+", script, re.S)
        if m:
            paths.update(re.findall(r"\b(?:requirements|tools)/[\w./-]+", m.group(0)))
    return paths


def test_runtime_key_covers_runtime_inputs():
    dockerfile = (ROOT / "docker/Dockerfile.rocm").read_text()
    assert dockerfile.count(MARKER) == 1
    assert context_sources(dockerfile.split(MARKER)[0]) <= key_paths("Kr")


def test_stack_key_covers_stack_inputs():
    dockerfile = (ROOT / "docker/Dockerfile.rocm_base").read_text()
    assert context_sources(dockerfile) <= key_paths("Ks", stack="rocm")


def run_build(
    tmp_path: Path, mode: str = "ci", existing: tuple[str, ...] = (), **env: str
) -> tuple[int, str, list[str]]:
    """build.sh <mode> against fake git, docker and buildkite-agent; returns (status,
    stderr, calls). `existing` are the image refs the registry already has.
    FAILURE=bake|missing|upload breaks the kernel symbol map; FAILURE=registry
    makes every registry lookup fail, FAILURE=pushed only lookups of pushed refs,
    FAILURE=appear those of APPEARS refs once they appear, FAILURE=fetch-main and
    FAILURE=merge-base the lookup of main. FAKE_CLOCK=<file> makes every `date` call
    advance a minute; REAL_SLEEP=1 makes `sleep` real."""
    (tmp_path / "docker").symlink_to(ROOT / "docker")
    fake = tmp_path / "bin"
    fake.mkdir()
    calls = tmp_path / "calls"
    calls.touch()
    registry = tmp_path / "registry"
    registry.write_text("".join(f"{ref}\n" for ref in existing))
    (tmp_path / "registry.beats").touch()
    # The merge base with main has the checkout's files, except where CHANGED names a
    # file (Dockerfile.rocm_base or requirements/rocm.txt) this branch changed.
    (fake / "git").write_text(
        """#!/bin/bash
case $1 in
  describe) echo v0.0.0-0-g0 ;;
  rev-parse) [[ $2 == HEAD ]] && echo "${HEAD_REV:-c1}" || echo "${2%%^*}" ;;
  log) echo 0 ;;
  archive) tar -ch docker ;;
  fetch) [[ $FAILURE == fetch-main && " $* " == *" main "* ]] && exit 1 ;;
  merge-base) [[ $FAILURE == merge-base ]] && exit 1; echo mb ;;
  show) cat "${2#*:}" ;;
  ls-tree)
    for f in "${@:4}"; do
      [[ $2 == HEAD && " $CHANGED " == *" $f "* ]] && echo "100644 blob changed $f"
    done ;;
esac
exit 0
"""
    )
    (fake / "docker").write_text(
        """#!/bin/bash
[[ $1 == buildx ]] || exit 0
case $2 in
  inspect) [[ -z $NO_BUILDER ]] || exit 1 ;;
  create) echo "builder ${*:3}" >> "$CALLS" ;;
  build)  # build.sh's heartbeat: a label-only image pushed to type=registry,name=<ref>
    args="$*"; name=${args##*name=}; name=${name%% *}
    echo "heartbeat $name" >> "$CALLS"
    echo "$name $(sed -n 's/^LABEL [^=]*=//p')" >> "$REGISTRY.beats" ;;
  imagetools)
    if [[ $3 == inspect && $4 == *:building-* ]]; then
      echo "beat? $4" >> "$CALLS"
      # BEAT_AGE: main's build of every image beat that many seconds ago.
      [[ -n $BEAT_AGE ]] && { echo $(( $(date +%s) - BEAT_AGE )); exit 0; }
      beat=$(awk -v r="$4" '$1 == r { b = $2 } END { print b }' "$REGISTRY.beats")
      [[ -n $beat ]] && echo "$beat" || exit 1
    elif [[ $3 == inspect ]]; then
      [[ $FAILURE == registry ]] && { echo "503 Service Unavailable" >&2; exit 1; }
      found=0
      while read -r ref; do [[ $4 == "$ref"* ]] && found=1; done < "$REGISTRY"
      # Refs starting with APPEARS exist from the APPEAR_AFTER-th lookup of one on
      # (main's build pushing them while this one waits).
      if [[ -n $APPEARS && $4 == "$APPEARS"* ]]; then
        n=$(( $(cat "$REGISTRY.n" 2>/dev/null || echo 0) + 1 )); echo $n > "$REGISTRY.n"
        (( n >= APPEAR_AFTER )) && found=1
        [[ $FAILURE == appear ]] && ((found)) && { echo "503" >&2; exit 1; }
      fi
      [[ $FAILURE == pushed ]] && ((found)) && { echo "503" >&2; exit 1; }
      ((found)) || { echo "$4: not found" >&2; exit 1; }
      echo "Digest: sha256:$(printf '0%.0s' {1..64})"
    else
      echo "create ${@: -1}" >> "$CALLS"
    fi ;;
  bake)
    echo "bake ${*:5}" >> "$CALLS"
    echo "cache: ${CACHE:-own only}" >&2
    [[ $5 == --metadata-file ]] && for t in stack-rocm stack-wheels runtime test-deps
    do
      printf '  "%s": {\\n    "containerimage.digest": "sha256:%064d"\\n  },\\n' "$t" 1
    done > "$6"
    case ${@: -1} in
      base|runtime|test-deps) printf '%s\n' "$TAG" "$TAG-wheels" >> "$REGISTRY" ;;
      version) mkdir -p .rocm-version && echo 1.0 > .rocm-version/vllm-version.txt ;;
      ci) mkdir -p .rocm-wheel && touch .rocm-wheel/vllm-0.whl ;;
      kernel-symbol-map)
        [[ $FAILURE == bake ]] && exit 1
        mkdir -p .rocm-kernel-map && cd .rocm-kernel-map
        [[ $FAILURE == missing ]] || echo map > kernel_symbol_map.rocm.json.gz ;;
    esac ;;
esac
exit 0
"""
    )
    (fake / "buildkite-agent").write_text(
        """#!/bin/bash
[[ $1 == meta-data && $2 == set ]] && echo "meta $3 $4" >> "$CALLS"
[[ $1 == meta-data ]] && { echo "ref@sha256:$(printf '0%.0s' {1..64})"; exit 0; }
echo "upload ${PWD##*/}/$3" >> "$CALLS"
[[ $FAILURE == upload && $3 == kernel_symbol_map.rocm.json.gz ]] && exit 1
exit 0
"""
    )
    # Skip the registry retry delays.
    (fake / "sleep").write_text(
        '#!/bin/bash\n[[ -z $REAL_SLEEP ]] || exec /bin/sleep "$@"\n'
    )
    (fake / "date").write_text(
        """#!/bin/bash
[[ -n $FAKE_CLOCK ]] || exec /bin/date "$@"
n=$(cat "$FAKE_CLOCK" 2>/dev/null || echo 0); echo $((n + 1)) > "$FAKE_CLOCK"
echo $((1000000000 + n * 60))
"""
    )
    for tool in fake.iterdir():
        tool.chmod(0o755)
    # A map left by an earlier build in this checkout must never be uploaded.
    (tmp_path / ".rocm-kernel-map").mkdir()
    (tmp_path / ".rocm-kernel-map/kernel_symbol_map.rocm.json.gz").write_text("stale")
    run_env = {
        "PATH": f"{fake}:{os.environ['PATH']}",
        "ROCM_CI_REPO": "rocm/vllm-ci",
        "BUILDKITE_BUILD_ID": "b1",
        "BUILDKITE_COMMIT": "c1",
        "CALLS": str(calls),
        "REGISTRY": str(registry),
        "FAILURE": "",
        "CHANGED": "",
        "APPEARS": "",
        "APPEAR_AFTER": "3",
        "BEAT_AGE": "",
        "FAKE_CLOCK": "",
        "REAL_SLEEP": "",
        "ROCM_TRUSTED_POLL_SECONDS": "0",
        "ROCM_TRUSTED_WAIT_MINUTES": "0",  # tests that wait set a deadline
        "ROCM_TRUSTED_START_GRACE_MINUTES": "0",
        **env,
    }
    result = subprocess.run(
        ["bash", str(ROOT / ".buildkite/scripts/rocm/build.sh"), "rocm", mode],
        cwd=tmp_path,
        env=run_env,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stderr, calls.read_text().splitlines()


run_ci_build = run_build


def built_tags(calls: list[str], stderr: str) -> list[str]:
    return [line.split(" as ", 1)[1] for line in stderr.splitlines() if " as " in line]


TRUSTED_KEYED = tuple(
    f"rocm/vllm-ci:{name}-rocm-" for name in ("stack", "runtime", "test-deps")
)


@pytest.mark.parametrize("mode", ["release", "release-base"])
def test_release_never_ships_images_from_the_ci_repo(tmp_path, mode):
    # The trusted CI repo has every key; release must still build its own into ECR.
    status, stderr, calls = run_build(
        tmp_path,
        mode,
        ROCM_CI_REPO="ecr/release",
        existing=TRUSTED_KEYED,
    )
    assert status == 0, stderr
    assert not any(
        "rocm/vllm-ci:" in line
        for line in stderr.splitlines()
        if line.startswith("Reusing")
    )
    assert [t.split(":")[1].rsplit("-", 1)[0] for t in built_tags(calls, stderr)] == [
        "stack-rocm",
        "runtime-rocm",
    ]
    assert all(t.startswith("ecr/release:") for t in built_tags(calls, stderr))
    assert ("bake vllm-openai" in calls) == (mode == "release")
    # Nor builds from the CI cache: CI agents also run PR code.
    assert "vllm-ci-cache" not in stderr and "cache: own only" in stderr


def test_ci_builds_reuse_the_trusted_images(tmp_path):
    status, stderr, calls = run_build(
        tmp_path,
        "base",
        ROCM_CI_REPO="rocm/vllm-dev",
        existing=TRUSTED_KEYED,
        BUILDKITE_PULL_REQUEST="7",
    )
    assert status == 0, stderr
    assert not built_tags(calls, stderr)
    assert stderr.count("Reusing rocm/vllm-ci:") == 3


@pytest.mark.parametrize(
    ("pull_request", "branch", "prefix"),
    [
        ("7", "feature", "pr7-"),
        ("false", "main", "br-main-"),
        ("false", "releases/v1.2", "br-releases-v1.2-"),
    ],
)
def test_untrusted_builds_get_their_own_namespace(
    tmp_path, pull_request, branch, prefix
):
    status, stderr, calls = run_build(
        tmp_path,
        "base",
        ROCM_CI_REPO="rocm/vllm-dev",
        BUILDKITE_PULL_REQUEST=pull_request,
        BUILDKITE_BRANCH=branch,
    )
    assert status == 0, stderr
    tags = built_tags(calls, stderr)
    assert tags and all(t.startswith(f"rocm/vllm-dev:{prefix}") for t in tags)


def test_ci_build_uploads_the_kernel_symbol_map_when_asked(tmp_path):
    status, stderr, calls = run_ci_build(tmp_path, VLLM_KERNEL_SYMBOL_MAP="1")
    assert status == 0, stderr
    assert calls == [
        "bake ci",
        "upload tmp_path/.rocm-wheel/vllm-*.whl".replace("tmp_path", tmp_path.name),
        "bake kernel-symbol-map",
        "upload .rocm-kernel-map/kernel_symbol_map.rocm.json.gz",
    ]


def test_ci_build_skips_the_kernel_symbol_map_by_default(tmp_path):
    status, stderr, calls = run_ci_build(tmp_path)
    assert status == 0, stderr
    assert "bake kernel-symbol-map" not in calls


@pytest.mark.parametrize("failure", ["bake", "missing", "upload"])
def test_a_kernel_symbol_map_failure_never_fails_the_image_build(tmp_path, failure):
    status, stderr, calls = run_ci_build(
        tmp_path, VLLM_KERNEL_SYMBOL_MAP="1", FAILURE=failure
    )
    assert status == 0, stderr
    assert "continuing without it" in stderr
    uploaded = "upload .rocm-kernel-map/kernel_symbol_map.rocm.json.gz" in calls
    assert uploaded == (failure == "upload")  # the stale map is never uploaded


@pytest.mark.parametrize("mode", ["base", "release", "release-base", "keys"])
def test_a_registry_error_fails_the_step_without_building(tmp_path, mode):
    # Not a miss: no bake, and nothing left running (run() waits for stderr's EOF).
    status, stderr, calls = run_build(
        tmp_path, mode, ROCM_CI_REPO="rocm/vllm-dev", FAILURE="registry"
    )
    assert status == 75, stderr
    assert "registry error" in stderr
    assert not any(call.startswith("bake") for call in calls)


def test_pushed_images_are_pinned_by_the_bake_digest_not_by_tag(tmp_path):
    # A tag lookup after the push could see a concurrent build's push of that key.
    status, stderr, _ = run_build(
        tmp_path, "base", ROCM_CI_REPO="rocm/vllm-dev", FAILURE="pushed"
    )
    assert status == 0, stderr


def test_build_refuses_a_checkout_that_is_not_the_commit(tmp_path):
    # The context and the keys come from HEAD: it must be the commit under test.
    status, stderr, calls = run_build(tmp_path, "base", HEAD_REV="c0")
    assert status == 1 and "HEAD is not BUILDKITE_COMMIT" in stderr
    assert not calls


@pytest.mark.skipif(
    Path("/run/buildkit/buildkitd.sock").exists(), reason="uses the host buildkitd"
)
@pytest.mark.parametrize("mode", ["base", "release"])
def test_only_release_builders_share_the_host_network(tmp_path, mode):
    # Release needs EC2 IMDS (sccache); CI builders also run PR code.
    _, stderr, calls = run_build(tmp_path, mode, NO_BUILDER="1")
    (builder,) = [call for call in calls if call.startswith("builder")]
    name = "vllm-rocm-release" if mode == "release" else "vllm-rocm"
    assert builder.startswith(f"builder --name {name} ")
    assert ("network=host" in builder) == (mode == "release"), stderr


def untrusted_base(tmp_path, **env):
    return run_build(
        tmp_path,
        "base",
        ROCM_CI_REPO="rocm/vllm-dev",
        BUILDKITE_PULL_REQUEST="7",
        **env,
    )


def test_a_pr_with_mains_keys_waits_for_mains_images(tmp_path):
    # Main's base build pushes the stack while the PR waits; by then its runtime and
    # test-deps are there too.
    status, stderr, calls = untrusted_base(
        tmp_path, APPEARS="rocm/vllm-ci:", BEAT_AGE="60", ROCM_TRUSTED_WAIT_MINUTES="5"
    )
    assert status == 0, stderr
    assert not built_tags(calls, stderr)
    assert "waiting for main's build" in stderr
    assert stderr.count("Main's build pushed rocm/vllm-ci:stack-rocm-") == 1
    assert stderr.count("Reusing rocm/vllm-ci:") == 2
    # The stack's -wheels sibling is pinned too, as when the stack is found at once.
    (wheels,) = [c for c in calls if c.startswith("meta rocm-rocm-stack-wheels ")]
    assert re.fullmatch(
        r"\S+ \S+ rocm/vllm-ci:stack-rocm-\w+-wheels@sha256:\w+", wheels
    )


def test_a_pr_builds_mains_images_itself_when_they_never_come(tmp_path):
    # Main's build keeps beating but never pushes: the deadline ends the wait.
    status, stderr, calls = untrusted_base(
        tmp_path, BEAT_AGE="60", ROCM_TRUSTED_WAIT_MINUTES="0"
    )
    assert status == 0, stderr
    assert "did not appear in time" in stderr
    assert [t.split(":")[1].rsplit("-", 1)[0] for t in built_tags(calls, stderr)] == [
        "pr7-stack-rocm",
        "pr7-runtime-rocm",
        "pr7-test-deps-rocm",
    ]


def test_a_pr_that_changes_the_stack_builds_it_without_waiting(tmp_path):
    status, stderr, calls = untrusted_base(
        tmp_path, CHANGED="docker/Dockerfile.rocm_base", ROCM_TRUSTED_WAIT_MINUTES="5"
    )
    assert status == 0, stderr
    assert "waiting for main's build" not in stderr
    assert len(built_tags(calls, stderr)) == 3


def test_a_pr_that_changes_requirements_waits_only_for_the_stack(tmp_path):
    status, stderr, calls = untrusted_base(
        tmp_path,
        CHANGED="requirements/rocm.txt",
        APPEARS="rocm/vllm-ci:stack-rocm-",
        BEAT_AGE="60",
        ROCM_TRUSTED_WAIT_MINUTES="5",
    )
    assert status == 0, stderr
    assert stderr.count("waiting for main's build") == 1
    assert [t.split(":")[1].rsplit("-", 1)[0] for t in built_tags(calls, stderr)] == [
        "pr7-runtime-rocm",
        "pr7-test-deps-rocm",
    ]


def test_trusted_builds_never_wait(tmp_path):
    status, stderr, calls = run_build(tmp_path, "base", ROCM_TRUSTED_WAIT_MINUTES="5")
    assert status == 0, stderr
    assert "waiting for main's build" not in stderr
    assert len(built_tags(calls, stderr)) == 3


@pytest.mark.parametrize("beat_age", ["", "3600"])
def test_without_a_live_build_of_mains_image_a_pr_builds_it_at_once(tmp_path, beat_age):
    # No heartbeat at all, or one an hour old: main is not building it.
    status, stderr, calls = untrusted_base(
        tmp_path, BEAT_AGE=beat_age, ROCM_TRUSTED_WAIT_MINUTES="300"
    )
    assert status == 0, stderr
    assert stderr.count("No build of rocm/vllm-ci:") == 3
    assert len(built_tags(calls, stderr)) == 3


def test_a_pr_gives_mains_build_time_to_start(tmp_path):
    # No heartbeat yet, but main's build pushes the images within the start grace.
    status, stderr, calls = untrusted_base(
        tmp_path,
        APPEARS="rocm/vllm-ci:",
        ROCM_TRUSTED_START_GRACE_MINUTES="10",
        ROCM_TRUSTED_WAIT_MINUTES="300",
    )
    assert status == 0, stderr
    assert "No build of" not in stderr
    assert not built_tags(calls, stderr)


def test_a_registry_error_while_waiting_fails_the_step_without_building(tmp_path):
    status, stderr, calls = untrusted_base(
        tmp_path,
        APPEARS="rocm/vllm-ci:",
        BEAT_AGE="60",
        FAILURE="appear",
        ROCM_TRUSTED_WAIT_MINUTES="5",
    )
    assert status == 75, stderr
    assert not built_tags(calls, stderr)


def test_trusted_builds_mark_what_they_build(tmp_path):
    status, stderr, calls = run_build(tmp_path, "base")
    assert status == 0, stderr
    beats = [c.split()[1] for c in calls if c.startswith("heartbeat ")]
    assert {b.split(":", 1)[1].rsplit("-", 1)[0] for b in beats} == {
        "building-stack-rocm",
        "building-runtime-rocm",
        "building-test-deps-rocm",
    }
    assert all(b.startswith("rocm/vllm-ci:building-") for b in beats)


@pytest.mark.parametrize(
    ("mode", "repo"),
    [
        ("base", "rocm/vllm-dev"),
        ("release", "ecr/release"),
        ("release-base", "ecr/release"),
    ],
)
def test_only_trusted_base_builds_mark(tmp_path, mode, repo):
    status, stderr, calls = run_build(
        tmp_path,
        mode,
        ROCM_CI_REPO=repo,
        BUILDKITE_PULL_REQUEST="7",
        CHANGED="docker/Dockerfile.rocm_base",
    )
    assert status == 0, stderr
    assert built_tags(calls, stderr)
    assert not any(c.startswith("heartbeat ") for c in calls)


@pytest.mark.parametrize("failure", ["fetch-main", "merge-base"])
def test_a_pr_that_cannot_read_main_builds_without_waiting(tmp_path, failure):
    status, stderr, calls = untrusted_base(
        tmp_path, FAILURE=failure, BEAT_AGE="60", ROCM_TRUSTED_WAIT_MINUTES="5"
    )
    assert status == 0, stderr
    assert "waiting for main's build" not in stderr
    assert len(built_tags(calls, stderr)) == 3


def test_a_pr_gives_main_the_start_grace_once_per_build(tmp_path):
    # Main is not building: after the stack's grace, the runtime and test-deps
    # build without waiting again.
    status, stderr, calls = untrusted_base(
        tmp_path,
        FAKE_CLOCK=str(tmp_path / "clock"),
        ROCM_TRUSTED_START_GRACE_MINUTES="5",
        ROCM_TRUSTED_WAIT_MINUTES="300",
    )
    assert status == 0, stderr
    polls = [c.split(":building-")[1].rsplit("-", 1)[0] for c in calls if "beat?" in c]
    assert polls.count("stack-rocm") > 1
    assert polls.count("runtime-rocm") == polls.count("test-deps-rocm") == 1
    assert len(built_tags(calls, stderr)) == 3


def test_the_heartbeat_does_not_hold_up_the_build(tmp_path):
    # The refresh loop must neither keep $(ensure ...) open nor outlive the bake.
    start = time.monotonic()
    status, stderr, _ = run_build(
        tmp_path, "base", REAL_SLEEP="1", ROCM_HEARTBEAT_SECONDS="20"
    )
    assert status == 0, stderr
    assert time.monotonic() - start < 15
