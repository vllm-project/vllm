# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ROCm image keys in .buildkite/scripts/rocm/build.sh must cover every
build-context file the keyed Dockerfile parts read, or editing that file would
ship a stale image."""

import os
import subprocess
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
    block_match = re.search(rf"^{name}=\$\(.*?sha256sum", script, re.M | re.S)
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
    makes every registry lookup fail, FAILURE=pushed only lookups of pushed refs."""
    (tmp_path / "docker").symlink_to(ROOT / "docker")
    fake = tmp_path / "bin"
    fake.mkdir()
    calls = tmp_path / "calls"
    calls.touch()
    registry = tmp_path / "registry"
    registry.write_text("".join(f"{ref}\n" for ref in existing))
    (fake / "git").write_text(
        """#!/bin/bash
case $1 in
  describe) echo v0.0.0-0-g0 ;;
  rev-parse) [[ $2 == HEAD ]] && echo "${HEAD_REV:-c1}" || echo "${2%%^*}" ;;
  log) echo 0 ;;
  archive) tar -ch docker ;;
esac
"""
    )
    (fake / "docker").write_text(
        """#!/bin/bash
[[ $1 == buildx ]] || exit 0
case $2 in
  inspect) [[ -z $NO_BUILDER ]] || exit 1 ;;
  create) echo "builder ${*:3}" >> "$CALLS" ;;
  imagetools)
    if [[ $3 == inspect ]]; then
      [[ $FAILURE == registry ]] && { echo "503 Service Unavailable" >&2; exit 1; }
      found=0
      while read -r ref; do [[ $4 == "$ref"* ]] && found=1; done < "$REGISTRY"
      [[ $FAILURE == pushed ]] && ((found)) && { echo "503" >&2; exit 1; }
      ((found)) || { echo "$4: not found" >&2; exit 1; }
      echo "Digest: sha256:$(printf '0%.0s' {1..64})"
    else
      echo "create ${@: -1}" >> "$CALLS"
    fi ;;
  bake)
    echo "bake ${*:5}" >> "$CALLS"
    echo "cache: ${CACHE:-own only}, to: ${CACHE_TO:-none}" >&2
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
[[ $1 == meta-data ]] && { echo "ref@sha256:$(printf '0%.0s' {1..64})"; exit 0; }
echo "upload ${PWD##*/}/$3" >> "$CALLS"
[[ $FAILURE == upload && $3 == kernel_symbol_map.rocm.json.gz ]] && exit 1
exit 0
"""
    )
    (fake / "sleep").write_text("#!/bin/bash\n")  # skip the registry retry delays
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


@pytest.mark.parametrize(("pull_request", "prefix"), [("7", "pr7-"), ("false", "br-")])
def test_only_pr_builds_read_back_their_rocm_vllm_dev_namespace(
    tmp_path, pull_request, prefix
):
    # Any PR can push to rocm/vllm-dev, e.g. the br-main- tags a nightly would reuse.
    status, stderr, calls = run_build(
        tmp_path,
        "base",
        ROCM_CI_REPO="rocm/vllm-dev",
        existing=(f"rocm/vllm-dev:{prefix}",),
        BUILDKITE_PULL_REQUEST=pull_request,
    )
    assert status == 0, stderr
    is_pr = pull_request != "false"
    assert (stderr.count("Reusing rocm/vllm-dev:") == 3) == is_pr
    assert (len(built_tags(calls, stderr)) == 3) != is_pr
    assert ("to: none" in stderr) != is_pr


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
