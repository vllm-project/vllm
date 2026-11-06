# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone CMake selection tests; run pytest with --noconftest."""

import subprocess
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parents[1] / "cmake" / "cuda_archs.cmake"


def intersect(tmp_path: Path, src: str, tgt: str) -> str:
    script = tmp_path / "run.cmake"
    script.write_text(
        f'include("{HELPER.as_posix()}")\n'
        'cuda_archs_loose_intersection(OUT "${SRC}" "${TGT}")\n'
        'message("RESULT:${OUT}")\n'
    )
    proc = subprocess.run(
        ["cmake", f"-DSRC={src}", f"-DTGT={tgt}", "-P", str(script)],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in proc.stderr.splitlines():
        if line.startswith("RESULT:"):
            return line.removeprefix("RESULT:")
    raise AssertionError(proc.stderr)


@pytest.mark.parametrize(
    "src,tgt,expected",
    [
        ("7.5;8.0;8.6;9.0;9.0a", "8.0;8.9;9.0", "8.0;8.6;9.0;9.0a"),
        ("8.0+PTX", "9.0", "8.0+PTX"),
        ("8.0", "9.0", ""),
        ("9.0", "9.0a;9.0", "9.0a;9.0"),
        ("9.0a", "9.0", "9.0a"),
        ("10.0a", "10.3", ""),
        ("10.0f", "10.0", "10.0f"),
        ("12.0f", "12.1a", "12.1a"),
        ("10.0f", "10.0a;10.7a", "10.0a;10.7a"),
        ("10.0f;10.7f", "10.7a;10.3a", "10.7a;10.3a"),
        ("10.0f;10.7f", "10.7", "10.7f"),
        ("10.0f", "10.1", ""),
        ("10.1f", "10.3", ""),
        ("10.1f", "10.1a", "10.1a"),
        ("10.0f+PTX", "10.3", "10.3f+PTX"),
        ("10.7f", "7.5;8.0;8.6;8.9;9.0;10.0;12.0", ""),
        ("10.7f", "10.0a", ""),
        ("12.1f", "12.0", ""),
        ("10.3f", "10.0a;10.7a", "10.7a"),
        ("10.0a", "10.0f", "10.0a"),
        ("10.0f", "10.7", "10.7f"),
        ("10.0f", "10.0", "10.0f"),
        ("8.0", "8.0f;8.9f", "8.0f;8.0"),
        ("11.0+PTX", "11.0a;12.0", "11.0a+PTX;11.0+PTX"),
        ("10.0;10.0f;10.0a", "10.0", "10.0;10.0a"),
        ("8.0;8.0+PTX", "8.0;8.9", "8.0+PTX"),
        ("10.0f;10.0f+PTX", "10.3", "10.3f+PTX"),
        ("8.0", "8.0;8.0", "8.0"),
        ("", "9.0", ""),
        ("9.0", "", ""),
        ("9.0a;", ";9.0;;", "9.0a"),
        ("10.0;10.0a", "10.0", "10.0;10.0a"),
        ("10.0;10.0f", "10.0", "10.0;10.0f"),
        ("10.0;10.0a", "10.0a", "10.0a"),
        ("10.0;10.0f", "10.0a", "10.0a"),
        ("10.0a", "10.0", "10.0a"),
        ("10.0+PTX;10.0a", "10.0", "10.0+PTX;10.0a"),
        ("10.0;10.0+PTX;10.0f+PTX", "10.0", "10.0+PTX;10.0f+PTX"),
        ("10.0;10.0a", "10.0;10.3", "10.0;10.0a"),
        ("10.0;10.0a", "10.0a;10.0", "10.0a;10.0"),
    ],
    ids=[
        "plain-loose",
        "ptx-cross-major",
        "plain-no-cross-major",
        "plain-takes-tgt-suffix",
        "a-exact",
        "a-not-higher-minor",
        "f-exact",
        "f-higher-minor-arch-specific",
        "f-each-target",
        "f-order-independent",
        "exact-family-precedes-generic-fallback",
        "f-sm101-own-family",
        "f-sm101-no-fallback",
        "f-sm101-exact",
        "f-keeps-ptx",
        "cu134-release-list",
        "lower-minor-arch-specific",
        "lower-minor-plain",
        "skips-lower-minor",
        "no-downgrade-a-to-f",
        "no-downgrade-f-to-plain",
        "no-downgrade-f-to-plain-same",
        "plain-source-reused",
        "ptx-kept-with-target-suffix",
        "strongest-source-suffix",
        "plain-ptx-tie",
        "family-ptx-tie",
        "deduplicated",
        "empty-src",
        "empty-tgt",
        "empty-list-elements",
        "plain-preserved-with-arch-source",
        "plain-preserved-with-family-source",
        "only-arch-target-does-not-add-plain",
        "only-arch-target-from-family-does-not-add-plain",
        "arch-source-does-not-invent-plain",
        "preserved-plain-keeps-ptx",
        "preserved-plain-ptx-tie",
        "plain-preserves-forward-compatible-code",
        "plain-preserved-in-target-order",
    ],
)
def test_cuda_archs_loose_intersection(tmp_path, src, tgt, expected):
    assert intersect(tmp_path, src, tgt) == expected
    # Source order must not decide which features or PTX are compiled.
    assert intersect(tmp_path, ";".join(reversed(src.split(";"))), tgt) == expected


@pytest.mark.parametrize("src,tgt", [("sm_90", "9.0"), ("9.0", "90")])
def test_invalid_arch_is_fatal(tmp_path, src, tgt):
    with pytest.raises(subprocess.CalledProcessError, match="non-zero exit"):
        intersect(tmp_path, src, tgt)


@pytest.mark.parametrize("suffix", ["a", "f"])
def test_suffixed_ptx_generates_sass_and_ptx(tmp_path, suffix):
    """Preserved PTX requests must reach both gencode flags unchanged."""
    utils = HELPER.with_name("utils.cmake")
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.26)\n"
        "project(arch_flags NONE)\n"
        f'include("{utils.as_posix()}")\n'
        f'cuda_archs_loose_intersection(OUT "11.0+PTX" "11.0{suffix}")\n'
        'set_gencode_flags_for_srcs(SRCS probe.cu CUDA_ARCHS "${OUT}")\n'
        "get_source_file_property(FLAGS probe.cu COMPILE_OPTIONS)\n"
        'message("FLAGS:${FLAGS}")\n'
    )
    proc = subprocess.run(
        ["cmake", "-S", str(tmp_path), "-B", str(tmp_path / "build")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert f"arch=compute_110{suffix},code=sm_110{suffix}" in proc.stderr
    assert f"arch=compute_110{suffix},code=compute_110{suffix}" in proc.stderr


@pytest.mark.parametrize("suffix", ["a", "f"])
def test_global_filter_keeps_explicit_plain_and_suffixed_targets(tmp_path, suffix):
    """Keep sm_100 code usable on B300 when sm_100a is also requested (#56548)."""
    assert (
        intersect(
            tmp_path,
            "7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0",
            f"7.5;8.0;8.6;8.7;8.9;9.0a;10.0;10.0{suffix};12.0;12.1",
        )
        == f"7.5;8.0;8.6;8.7;8.9;9.0a;10.0;10.0{suffix};12.0"
    )
