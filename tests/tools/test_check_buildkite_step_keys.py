# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Buildkite step key pre-commit check."""

import textwrap
from pathlib import Path

import pytest
import yaml

from tools.pre_commit.check_buildkite_step_keys import (
    device_job_keys,
    device_key_problems,
    device_key_tag,
    device_vendor,
    duplicate_keys,
    main,
    scan,
)

SOURCE = Path("synthetic.yaml")


def _steps(body: str) -> list:
    return yaml.safe_load(textwrap.dedent(body))


def _keyless(body: str) -> list[str]:
    return scan(_steps(body), SOURCE)


def _keys(body: str) -> list[tuple[str, str]]:
    (step,) = _steps(body)
    return [(job, key) for job, key, _ in device_job_keys(step)]


def _problems(body: str) -> list[str]:
    return device_key_problems(_steps(body))


def test_step_without_key_is_reported():
    assert _keyless(
        """
        - label: Keyed
          key: keyed
        - label: Keyless
        - wait
        """
    ) == ["Keyless"]


def test_nested_steps_are_rejected():
    with pytest.raises(SystemExit):
        _keyless(
            """
            - label: Group
              key: group
              steps:
                - label: Child
                  key: child
            """
        )


@pytest.mark.parametrize(
    ("device", "vendor", "tag"),
    [
        ("h100", "nvidia", "h100"),
        ("h200_35gb", "nvidia", "h200"),
        ("h200_18gb", "nvidia", "h200"),
        ("b200-k8s", "nvidia", "b200"),
        ("dgx-spark", "nvidia", "dgx-spark"),
        ("cpu-small", "cpu", "cpu"),
        ("cpu", "cpu", "cpu"),
        ("mi355_4", "amd", "mi355"),
        ("mi300_dpx", "amd", "mi300-dpx"),
        ("mi300_dpx_2", "amd", "mi300-dpx"),
        ("r9700_1", "amd", "r9700"),
    ],
)
def test_device_vendor_and_key_tag(device, vendor, tag):
    assert device_vendor(device) == vendor
    assert device_key_tag(device) == tag


def test_first_device_of_each_side_takes_the_step_key():
    assert _keys(
        """
        - label: V1 Attention
          key: v1-attention
          devices:
            h100:
            mi300_1:
        """
    ) == [("h100", "v1-attention"), ("mi300_1", "amd-v1-attention")]


def test_other_devices_add_their_key_tag():
    assert _keys(
        """
        - label: Fused MoE
          key: fused-moe
          devices:
            h100:
            b200-k8s:
              same_as: h100
            cpu-small:
            mi355_1:
            mi300_dpx:
        """
    ) == [
        ("h100", "fused-moe"),
        ("b200-k8s", "fused-moe-b200"),
        ("cpu-small", "fused-moe-cpu"),
        ("mi355_1", "amd-fused-moe"),
        ("mi300_dpx", "amd-fused-moe-mi300-dpx"),
    ]


@pytest.mark.parametrize(
    ("devices", "expected"),
    [
        # An AMD-only step with no pinned key still takes the `amd-` prefix.
        (
            ["mi300_1"],
            [("mi300_1", "amd-kernels")],
        ),
        (
            ["mi355_1", "mi300_1"],
            [("mi355_1", "amd-kernels"), ("mi300_1", "amd-kernels-mi300")],
        ),
        # A CPU device listed first is the step's first non-AMD job.
        (
            ["cpu-small", "h100"],
            [("cpu-small", "kernels"), ("h100", "kernels-h100")],
        ),
        (
            ["cpu-small", "mi300_1", "h100"],
            [
                ("cpu-small", "kernels"),
                ("mi300_1", "amd-kernels"),
                ("h100", "kernels-h100"),
            ],
        ),
    ],
)
def test_unpinned_keys_follow_device_order(devices, expected):
    entries = "".join(f"\n    {device}:" for device in devices)
    assert _keys(f"- label: Kernels\n  key: kernels\n  devices:{entries}\n") == (
        expected
    )


def test_variants_and_pinned_keys():
    assert _keys(
        """
        - label: Fusion E2E
          key: fusion-e2e
          devices:
            mi355_2:
              key: fusion-e2e-amd
              variants:
                eager:
                  env:
                    VLLM_ENFORCE_EAGER: "1"
                extended:
                  key: fusion-e2e-extended
                  timeout_in_minutes: 90
        """
    ) == [
        ("mi355_2", "fusion-e2e-amd"),
        ("mi355_2 variant eager", "fusion-e2e-amd-eager"),
        ("mi355_2 variant extended", "fusion-e2e-extended"),
    ]


def test_same_as_copies_no_key_and_no_variants():
    assert _keys(
        """
        - label: Kernels
          key: kernels
          devices:
            mi355_1:
              key: kernels-mi355
              variants:
                eager:
                  env:
                    VLLM_ENFORCE_EAGER: "1"
            mi300_1:
              same_as: mi355_1
        """
    ) == [
        ("mi355_1", "kernels-mi355"),
        ("mi355_1 variant eager", "kernels-mi355-eager"),
        ("mi300_1", "amd-kernels-mi300"),
    ]


def test_shared_key_tag_needs_a_pin():
    body = """
        - label: Kernels
          key: kernels
          devices:
            mi355_1:
            mi355_4:
        """
    assert _problems(body) == [
        "Kernels (mi355_4): shares the key tag `mi355` with mi355_1; pin `key:` on one"
    ]
    assert not _problems(
        """
        - label: Kernels
          key: kernels
          devices:
            mi355_1:
            mi355_4:
              key: kernels-4
        """
    )


def test_two_jobs_on_one_key_are_reported():
    assert _problems(
        """
        - label: Kernels
          key: kernels
          devices:
            h100:
              variants:
                b200:
                  timeout_in_minutes: 5
            b200-k8s:
        """
    ) == ["Kernels (b200-k8s): `kernels-b200` is also h100 variant b200's key"]


def test_empty_pinned_key_is_reported():
    assert _problems(
        """
        - label: Kernels
          key: kernels
          devices:
            mi300_dpx:
              key: ""
        """
    ) == ["Kernels (mi300_dpx): `key:` is empty; drop it or name the job's key"]


@pytest.mark.parametrize("value", ['""', "null", "' '"])
def test_empty_pinned_variant_key_is_reported(value):
    (problem,) = _problems(
        f"""
        - label: Kernels
          key: kernels
          devices:
            h100:
              variants:
                eager:
                  key: {value}
        """
    )
    assert problem == (
        "Kernels (h100 variant eager): `key:` is empty; drop it or name the job's key"
    )


def test_duplicate_keys_across_files(tmp_path):
    first, second = tmp_path / "a.yaml", tmp_path / "b.yaml"
    first.write_text(
        textwrap.dedent(
            """
            steps:
            - label: Old
              key: kernels
              mirror:
                amd:
                  device: mi355_1
            """
        )
    )
    second.write_text(
        textwrap.dedent(
            """
            steps:
            - label: New
              key: kernels
              devices:
                mi355_1:
            """
        )
    )
    assert duplicate_keys([first, second]) == [
        f"`amd-kernels` is the key of 2 jobs, in {first}, {second}"
    ]
    # Two older steps on one key are not this check's to report.
    assert duplicate_keys([first, first]) == []


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body))


DEVICE_STEP = """
    steps:
    - label: Kernels
      key: kernels
      devices:
        h100:
    """


def test_keys_are_unique_per_pipeline(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / ".buildkite/ci_config.yaml", "job_dirs:\n- .buildkite/cuda\n")
    _write(
        tmp_path / ".buildkite/ci_config_intel.yaml",
        "job_dirs:\n- .buildkite/intel\n",
    )
    _write(tmp_path / ".buildkite/cuda/kernels.yaml", DEVICE_STEP)
    # Another pipeline may publish a job on a key the CUDA one uses.
    _write(tmp_path / ".buildkite/intel/kernels.yaml", DEVICE_STEP)
    assert main() == 0

    # Within one pipeline, two files on one key are reported.
    _write(tmp_path / ".buildkite/cuda/more.yaml", DEVICE_STEP)
    assert main() == 1
    err = capsys.readouterr().err
    assert "`kernels` is the key of 2 jobs" in err
    assert ".buildkite/cuda/kernels.yaml" in err
    assert ".buildkite/cuda/more.yaml" in err
    assert ".buildkite/intel/" not in err
