# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = REPO_ROOT / "tools" / "pre_commit" / "check_rocm_docker_config.py"


def load_module():
    spec = spec_from_file_location("check_rocm_docker_config", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_dockerfile(tmp_path: Path, *, base_image: str, rocm_arch: str) -> Path:
    dockerfile = tmp_path / "Dockerfile.rocm_base"
    dockerfile.write_text(
        "\n".join(
            [
                f"ARG BASE_IMAGE={base_image}",
                f"ARG PYTORCH_ROCM_ARCH={rocm_arch}",
                "",
            ]
        )
    )
    return dockerfile


def test_accepts_gfx115x_with_rocm_7_2_or_newer(tmp_path: Path):
    module = load_module()
    dockerfile = write_dockerfile(
        tmp_path,
        base_image="rocm/dev-ubuntu-22.04:7.2.1-complete",
        rocm_arch="gfx90a;gfx942;gfx1150;gfx1151",
    )

    assert module.validate_rocm_docker_config(dockerfile) == []


def test_accepts_pre_7_2_base_when_gfx115x_is_not_advertised(tmp_path: Path):
    module = load_module()
    dockerfile = write_dockerfile(
        tmp_path,
        base_image="rocm/dev-ubuntu-22.04:7.0.2-complete",
        rocm_arch="gfx90a;gfx942;gfx1100;gfx1101",
    )

    assert module.validate_rocm_docker_config(dockerfile) == []


def test_rejects_gfx115x_with_pre_7_2_rocm_base(tmp_path: Path):
    module = load_module()
    dockerfile = write_dockerfile(
        tmp_path,
        base_image="rocm/dev-ubuntu-22.04:7.1.1-complete",
        rocm_arch="gfx90a;gfx942;gfx1150",
    )

    errors = module.validate_rocm_docker_config(dockerfile)

    assert len(errors) == 1
    assert "gfx1150" in errors[0]
    assert "ROCm 7.2+" in errors[0]
    assert "#31333" in errors[0]
