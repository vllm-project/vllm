# SPDX-License-Identifier: Apache-2.0

import importlib
import os
import sys
from shutil import which
from textwrap import dedent

from setuptools.build_meta import *


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

# cannot import envs directly because it depends on vllm, which is not installed yet
envs = load_module_from_path('envs', os.path.join(ROOT_DIR, 'vllm', 'envs.py'))
VLLM_TARGET_DEVICE = envs.VLLM_TARGET_DEVICE
VLLM_USE_PRECOMPILED = envs.VLLM_USE_PRECOMPILED


def _get_requires_for_build_extensions() -> list[str]:
    """ returns the requirements for extensions builds"""
    with open("requirements-build.txt") as fh:
        return [line.strip() for line in fh.readlines()]


def _get_requires_for_basic_build() -> list[str]:
    """ returns the base requirements"""
    return [
        req for req in _get_requires_for_build_extensions()
        if "setuptools" in req
    ]


def _check_for_env_var(key: str, expected_value: str, strict: bool = False):
    """Print a warning when the env var's value doesn't match the expected value.

    When strict is set to True, raises SetupError instead of warning.
    """
    value = os.getenv(key)
    if value and value == expected_value:
        return

    warning = (
        f"{key} is not defined, but {'is' if strict else 'might be'} required for this build."
        if value is None else
        f"{key} is set to {value}, but {expected_value} is suggested.")

    if strict:
        from setuptools.errors import SetupError

        raise SetupError(warning)

    msg = dedent(
        """
        ***
        {warning}
        If the build fails, try setting

            {key}={suggested_value}

        in your environment before starting the build.
        ***""", )

    import warnings

    warnings.warn(
        msg.format(warning=warning, key=key, suggested_value=expected_value),
        stacklevel=2,
    )


def _check_for_extra_index_url(expected_value: str, strict: bool = False):
    """Print a warning when the env var's value doesn't match the expected value.

    When strict is set to True, raises SetupError instead of warning.
    """
    has_uv = which("uv")
    if has_uv:
        _check_for_env_var("UV_EXTRA_INDEX_URL", expected_value, strict=strict)
        # need to match pip's index behaviour,
        # see https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes
        _check_for_env_var("UV_INDEX_STRATEGY",
                           "unsafe-best-match",
                           strict=strict)
    else:
        _check_for_env_var("PIP_EXTRA_INDEX_URL",
                           expected_value,
                           strict=strict)


def get_requires_for_build_wheel(config_settings=None) -> list[str]:
    """ Dynamically computes the wheel build requirements based on VLLM_TARGET_DEVICE

    torch versions here will have to be kept in sync with the corresponding `requirements-<device>.txt` files.`
    """
    requirements_extras: list[str] = []

    if VLLM_TARGET_DEVICE == "cpu" or VLLM_TARGET_DEVICE == "openvino":
        _check_for_extra_index_url("https://download.pytorch.org/whl/cpu")

        requirements_extras.append("torch==2.5.1")
    elif VLLM_TARGET_DEVICE == "cuda":
        from platform import machine as _machine

        machine = _machine()
        if machine == "aarch64":  # GH200
            _check_for_extra_index_url(
                "https://download.pytorch.org/whl/nightly/cu126")
            requirements_extras.append("torch==2.7.0.dev20250121+cu126")
        elif machine == "x86_64":
            requirements_extras.append("torch==2.5.1")
        else:
            from setuptools.errors import SetupError
            raise SetupError(f"{machine=} is not supported")

            requirements_extras.append("torch==2.5.1")
    elif VLLM_TARGET_DEVICE == "rocm":
        rocm_supported_versions = ("6.2", )
        requested_rocm_version = os.getenv("VLLM_ROCM_VERSION")
        if not requested_rocm_version:
            raise RuntimeError("Set ROCM_VERSION env var. "
                               f"Supported versions={rocm_supported_versions}")
        if requested_rocm_version not in rocm_supported_versions:
            raise ValueError("Invalid ROCM_VERSION. "
                             f"Supported versions={rocm_supported_versions}")

        _check_for_extra_index_url(
            f"https://download.pytorch.org/whl/nightly/rocm{requested_rocm_version}"
        )
        requirements_extras.extend([
            f"torch==2.5.1+rocm{requested_rocm_version}"
            f"torchvision==0.20.1+rocm${requested_rocm_version}",
        ])
    elif VLLM_TARGET_DEVICE == "neuron":
        _check_for_extra_index_url(
            expected_value="https://pip.repos.neuron.amazonaws.com")
        requirements_extras.extend([
            "torch-neuronx>=2.1.2",
            "neuronx-cc==2.15.*",
        ])
    elif VLLM_TARGET_DEVICE == "tpu":
        _check_for_env_var(
            "PIP_FIND_LINKS",
            expected_value=
            "https://storage.googleapis.com/libtpu-releases/index.html https://storage.googleapis.com/jax-releases/jax_nightly_releases.html https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html",
        )
        torch_xla_base = (
            "https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/"
        )
        requirements_extras.extend([
            "torch==2.6.0.dev20241126+cpu",
            "torch_xla[tpu,pallas]",
        ])
        for python_version in ("3.11", "3.10", "3.9"):
            pyv = python_version.replace(".", "")
            torch_xla_version = "torch_xla-2.6.0.dev20241126"
            req_str = f'torch_xla[tpu] @ {torch_xla_base}/{torch_xla_version}-cp{pyv}-cp{pyv}-linux_x86_64.whl ; python_version == "{python_version}"'
            requirements_extras.append(req_str)
    elif VLLM_TARGET_DEVICE == "xpu":
        requirements_extras.extend([
            "torch @ https://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/ipex_dev/xpu/torch-2.5.0a0%2Bgite84e33f-cp310-cp310-linux_x86_64.whl",
            "intel-extension-for-pytorch @ https://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/ipex_dev/xpu/intel_extension_for_pytorch-2.5.10%2Bgit9d489a8-cp310-cp310-linux_x86_64.whl",
            "oneccl_bind_pt @ https://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/ipex_dev/xpu/oneccl_bind_pt-2.5.0%2Bxpu-cp310-cp310-linux_x86_64.whl",
        ])
    elif VLLM_TARGET_DEVICE == "hpu":  # noqa: SIM114
        pass
    elif VLLM_TARGET_DEVICE == "empty":
        pass
    else:
        raise RuntimeError(
            f"Unknown runtime environment {VLLM_TARGET_DEVICE=}")

    return [
        *_get_requires_for_build_extensions(),
        *requirements_extras,
    ]


def get_requires_for_build_sdist(config_settings=None):
    return _get_requires_for_basic_build()


def get_requires_for_build_editable(config_settings=None):
    if VLLM_USE_PRECOMPILED:
        return _get_requires_for_basic_build()

    return get_requires_for_build_wheel(config_settings)
