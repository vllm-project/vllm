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


def _read_requirements(file: str):
    """ Reads requirements.txt files recursively descending into files included with -r

    - ignores comments
    - ignores empty lines
    """
    requirements: list[str] = []

    with open(file) as fh:
        for line in fh.readlines():
            line = (
                line.strip()  # remove newlines
                .split("#")[0]  # remove comments
            )
            if not line:
                continue

            if line.startswith("-r "):  # resolve other requirements.txt files
                requirements += _read_requirements(line.split()[1])
                continue

            requirements.append(line)

    return requirements


def _get_requires_for_build_extensions() -> list[str]:
    """ returns the requirements for extensions builds"""
    return _read_requirements("requirements-build.txt")


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


def get_requires_for_build_wheel(config_settings=None) -> list[str]:
    """ Dynamically computes the wheel build requirements based on VLLM_TARGET_DEVICE

    torch versions here will have to be kept in sync with the corresponding `requirements-<device>.txt` files.`
    """
    requirements_extras: list[str] = []

    if VLLM_TARGET_DEVICE == "cpu" or VLLM_TARGET_DEVICE == "openvino":
        from platform import machine

        if machine() == "ppc64le":
            requirements_extras.extend(
                _read_requirements("requirements/torch-ppc64le.txt"))
        else:
            requirements_extras.extend(
                _read_requirements("requirements/torch-cpu.txt"))
    elif VLLM_TARGET_DEVICE == "cuda":
        from platform import machine, system

        if machine() == "aarch64" and system() == "Linux":
            requirements_extras.extend(
                _read_requirements("requirements/torch-cuda-aarch64.txt"))
        else:
            requirements_extras.extend(
                _read_requirements("requirements/torch-cuda.txt"))
    elif VLLM_TARGET_DEVICE == "rocm":
        requirements_extras.extend(
            _read_requirements("requirements/torch-rocm.txt"))
    elif VLLM_TARGET_DEVICE == "neuron":
        requirements_extras.extend(
            _read_requirements("requirements/torch-neuron.txt"))
    elif VLLM_TARGET_DEVICE == "tpu":
        requirements_extras.extend(
            _read_requirements("requirements/torch-tpu.txt"))
    elif VLLM_TARGET_DEVICE == "xpu":
        requirements_extras.extend(
            _read_requirements("requirements/torch-xpu.txt"))
    elif VLLM_TARGET_DEVICE == "hpu":  # noqa: SIM114
        pass
    elif VLLM_TARGET_DEVICE == "empty":
        pass
    else:
        raise RuntimeError(
            f"Unknown runtime environment {VLLM_TARGET_DEVICE=}")

    has_uv = which("uv")
    if has_uv and any("--extra-index-url" in req
                      for req in requirements_extras):
        _check_for_env_var("UV_INDEX_STRATEGY",
                           "unsafe-best-match",
                           strict=True)

    if has_uv and any("--pre" in req for req in requirements_extras):
        _check_for_env_var("UV_PRERELEASE", "allow", strict=True)

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
