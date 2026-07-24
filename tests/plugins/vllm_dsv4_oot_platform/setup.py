# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from setuptools import setup

setup(
    name="vllm_dsv4_oot_platform",
    version="0.1",
    packages=["vllm_dsv4_oot_platform"],
    entry_points={
        "vllm.platform_plugins": [
            "dsv4_oot_platform_plugin = vllm_dsv4_oot_platform:dsv4_oot_platform_plugin"  # noqa
        ],
    },
)
