# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from setuptools import setup

setup(
    name="vllm_add_predictable_models",
    version="0.1",
    packages=["vllm_add_predictable_models"],
    entry_points={
        "vllm.general_plugins": [
            "register_predictable_models = vllm_add_predictable_models:register"
        ]
    },
)
