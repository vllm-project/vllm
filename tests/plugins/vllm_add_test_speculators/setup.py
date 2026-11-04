# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from setuptools import setup

setup(
    name="vllm_add_test_speculators",
    version="0.1",
    packages=["vllm_add_test_speculators"],
    entry_points={
        "vllm.general_plugins": [
            "register_test_speculators = vllm_add_test_speculators:register"
        ]
    },
)
