# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from setuptools import setup

setup(
    name="prithvi_io_processor_plugin",
    version="0.1",
    packages=["prithvi_io_processor"],
    entry_points={
        "vllm.io_processor_plugins": [
            "prithvi_to_tiff = prithvi_io_processor:register_prithvi",  # noqa: E501
        ]
    },
)
