# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from setuptools import setup

setup(
    name="bge-m3-sparse-plugin",
    version="0.1",
    packages=["bge_m3_sparse_processor"],
    entry_points={
        "vllm.io_processor_plugins": [
            "bge_m3_sparse_plugin = bge_m3_sparse_processor:register_bge_m3_sparse_embeddings_processor",  # noqa: E501
        ]
    },
)
