# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.pooling.base.io_processor import PoolingIOProcessor


class ClassifyIOProcessor(PoolingIOProcessor):
    name = "classification"
