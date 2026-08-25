# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from ..base.io_processor import PoolingIOProcessor


class ClassifyIOProcessor(PoolingIOProcessor):
    name = "classify"


class TokenClassifyIOProcessor(PoolingIOProcessor):
    name = "token_classify"
