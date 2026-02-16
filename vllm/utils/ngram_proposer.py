# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Provides lazy import of the ngram_proposer module."""

from typing import TYPE_CHECKING

from vllm.utils.import_utils import LazyLoader

if TYPE_CHECKING:
    # if type checking, eagerly import the module
    import vllm.v1.spec_decode.ngram_proposer as ngp
else:
    ngp = LazyLoader("ngp", globals(), "vllm.v1.spec_decode.ngram_proposer")
