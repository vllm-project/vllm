# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.warmup.step5_dsa_warmup import _step5_dsa_impl


def test_step5_warmup_finds_rocm_csa_indexer():
    indexer = object()

    class RocmCSALayer:
        def __init__(self):
            self.indexer = indexer
            self.csa_tail_ring = object()

    assert _step5_dsa_impl({"step5.csa": RocmCSALayer()}) is indexer
