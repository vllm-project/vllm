# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.models.deepseek_v4.nvidia import flashmla


def test_indexed_d512_split_topk_keeps_small_c128a_prefills() -> None:
    assert not flashmla._is_indexed_d512_split_topk(128)
    assert flashmla._is_indexed_d512_split_topk(256)
    assert flashmla._is_indexed_d512_split_topk(512)
    assert flashmla._is_indexed_d512_split_topk(1152)
    assert not flashmla._is_indexed_d512_split_topk(1280)
