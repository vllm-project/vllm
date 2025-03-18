# SPDX-License-Identifier: Apache-2.0
from vllm.v1.worker.tpu_model_runner import (_get_padded_token_len,
                                             _get_paddings)


def test_get_paddings():
    min_token_size, max_token_size, padding_gap = 16, 512, 64
    expected_paddings = [16, 32, 64, 128, 192, 256, 320, 384, 448, 512]
    actual_paddings = _get_paddings(min_token_size, max_token_size,
                                    padding_gap)
    assert actual_paddings == expected_paddings


def test_get_padded_token_len():
    min_token_size, max_token_size, padding_gap = 16, 512, 64
    paddings = _get_paddings(min_token_size, max_token_size, padding_gap)
    assert _get_padded_token_len(paddings, 1) == 16
    assert _get_padded_token_len(paddings, 16) == 16
    assert _get_padded_token_len(paddings, 20) == 32
    assert _get_padded_token_len(paddings, 300) == 320
    assert _get_padded_token_len(paddings, 512) == 512
