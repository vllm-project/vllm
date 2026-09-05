# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config.watermarking import WatermarkPRFName
from vllm.v1.watermarking import GumbelWatermarkDetector

_NON_WATERMARKED_TOKEN_IDS = list(range(64))


@pytest.mark.parametrize(
    ("prf_name", "watermarked_token_ids"),
    [
        (
            "philox",
            [
                85,
                56,
                123,
                77,
                105,
                62,
                9,
                104,
                16,
                98,
                123,
                22,
                16,
                35,
                47,
                127,
                77,
                67,
                19,
                2,
                10,
                8,
                75,
                22,
                127,
                30,
                95,
                9,
                101,
                14,
                113,
                122,
                85,
                66,
                122,
                50,
                38,
                119,
                15,
                58,
                68,
                62,
                24,
                126,
                80,
                126,
                68,
                83,
                81,
                126,
                62,
                31,
                18,
                81,
                64,
                47,
                22,
                70,
                47,
                109,
                54,
                21,
                65,
                38,
            ],
        ),
        (
            "hmac_sha256",
            [
                116,
                1,
                11,
                21,
                75,
                65,
                2,
                80,
                49,
                65,
                11,
                60,
                56,
                39,
                120,
                119,
                27,
                96,
                123,
                10,
                45,
                16,
                14,
                54,
                43,
                93,
                72,
                11,
                30,
                98,
                37,
                69,
                14,
                57,
                120,
                108,
                13,
                76,
                112,
                49,
                89,
                112,
                85,
                32,
                28,
                21,
                34,
                61,
                121,
                119,
                25,
                30,
                5,
                7,
                41,
                55,
                61,
                91,
                59,
                30,
                36,
                120,
                93,
                66,
            ],
        ),
    ],
)
def test_detection_on_known_sequences(
    prf_name: WatermarkPRFName, watermarked_token_ids: list[int]
):
    detector = GumbelWatermarkDetector(key=42, context_width=4, prf=prf_name)

    assert detector.detect(watermarked_token_ids).is_watermarked
    assert not detector.detect(_NON_WATERMARKED_TOKEN_IDS).is_watermarked


@pytest.mark.parametrize("prf_name", ["philox", "hmac_sha256"])
def test_detector_rejects_empty_input(prf_name: WatermarkPRFName):
    detection = GumbelWatermarkDetector(key=42, prf=prf_name).detect([])

    assert detection.score == 0
    assert detection.p_value == 1
    assert not detection.is_watermarked
