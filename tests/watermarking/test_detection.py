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
                5,
                50,
                19,
                29,
                41,
                38,
                113,
                35,
                9,
                22,
                1,
                102,
                65,
                117,
                107,
                99,
                36,
                25,
                92,
                73,
                54,
                6,
                123,
                29,
                107,
                71,
                90,
                55,
                89,
                87,
                78,
                36,
                12,
                16,
                17,
                127,
                123,
                60,
                79,
                41,
                85,
                117,
                92,
                87,
                92,
                93,
                32,
                30,
                84,
                88,
                94,
                54,
                116,
                43,
                22,
                92,
                23,
                76,
                12,
                127,
                28,
                61,
                74,
                91,
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
