# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.watermarking import WatermarkPRFName
from vllm.v1.watermarking.prfs.base import WatermarkPRF
from vllm.v1.watermarking.prfs.hmac_sha256 import HMACSHA256PRF
from vllm.v1.watermarking.prfs.philox import PhiloxPRF


def create_prf(name: WatermarkPRFName, key: int) -> WatermarkPRF:
    if name == "philox":
        return PhiloxPRF(key)
    if name == "hmac_sha256":
        return HMACSHA256PRF(key)
    raise ValueError(f"Unknown watermark PRF: {name}")


__all__ = [
    "HMACSHA256PRF",
    "PhiloxPRF",
    "WatermarkPRF",
    "create_prf",
]
