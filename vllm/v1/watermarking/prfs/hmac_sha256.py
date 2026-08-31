# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""HMAC-SHA-256 following https://www.rfc-editor.org/rfc/rfc2104."""

import hashlib
import hmac
import struct

import torch

from vllm.v1.watermarking.prfs.base import WatermarkPRF, uint32_to_uniform

_HMAC_DOMAIN = b"vllm-watermark-hmac-sha256-v1\0"


class HMACSHA256PRF(WatermarkPRF):
    """Cryptographically secure reference watermark PRF."""

    version = "hmac-sha256-v1"

    def __init__(self, key: int) -> None:
        if not 0 <= key <= 2**256 - 1:
            raise ValueError("HMAC-SHA-256 keys must fit in 256 bits")
        self.key = key
        self._key_bytes = key.to_bytes(32, byteorder="little")

    def uniform(self, contexts: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        output_shape = torch.broadcast_shapes(
            contexts.shape[:-1] + (1,), token_ids.shape
        )
        expanded_contexts = torch.broadcast_to(
            contexts.unsqueeze(-2), output_shape + (contexts.shape[-1],)
        )
        expanded_tokens = torch.broadcast_to(token_ids, output_shape)
        context_rows = expanded_contexts.detach().cpu().reshape(-1, contexts.shape[-1])
        token_values = expanded_tokens.detach().cpu().reshape(-1)

        words = []
        for context, token_id in zip(
            context_rows.tolist(), token_values.tolist(), strict=True
        ):
            message = bytearray(_HMAC_DOMAIN)
            message.extend(struct.pack("<I", len(context)))
            for context_token in context:
                message.extend(struct.pack("<q", context_token))
            message.extend(struct.pack("<q", token_id))
            digest = hmac.digest(self._key_bytes, message, hashlib.sha256)
            words.append(int.from_bytes(digest[:4], byteorder="little"))

        values = torch.tensor(words, dtype=torch.int64, device=contexts.device).reshape(
            output_shape
        )
        return uint32_to_uniform(values)
