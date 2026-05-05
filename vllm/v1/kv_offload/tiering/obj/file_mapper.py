# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Maps OffloadKeys to S3 object keys."""


class FileMapper:
    """Maps a block hash to an S3 object key.

    Key format:
        {key_prefix}/{model_name}/block_size_{N}/tp_{tp}_pp_{pp}_pcp_{pcp}
        /rank_{rank}/{dtype}/{hash[:3]}/{hash[3:5]}/{hash}.bin
    """

    def __init__(
        self,
        model_name: str,
        gpu_block_size: int,
        tp_size: int,
        pp_size: int,
        pcp_size: int,
        rank: int,
        dtype: str,
        key_prefix: str = "",
    ):
        prefix = f"{key_prefix}/" if key_prefix else ""
        self._base = (
            f"{prefix}{model_name}"
            f"/block_size_{gpu_block_size}"
            f"/tp_{tp_size}_pp_{pp_size}_pcp_{pcp_size}"
            f"/rank_{rank}"
            f"/{dtype}"
        )

    def get_key(self, block_hash: bytes) -> str:
        h = int.from_bytes(block_hash, "big")
        hex_hash = f"{h & ((1 << 64) - 1):016x}"
        return f"{self._base}/{hex_hash[:3]}/{hex_hash[3:5]}/{hex_hash}.bin"
