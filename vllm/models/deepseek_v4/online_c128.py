# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 C128 online compression state."""

from __future__ import annotations

import numpy as np
import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.platforms import current_platform

ONLINE_C128_COMPRESS_RATIO = 128
ONLINE_C128_HEAD_DIM = 512
ONLINE_C128_NUM_STATE_VECTORS = 3
ONLINE_C128_STATE_DTYPE = torch.float32

# Descriptor columns (int32):
#   0: row_base   - first token row for this segment
#   1: num_rows   - number of token rows in this segment
#   2: read_row   - online-state row to seed from, or -1
#   3: emit_token - token row to write compressed_kv to, or -1
#   4: write_row  - online-state row to write the trailing carry to, or -1
SEGMENT_NUM_COLS = 5


def online_c128_compress_enabled() -> bool:
    return bool(envs.VLLM_USE_ONLINE_C128_COMPRESS)


def _is_sm90() -> bool:
    return current_platform.is_cuda() and current_platform.is_device_capability(90)


def assert_online_c128_supported(
    vllm_config: VllmConfig,
    *,
    compress_ratio: int,
    head_dim: int,
) -> None:
    """Fail closed for the first online C128 support envelope."""
    if not _is_sm90():
        raise ValueError(
            "VLLM_USE_ONLINE_C128_COMPRESS is only supported on CUDA SM90 "
            "(Hopper)."
        )
    if compress_ratio != ONLINE_C128_COMPRESS_RATIO:
        raise ValueError(
            "VLLM_USE_ONLINE_C128_COMPRESS requires compress_ratio == "
            f"{ONLINE_C128_COMPRESS_RATIO}, got {compress_ratio}."
        )
    if head_dim != ONLINE_C128_HEAD_DIM:
        raise ValueError(
            "VLLM_USE_ONLINE_C128_COMPRESS requires head_dim == "
            f"{ONLINE_C128_HEAD_DIM}, got {head_dim}."
        )

    parallel_config = vllm_config.parallel_config
    if getattr(parallel_config, "decode_context_parallel_size", 1) > 1:
        raise ValueError(
            "VLLM_USE_ONLINE_C128_COMPRESS does not support decode context "
            "parallelism (DCP)."
        )
    if getattr(parallel_config, "prefill_context_parallel_size", 1) > 1:
        raise ValueError(
            "VLLM_USE_ONLINE_C128_COMPRESS does not support prefill context "
            "parallelism (PCP)."
        )

    if getattr(vllm_config, "speculative_config", None) is not None:
        raise ValueError(
            "VLLM_USE_ONLINE_C128_COMPRESS does not support speculative "
            "decoding in this path."
        )

    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is not None and getattr(
        kv_transfer_config, "kv_connector", None
    ):
        raise ValueError(
            "VLLM_USE_ONLINE_C128_COMPRESS does not support KV transfer / "
            "disaggregated prefill."
        )

    from vllm.config.compilation import CUDAGraphMode

    cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
    if cudagraph_mode not in (None, CUDAGraphMode.NONE):
        raise ValueError(
            "VLLM_USE_ONLINE_C128_COMPRESS does not support CUDA graphs yet; "
            "set cudagraph_mode to NONE."
        )


class DeepseekOnlineC128State(torch.nn.Module):
    """Dense per-request running state for one C128 compressor layer."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        head_dim: int,
        layer_index: int,
        device: torch.device | str,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.layer_index = layer_index
        self.num_state_vectors = ONLINE_C128_NUM_STATE_VECTORS
        self.row_width = self.num_state_vectors * head_dim
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs

        self.state = torch.empty(
            (self.max_num_reqs, self.row_width),
            dtype=ONLINE_C128_STATE_DTYPE,
            device=device,
        )
        self.reset_all()

    @property
    def device(self) -> torch.device:
        return self.state.device

    def reset_all(self) -> None:
        self.state[:, : self.head_dim] = float("-inf")
        self.state[:, self.head_dim :] = 0.0

    def reset_rows(self, req_state_indices: torch.Tensor) -> None:
        if req_state_indices.numel() == 0:
            return
        rows = req_state_indices.to(self.state.device, dtype=torch.long)
        rows = rows[rows >= 0]
        if rows.numel() == 0:
            return
        self.state[rows, : self.head_dim] = float("-inf")
        self.state[rows, self.head_dim :] = 0.0


class OnlineC128Plan:
    """Host-built segment plan for one forward step."""

    def __init__(
        self,
        emit_segments: torch.Tensor,
        update_segments: torch.Tensor,
        reset_rows: torch.Tensor,
    ):
        self.emit_segments = emit_segments
        self.update_segments = update_segments
        self.reset_rows = reset_rows

    @property
    def is_empty(self) -> bool:
        return (
            self.emit_segments.shape[0] == 0
            and self.update_segments.shape[0] == 0
            and self.reset_rows.shape[0] == 0
        )


def plan_online_c128_segments(
    query_start_loc_cpu: np.ndarray,
    seq_lens_cpu: np.ndarray,
    req_state_indices_cpu: np.ndarray,
    max_num_reqs: int,
    device: torch.device | str,
    bank_id: int = 0,
    compress_ratio: int = ONLINE_C128_COMPRESS_RATIO,
    req_mask: np.ndarray | None = None,
) -> OnlineC128Plan:
    """Build emit/update/reset segments from CPU batch metadata."""
    del max_num_reqs
    if bank_id != 0:
        raise ValueError("online C128 only supports committed bank 0.")

    emit: list[list[int]] = []
    update: list[list[int]] = []
    reset: list[int] = []
    num_reqs = len(req_state_indices_cpu)

    for req in range(num_reqs):
        if req_mask is not None and not bool(req_mask[req]):
            continue
        rsi = int(req_state_indices_cpu[req])
        if rsi < 0:
            continue
        row_start = int(query_start_loc_cpu[req])
        row_end = int(query_start_loc_cpu[req + 1])
        query_len = row_end - row_start
        if query_len <= 0:
            continue

        seq_end = int(seq_lens_cpu[req])
        first_pos = seq_end - query_len
        carry_len = first_pos % compress_ratio
        cur_row = row_start
        cur_pos = first_pos
        seeded_from_bank = carry_len > 0
        to_boundary = compress_ratio - (cur_pos % compress_ratio)
        last_was_boundary = False

        while cur_row < row_end:
            seg_rows = min(to_boundary, row_end - cur_row)
            closes_chunk = seg_rows == to_boundary
            read_row = rsi if seeded_from_bank else -1
            if closes_chunk:
                emit_token = cur_row + seg_rows - 1
                emit.append([cur_row, seg_rows, read_row, emit_token, -1])
                last_was_boundary = True
            else:
                update.append([cur_row, seg_rows, read_row, -1, rsi])
                last_was_boundary = False

            cur_row += seg_rows
            cur_pos += seg_rows
            seeded_from_bank = False
            to_boundary = compress_ratio

        if last_was_boundary:
            reset.append(rsi)

    emit_t = torch.tensor(
        emit if emit else [], dtype=torch.int32, device=device
    ).reshape(-1, SEGMENT_NUM_COLS)
    update_t = torch.tensor(
        update if update else [], dtype=torch.int32, device=device
    ).reshape(-1, SEGMENT_NUM_COLS)
    reset_t = torch.tensor(reset, dtype=torch.int32, device=device)
    return OnlineC128Plan(emit_t, update_t, reset_t)


_ONLINE_C128_STATES: list[DeepseekOnlineC128State] = []


def register_online_c128_state(state: DeepseekOnlineC128State) -> None:
    _ONLINE_C128_STATES.append(state)


def clear_online_c128_states() -> None:
    _ONLINE_C128_STATES.clear()


def reset_online_c128_state_rows(req_state_indices: torch.Tensor) -> None:
    if not online_c128_compress_enabled():
        return
    for state in _ONLINE_C128_STATES:
        state.reset_rows(req_state_indices)
