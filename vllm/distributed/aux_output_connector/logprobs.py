"""Encoding helpers for AuxOutput logprob artifacts."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import torch

from vllm.v1.outputs import LogprobsLists, LogprobsTensors

_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LogprobRows:
    positions: np.ndarray
    token_ids: np.ndarray
    values: np.ndarray
    ranks: np.ndarray


def rows_from_lists(value: LogprobsLists, start: int) -> LogprobRows:
    return LogprobRows(
        np.arange(start, start + len(value.logprobs), dtype=np.int64),
        np.asarray(value.logprob_token_ids, dtype=np.int32),
        np.asarray(value.logprobs, dtype=np.float32),
        np.asarray(value.sampled_token_ranks, dtype=np.int32),
    )


def rows_from_tensors(value: LogprobsTensors, start: int) -> LogprobRows:
    return LogprobRows(
        np.arange(start, start + value.logprobs.shape[0], dtype=np.int64),
        value.logprob_token_ids.detach().cpu().numpy().astype(np.int32, copy=False),
        value.logprobs.detach().cpu().numpy().astype(np.float32, copy=False),
        value.selected_token_ranks.detach().cpu().numpy().astype(np.int32, copy=False),
    )


def concat_rows(chunks: list[LogprobRows]) -> LogprobRows | None:
    if not chunks:
        return None
    positions = np.concatenate([chunk.positions for chunk in chunks])
    order = np.argsort(positions, kind="stable")
    sorted_positions = positions[order]
    keep = np.ones(len(order), dtype=bool)
    if len(order) > 1:
        keep[:-1] = sorted_positions[:-1] != sorted_positions[1:]
    order = order[keep]
    return LogprobRows(
        positions[order],
        np.concatenate([chunk.token_ids for chunk in chunks])[order],
        np.concatenate([chunk.values for chunk in chunks])[order],
        np.concatenate([chunk.ranks for chunk in chunks])[order],
    )


def encode_rows(rows: LogprobRows) -> bytes:
    stream = io.BytesIO()
    np.savez(
        stream,
        schema_version=np.asarray(_SCHEMA_VERSION, dtype=np.int32),
        positions=rows.positions,
        token_ids=rows.token_ids,
        values=rows.values,
        ranks=rows.ranks,
    )
    return stream.getvalue()


def decode_rows(payload: bytes) -> LogprobRows:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            schema_version = data["schema_version"]
            if schema_version.shape != () or schema_version.item() != _SCHEMA_VERSION:
                raise ValueError("unsupported auxiliary logprobs schema version")
            rows = LogprobRows(
                data["positions"].astype(np.int64, copy=False),
                data["token_ids"].astype(np.int32, copy=False),
                data["values"].astype(np.float32, copy=False),
                data["ranks"].astype(np.int32, copy=False),
            )
    except (KeyError, OSError, ValueError, TypeError) as error:
        if isinstance(error, ValueError) and str(error).startswith(
            "unsupported auxiliary"
        ):
            raise
        raise ValueError("malformed auxiliary logprobs artifact") from error
    num_rows = len(rows.positions)
    if (
        rows.positions.ndim != 1
        or rows.token_ids.ndim != 2
        or rows.values.shape != rows.token_ids.shape
        or rows.ranks.shape != (num_rows,)
        or rows.token_ids.shape[0] != num_rows
    ):
        raise ValueError("malformed auxiliary logprobs artifact")
    return rows


def rows_to_tensors(rows: LogprobRows) -> LogprobsTensors:
    return LogprobsTensors(
        torch.from_numpy(rows.token_ids),
        torch.from_numpy(rows.values),
        torch.from_numpy(rows.ranks),
    )


def rows_to_lists(rows: LogprobRows) -> LogprobsLists:
    return LogprobsLists(rows.token_ids, rows.values, rows.ranks)
