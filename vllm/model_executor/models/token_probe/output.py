# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import logging
import os
from collections.abc import Mapping, Sequence
from urllib.parse import quote

import numpy as np

import vllm.envs as envs

from .loader import read_probe_config, read_probe_labels

logger = logging.getLogger(__name__)


def make_token_probe_probs(
    scores: list[list[float]],
    labels: tuple[str, ...],
) -> list[dict[str, float]]:
    label_names = list(labels)
    for index in range(len(label_names), len(scores[0])):
        label_names.append(f"label_{index}")

    probabilities = []
    for score_row in scores:
        probabilities.append(dict(zip(label_names, score_row)))
    return probabilities


class TokenProbeBatch:
    def __init__(
        self,
        manager: "TokenProbeOutputManager",
        scores: np.ndarray | None,
        req_ids: Sequence[str],
        num_scheduled_tokens: Mapping[str, int],
    ) -> None:
        self.manager = manager
        self.scores: list[list[float]] | None = None
        if scores is not None:
            self.scores = scores.tolist()
        self.offsets: dict[str, int] = {}
        if self.scores is None:
            return

        offset = 0
        for req_id in req_ids:
            self.offsets[req_id] = offset
            offset += num_scheduled_tokens[req_id]
        if offset != len(self.scores):
            raise RuntimeError(
                "token probe score rows do not match scheduled tokens: "
                f"got {len(self.scores)}, expected {offset}"
            )

    def record(
        self,
        *,
        request_id: str,
        num_tokens_scheduled: int,
        is_prefill: bool,
        is_spec_decode: bool,
        num_raw_output_tokens: int,
        num_output_tokens: int,
    ) -> list[dict[str, float]] | None:
        req_offset = self.offsets.get(request_id)
        if self.scores is None or req_offset is None:
            return None

        request_scores = self.scores[req_offset : req_offset + num_tokens_scheduled]
        selected: list[list[float]] | None = None
        if is_prefill:
            if not self.manager.prefill_enabled:
                return None
            if self.manager.save_dir:
                selected = request_scores
            elif num_output_tokens and len(request_scores):
                selected = request_scores[-1:]
        else:
            num_rows = (
                num_raw_output_tokens if self.manager.save_dir else num_output_tokens
            )
            num_rows = min(num_rows, len(request_scores))
            if num_rows:
                selected = (
                    request_scores[:num_rows]
                    if is_spec_decode
                    else request_scores[-num_rows:]
                )

        if not selected:
            return None
        if self.manager.save_dir:
            self.manager.saved_scores.setdefault(request_id, []).extend(selected)
            return None
        return make_token_probe_probs(selected, self.manager.labels)


class TokenProbeOutputManager:
    def __init__(self, ckpt_path: str | None) -> None:
        self.save_dir = envs.VLLM_TOKEN_PROBE_SAVE_DIR
        self.prefill_enabled = envs.VLLM_ENABLE_TOKEN_PROBE_PREFILL
        self.labels: tuple[str, ...] = ()
        self.config: dict = {}
        self.saved_scores: dict[str, list[list[float]]] = {}
        if ckpt_path is not None:
            self.config = read_probe_config(ckpt_path)
            self.labels = read_probe_labels(ckpt_path)
        self.empty_batch = TokenProbeBatch(self, None, (), {})

    def start_batch(
        self,
        *,
        scores: np.ndarray | None,
        req_ids: Sequence[str],
        num_scheduled_tokens: Mapping[str, int],
    ) -> TokenProbeBatch:
        if scores is None:
            return self.empty_batch
        return TokenProbeBatch(self, scores, req_ids, num_scheduled_tokens)

    def save(self, request_id: str) -> None:
        scores = self.saved_scores.pop(request_id, None)
        if not self.save_dir or not scores:
            return
        try:
            from safetensors.numpy import save_file

            os.makedirs(self.save_dir, exist_ok=True)
            metadata = {
                "rid": request_id,
                "labels": json.dumps(list(self.labels)),
            }
            for key in ("model_type", "base_model_layer_ids", "hidden_size"):
                if key in self.config:
                    metadata[key] = json.dumps(self.config[key])
            filename = quote(request_id, safe="")
            if len(filename) > 200:
                filename = hashlib.sha256(request_id.encode()).hexdigest()
            save_file(
                {"scores": np.asarray(scores, dtype=np.float32)},
                os.path.join(self.save_dir, f"{filename}.safetensors"),
                metadata=metadata,
            )
        except Exception:
            logger.exception(
                "Failed to save token probe result for request %s",
                request_id,
            )
