# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU CTC alignment ported from NVIDIA-NeMo/Speech, revision 0d1ec2f89764.

Preserves the reference speaker assignment, anchor-guided banded search,
multilingual units and 10 ms diarization boundary refinement. No NeMo runtime
is required. The checkpoint's separate alignment_config calibration is not
part of that reference algorithm.
"""

from __future__ import annotations

import math
import unicodedata
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Callable, Sequence
from itertools import groupby
from typing import Any, TypedDict

import numpy as np
import regex as re
import torch

try:
    from numba import jit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


class _CTCStream(TypedDict):
    speaker_tag: int | None
    words: list[dict[str, Any]]
    labels: list[int]
    state_to_word: list[int | None]


class MultiSpeakerSOTWordTimestampAligner:
    _SPEAKER_TAG_RE = re.compile("<spk:(\\d+)>", flags=re.IGNORECASE)
    _CTC_WINDOW_MAX_ELEMENTS = 16 * 1024 * 1024

    def __init__(
        self,
        *,
        ctc_decoder: Any | None = None,
        tokenizer: Any | None = None,
        blank_id: int | None = None,
        ctc_frame_seconds: float | None = None,
        sortformer_frame_seconds: float | None = None,
        speaker_activity_threshold: float = 0.5,
        speaker_logprob_weight: float = 1.0,
        maximum_token_len: float = 1.0,
        max_word_frames: int = 12,
        timestamp_language: str | None = None,
        epsilon: float = 1e-06,
        online_inference_length: int = 0,
        chunk_left_context: int = 0,
        chunk_right_context: int = 0,
    ) -> None:
        if speaker_logprob_weight < 0:
            raise ValueError("speaker_logprob_weight must be non-negative.")
        if not 0 <= speaker_activity_threshold <= 1:
            raise ValueError("speaker_activity_threshold must be between zero and one.")
        if maximum_token_len <= 0:
            raise ValueError("maximum_token_len must be positive.")
        self.ctc_decoder = ctc_decoder
        if (
            isinstance(max_word_frames, bool)
            or not isinstance(max_word_frames, int)
            or max_word_frames < 1
        ):
            raise ValueError("max_word_frames must be a positive integer.")
        self.max_word_frames = max_word_frames
        self.timestamp_language = timestamp_language
        self.tokenizer = tokenizer
        self.blank_id = blank_id
        self.ctc_frame_seconds = ctc_frame_seconds
        self.sortformer_frame_seconds = sortformer_frame_seconds
        self.speaker_activity_threshold = float(speaker_activity_threshold)
        self.speaker_logprob_weight = float(speaker_logprob_weight)
        self.maximum_token_len = float(maximum_token_len)
        self.epsilon = float(epsilon)
        self.online_inference_length = int(online_inference_length)
        self.chunk_left_context = int(chunk_left_context)
        self.chunk_right_context = int(chunk_right_context)

    @classmethod
    def clean_sot_transcript(
        cls, transcript: str, special_tokens: Sequence[str]
    ) -> str:
        for token in special_tokens:
            if (
                cls._SPEAKER_TAG_RE.fullmatch(token) is None
                and re.fullmatch("<(?:ev|sty):[^<>]*>", token) is None
            ):
                transcript = transcript.replace(token, " ")
        return " ".join(transcript.split())

    @classmethod
    def parse_sot_words(cls, transcript: str) -> list[dict[str, Any]]:
        transcript = transcript.replace("—", " ")
        words: list[dict[str, Any]] = []
        parts = cls._SPEAKER_TAG_RE.split(transcript)
        speaker_tag: int | None = 0 if len(parts) == 1 else None
        turn_index: int | None = None
        for index, part in enumerate(parts):
            if index % 2:
                speaker_tag, turn_index = (int(part), index // 2)
                continue
            for word in part.split():
                words.append(
                    dict(
                        word=word,
                        speaker_tag=speaker_tag,
                        turn_index=turn_index,
                        word_index=len(words),
                    )
                )
        return words

    def _prepare(
        self,
        ctc_shape: Sequence[int],
        read_columns: Callable[[list[list[int]], list[int]], list[torch.Tensor]],
        sortformer_sigmoids: torch.Tensor | None,
        sot_transcripts: Sequence[str],
        *,
        ctc_lengths: torch.Tensor | None = None,
        sortformer_lengths: torch.Tensor | None = None,
        audio_durations: Sequence[float | None] | None = None,
        time_offsets: Sequence[float] | None = None,
        speaker_logprob_weight: float | None = None,
        diarization_labels: torch.Tensor | None = None,
        diarization_lengths: torch.Tensor | None = None,
        diarization_frame_seconds: float = 0.01,
    ) -> dict[str, Any]:
        batch_size, max_ctc_frames, vocab_size = ctc_shape
        if len(sot_transcripts) != batch_size:
            raise ValueError("sot_transcripts must contain one string per batch item.")
        ctc_lengths_list = self._lengths(ctc_lengths, batch_size, max_ctc_frames)
        durations = (
            list(audio_durations)
            if audio_durations is not None
            else [None] * batch_size
        )
        offsets = list(time_offsets) if time_offsets is not None else [0.0] * batch_size
        if len(durations) != batch_size or len(offsets) != batch_size:
            raise ValueError(
                "audio_durations and time_offsets must match the batch size."
            )
        diarization_lengths_list: Sequence[int | None] = [None] * batch_size
        if diarization_labels is not None:
            if (
                diarization_labels.ndim != 3
                or diarization_labels.shape[0] != batch_size
            ):
                raise ValueError(
                    "diarization_labels must have shape (batch, speakers, 10ms_frames)."
                )
            if diarization_labels.dtype != torch.bool:
                raise TypeError("diarization_labels must use boolean storage.")
            diarization_lengths_list = self._lengths(
                diarization_lengths, batch_size, diarization_labels.shape[2]
            )
        elif diarization_lengths is not None:
            raise ValueError("diarization_lengths require diarization_labels.")
        if diarization_frame_seconds <= 0:
            raise ValueError("diarization_frame_seconds must be positive.")
        speaker_lengths_list: Sequence[int | None] = [None] * batch_size
        if sortformer_sigmoids is not None:
            if (
                sortformer_sigmoids.ndim != 3
                or sortformer_sigmoids.shape[0] != batch_size
            ):
                raise ValueError(
                    "sortformer_sigmoids must have shape (batch, frames, speakers)."
                )
            speaker_lengths_list = self._lengths(
                sortformer_lengths, batch_size, sortformer_sigmoids.shape[1]
            )
        blank_id = self._resolve_blank_id(vocab_size)
        speaker_logprob_weight = (
            self.speaker_logprob_weight
            if speaker_logprob_weight is None
            else float(speaker_logprob_weight)
        )
        if speaker_logprob_weight < 0:
            raise ValueError("speaker_logprob_weight must be non-negative.")
        records: list[dict[str, Any]] = []
        columns_per_record: list[list[int]] = []
        for index, transcript in enumerate(sot_transcripts):
            ctc_length = ctc_lengths_list[index]
            speaker_length = speaker_lengths_list[index]
            duration = durations[index]
            if duration is not None:
                ctc_step = float(duration) / ctc_length
            elif self.ctc_frame_seconds is not None:
                ctc_step = float(self.ctc_frame_seconds)
            else:
                raise ValueError(
                    f"Record {index} has no duration or configured CTC frame step."
                )
            sortformer_step = (
                None
                if speaker_length is None
                else self._frame_seconds(
                    speaker_length,
                    durations[index],
                    self.sortformer_frame_seconds or None,
                    ctc_step,
                )
            )
            words = self._tokenize_words(
                _timestamp_units(
                    self.parse_sot_words(transcript), self.timestamp_language or "auto"
                ),
                blank_id,
            )
            streams: list[_CTCStream] = []
            for speaker_tag, speaker_words in self._group_words_by_speaker(
                words
            ).items():
                labels, state_to_word = self._build_ctc_target(speaker_words, blank_id)
                tokens = labels[1::2]
                minimum_frames = len(tokens) + sum(
                    (left == right for left, right in zip(tokens, tokens[1:]))
                )
                if minimum_frames > ctc_length:
                    raise ValueError(
                        f"Speaker {speaker_tag!r} transcript exceeds the CTC timeline."
                    )
                streams.append(
                    {
                        "speaker_tag": speaker_tag,
                        "words": speaker_words,
                        "labels": labels,
                        "state_to_word": state_to_word,
                    }
                )
            columns = [blank_id] + sorted(
                {label for stream in streams for label in stream["labels"]} - {blank_id}
            )
            column_of = {label: column for column, label in enumerate(columns)}
            for stream in streams:
                stream["labels"] = [column_of[label] for label in stream["labels"]]
            columns_per_record.append(columns)
            records.append(
                {
                    "words": words,
                    "streams": streams,
                    "ctc_log_probs": None,
                    "ctc_length": ctc_length,
                    "ctc_step": ctc_step,
                    "speaker_probs": None
                    if sortformer_sigmoids is None
                    else sortformer_sigmoids[index, :speaker_length]
                    .detach()
                    .float()
                    .cpu(),
                    "speaker_length": speaker_length,
                    "sortformer_step": sortformer_step,
                    "diarization_labels": None
                    if diarization_labels is None
                    else diarization_labels[index, :, : diarization_lengths_list[index]]
                    .detach()
                    .cpu(),
                    "audio_duration": durations[index],
                    "time_offset": float(offsets[index]),
                }
            )
        for record, log_probs in zip(
            records,
            read_columns(columns_per_record, ctc_lengths_list) if records else (),
        ):
            record["ctc_log_probs"] = log_probs
        return {
            "config": {
                "speaker_activity_threshold": self.speaker_activity_threshold,
                "speaker_logprob_weight": speaker_logprob_weight,
                "maximum_token_len": self.maximum_token_len,
                "max_word_frames": self._ctc_frames(self.max_word_frames),
                "timestamp_language": self.timestamp_language,
                "epsilon": self.epsilon,
            },
            "num_speaker_columns": None
            if sortformer_sigmoids is None
            else sortformer_sigmoids.shape[-1],
            "diarization_frame_seconds": None
            if diarization_labels is None
            else float(diarization_frame_seconds),
            "diarization_max_speaker_count": None
            if diarization_labels is None
            else diarization_labels.shape[1],
            "records": records,
        }

    def align_prepared(self, prepared: dict[str, Any]) -> list[dict[str, Any]]:
        return _align_banded_ctc(self, prepared)

    @staticmethod
    def diarization_segments(
        labels: torch.Tensor,
        *,
        frame_seconds: float,
        time_offset: float,
        audio_duration: float | None,
    ) -> list[dict[str, Any]]:
        segments = []
        for speaker, activity in enumerate(labels):
            padded = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int8, device="cpu"),
                    activity.to(torch.int8),
                    torch.zeros(1, dtype=torch.int8, device="cpu"),
                ]
            )
            transitions = padded[1:] - padded[:-1]
            starts = torch.nonzero(transitions == 1).flatten().tolist()
            ends = torch.nonzero(transitions == -1).flatten().tolist()
            for start_frame, end_frame in zip(starts, ends):
                start = time_offset + start_frame * frame_seconds
                end = time_offset + end_frame * frame_seconds
                if audio_duration is not None:
                    end = min(end, time_offset + float(audio_duration))
                if end > start:
                    segments.append({"speaker": speaker, "start": start, "end": end})
        return sorted(
            segments,
            key=lambda segment: (segment["start"], segment["end"], segment["speaker"]),
        )

    @staticmethod
    def _result(
        *,
        rows: Sequence[dict[str, Any]],
        mapping: dict[int, int | None],
        ctc_step: float,
        sortformer_step: float | None,
        time_offset: float,
        ctc_frames: int,
        sortformer_frames: int | None,
        preliminary_scores: dict[int | None, float],
        final_scores: dict[int | None, float],
        assignment_scores: dict[int, list[float]],
        diarization_timestamps: Sequence[dict[str, Any]],
        diarization_frame_seconds: float | None,
        diarization_max_speaker_count: int | None,
    ) -> dict[str, Any]:
        public_rows = MultiSpeakerSOTWordTimestampAligner._public_word_timestamps(rows)
        return {
            "speaker_word_timestamps": public_rows,
            "diarization_timestamps": list(diarization_timestamps),
            "diarization_frame_seconds": diarization_frame_seconds,
            "diarization_max_speaker_count": diarization_max_speaker_count,
            "diarization_activity_threshold": 0.5
            if diarization_frame_seconds is not None
            else None,
            "speaker_tag_to_sortformer_column": mapping,
            "alignment_mode": "banded",
            "requested_alignment_mode": "banded",
            "speaker_assignment_mode": "optimal",
            "ctc_frame_seconds": ctc_step,
            "sortformer_frame_seconds": sortformer_step,
            "time_offset": time_offset,
            "num_ctc_frames": ctc_frames,
            "num_sortformer_frames": sortformer_frames,
            "alignment_diagnostics": {
                "preliminary_ctc_path_scores": preliminary_scores,
                "final_path_scores": final_scores,
                "speaker_assignment_scores": assignment_scores,
            },
        }

    def _ctc_frames(self, encoder_frames: int | torch.Tensor) -> int | torch.Tensor:
        return encoder_frames * int(getattr(self.ctc_decoder, "upsampling_factor", 1))

    @staticmethod
    def _group_words_by_speaker(
        words: Sequence[dict[str, Any]],
    ) -> dict[int | None, list[dict[str, Any]]]:
        grouped: dict[int | None, list[dict[str, Any]]] = {}
        for word in words:
            grouped.setdefault(word["speaker_tag"], []).append(word)
        return grouped

    @staticmethod
    def _public_word_timestamps(
        words: Sequence[dict[str, Any]],
    ) -> dict[int | None, list[dict[str, Any]]]:
        grouped: dict[int | None, list[dict[str, Any]]] = {}
        for word in words:
            speaker = word["speaker_tag"]
            grouped.setdefault(speaker, []).append(
                {
                    "word": word["word"],
                    "speaker": speaker,
                    "start": word["start"],
                    "end": word["end"],
                }
            )
        return grouped

    def _tokenize_words(
        self, words: Sequence[dict[str, Any]], blank_id: int
    ) -> list[dict[str, Any]]:
        tokenized = {}
        for stream_words in self._group_words_by_speaker(words).values():
            for word, token_ids in zip(
                stream_words, self._tokenize_word_stream(stream_words, blank_id)
            ):
                tokenized[word["word_index"]] = {**word, "token_ids": token_ids}
        return [tokenized[word["word_index"]] for word in words]

    def _tokenize_word_stream(
        self, words: Sequence[dict[str, Any]], blank_id: int
    ) -> list[list[int]]:
        tokenizer = self.tokenizer
        if tokenizer is None:
            raise ValueError("CTC tokenizer is required to prepare a transcript")
        if any(word.get("alignment_unit") == "character" for word in words):
            return [
                _character_token_ids(tokenizer.tokenizer, word["word"], blank_id)
                if word.get("alignment_unit") == "character"
                else self._tokenize_word_stream([word], blank_id)[0]
                for word in words
            ]
        text = " ".join(word["word"] for word in words)
        token_ids = [int(token_id) for token_id in tokenizer.text_to_ids(text)]
        if not token_ids or min(token_ids) < 0 or max(token_ids) >= blank_id:
            raise ValueError(
                "Tokenizer produced IDs outside the non-blank CTC vocabulary."
            )
        ids_of = {}
        for word in words:
            if word["word"] not in ids_of:
                ids_of[word["word"]] = [
                    int(token_id) for token_id in tokenizer.text_to_ids(word["word"])
                ]
        ids_by_word = [list(ids_of[word["word"]]) for word in words]
        if any(not token_ids for token_ids in ids_by_word):
            raise ValueError("SentencePiece produced an empty word tokenization.")
        if [token_id for word_ids in ids_by_word for token_id in word_ids] != token_ids:
            raise ValueError(
                "Per-word and full-stream SentencePiece tokenization disagree."
            )
        return ids_by_word

    @staticmethod
    def _build_ctc_target(
        words: Sequence[dict[str, Any]], blank_id: int
    ) -> tuple[list[int], list[int | None]]:
        labels = [blank_id]
        state_to_word: list[int | None] = [None]
        for word_index, word in enumerate(words):
            for token_id in word["token_ids"]:
                labels.extend((token_id, blank_id))
                state_to_word.extend((word_index, None))
        return (labels, state_to_word)

    def _word_rows_from_path(
        self,
        words: Sequence[dict[str, Any]],
        state_to_word: Sequence[int | None],
        path: np.ndarray,
        ctc_step: float,
        time_offset: float,
    ) -> list[dict[str, Any]]:
        frames_by_word: list[list[int]] = [[] for _ in words]
        for frame, state in enumerate(path.tolist()):
            if state < 0:
                continue
            word_index = state_to_word[state]
            if word_index is not None:
                frames_by_word[word_index].append(frame)
        rows = []
        for word, frames in zip(words, frames_by_word):
            if not frames:
                raise ValueError(f"CTC path did not visit word {word['word']!r}.")
            start = time_offset + frames[0] * ctc_step
            end = min(
                time_offset + (frames[-1] + 1) * ctc_step,
                start + self.maximum_token_len,
            )
            rows.append({**word, "start": start, "end": end})
        return rows

    @staticmethod
    def _resample_speaker_probs(
        speaker_probs: torch.Tensor, target_frames: int
    ) -> torch.Tensor:
        if speaker_probs.shape[0] == target_frames:
            return speaker_probs
        if speaker_probs.shape[0] == 1:
            return speaker_probs.expand(target_frames, -1)
        positions = torch.linspace(
            0,
            speaker_probs.shape[0] - 1,
            target_frames,
            dtype=torch.float32,
            device="cpu",
        )
        lower = positions.floor().long()
        upper = positions.ceil().long()
        fraction = (positions - lower).unsqueeze(1)
        return speaker_probs[lower] * (1 - fraction) + speaker_probs[upper] * fraction

    @staticmethod
    def _lengths(
        values: torch.Tensor | None, batch_size: int, maximum: int
    ) -> list[int]:
        if values is None:
            return [maximum] * batch_size
        lengths = [
            int(value) for value in torch.as_tensor(values, device="cpu").reshape(-1)
        ]
        if len(lengths) != batch_size or any(
            length <= 0 or length > maximum for length in lengths
        ):
            raise ValueError(
                "Padded sequence lengths must match the batch and time dimension."
            )
        return lengths

    def _resolve_blank_id(self, vocab_size: int) -> int:
        if self.blank_id is not None:
            blank_id = int(self.blank_id)
        elif self.ctc_decoder is not None:
            blank_id = int(self.ctc_decoder.num_classes_with_blank) - 1
        else:
            blank_id = vocab_size - 1
        if not 0 <= blank_id < vocab_size:
            raise ValueError(f"blank_id={blank_id} is outside the CTC vocabulary.")
        return blank_id

    @staticmethod
    def _frame_seconds(
        length: int,
        audio_duration: float | None,
        configured: float | None,
        default: float,
    ) -> float:
        if audio_duration is not None:
            return float(audio_duration) / length
        return float(configured) if configured is not None else default


def _one_pass_speaker_mapping(ctc, streams, speaker, threshold, greedy):
    if not streams:
        return {}
    if speaker.shape[1] == 0:
        return {stream["speaker_tag"]: None for stream in streams}
    from scipy.optimize import linear_sum_assignment

    peaks = (greedy != 0) & np.r_[True, greedy[1:] != greedy[:-1]]
    observed = np.stack(
        [
            np.bincount(
                greedy[peaks & (speaker[:, c] >= threshold)], minlength=ctc.shape[1]
            )
            for c in range(speaker.shape[1])
        ]
    )
    expected = np.stack(
        [
            np.bincount(np.asarray(s["labels"])[1::2], minlength=ctc.shape[1])
            for s in streams
        ]
    )
    idf = np.log1p(len(observed) / (1 + (observed > 0).sum(axis=0)))
    a = np.sqrt(expected * idf)
    b = np.sqrt(observed * idf)
    similarity = (
        a
        @ b.T
        / np.maximum(
            1e-08,
            np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :],
        )
    )
    similarity -= 0.1 * np.abs(
        np.log(
            (expected.sum(axis=1)[:, None] + 1) / (observed.sum(axis=1)[None, :] + 1)
        )
    )
    rows, cols = linear_sum_assignment(-similarity)
    mapping = {s["speaker_tag"]: None for s in streams}
    mapping.update({streams[r]["speaker_tag"]: int(c) for r, c in zip(rows, cols)})
    return mapping


def _banded_ctc_search(logp, labels, positions, activity_prior, greedy=None):
    frames, states = (len(positions), len(labels))
    half = states if frames < 1000 else min(states, math.ceil(0.1 * states + 100))
    width = min(states, 2 * half + 1)
    starts = np.clip(
        np.rint(np.linspace(0, states - 1, frames)).astype(np.int64) - half,
        0,
        states - width,
    )
    target = labels[1::2]
    if frames < len(target) + np.count_nonzero(target[1:] == target[:-1]):
        return (np.full(frames, -1, np.int64), -np.inf, starts)
    if not frames:
        return (np.full(frames, -1, np.int64), -np.inf, starts)
    greedy = logp[positions].argmax(axis=1) if greedy is None else greedy[positions]
    starts = _ctc_anchor_band_starts(greedy, target, states, half, width)
    previous, pointers = _ctc_banded_forward(
        logp, labels, positions, activity_prior, starts, width
    )
    path, score = _ctc_banded_backtrace(previous, pointers, starts, states)
    return (path, score, starts)


def _ctc_anchor_band_starts(greedy, target, states, half, width):
    frames = len(greedy)
    if not frames:
        return np.empty(0, dtype=np.int64)
    keep = (greedy != 0) & np.r_[True, greedy[1:] != greedy[:-1]]
    observed = greedy[keep]
    times = np.flatnonzero(keep)
    left, right = (_ctc_trigram_positions(observed), _ctc_trigram_positions(target))
    pairs = sorted(
        (
            (int(times[v[0] + 1]), 2 * (right[k][0] + 1) + 1)
            for k, v in left.items()
            if len(v) == 1 and len(right.get(k, [])) == 1
        )
    )
    tails: list[int] = []
    tail_ids: list[int] = []
    parent: list[int] = []
    for i, (_, state) in enumerate(pairs):
        j = bisect_left(tails, state)
        parent.append(tail_ids[j - 1] if j else -1)
        if j == len(tails):
            tails.append(state)
            tail_ids.append(i)
        else:
            tails[j] = state
            tail_ids[j] = i
    chosen = []
    k = tail_ids[-1] if tail_ids else -1
    while k >= 0:
        chosen.append(pairs[k])
        k = parent[k]
    chosen = chosen[::-1]
    points = (
        [(0, 0)]
        + [(t, s) for t, s in chosen if 0 < t < frames - 1]
        + [(frames - 1, states - 1)]
    )
    center = np.interp(
        np.arange(frames), [v[0] for v in points], [v[1] for v in points]
    )
    starts = np.clip(np.rint(center).astype(np.int64) - half, 0, states - width)
    starts = np.minimum.accumulate(starts - 2 * np.arange(frames)) + 2 * np.arange(
        frames
    )
    starts = np.maximum(
        starts, np.maximum(0, states - width - 2 * (frames - 1 - np.arange(frames)))
    )
    return starts


def _ctc_banded_forward(logp, labels, positions, activity_prior, starts, width):
    if not NUMBA_AVAILABLE:
        return _ctc_banded_forward_numpy(
            logp, labels, positions, activity_prior, starts, width
        )
    labels = np.ascontiguousarray(labels, dtype=np.int64)
    can_skip = (labels != 0) & (labels != np.r_[labels[:2], labels[:-2]])
    return _ctc_banded_forward_compiled(
        np.ascontiguousarray(logp),
        labels,
        np.ascontiguousarray(positions, dtype=np.int64),
        np.zeros(1, np.float32)
        if activity_prior is None
        else np.ascontiguousarray(activity_prior),
        activity_prior is not None,
        np.ascontiguousarray(starts, dtype=np.int64),
        int(width),
        can_skip,
    )


def _ctc_banded_forward_numpy(logp, labels, positions, activity_prior, starts, width):
    frames = len(positions)
    previous = np.full(width, -np.inf, np.float32)
    previous[: min(2, width)] = logp[positions[0], labels[: min(2, width)]]
    if activity_prior is not None:
        previous[: min(2, width)] += np.where(
            labels[: min(2, width)] != 0, activity_prior[positions[0]], 0
        )
    pointers = np.zeros((frames, width), np.uint8)
    padded = np.full(width + 4, -np.inf, np.float32)
    can_skip = (labels != 0) & (labels != np.r_[labels[:2], labels[:-2]])
    for t in range(1, frames):
        start = int(starts[t])
        delta = start - int(starts[t - 1])
        padded[2 : width + 2] = previous
        best = padded[2 + delta : 2 + delta + width].copy()
        one = padded[1 + delta : 1 + delta + width]
        two = padded[delta : delta + width]
        move = pointers[t]
        take = one > best
        np.copyto(best, one, where=take)
        move[take] = 1
        take = (two > best) & can_skip[start : start + width]
        np.copyto(best, two, where=take)
        move[take] = 2
        target = labels[start : start + width]
        previous = best + logp[positions[t], target]
        if activity_prior is not None:
            previous += np.where(target != 0, activity_prior[positions[t]], 0)
    return (previous, pointers)


if NUMBA_AVAILABLE:

    @jit(nopython=True, nogil=True, cache=True)
    def _ctc_banded_forward_compiled(
        logp, labels, positions, activity_prior, has_prior, starts, width, can_skip
    ):
        frames = positions.shape[0]
        previous = np.full(width, -np.inf, np.float32)
        current = np.empty(width, np.float32)
        pointers = np.zeros((frames, width), np.uint8)
        first = positions[0]
        for j in range(min(2, width)):
            previous[j] = logp[first, labels[j]]
            if has_prior and labels[j] != 0:
                previous[j] = previous[j] + activity_prior[first]
        for t in range(1, frames):
            start = starts[t]
            delta = start - starts[t - 1]
            position = positions[t]
            for j in range(width):
                k = j + delta
                best = previous[k] if k < width else np.float32(-np.inf)
                move = 0
                one = previous[k - 1] if 0 <= k - 1 < width else np.float32(-np.inf)
                if one > best:
                    best = one
                    move = 1
                two = previous[k - 2] if 0 <= k - 2 < width else np.float32(-np.inf)
                if two > best and can_skip[start + j]:
                    best = two
                    move = 2
                pointers[t, j] = move
                label = labels[start + j]
                current[j] = best + logp[position, label]
                if has_prior and label != 0:
                    current[j] = current[j] + activity_prior[position]
            previous, current = (current, previous)
        return (previous, pointers)


def _ctc_banded_backtrace(previous, pointers, starts, states):
    frames, width = pointers.shape
    path = np.full(frames, -1, np.int64)
    ends = np.array([states - 1, states - 2])
    local = ends - starts[-1]
    valid = (local >= 0) & (local < width)
    ends = ends[valid]
    local = local[valid]
    best = previous[local].argmax()
    score = float(previous[local[best]])
    if np.isfinite(score):
        state = int(ends[best])
        for t in range(frames - 1, -1, -1):
            path[t] = state
            if t:
                state -= int(pointers[t, state - starts[t]])
    return (path, score)


def _refine_word_timestamps(result, max_frames=12):
    if (
        isinstance(max_frames, bool)
        or not isinstance(max_frames, int)
        or max_frames < 1
    ):
        raise ValueError("max_frames must be a positive integer")
    step = float(result["ctc_frame_seconds"])
    if not math.isfinite(step) or step <= 0:
        raise ValueError("ctc_frame_seconds must be positive and finite")
    limit = max_frames * min(step, 0.08)
    segments = defaultdict(list)
    for segment in result.get("diarization_timestamps", []):
        if segment["end"] > segment["start"]:
            segments[int(segment["speaker"])].append((segment["start"], segment["end"]))
    for intervals in segments.values():
        intervals.sort()
    mapping = result.get("speaker_tag_to_sortformer_column", {})
    refined = {}
    moved = unmatched = count = 0
    for tag, source in result["speaker_word_timestamps"].items():
        words = sorted((dict(w) for w in source), key=lambda w: w["start"])
        column = mapping.get(tag, mapping.get(str(tag)))
        intervals = segments.get(column, [])
        starts = [s for s, _ in intervals]
        memberships: list[int | None] = []
        index: int | None
        for word in words:
            index = bisect_right(starts, word["start"]) - 1
            memberships.append(
                index if index >= 0 and word["start"] < intervals[index][1] else None
            )
        seen = set()
        for word, index in zip(words, memberships):
            if index is None:
                unmatched += 1
            elif index not in seen:
                onset = intervals[index][0]
                if 0 < word["start"] - onset <= limit:
                    word["start"] = onset
                    moved += 1
                seen.add(index)
        for i, (word, index) in enumerate(zip(words, memberships)):
            if index is None:
                end = words[i + 1]["start"] if i + 1 < len(words) else word["end"]
            elif i + 1 < len(words) and memberships[i + 1] == index:
                end = words[i + 1]["start"]
            else:
                end = intervals[index][1]
            word["end"] = max(word["start"], min(end, word["start"] + limit))
        refined[tag] = words
        count += len(words)
    return {
        **result,
        "speaker_word_timestamps": refined,
        "word_timestamp_refinement": {
            "method": "diarization_boundary_refinement",
            "max_frames": max_frames,
            "max_seconds": limit,
            "words": count,
            "onsets_moved": moved,
            "unmatched_words": unmatched,
        },
    }


def _align_banded_ctc(aligner, prepared):
    results = []
    for record in prepared["records"]:
        ctc = record["ctc_log_probs"].numpy()
        greedy = ctc.argmax(axis=1)
        diagnostics = []
        frames = record["ctc_length"]
        probs = record["speaker_probs"]
        tags = [s["speaker_tag"] for s in record["streams"]]
        if probs is None:
            speaker = None
            mapping = {tag: None for tag in tags}
        else:
            speaker = aligner._resample_speaker_probs(probs.clamp(0, 1), frames)
            mapping = _one_pass_speaker_mapping(
                ctc,
                record["streams"],
                speaker.numpy(),
                aligner.speaker_activity_threshold,
                greedy,
            )
        rows, scores, failures = ([], {}, [])
        for stream in record["streams"]:
            labels = np.asarray(stream["labels"], np.int64)
            column = mapping[stream["speaker_tag"]]
            activity = (
                np.ones(frames)
                if speaker is None
                else np.zeros(frames)
                if column is None
                else speaker[:, column].numpy()
            )
            positions = np.flatnonzero(activity >= 0.01)
            prior = prepared["config"]["speaker_logprob_weight"] * np.log(
                np.maximum(activity, aligner.epsilon)
            )
            path, score, _ = _banded_ctc_search(ctc, labels, positions, prior, greedy)
            failed = not np.isfinite(score)
            half = (
                len(labels)
                if len(positions) < 1000
                else min(len(labels), math.ceil(0.1 * len(labels) + 100))
            )
            diagnostics.append(
                dict(
                    speaker=stream["speaker_tag"],
                    frames=frames,
                    retained_frames=len(positions),
                    states=len(labels),
                    half_width=half,
                    attempts=1,
                    fallback=False,
                    failed=failed,
                    mapping=column,
                    visited_cells=len(positions) * min(len(labels), 2 * half + 1),
                    full_cells=frames * len(labels),
                )
            )
            scores[stream["speaker_tag"]] = score if not failed else None
            if failed:
                failures.append(
                    {
                        "speaker": stream["speaker_tag"],
                        "reason": "no_legal_path_in_anchor_band",
                        "words": [w["word"] for w in stream["words"]],
                    }
                )
                continue
            fullpath = np.full(frames, -1, np.int64)
            fullpath[positions] = path
            rows.extend(
                aligner._word_rows_from_path(
                    stream["words"],
                    stream["state_to_word"],
                    fullpath,
                    record["ctc_step"],
                    record["time_offset"],
                )
            )
        diarization = (
            []
            if record["diarization_labels"] is None
            else aligner.diarization_segments(
                record["diarization_labels"],
                frame_seconds=prepared["diarization_frame_seconds"],
                time_offset=record["time_offset"],
                audio_duration=record["audio_duration"],
            )
        )
        result = aligner._result(
            rows=rows,
            mapping=mapping,
            ctc_step=record["ctc_step"],
            sortformer_step=record["sortformer_step"],
            time_offset=record["time_offset"],
            ctc_frames=frames,
            sortformer_frames=record["speaker_length"],
            preliminary_scores={},
            final_scores=scores,
            assignment_scores={},
            diarization_timestamps=diarization,
            diarization_frame_seconds=prepared["diarization_frame_seconds"],
            diarization_max_speaker_count=prepared["diarization_max_speaker_count"],
        )
        result["speaker_assignment_mode"] = "token_distribution"
        result["alignment_diagnostics"]["failed_streams"] = failures
        result["alignment_diagnostics"]["one_pass"] = True
        result["alignment_diagnostics"]["banded"] = diagnostics
        results.append(
            _refine_word_timestamps(result, prepared["config"]["max_word_frames"])
        )
    return results


def _lexical_characters(text):
    return [
        c
        for c in unicodedata.normalize("NFC", text)
        if unicodedata.category(c)[0] in "LMN"
    ]


def _timestamp_units(words, language):
    if language == "auto":
        names = [
            unicodedata.name(c, "")
            for c in {
                c
                for word in words
                if not re.fullmatch("<[^<>]+>", word["word"])
                for c in word["word"]
            }
        ]
        language = next(
            (
                lang
                for lang, prefixes in (
                    ("ko", ("HANGUL",)),
                    ("ja", ("HIRAGANA", "KATAKANA")),
                    ("zh", ("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")),
                    ("th", ("THAI",)),
                )
                if any(name.startswith(prefixes) for name in names)
            ),
            None,
        )
    if language not in ("zh", "ja", "th"):
        return words
    output: list[dict[str, Any]] = []
    if language == "th":
        try:
            from pythainlp.tokenize import word_tokenize
        except ImportError as error:
            raise ImportError(
                "Thai timestamp units require pythainlp==5.3.4."
            ) from error
    for (_, _, special), group in groupby(
        words,
        key=lambda w: (
            w.get("speaker_tag"),
            w.get("turn_index"),
            bool(re.fullmatch("<[^<>]+>", w["word"])),
        ),
    ):
        group_items = list(group)
        if special:
            for word in group_items:
                output.append({**word, "word_index": len(output)})
        elif language == "th":
            text = unicodedata.normalize(
                "NFKC", "".join(w["word"] for w in group_items)
            ).casefold()
            text = "".join(c for c in text if unicodedata.category(c)[0] in "LMN")
            for unit in word_tokenize(text, engine="newmm", keep_whitespace=False):
                output.append(
                    {
                        **group_items[0],
                        "word": unit,
                        "word_index": len(output),
                        "alignment_unit": "word",
                    }
                )
        else:
            for word in group_items:
                for unit in _lexical_characters(word["word"]):
                    output.append(
                        {
                            **word,
                            "word": unit,
                            "word_index": len(output),
                            "alignment_unit": "character",
                        }
                    )
    return output


def _character_token_ids(processor, character, blank_id):
    index = processor.piece_to_id(character)
    ids = (
        [index]
        if index != processor.unk_id()
        else [
            token
            for token in processor.encode(character)
            if processor.id_to_piece(token) != "▁"
        ]
    )
    if (
        not ids
        or processor.unk_id() in ids
        or min(ids) < 0
        or (max(ids) >= blank_id)
        or ("".join(_lexical_characters(processor.decode(ids))) != character)
    ):
        raise ValueError(f"Unrepresentable character: {character!r}")
    return ids


def _ctc_trigram_positions(tokens):
    table = defaultdict(list)
    for i in range(len(tokens) - 2):
        table[tuple(tokens[i : i + 3])].append(i)
    return table
