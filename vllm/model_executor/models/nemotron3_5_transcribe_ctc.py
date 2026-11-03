# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native deferred CTC head and the frozen NeMo v1 alignment adapter."""

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .nemotron3_5_transcribe_alignment import (
    MultiSpeakerSOTWordTimestampAligner,
    _timestamp_units,
)
from .nemotron3_5_transcribe_audio import TransformerEncoder


class CTCTokenizer:
    """The checkpoint's independent, non-legacy SentencePiece tokenizer."""

    def __init__(self, proto: bytes, vocabulary: Sequence[str]) -> None:
        import sentencepiece as spm

        self.tokenizer = spm.SentencePieceProcessor(model_proto=proto)
        pieces = [self.tokenizer.id_to_piece(i) for i in range(len(self.tokenizer))]
        if pieces != list(vocabulary):
            raise ValueError("CTC tokenizer does not match the classifier vocabulary")

    def text_to_ids(self, text: str) -> list[int]:
        # NeMo's non-legacy tokenizer with ignore_extra_whitespaces=True.
        return self.tokenizer.encode_as_ids(text)


class NemotronCTCHead(nn.Module):
    """Learned 80-to-40 ms upsampling, Transformer bridge and CTC classifier."""

    def __init__(self, cfg: dict[str, Any], prefix: str = "") -> None:
        super().__init__()
        self.cfg = cfg
        self.vocabulary = cfg["vocabulary"]
        width = int(cfg["feat_in"])
        self.num_classes_with_blank = int(cfg["num_classes"]) + 1
        self.upsampling_factor = int(cfg.get("upsampling_factor", 1))
        if (
            not cfg.get("use_transformer", True)
            or self.upsampling_factor != 2
            or cfg.get("upsampling_mode", "conv_transpose") != "conv_transpose"
            or not cfg.get("residual", False)
        ):
            raise ValueError(
                "CTC timestamps require the released Transformer/40ms head"
            )
        if len(self.vocabulary) + 1 != self.num_classes_with_blank:
            raise ValueError("Invalid CTC vocabulary size")
        self.register_buffer("upsample_factor", torch.tensor(self.upsampling_factor))
        self.upsample = nn.ConvTranspose1d(width, width, kernel_size=2, stride=2)
        self.register_buffer(
            "residual_scale", torch.tensor(float(cfg.get("residual_scale", 0.1)))
        )
        transformer_config = {
            "feat_in": width,
            "d_model": width,
            "n_heads": int(cfg["n_heads"]),
            "n_layers": int(cfg["n_layers"]),
            "ff_expansion": float(cfg["ff_expansion"]),
            "qkv_bias": bool(cfg.get("qkv_bias", False)),
            "qk_norm": bool(cfg.get("qk_norm", True)),
            "pre_block_norm": bool(cfg.get("pre_block_norm", True)),
            "subsampling_factor": 1,
            "rope_base": float(cfg.get("rope_base", 10000.0)),
            "rotary_fraction": float(cfg.get("rotary_fraction", 0.5)),
            "xscaling": bool(cfg.get("xscaling", False)),
        }
        self.transformer = TransformerEncoder(
            transformer_config, None, f"{prefix}.transformer"
        )
        self.transformer.pre_encode = nn.Identity()
        self.decoder_layers = nn.Sequential(
            nn.Conv1d(width, self.num_classes_with_blank, kernel_size=1)
        )
        self.tokenizer: CTCTokenizer | None = None
        self._loaded: set[str] = set()

    def load_weight(self, name: str, tensor: torch.Tensor) -> None:
        if name == "tokenizer.model_proto":
            if tensor.dtype != torch.uint8 or tensor.numel() == 0:
                raise ValueError("CTC tokenizer.model_proto must be nonempty uint8")
            self.tokenizer = CTCTokenizer(
                tensor.detach().cpu().numpy().tobytes(), self.vocabulary
            )
            return
        if not name.startswith("decoder."):
            raise ValueError(f"Unexpected CTC checkpoint tensor {name!r}")
        name = name.removeprefix("decoder.")
        targets = dict(self.named_parameters()) | dict(self.named_buffers())
        target = targets.get(name)
        if target is None or target.shape != tensor.shape:
            raise ValueError(f"Unexpected CTC checkpoint tensor or shape {name!r}")
        loader = getattr(target, "weight_loader", None)
        if loader is None:
            with torch.no_grad():
                target.copy_(tensor.to(device=target.device, dtype=target.dtype))
        else:
            loader(target, tensor)
        self._loaded.add(name)

    def validate_loaded(self) -> None:
        expected = set(dict(self.named_parameters())) | set(dict(self.named_buffers()))
        missing = expected - self._loaded
        if missing or self.tokenizer is None:
            raise ValueError(
                f"CTC timestamp checkpoint is incomplete: {sorted(missing)}"
            )

    def trunk(self, states: torch.Tensor, lengths_cpu: torch.Tensor) -> torch.Tensor:
        encoded = self.upsample(states)
        lengths_cpu = lengths_cpu * self.upsampling_factor
        residual = encoded
        encoded, _ = self.transformer(
            encoded.transpose(1, 2),
            lengths_cpu.to(device=encoded.device),
            bypass_pre_encode=True,
            lengths_cpu=lengths_cpu,
        )
        return residual + self.residual_scale.to(encoded.dtype) * encoded

    def classify(self, states: torch.Tensor) -> torch.Tensor:
        # Preserve NeMo: full-vocabulary softmax in model dtype, then FP32 storage.
        return F.log_softmax(self.decoder_layers(states).transpose(1, 2), dim=-1)

    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        return lengths * self.upsampling_factor

    @torch.inference_mode()
    def read_columns(
        self,
        tensors: dict[str, torch.Tensor],
        columns: list[list[int]],
        lengths: list[int],
        *,
        core_length: int,
        left_context: int,
        right_context: int,
    ) -> list[torch.Tensor]:
        """Keep one request's columns on CPU, never a whole-record GPU output."""
        if len(columns) != 1 or len(lengths) != 1:
            raise ValueError("Deferred CTC supports one audio per alignment")
        states = tensors["asr_encoded"]
        input_length = int(tensors["asr_encoded_lengths"][0])
        device = self.upsample.weight.device
        dtype = self.upsample.weight.dtype
        index = torch.tensor(columns[0], dtype=torch.long, device=device)
        # Pageable final storage keeps one-hour captures from pinning gigabytes.
        host = torch.empty(
            (lengths[0], len(columns[0])), dtype=torch.float32, device="cpu"
        )
        factor = self.upsampling_factor
        step = max(1, (16 * 1024 * 1024) // self.num_classes_with_blank)
        for start in range(0, input_length, core_length):
            end = min(start + core_length, input_length)
            context_start = max(0, start - left_context)
            context_end = min(input_length, end + right_context)
            window = states[:, :, context_start:context_end].to(
                device=device, dtype=dtype
            )
            window_lengths = torch.tensor(
                [context_end - context_start], dtype=torch.long, device="cpu"
            )
            trunk = self.trunk(window, window_lengths)
            core_start = (start - context_start) * factor
            core = trunk[:, :, core_start : core_start + (end - start) * factor]
            for offset in range(0, core.shape[-1], step):
                log_probs = self.classify(core[:, :, offset : offset + step])
                kept = log_probs[0].index_select(-1, index).float()
                first = start * factor + offset
                # A bounded synchronous D2H copy avoids a full-duration GPU tensor
                # and unbounded queued buffers while the offline engine is idle.
                host[first : first + kept.shape[0]].copy_(kept)
                del log_probs, kept
            del core, trunk, window
        return [host]

    def make_aligner(
        self, online: dict[str, int], language: str = "auto"
    ) -> MultiSpeakerSOTWordTimestampAligner:
        if self.tokenizer is None:
            raise ValueError("CTC tokenizer has not been loaded")
        return MultiSpeakerSOTWordTimestampAligner(
            ctc_decoder=self,
            tokenizer=self.tokenizer,
            blank_id=self.num_classes_with_blank - 1,
            ctc_frame_seconds=0.04,
            timestamp_language=language,
            online_inference_length=online["window_frames"],
            chunk_left_context=online["left_context_frames"],
            chunk_right_context=online["right_context_frames"],
        )


def alignment_speaker_priors(
    text: str,
    sortformer_sigmoids: torch.Tensor,
    diarization_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use union speech activity for transcripts without speaker identities."""
    if MultiSpeakerSOTWordTimestampAligner._SPEAKER_TAG_RE.search(text):
        return sortformer_sigmoids, diarization_labels
    return (
        sortformer_sigmoids.amax(dim=-1, keepdim=True),
        diarization_labels.any(dim=1, keepdim=True),
    )


def estimate_alignment_workspace(
    aligner: MultiSpeakerSOTWordTimestampAligner, text: str, frames: int
) -> int:
    """Bound compact FP32 columns, one stream's byte backtrace and transients."""
    if aligner.blank_id is None:
        raise ValueError("A CTC blank ID is required for alignment")
    words = aligner._tokenize_words(
        _timestamp_units(
            aligner.parse_sot_words(text), aligner.timestamp_language or "auto"
        ),
        aligner.blank_id,
    )
    columns = {aligner.blank_id}
    largest_backtrace = 0
    for stream in aligner._group_words_by_speaker(words).values():
        tokens = [token for word in stream for token in word["token_ids"]]
        required = len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:]))
        if required > frames:
            raise ValueError("Transcript is too long for the CTC timeline")
        columns.update(tokens)
        states = 2 * len(tokens) + 1
        half = states if frames < 1000 else min(states, math.ceil(0.10 * states + 100))
        largest_backtrace = max(largest_backtrace, frames * min(states, 2 * half + 1))
    return frames * len(columns) * 4 + largest_backtrace + 2 * 1024**3


def public_words(result: dict[str, Any], duration_s: float) -> dict[str, Any]:
    failures = result.get("alignment_diagnostics", {}).get("failed_streams", [])
    if failures:
        return {"words": [], "error": "no_legal_ctc_alignment_path"}
    words = []
    for stream in result["speaker_word_timestamps"].values():
        for word in stream:
            start = max(0.0, min(duration_s, float(word["start"])))
            end = max(start, min(duration_s, float(word["end"])))
            start = min(duration_s, round(start, 3))
            end = max(start, min(duration_s, round(end, 3)))
            words.append({"word": word["word"], "start": start, "end": end})
    words.sort(key=lambda w: (w["start"], w["end"]))
    return {"words": words, "error": None}
