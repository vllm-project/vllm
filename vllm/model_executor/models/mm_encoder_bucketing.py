# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoder length bucketing for padded multimodal (audio) encoders in vLLM.

When one batch mixes short and long clips, a padded encoder computes every item
over the padded batch length (Qwen2-Audio pads every mel to a fixed 3000-frame
window): a 2 s clip batched with a 30 s clip computes far more frames than it
needs, entirely on padding. Length bucketing re-groups items by their true
lengths into discrete length "rungs" and computes each rung over its own window,
so short clips stop paying for long ones. It moves from *fixed padding* to
*padding sized to the actual audio length* -- it never removes content, only the
padding around it.

Bucketing runs in `EncoderRunner.execute_mm_encoder`, *before* the
CUDA-graph-or-eager decision, at the fbank/feature stage (right when the padded
features enter, before any conv/attention compute). Each rung is a fixed-shape
sub-batch, so bucketing and CUDA graphs are complementary rather than exclusive:
a rung's fixed shape is exactly what one captured graph needs, so every rung can
be captured and replayed on its own. This is why bucketing lives at the
scheduler layer and not inside an encoder forward -- anything done inside the
forward is invisible to the CUDA-graph path.

This is complementary to vLLM's packed-varlen path (`MMEncoderAttention` +
`cu_seqlens`). Varlen packs items into one flat sequence and removes padding
losslessly, but only for encoders whose function is unchanged by packing. For
padded encoders that cannot be packed (notably unmasked ones) this recovers the
waste the packing path cannot. The two cover disjoint encoder classes and
neither replaces the other.

Numerics are classified per encoder class:

- mask-aware (explicit attention mask): the mask puts padding outside the
  function, so shrinking the padding is content-exact; in bf16 the residual is
  ULP-level kernel rounding, the same class of non-batch-invariance vLLM
  already documents.
- unmasked + relative PE: approx-lossless.
- unmasked + absolute PE: lossy by construction (padding is attended) -- opt-in
  and off by default.

The feature is opt-in via `additional_config.audio_bucket_seconds` (a ladder of
seconds, e.g. `[2, 4, 8, 16, 30]`) and engages only for models that are both
listed in `_VERIFIED_FOR_BUCKETING` (the maintainers' gate restricting the
feature to adapted models) and declare an `EncoderBucketingSpec` carrying the
model's own length/crop semantics and frame-rate knobs. A model still being
adapted stays off until maintainers verify it and add it to the whitelist. With
the feature off, no shim is installed and the original forward runs untouched
(byte-identical); a per-batch waste gate keeps bucketing from ever computing more
than the untouched forward.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping
from dataclasses import dataclass

from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)

_CONFIG_KEY = "audio_bucket_seconds"
_CONFIG_PATH = f"additional_config.{_CONFIG_KEY}"

# Losslessness levels a whitelisted model may guarantee for its crop (see the
# module docstring): bit-exact for mask-aware encoders, output-lossless-within-
# a-1-frame-grid for unmasked + relative-PE encoders. Approx is the escape hatch
# for mixed models (e.g. FunASR: bit-exact main encoder + unmasked adaptor).
LOSSLESS_BIT_EXACT = "bit_exact"
LOSSLESS_OUTPUT_LE_1FRAME = "output_le_1frame"
LOSSLESS_APPROX = "approx"

# Model types verified for length bucketing, mapped to their losslessness level.
# This is the gate that restricts the feature to models that have been adapted
# (see the module docstring). A model must be listed here AND declare an
# `EncoderBucketingSpec` for bucketing to engage; a model still being adapted
# stays off until maintainers add it here. This keeps the feature from ever
# silently corrupting an unadapted encoder just because the config is set.
_VERIFIED_FOR_BUCKETING: dict[str, str] = {
    "qwen2_audio": LOSSLESS_BIT_EXACT,
    "canary": LOSSLESS_BIT_EXACT,
    "fireredasr2": LOSSLESS_BIT_EXACT,
    "cohere_asr": LOSSLESS_BIT_EXACT,
}


# ---------------------------------------------------------- fixed-window shim
#
# Some encoders pin their window length inside `forward` -- an expected-length
# assertion plus a fixed positional-embedding shape -- so a shorter rung window
# is rejected even though the math is well defined for variable lengths. The
# recipe is model-specific (it mirrors that encoder's own forward minus the
# fixed-length assumption), so each adapted model registers its own shim here at
# import time (see `qwen2_audio.py`). The shim is installed on the encoder module
# only while bucketing is active so rung-length inputs are accepted; for a
# full-length input it is byte-identical to the original forward in eval mode.
#
# Keyed by the wrapped class's exact name -- no MRO walk. A subclass that
# changes forward semantics must not inherit the base recipe. Classes without a
# verified variable-length recipe keep their original forward (which also serves
# the fallback path).
_FORWARD_SHIMS: dict[str, object] = {}


# ------------------------------------------------------------- length ladder


@dataclass(frozen=True)
class LengthBucketingConfig:
    """A discrete length ladder used to bucket variable-length inputs.

    Args:
        buckets: Rung lengths in feature frames, sorted ascending.
    """

    buckets: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.buckets:
            raise ValueError("LengthBucketingConfig.buckets must be non-empty")
        if tuple(self.buckets) != tuple(sorted(self.buckets)):
            raise ValueError(
                "LengthBucketingConfig.buckets must be sorted ascending"
            )

    @property
    def max_length(self) -> int:
        """The largest rung: the window used when no rung fits."""
        return self.buckets[-1]

    def padded_length(self, length: int) -> int:
        """Return the smallest rung `>=` `length`, else the largest rung.

        Args:
            length: True length in feature frames.

        Returns:
            The rung window this length should compute over.
        """
        for rung in self.buckets:
            if rung >= length:
                return rung
        return self.max_length


def _ladder_seconds(
    additional_config: Mapping[str, object] | None,
) -> tuple[float, ...] | None:
    """Read `additional_config.audio_bucket_seconds` as a seconds ladder.

    Args:
        additional_config: The `VllmConfig.additional_config` mapping in effect,
            or `None` to disable bucketing.

    Returns:
        Sorted unique seconds, or `None` when bucketing is not requested.

    Raises:
        ValueError: If the config value is not a list of positive numbers.
    """
    raw = (
        additional_config.get(_CONFIG_KEY)
        if isinstance(additional_config, Mapping)
        else None
    )
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{_CONFIG_PATH} must be a list of seconds, got {raw!r}")
    values: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{_CONFIG_PATH} entries must be numbers, got {item!r}")
        value = float(item)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{_CONFIG_PATH} entries must be > 0, got {item!r}")
        values.append(value)
    seconds = sorted(set(values))
    return tuple(seconds) if seconds else None


def _rungs_from_seconds(
    seconds: tuple[float, ...], frames_per_second: int, align_to: int = 1
) -> tuple[int, ...]:
    """Convert a seconds ladder to mel-frame rungs at the model's frame rate.

    Args:
        seconds: Sorted ascending bucket edges in seconds.
        frames_per_second: Mel frames per second of input audio.
        align_to: Align each rung up to a multiple of this many frames (e.g. a
            conv/merge downsampling factor). `1` disables alignment.

    Returns:
        Rung lengths in mel frames (at least one frame per bucket).
    """
    rungs = [max(1, int(round(s * frames_per_second))) for s in seconds]
    if align_to > 1:
        rungs = [((r + align_to - 1) // align_to) * align_to for r in rungs]
    return tuple(rungs)


# ---------------------------------------------------------------- model spec


class EncoderBucketingSpec:
    """Per-model length extraction and field cropping for encoder bucketing.

    The encoder-owning model subclasses this and passes it to
    `install_encoder_bucketing`, keeping the model's own length semantics (which
    field carries the per-item length, how to crop each time-varying field to a
    rung) and frame-rate knobs next to the model code. The generic bucketer stays
    free of per-model field knowledge.

    Subclasses set `frames_per_second` (mel frames per second of input audio),
    `context_margin` (extra frames past the published length so a rung window
    never clips anything the encoder reads), and `align_to` (align each rung up
    to a multiple of this many frames, e.g. a conv/merge downsampling factor),
    and implement the three methods.
    """

    frames_per_second: int
    context_margin: int = 0
    align_to: int = 1

    def item_lengths(
        self, mm_kwargs_batch: Mapping[str, object]
    ) -> list[int] | None:
        """Per-item true lengths in feature frames.

        Args:
            mm_kwargs_batch: The batched encoder kwargs (tensors over `num_items`).

        Returns:
            One length per item, or `None` if not derivable (skip bucketing).
        """
        raise NotImplementedError

    def padded_time(self, mm_kwargs_batch: Mapping[str, object]) -> int:
        """The padded time extent of the batch (the no-bucket compute per item).

        Args:
            mm_kwargs_batch: The batched encoder kwargs.

        Returns:
            Feature frames each item occupies without bucketing.
        """
        raise NotImplementedError

    def crop(
        self,
        mm_kwargs_batch: Mapping[str, object],
        indices: list[int],
        rung: int,
    ) -> dict[str, object]:
        """Crop to the given rows, each truncated to `rung` feature frames.

        The frames past an item's real content are the producer's own padding and
        are carried through (never zero-filled), so the encoder sees input it saw
        in training. Content is never truncated: callers only pass `rung` values
        at least as large as every selected item's length.

        Args:
            mm_kwargs_batch: The batched encoder kwargs.
            indices: Batch-row indices to select, in output order.
            rung: Window length in feature frames to truncate each row to.

        Returns:
            Keyword arguments for the rung's fixed-shape sub-batch.
        """
        raise NotImplementedError


# ----------------------------------------------------------------- bucketer


class EncoderBucketing:
    """Groups a batched encoder input into fixed-length rung sub-batches.

    Args:
        spec: The model's length/crop semantics.
        ladder: Rung ladder, or `None` to disable bucketing.
    """

    def __init__(
        self,
        spec: EncoderBucketingSpec,
        ladder: LengthBucketingConfig | None,
    ):
        self.spec = spec
        self.ladder = ladder

    @property
    def active(self) -> bool:
        """Whether bucketing is enabled (a rung ladder is configured)."""
        return self.ladder is not None

    def bucket(
        self,
        mm_kwargs_batch: Mapping[str, object],
        num_items: int,
    ) -> list[tuple[list[int], Mapping[str, object]]]:
        """Split a batch into fixed-length rung sub-batches.

        Falls back to a single full batch when bucketing cannot help: no
        lengths, an item longer than the ladder (never truncate content), or the
        waste gate (the capped windows would not reduce computed frames). Every
        item appears in exactly one returned sub-batch.

        Args:
            mm_kwargs_batch: The batched encoder kwargs.
            num_items: Number of items in the batch.

        Returns:
            `(indices, cropped_kwargs)` pairs; `indices` are batch-row positions
            and `cropped_kwargs` is that rung's fixed-shape sub-batch.
        """
        full = [(list(range(num_items)), mm_kwargs_batch)]
        if not self.active or num_items == 0:
            return full
        lengths = self.spec.item_lengths(mm_kwargs_batch)
        if lengths is None or len(lengths) != num_items:
            return full
        time_extent = self.spec.padded_time(mm_kwargs_batch)
        if time_extent <= 0:
            return full
        margin = self.spec.context_margin
        # An item longer than the ladder has no safe rung; run the untouched
        # forward rather than truncate content.
        if max(lengths) + margin > self.ladder.max_length:
            return full
        windows = [
            min(self.ladder.padded_length(length + margin), time_extent)
            for length in lengths
        ]
        # Waste gate: only bucket when the capped windows strictly reduce the
        # total computed frames. Single, uniform, and no-shrink batches run the
        # untouched forward -- the feature is never slower than off.
        if sum(windows) >= num_items * time_extent:
            return full
        groups: dict[int, list[int]] = {}
        for index, window in enumerate(windows):
            groups.setdefault(window, []).append(index)
        return [
            (indices, self.spec.crop(mm_kwargs_batch, indices, rung))
            for rung, indices in groups.items()
        ]


# -------------------------------------------------------------- install


def install_encoder_bucketing(
    model: nn.Module,
    encoders: Mapping[str, nn.Module],
    spec: EncoderBucketingSpec,
    *,
    model_type: str,
    additional_config: Mapping[str, object] | None = None,
) -> list[str]:
    """Install variable-length encoder shims and register the bucketer.

    The encoder-owning model passes its encoder(s) as a `label -> module` mapping
    and its `EncoderBucketingSpec` from its constructor. Bucketing engages only
    when the model is in `_VERIFIED_FOR_BUCKETING` (the gate restricting the
    feature to adapted models), the spec is declared, and
    `additional_config.audio_bucket_seconds` is set. When it engages, any encoder
    with a fixed-window recipe gets the variable-length shim so rung inputs are
    accepted, and the bucketer is registered on `model` for
    `EncoderRunner.execute_mm_encoder` to use before the CUDA-graph decision.
    Idempotent per `model`.

    Args:
        model: The encoder-owning model to register the bucketer on.
        encoders: `label -> encoder module` map declared by the model.
        spec: The model's length/crop semantics and frame-rate knobs.
        model_type: The model's `model_type`, checked against the whitelist.
        additional_config: The `VllmConfig.additional_config` mapping carrying
            the opt-in bucketing ladder.

    Returns:
        The declared encoder labels.

    Raises:
        ValueError: If `encoders` is empty or a declared encoder is not a module.
        RuntimeError: If the spec does not set `frames_per_second` when enabled.
    """
    if model is None or getattr(model, "_encoder_bucketing_installed", False):
        return []
    if not encoders:
        raise ValueError(
            "install_encoder_bucketing: declare at least one encoder (label -> module)"
        )
    for label, module in encoders.items():
        if not isinstance(module, nn.Module):
            raise ValueError(
                f"install_encoder_bucketing: declared encoder {label!r} is not "
                f"an nn.Module, got {type(module).__name__}"
            )

    ladder = None
    lossless = None
    seconds = _ladder_seconds(additional_config)
    if seconds is not None:
        lossless = _VERIFIED_FOR_BUCKETING.get(model_type)
        if lossless is None:
            logger.warning(
                "Encoder length bucketing requested for model_type %r, which is "
                "not verified for bucketing; leaving it disabled. Only adapted "
                "models are safe here. Verified: %s",
                model_type,
                sorted(_VERIFIED_FOR_BUCKETING),
            )
        else:
            frames_per_second = spec.frames_per_second
            if not isinstance(frames_per_second, int):
                raise RuntimeError(
                    "EncoderBucketingSpec must set frames_per_second"
                )
            ladder = LengthBucketingConfig(
                _rungs_from_seconds(seconds, frames_per_second, spec.align_to)
            )

    manifest: list[tuple[nn.Module, object]] = []
    if ladder is not None:
        for label, module in encoders.items():
            shim_fn = _FORWARD_SHIMS.get(type(module).__name__)
            if shim_fn is not None:
                manifest.append((module, module.forward))
                module.forward = functools.partial(shim_fn, module)
                logger.info(
                    "Bucketed encoder %r: fixed-window class, variable-length "
                    "recipe installed.",
                    label,
                )

    model._encoder_bucketing = EncoderBucketing(spec, ladder)
    model._encoder_bucketing_manifest = manifest
    model._encoder_bucketing_installed = True
    if ladder is not None:
        logger.info(
            "Encoder length bucketing enabled over rungs %s for modules %s "
            "(lossless level: %s)",
            ladder.buckets,
            list(encoders),
            lossless,
        )
    else:
        logger.info(
            "Encoder length bucketing disabled; enable it with additional_config.%s "
            "(e.g. [2, 4, 8, 16, 30]).",
            _CONFIG_KEY,
        )
    return list(encoders)


def uninstall_encoder_bucketing(model: nn.Module) -> list[str]:
    """Restore every shimmed `forward`; returns the restored module class names.

    Rollback for `install_encoder_bucketing` (test isolation, teardown).

    Args:
        model: The model previously passed to `install_encoder_bucketing`.

    Returns:
        Class names of the encoders whose `forward` was restored.
    """
    manifest = getattr(model, "_encoder_bucketing_manifest", None) or []
    restored: list[str] = []
    for module, original in manifest:
        module.forward = original
        restored.append(type(module).__name__)
    if manifest:
        del model._encoder_bucketing_manifest
    if getattr(model, "_encoder_bucketing", None) is not None:
        del model._encoder_bucketing
    if getattr(model, "_encoder_bucketing_installed", False):
        del model._encoder_bucketing_installed
    return restored
