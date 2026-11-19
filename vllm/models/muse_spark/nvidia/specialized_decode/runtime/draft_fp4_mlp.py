# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in static scales and BF16 calibration for DSpark's NVFP4 W4A16 MLP
bodies (`cutedsl/draft/fp4_mlp.py`)."""

import atexit
import hashlib
import json
import math
import threading
import weakref
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import atomic_writer

from . import DraftMlpNvfp4

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig

logger = init_logger(__name__)

OPTION = "draft_mlp_nvfp4"
SCHEMA_VERSION = 1
CALIBRATION_WRITE_S = 10.0


def _sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


@dataclass(frozen=True)
class ModelIdentity:
    model: str
    revision: str | None
    config_sha256: str
    safetensors_index_sha256: dict[str, str]

    @classmethod
    def from_model_config(cls, config: "ModelConfig") -> "ModelIdentity":
        """Bind the caller's model/revision and the original checkpoint metadata,
        both read at the config's pinned snapshot, the local one where cached."""
        from vllm.transformers_utils.repo_utils import hf_api, try_get_local_file

        source = config.hf_config_path or config.model
        revision = getattr(config.hf_config, "_commit_hash", None) or config.revision
        root = Path(source)
        if root.is_file():
            config_path = root
        elif root.is_dir():
            config_path = root / "config.json"
        else:
            cached = try_get_local_file(source, "config.json", revision=revision)
            if not isinstance(cached, Path):
                raise ValueError(f"{OPTION}: config_sha256 unavailable for {source}")
            config_path = Path(cached)
        weights = Path(config.model)
        if weights.is_dir():
            indices = sorted(weights.glob("*.safetensors.index.json"))
        else:
            from huggingface_hub.utils import (
                EntryNotFoundError,
                LocalEntryNotFoundError,
            )

            name = "model.safetensors.index.json"
            if source != config.model:
                # The commit hash pins the config's repo, not the weights'.
                revision = config.revision
            index = try_get_local_file(config.model, name, revision=revision)
            if index is None:
                try:
                    index = Path(
                        hf_api().hf_hub_download(
                            config.model,
                            name,
                            revision=revision,
                            token=config.hf_token,
                        )
                    )
                except LocalEntryNotFoundError as e:
                    raise ValueError(
                        f"{OPTION}: safetensors_index_sha256 unavailable; "
                        "stage the checkpoint locally before calibrating or serving"
                    ) from e
                except EntryNotFoundError:
                    index = None
            indices = [index] if isinstance(index, Path) else []
        return cls(
            config.model,
            str(config.revision) if config.revision is not None else None,
            _sha256(config_path),
            {p.name: _sha256(p) for p in indices},
        )


@dataclass(frozen=True)
class DraftMlpIdentity:
    target: ModelIdentity
    draft: ModelIdentity

    @classmethod
    def from_vllm_config(cls, config: "VllmConfig") -> "DraftMlpIdentity":
        spec = config.speculative_config
        if spec is None or spec.target_model_config is None:
            raise ValueError(f"{OPTION}: target_model_config is required")
        if spec.draft_model_config is None:
            raise ValueError(f"{OPTION}: draft_model_config is required")
        return cls(
            ModelIdentity.from_model_config(spec.target_model_config),
            ModelIdentity.from_model_config(spec.draft_model_config),
        )


def scales_path(prefix: str, rank: int) -> str:
    return f"{prefix}.tp{rank}.json"


def _manifest_header(
    identity: DraftMlpIdentity, tp: int, rank: int, num_layers: int
) -> dict:
    """The fields a scales file carries besides its amax."""
    return {
        "version": SCHEMA_VERSION,
        **asdict(identity),
        "tp": tp,
        "rank": rank,
        "num_layers": num_layers,
    }


def load_amax(
    prefix: str,
    rank: int,
    tp: int,
    num_layers: int,
    identity: DraftMlpIdentity,
) -> list[tuple[float, float]]:
    """Load this rank's positive amax pairs, refusing a mismatched manifest."""
    path = scales_path(prefix, rank)
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: manifest must be an object")
    expected = _manifest_header(identity, tp, rank, num_layers)

    def check(actual: dict, wanted: dict, parent: str = "") -> None:
        for name, value in wanted.items():
            field = f"{parent}.{name}" if parent else name
            if name not in actual:
                raise ValueError(f"{path}: {field} missing")
            got = actual[name]
            if isinstance(value, dict):
                if not isinstance(got, dict):
                    raise ValueError(f"{path}: {field} mismatch")
                check(got, value, field)
                if set(got) != set(value):
                    raise ValueError(f"{path}: {field} mismatch")
            elif type(got) is not type(value) or got != value:
                raise ValueError(f"{path}: {field} mismatch: {got!r} != {value!r}")

    check(data, expected)
    layers = data.get("layers")
    try:
        valid = (
            isinstance(layers, list)
            and len(layers) == num_layers
            and all(
                isinstance(pair, list)
                and len(pair) == 2
                and all(
                    type(v) in (int, float) and math.isfinite(v) and v > 0 for v in pair
                )
                for pair in layers
            )
        )
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(
            f"{path}: layers needs {num_layers} finite positive amax pairs"
        )
    assert isinstance(layers, list)
    return [(float(x), float(inter)) for x, inter in layers]


@dataclass(frozen=True)
class DraftMlpNvfp4Setup:
    """`draft_mlp_nvfp4` bound to the deployment: the option, the identity its
    scales files carry and, with `scales`, this rank's amax pairs."""

    option: DraftMlpNvfp4
    identity: DraftMlpIdentity
    tp: int
    rank: int
    amax: list[tuple[float, float]] | None

    @classmethod
    def build(
        cls, vllm_config: "VllmConfig", option: DraftMlpNvfp4
    ) -> "DraftMlpNvfp4Setup":
        """Checks what the configuration decides (the fused draft all-reduce's
        TP sizes, the MLP shapes), then reads the identity and the amax."""
        from vllm.distributed import get_tensor_model_parallel_rank

        from ..cutedsl.common.policy import FUSED_TPS
        from ..cutedsl.draft.fp4_mlp import DraftW4A16Mlp

        tp = vllm_config.parallel_config.tensor_parallel_size
        if tp not in FUSED_TPS:
            raise ValueError(
                f"{OPTION} needs the fused draft all-reduce, which TP{tp} lacks"
            )
        spec = vllm_config.speculative_config
        assert spec is not None and spec.draft_model_config is not None
        draft = spec.draft_model_config.hf_config
        reason = DraftW4A16Mlp.unsupported(
            draft.hidden_size, draft.intermediate_size_mlp // tp
        )
        if reason is not None:
            raise ValueError(f"{OPTION}: {reason}")
        identity = DraftMlpIdentity.from_vllm_config(vllm_config)
        rank = get_tensor_model_parallel_rank()
        amax = None
        if option.mode == "scales":
            amax = load_amax(option.prefix, rank, tp, draft.num_hidden_layers, identity)
        return cls(option, identity, tp, rank, amax)

    def calibration(
        self, num_layers: int, device: torch.device
    ) -> "DraftMlpCalibration | None":
        """With `calibrate`, a calibration recording into the option's files."""
        if self.option.mode != "calibrate":
            return None
        return DraftMlpCalibration(
            self.option.prefix, self.rank, self.tp, num_layers, device, self.identity
        )


class DraftMlpCalibration:
    """Capture-safe BF16 amax.

    GPU updates copy into pinned host tensors once a step, after the last
    layer's. The writer never calls CUDA; periodic files are eventually
    consistent snapshots, not stream barriers. Synchronize the producing
    stream before an exact final `write` or `close`.
    """

    def __init__(
        self,
        prefix: str,
        rank: int,
        tp: int,
        num_layers: int,
        device: torch.device,
        identity: DraftMlpIdentity,
    ) -> None:
        self.path = scales_path(prefix, rank)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.header = _manifest_header(identity, tp, rank, num_layers)
        self.last_layer = num_layers - 1
        self.amax = torch.zeros((num_layers, 2), dtype=torch.float32, device=device)
        self.calls = torch.zeros((num_layers,), dtype=torch.int64, device=device)
        self.host_amax = torch.zeros_like(self.amax, device="cpu").pin_memory()
        self.host_calls = torch.zeros_like(self.calls, device="cpu").pin_memory()
        self._stop = threading.Event()
        self._closed = False
        self._write_lock = threading.Lock()
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()
        atexit.register(self.close)
        _calibrations.add(self)

    def observe(self, layer: int, x: torch.Tensor, inter: torch.Tensor) -> None:
        row = self.amax[layer]
        for column, value in enumerate((x, inter)):
            torch.maximum(row[column], value.abs().amax().float(), out=row[column])
        self.calls[layer] += 1
        if layer == self.last_layer:
            self.host_amax.copy_(self.amax, non_blocking=True)
            self.host_calls.copy_(self.calls, non_blocking=True)

    def reset(self) -> None:
        """Forget every observation so far."""
        torch.accelerator.synchronize()
        self.amax.zero_()
        self.calls.zero_()
        with self._write_lock:
            self.host_amax.zero_()
            self.host_calls.zero_()
        torch.accelerator.synchronize()

    def write(self) -> None:
        with self._write_lock:
            data = {
                **self.header,
                "layers": self.host_amax.tolist(),
                "calls": self.host_calls.tolist(),
            }
            with atomic_writer(self.path) as f:
                json.dump(data, f)

    def close(self) -> None:
        if self._closed:
            return
        self._stop.set()
        self._thread.join()
        self.write()
        self._closed = True
        atexit.unregister(self.close)

    def _writer(self) -> None:
        while not self._stop.wait(CALIBRATION_WRITE_S):
            try:
                self.write()
            except OSError as e:
                logger.warning("draft MLP calibration write failed: %s", e)


_calibrations: "weakref.WeakSet[DraftMlpCalibration]" = weakref.WeakSet()


def reset_calibrations() -> None:
    """Reset this process's calibrations. Profiling, warm-up and capture run
    the draft (its captured steps included) on dummy inputs, which must not
    set static scales: the model runner state calls this before the first
    request after them."""
    for calibration in list(_calibrations):
        calibration.reset()
