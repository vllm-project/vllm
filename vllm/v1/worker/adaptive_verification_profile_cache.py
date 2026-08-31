# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent, fail-closed cache for adaptive-verification cost curves."""

import contextlib
import dataclasses
import enum
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.async_utils import StepTimingSample
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)

PROFILE_CACHE_SCHEMA_VERSION = 1
_SENTINEL_REPLAYS = 2
_SENTINEL_MIN_RATIO = 0.4
_SENTINEL_MAX_RATIO = 2.5

CostCurve = list[tuple[int, float]]
CachedCurves = tuple[CostCurve, CostCurve]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(b"\0")
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _model_identity(runner: "GPUModelRunner") -> dict[str, Any]:
    model_config = runner.model_config
    model = str(model_config.model)
    hf_config = model_config.hf_config.to_dict()
    hf_config_digest = _digest(hf_config)
    model_path = Path(model).expanduser()
    if model_path.is_dir():
        config_path = model_path / "config.json"
        index_paths = list(model_path.glob("*.index.json"))
        if not config_path.is_file() or not index_paths:
            raise ValueError(
                "local models require config.json and at least one *.index.json "
                "for adaptive profile caching"
            )
        return {
            "kind": "local",
            "config_index_digest": _file_digest([config_path, *index_paths]),
            "hf_config_digest": hf_config_digest,
        }

    resolved_revision = getattr(model_config.hf_config, "_commit_hash", None)
    if not resolved_revision:
        raise ValueError(
            "remote models require a resolved commit hash for "
            "adaptive profile caching"
        )
    return {
        "kind": "remote",
        "model": model,
        "revision": str(model_config.revision or ""),
        "resolved_revision": str(resolved_revision),
        "code_revision": str(model_config.code_revision or ""),
        "tokenizer_revision": str(model_config.tokenizer_revision or ""),
        "hf_config_digest": hf_config_digest,
    }


def _package_version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return "not-installed"


def _module_types(module: torch.nn.Module | None) -> list[str]:
    if module is None:
        return []
    return sorted(
        {
            f"{type(child).__module__}.{type(child).__qualname__}"
            for child in module.modules()
        }
    )


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_jsonable(item) for item in value]
        return sorted(items, key=_canonical_json)
    if isinstance(value, enum.Enum):
        return str(value.value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, (Path, torch.device, torch.dtype)):
        return str(value)
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _module_backend_state(module: torch.nn.Module | None) -> list[str]:
    """Capture resolved auto-selected kernel enums/classes from loaded modules."""
    if module is None:
        return []
    backend_attributes = {
        "backend",
        "experts_cls",
        "fp8_backend",
        "int8_backend",
        "linear_backend",
        "moe_backend",
        "mxfp4_backend",
        "nvfp4_backend",
        "unquantized_backend",
        "w4a8_backend",
        "wna16_backend",
    }
    signatures: set[str] = set()
    for child in module.modules():
        owner = f"{type(child).__module__}.{type(child).__qualname__}"
        objects = [("module", child)]
        quant_method = getattr(child, "quant_method", None)
        if quant_method is not None:
            objects.append(("quant_method", quant_method))
        for role, value in objects:
            state = vars(value)
            selected = {
                key: _jsonable(state[key])
                for key in sorted(backend_attributes & state.keys())
            }
            if not selected and role == "module":
                continue
            selected.update(
                {
                    "owner": owner,
                    "role": role,
                    "type": f"{type(value).__module__}.{type(value).__qualname__}",
                }
            )
            signatures.add(_canonical_json(selected).decode())
    return sorted(signatures)


def build_profile_cache_factors(
    runner: "GPUModelRunner", capture_sizes: list[int]
) -> dict[str, Any]:
    """Build a strict fingerprint input after runtime backends are resolved."""
    from vllm import __version__ as vllm_version

    capability = current_platform.get_device_capability()
    parallel = runner.parallel_config
    scheduler = runner.scheduler_config
    compilation = runner.compilation_config
    speculative = runner.speculative_config
    kernel = runner.vllm_config.kernel_config
    attn_groups = getattr(runner, "attn_groups", ())
    kv_connector = getattr(runner, "kv_connector", None)

    return {
        "schema": PROFILE_CACHE_SCHEMA_VERSION,
        "model": _model_identity(runner),
        "vllm_config": runner.vllm_config.compute_hash(),
        "hardware": {
            "device_name": current_platform.get_device_name(),
            "device_capability": str(capability) if capability else "",
            "device_total_memory": current_platform.get_device_total_memory(),
        },
        "software": {
            "vllm": vllm_version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda or "",
            "hip": torch.version.hip or "",
            "cudnn": str(torch.backends.cudnn.version() or ""),
            "flashinfer": _package_version("flashinfer-python", "flashinfer"),
        },
        "parallel": {
            "tp": parallel.tensor_parallel_size,
            "pp": parallel.pipeline_parallel_size,
            "dp": parallel.data_parallel_size,
            "dcp": parallel.decode_context_parallel_size,
        },
        "backends": {
            "attention": sorted(group.backend.get_name() for group in attn_groups),
            "kv": [_jsonable(group.kv_cache_spec) for group in attn_groups],
            "kv_connector": (
                f"{type(kv_connector).__module__}.{type(kv_connector).__qualname__}"
            ),
            "moe_setting": kernel.moe_backend,
            "linear_setting": kernel.linear_backend,
            # Class selection makes auto-selected MoE/linear implementations part
            # of the key without relying on private per-kernel enum attributes.
            "target_module_types": _module_types(runner.model),
            "draft_module_types": _module_types(runner.get_draft_model()),
            "target_kernel_selection": _module_backend_state(runner.model),
            "draft_kernel_selection": _module_backend_state(runner.get_draft_model()),
        },
        "profile": {
            "k": speculative.num_speculative_tokens,
            "max_num_seqs": scheduler.max_num_seqs,
            "max_num_batched_tokens": scheduler.max_num_batched_tokens,
            "max_model_len": runner.model_config.max_model_len,
            "kv_cache_dtype": str(runner.kv_cache_dtype),
            "cudagraph_mode": str(compilation.cudagraph_mode),
            "capture_sizes": list(capture_sizes),
            "context_len": envs.VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN,
        },
    }


def profile_cache_fingerprint(factors: dict[str, Any]) -> str:
    return _digest(factors)


def _cache_path(fingerprint: str, cache_root: str | None = None) -> str:
    root = envs.VLLM_CACHE_ROOT if cache_root is None else cache_root
    return os.path.join(
        root, "adaptive_verification", f"profile_{fingerprint}.json"
    )


def _valid_curve(value: Any) -> CostCurve | None:
    if not isinstance(value, list) or not value:
        return None
    curve: CostCurve = []
    previous_x = 0
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        x, y = point
        if (
            not isinstance(x, int)
            or isinstance(x, bool)
            or x <= previous_x
            or not isinstance(y, (int, float))
            or isinstance(y, bool)
            or not math.isfinite(y)
            or y <= 0
        ):
            return None
        curve.append((x, float(y)))
        previous_x = x
    return curve


def load_profile_cache(
    factors: dict[str, Any], cache_root: str | None = None
) -> CachedCurves | None:
    fingerprint = profile_cache_fingerprint(factors)
    path = _cache_path(fingerprint, cache_root)
    try:
        with open(path) as file:
            payload = json.load(file)
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        logger.warning("Ignoring unreadable adaptive profile cache %s: %s", path, error)
        return None

    if not isinstance(payload, dict):
        logger.warning("Ignoring invalid adaptive profile cache %s", path)
        return None
    checksum = payload.pop("checksum", None)
    if (
        payload.get("schema") != PROFILE_CACHE_SCHEMA_VERSION
        or payload.get("fingerprint") != fingerprint
        or payload.get("factors") != factors
        or not isinstance(checksum, str)
        or checksum != _digest(payload)
    ):
        logger.warning("Ignoring invalid adaptive profile cache %s", path)
        return None
    draft_curve = _valid_curve(payload.get("draft_curve"))
    verify_curve = _valid_curve(payload.get("verify_curve"))
    if draft_curve is None or verify_curve is None:
        logger.warning("Ignoring malformed adaptive profile curves in %s", path)
        return None
    return draft_curve, verify_curve


def save_profile_cache(
    factors: dict[str, Any],
    curves: CachedCurves,
    cache_root: str | None = None,
) -> None:
    """Atomically save exact calibrated curves; failures never block startup."""
    draft_curve = _valid_curve(curves[0])
    verify_curve = _valid_curve(curves[1])
    if draft_curve is None or verify_curve is None:
        logger.warning("Not saving invalid adaptive profile curves")
        return
    fingerprint = profile_cache_fingerprint(factors)
    path = _cache_path(fingerprint, cache_root)
    payload = {
        "schema": PROFILE_CACHE_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "factors": factors,
        "draft_curve": draft_curve,
        "verify_curve": verify_curve,
    }
    payload["checksum"] = _digest(payload)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "w") as file:
            json.dump(payload, file, sort_keys=True, separators=(",", ":"))
        os.replace(tmp, path)
        logger.info("Saved adaptive verification profile cache %s", path)
    except (OSError, TypeError, ValueError) as error:
        logger.warning("Failed to save adaptive profile cache %s: %s", path, error)
        with contextlib.suppress(OSError):
            os.unlink(tmp)


def sentinel_batches(capture_sizes: list[int]) -> list[dict[str, int]]:
    """Two graph-replay probes at the largest calibrated graph shape."""
    if not capture_sizes:
        return []
    batch = {
        "num_tokens": capture_sizes[-1],
        "context_len": envs.VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN,
    }
    return [batch.copy() for _ in range(_SENTINEL_REPLAYS)]


def validate_profile_sentinel(
    samples: list["StepTimingSample"], curves: CachedCurves
) -> bool:
    """Accept only when both measured stages remain near their cached points."""
    if len(samples) != _SENTINEL_REPLAYS:
        return False
    draft = dict(curves[0])
    verify = dict(curves[1])
    for sample in samples:
        if not sample.full_cudagraph:
            return False
        expected_draft = draft.get(sample.num_reqs)
        expected_verify = verify.get(sample.num_target_tokens)
        if expected_draft is None or expected_verify is None:
            return False
        for actual, expected in (
            (sample.drafter_ms, expected_draft),
            (sample.forward_ms, expected_verify),
        ):
            ratio = actual / expected
            if (
                not math.isfinite(actual)
                or actual <= 0
                or ratio < _SENTINEL_MIN_RATIO
                or ratio > _SENTINEL_MAX_RATIO
            ):
                return False
    return True


def initialize_adaptive_verification_profile(
    runner: "GPUModelRunner", capture_sizes: list[int]
) -> None:
    """Load+validate cached curves, or run and persist full calibration.

    Every TP rank fingerprints its resolved runtime and participates in the
    sentinel. Rank 0 alone performs filesystem I/O. This keeps the number of
    distributed dummy runs identical and rejects heterogeneous/stale ranks.
    """
    from vllm.distributed.parallel_state import get_tp_group

    manager = runner.adaptive_verification
    assert manager is not None
    manager.configure_profile(capture_sizes)

    factors: dict[str, Any] | None = None
    curves: CachedCurves | None = None
    tp_group = get_tp_group()
    if envs.VLLM_ENABLE_ADAPTIVE_VERIFICATION_PROFILE_CACHE:
        try:
            factors = build_profile_cache_factors(runner, capture_sizes)
            local_fingerprint = profile_cache_fingerprint(factors)
        # This optimization must never make a previously valid boot fail.
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "Adaptive profile cache unavailable on this rank: %s", error
            )
            local_fingerprint = None

        source_fingerprint = tp_group.broadcast_object(local_fingerprint, src=0)
        agrees = torch.tensor(
            int(
                local_fingerprint is not None
                and local_fingerprint == source_fingerprint
            ),
            dtype=torch.int32,
            device=runner.device,
        )
        all_agree = int(tp_group.all_reduce(agrees).item()) == tp_group.world_size
        if not all_agree:
            factors = None
            logger.warning(
                "Adaptive profile cache fingerprints differ across TP ranks; "
                "falling back to full profiling"
            )
        elif tp_group.rank_in_group == 0:
            assert factors is not None
            curves = load_profile_cache(factors)
        curves = tp_group.broadcast_object(curves, src=0)

    if curves is not None:
        with runner.step_timing.collect() as sentinel_samples:
            for batch in sentinel_batches(capture_sizes):
                runner._dummy_run(**batch)
        valid = torch.tensor(
            int(validate_profile_sentinel(sentinel_samples, curves)),
            dtype=torch.int32,
            device=runner.device,
        )
        valid_count = int(tp_group.all_reduce(valid).item())
        if valid_count == tp_group.world_size:
            manager.set_cost_curves(*curves)
            logger.info(
                "Applied adaptive verification profile cache after %d sentinel "
                "replays (fingerprint %s)",
                _SENTINEL_REPLAYS,
                profile_cache_fingerprint(factors)[:16] if factors else "unknown",
            )
            return
        logger.warning(
            "Adaptive profile cache failed its GPU sentinel; falling back to "
            "full profiling"
        )

    with runner.step_timing.collect() as timings:
        for batch in manager.batches_to_profile(capture_sizes):
            runner._dummy_run(**batch)
    calibrated_curves = manager.set_initial_cost_curves(timings)
    if factors is not None and tp_group.rank_in_group == 0:
        save_profile_cache(factors, calibrated_curves)
