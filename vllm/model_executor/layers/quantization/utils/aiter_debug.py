import contextlib
import contextvars
import itertools
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import torch

from vllm.forward_context import get_forward_context, is_forward_context_available

_CALL_COUNTER = itertools.count()
_DUMP_COUNTER = itertools.count()
_CURRENT_CALL: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "aiter_blockscale_debug_call", default=None
)
_DUMP_LOCK = threading.Lock()
_WEIGHT_FILE_CACHE: dict[tuple[int | None, tuple[int, ...], tuple[int, ...], str], str] = {}


def debug_enabled() -> bool:
    return bool(
        os.environ.get("AITER_DEBUG_TENSORS")
        or os.environ.get("AITER_DEBUG_TRACE_PATH")
        or os.environ.get("AITER_DEBUG_DUMP_DIR")
    )


def _stream_id() -> int | None:
    if not torch.cuda.is_available():
        return None
    try:
        stream = torch.cuda.current_stream()
    except Exception:
        return None
    for attr in ("cuda_stream", "hip_stream"):
        value = getattr(stream, attr, None)
        if value is not None:
            try:
                return int(value)
            except Exception:
                return None
    return None


def _storage_ptr(tensor: torch.Tensor) -> int | None:
    try:
        return int(tensor.untyped_storage().data_ptr())
    except Exception:
        return None


def _tensor_preview(tensor: torch.Tensor, limit: int = 8) -> dict[str, Any]:
    if tensor.numel() == 0:
        return {"sample": [], "checksum": 0.0}
    flat = tensor.detach().reshape(-1)[:limit].to(device="cpu")
    if tensor.is_floating_point() or tensor.is_complex():
        checksum_tensor = flat.to(torch.float32)
        sample = checksum_tensor.tolist()
        checksum = float(checksum_tensor.sum().item())
    else:
        checksum_tensor = flat.to(torch.int64)
        sample = checksum_tensor.tolist()
        checksum = float(checksum_tensor.sum().item())
    return {"sample": sample, "checksum": checksum}


def tensor_metadata(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    preview = _tensor_preview(tensor)
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "contiguous": bool(tensor.is_contiguous()),
        "numel": int(tensor.numel()),
        "data_ptr": int(tensor.data_ptr()),
        "storage_ptr": _storage_ptr(tensor),
        "storage_offset": int(tensor.storage_offset()),
        "sample": preview["sample"],
        "checksum": preview["checksum"],
    }


def _forward_context_metadata() -> dict[str, Any]:
    if not is_forward_context_available():
        return {}
    try:
        ctx = get_forward_context()
    except Exception:
        return {}
    batch = ctx.batch_descriptor
    return {
        "batch_descriptor": {
            "num_tokens": getattr(batch, "num_tokens", None),
            "num_reqs": getattr(batch, "num_reqs", None),
            "uniform": getattr(batch, "uniform", None),
            "has_lora": getattr(batch, "has_lora", None),
            "num_active_loras": getattr(batch, "num_active_loras", None),
        }
        if batch is not None
        else None,
        "cudagraph_runtime_mode": str(ctx.cudagraph_runtime_mode),
        "has_ubatch_slices": ctx.ubatch_slices is not None,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _append_jsonl(path: str, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=_json_default) + "\n")


def current_call() -> dict[str, Any] | None:
    return _CURRENT_CALL.get()


@contextlib.contextmanager
def blockscale_call_context(
    *,
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
) -> Any:
    if not debug_enabled():
        yield None
        return

    call = {
        "call_id": f"blockscale-{next(_CALL_COUNTER):06d}",
        "layer_prefix": getattr(layer, "prefix", None),
        "thread_id": threading.get_ident(),
        "stream_id": _stream_id(),
        "forward_context": _forward_context_metadata(),
    }
    token = _CURRENT_CALL.set(call)
    emit_trace(
        "blockscale.call.begin",
        tensors={
            "x": x,
            "weight": weight,
            "weight_scale": weight_scale,
            "bias": bias,
        },
        extra={
            "layer_prefix": call["layer_prefix"],
            "thread_id": call["thread_id"],
            "stream_id": call["stream_id"],
        },
    )
    try:
        yield call
    finally:
        emit_trace(
            "blockscale.call.end",
            extra={
                "layer_prefix": call["layer_prefix"],
                "thread_id": call["thread_id"],
                "stream_id": _stream_id(),
            },
        )
        _CURRENT_CALL.reset(token)


def emit_trace(
    stage: str,
    *,
    tensors: dict[str, torch.Tensor | None] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    if not debug_enabled():
        return

    call = current_call() or {}
    payload = {
        "timestamp_ms": int(time.time() * 1000),
        "stage": stage,
        "call_id": call.get("call_id"),
        "layer_prefix": call.get("layer_prefix"),
        "thread_id": call.get("thread_id", threading.get_ident()),
        "stream_id": _stream_id(),
        "forward_context": call.get("forward_context") or _forward_context_metadata(),
    }
    if tensors:
        payload["tensors"] = {
            name: tensor_metadata(tensor) for name, tensor in tensors.items()
        }
    if extra:
        payload["extra"] = extra

    trace_path = os.environ.get("AITER_DEBUG_TRACE_PATH")
    if trace_path:
        _append_jsonl(trace_path, payload)

    if os.environ.get("AITER_DEBUG_TENSORS"):
        print(json.dumps(payload, default=_json_default), flush=True)


def _dump_limit() -> int:
    raw = os.environ.get("AITER_DEBUG_DUMP_LIMIT", "0")
    try:
        return max(int(raw), 0)
    except ValueError:
        return 0


def maybe_dump_blockscale_call(
    stage: str,
    *,
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    output: torch.Tensor | None,
    extra: dict[str, Any] | None = None,
) -> str | None:
    dump_root = os.environ.get("AITER_DEBUG_DUMP_DIR")
    if not dump_root:
        return None

    dump_dir = Path(dump_root)
    dump_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dump_dir / "manifest.jsonl"

    with _DUMP_LOCK:
        dump_index = next(_DUMP_COUNTER)
        limit = _dump_limit()
        if limit and dump_index >= limit:
            return None

        call = current_call() or {}
        weight_key = (
            _storage_ptr(B),
            tuple(B.shape),
            tuple(B.stride()),
            str(B.dtype),
        )
        weight_file = _WEIGHT_FILE_CACHE.get(weight_key)
        if weight_file is None:
            weight_file = f"blockscale_weight_{len(_WEIGHT_FILE_CACHE):06d}.pt"
            torch.save(
                {
                    "WQ": B.detach().cpu(),
                    "w_scale": Bs.detach().cpu(),
                    "meta": {
                        "stage": stage,
                        "call_id": call.get("call_id"),
                        "weight": tensor_metadata(B),
                        "w_scale": tensor_metadata(Bs),
                    },
                },
                dump_dir / weight_file,
            )
            _WEIGHT_FILE_CACHE[weight_key] = weight_file

        activation_file = f"gemm_a8w8_blockscale_{dump_index:06d}.pt"
        torch.save(
            {
                "XQ": A.detach().cpu(),
                "x_scale": As.detach().cpu(),
                "output": output.detach().cpu() if output is not None else None,
                "weight_file": weight_file,
                "stage": stage,
                "call_id": call.get("call_id"),
                "extra": extra or {},
                "meta": {
                    "input": tensor_metadata(A),
                    "input_scale": tensor_metadata(As),
                    "output": tensor_metadata(output),
                    "stream_id": _stream_id(),
                    "thread_id": threading.get_ident(),
                    "forward_context": call.get("forward_context")
                    or _forward_context_metadata(),
                },
            },
            dump_dir / activation_file,
        )

        _append_jsonl(
            str(manifest_path),
            {
                "timestamp_ms": int(time.time() * 1000),
                "stage": stage,
                "call_id": call.get("call_id"),
                "activation_file": activation_file,
                "weight_file": weight_file,
                "stream_id": _stream_id(),
                "thread_id": threading.get_ident(),
                "extra": extra or {},
            },
        )
        return str(dump_dir / activation_file)
