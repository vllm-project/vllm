# SPDX-License-Identifier: Apache-2.0

"""Type codecs for trace argument serialization.

The trace recorder needs to serialize arbitrary Python values that
appear as arguments to decorated functions.  Msgpack natively handles
``int``, ``float``, ``str``, ``bytes``, ``bool``, ``None``, ``list``,
``tuple``, ``dict``.  Anything else needs an explicit codec.

A codec is a pair ``(encode, decode)`` keyed on a Python type.  At
encode time the value is wrapped in a ``{"__t__": tag, "v": payload}``
dict so the decoder can recognize it without losing the round-trip.

This registry is shared between the recorder (PR1, encode-only path
exercised) and the replay driver (PR2, decode-only path).  Both halves
ship together to keep the format and behavior coherent.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Any, Callable

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import (
    AttnWindowDesc,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchHandle,
    PrefetchMode,
    PrefetchRequestSpec,
    TrimPolicy,
)


@dataclass(frozen=True)
class TypeCodec:
    """Encode/decode pair for a single Python type."""

    tag: str
    encode: Callable[[Any], Any]
    decode: Callable[[Any], Any]


# Tag dispatch table populated by ``register_codec``.  Keyed by type.
_BY_TYPE: dict[type, TypeCodec] = {}
# Tag dispatch table for decode.  Keyed by tag string.
_BY_TAG: dict[str, TypeCodec] = {}

_WRAP_KEY = "__t__"
_VALUE_KEY = "v"


def register_codec(t: type, codec: TypeCodec) -> None:
    """Register a codec for type ``t``.

    Raises:
        ValueError: If ``t`` or ``codec.tag`` is already registered.
    """
    if t in _BY_TYPE:
        raise ValueError(f"codec already registered for type {t!r}")
    if codec.tag in _BY_TAG:
        raise ValueError(f"codec tag {codec.tag!r} already in use")
    _BY_TYPE[t] = codec
    _BY_TAG[codec.tag] = codec


# ---------------------------------------------------------------------------
# Encode / decode entry points
# ---------------------------------------------------------------------------

# Native msgpack types pass through unchanged.  Everything else must be
# wrapped via a registered codec.
_PASSTHROUGH = (int, float, str, bytes, bool, type(None))


def encode_value(v: Any) -> Any:
    """Encode ``v`` to a msgpack-friendly representation.

    Recursively encodes lists, tuples, and dicts.  Tuples are preserved
    via a tag so they can be decoded back to tuples (msgpack would
    otherwise round-trip them as lists).

    Raises:
        TypeError: If ``v`` is of a type with no registered codec.
    """
    # Codec lookup by exact type takes priority so that registered
    # types which happen to subclass ``tuple`` (e.g. ``torch.Size``) are
    # handled by their codec rather than the generic tuple branch.
    codec = _BY_TYPE.get(type(v))
    if codec is not None:
        return {_WRAP_KEY: codec.tag, _VALUE_KEY: codec.encode(v)}

    if isinstance(v, _PASSTHROUGH):
        return v
    if isinstance(v, list):
        return [encode_value(x) for x in v]
    if isinstance(v, tuple):
        return {_WRAP_KEY: "tuple", _VALUE_KEY: [encode_value(x) for x in v]}
    if isinstance(v, dict):
        # Dict keys must already be strings/ints for msgpack.  We do not
        # encode keys, only values, to keep the on-wire form readable.
        return {k: encode_value(x) for k, x in v.items()}

    raise TypeError(
        f"trace.codecs: no codec registered for type {type(v).__name__!r} "
        f"(value={v!r}). Register one via register_codec() or extend "
        f"the default registry."
    )


def decode_value(v: Any) -> Any:
    """Decode a msgpack-deserialized value back to its native form.

    Raises:
        ValueError: If a wrapped value carries an unknown tag.
    """
    if isinstance(v, list):
        return [decode_value(x) for x in v]
    if isinstance(v, dict):
        tag = v.get(_WRAP_KEY)
        if tag is None:
            return {k: decode_value(x) for k, x in v.items()}
        if tag == "tuple":
            return tuple(decode_value(x) for x in v[_VALUE_KEY])
        codec = _BY_TAG.get(tag)
        if codec is None:
            raise ValueError(f"trace.codecs: unknown tag {tag!r}")
        return codec.decode(v[_VALUE_KEY])
    return v


def encode_args(args: dict[str, Any]) -> dict[str, Any]:
    """Encode an argument dict for serialization."""
    return {k: encode_value(v) for k, v in args.items()}


def decode_args(args: dict[str, Any]) -> dict[str, Any]:
    """Decode an argument dict back to native values."""
    return {k: decode_value(v) for k, v in args.items()}


# ---------------------------------------------------------------------------
# Default codecs for LMCache types
# ---------------------------------------------------------------------------


def _enc_object_key(k: ObjectKey) -> dict[str, Any]:
    return {
        "chunk_hash": k.chunk_hash,
        "model_name": k.model_name,
        "kv_rank": k.kv_rank,
        "object_group_id": k.object_group_id,
    }


def _dec_object_key(d: dict[str, Any]) -> ObjectKey:
    return ObjectKey(
        chunk_hash=d["chunk_hash"],
        model_name=d["model_name"],
        kv_rank=d["kv_rank"],
        object_group_id=d.get("object_group_id", 0),
    )


def _enc_layout_desc(d: MemoryLayoutDesc) -> dict[str, Any]:
    return {
        "shapes": [list(s) for s in d.shapes],
        "dtypes": [str(dt) for dt in d.dtypes],
    }


# Mapping from str(torch.dtype) back to the dtype object.  Built lazily
# the first time a layout desc is decoded.
_DTYPE_BY_NAME: dict[str, torch.dtype] = {}


def _resolve_dtype(name: str) -> torch.dtype:
    if not _DTYPE_BY_NAME:
        for attr in dir(torch):
            obj = getattr(torch, attr)
            if isinstance(obj, torch.dtype):
                _DTYPE_BY_NAME[str(obj)] = obj
    dtype = _DTYPE_BY_NAME.get(name)
    if dtype is None:
        raise ValueError(f"trace.codecs: unknown torch dtype {name!r}")
    return dtype


def _dec_layout_desc(d: dict[str, Any]) -> MemoryLayoutDesc:
    return MemoryLayoutDesc(
        shapes=[torch.Size(s) for s in d["shapes"]],
        dtypes=[_resolve_dtype(dt) for dt in d["dtypes"]],
    )


def _enc_prefetch_handle(h: PrefetchHandle) -> dict[str, Any]:
    return {
        "prefetch_request_id": h.prefetch_request_id,
        "external_request_id": h.external_request_id,
        # Derived count kept for readable traces; decode rebuilds from indices.
        "l1_prefix_hit_count": len(h.l1_found_indices),
        "l1_found_indices": list(h.l1_found_indices),
        "total_requested_keys": h.total_requested_keys,
        "submit_time": h.submit_time,
        "l2_orig_indices": list(h.l2_orig_indices),
    }


def _dec_prefetch_handle(d: dict[str, Any]) -> PrefetchHandle:
    return PrefetchHandle(
        prefetch_request_id=d["prefetch_request_id"],
        external_request_id=d["external_request_id"],
        l1_found_indices=tuple(d["l1_found_indices"]),
        total_requested_keys=d["total_requested_keys"],
        submit_time=d["submit_time"],
        l2_orig_indices=tuple(d.get("l2_orig_indices", ())),
    )


def _enc_torch_size(s: torch.Size) -> list[int]:
    return list(s)


def _dec_torch_size(s: list[int]) -> torch.Size:
    return torch.Size(s)


def _enc_torch_dtype(dt: torch.dtype) -> str:
    return str(dt)


def _dec_torch_dtype(name: str) -> torch.dtype:
    return _resolve_dtype(name)


def _enc_trim_policy(p: TrimPolicy) -> str:
    return p.name


def _dec_trim_policy(name: str) -> TrimPolicy:
    return TrimPolicy[name]


def _enc_prefetch_mode(m: PrefetchMode) -> str:
    return m.name


def _dec_prefetch_mode(name: str) -> PrefetchMode:
    return PrefetchMode[name]


def _enc_attn_window(d: AttnWindowDesc) -> list[int]:
    return list(d.num_chunks_in_sw)


def _dec_attn_window(num_chunks_in_sw: list[int]) -> AttnWindowDesc:
    return AttnWindowDesc(num_chunks_in_sw=list(num_chunks_in_sw))


def _enc_prefetch_request_spec(s: PrefetchRequestSpec) -> dict[str, Any]:
    # Delegate each field to its registered codec so the struct survives
    # component-type changes (keys/layout/policy/attn/mode all have codecs).
    return {
        "keys": [encode_value(k) for k in s.keys],
        "layout_desc": encode_value(s.layout_desc),
        "extra_count": s.extra_count,
        "policy": encode_value(s.policy),
        "attn_desc": encode_value(s.attn_desc),
        "mode": encode_value(s.mode),
    }


def _dec_prefetch_request_spec(d: dict[str, Any]) -> PrefetchRequestSpec:
    return PrefetchRequestSpec(
        keys=[decode_value(k) for k in d["keys"]],
        layout_desc=decode_value(d["layout_desc"]),
        extra_count=d["extra_count"],
        policy=decode_value(d["policy"]),
        attn_desc=decode_value(d["attn_desc"]),
        mode=decode_value(d["mode"]),
    )


def _enc_set(s: set) -> list:
    return [encode_value(x) for x in s]


def _dec_set(items: list) -> set:
    return {decode_value(x) for x in items}


register_codec(
    ObjectKey,
    TypeCodec(tag="ObjectKey", encode=_enc_object_key, decode=_dec_object_key),
)
register_codec(
    MemoryLayoutDesc,
    TypeCodec(
        tag="MemoryLayoutDesc",
        encode=_enc_layout_desc,
        decode=_dec_layout_desc,
    ),
)
register_codec(
    PrefetchHandle,
    TypeCodec(
        tag="PrefetchHandle",
        encode=_enc_prefetch_handle,
        decode=_dec_prefetch_handle,
    ),
)
register_codec(
    torch.Size,
    TypeCodec(tag="torch.Size", encode=_enc_torch_size, decode=_dec_torch_size),
)
register_codec(
    torch.dtype,
    TypeCodec(tag="torch.dtype", encode=_enc_torch_dtype, decode=_dec_torch_dtype),
)
register_codec(
    AttnWindowDesc,
    TypeCodec(tag="AttnWindowDesc", encode=_enc_attn_window, decode=_dec_attn_window),
)
register_codec(
    TrimPolicy,
    TypeCodec(tag="TrimPolicy", encode=_enc_trim_policy, decode=_dec_trim_policy),
)
register_codec(
    PrefetchMode,
    TypeCodec(tag="PrefetchMode", encode=_enc_prefetch_mode, decode=_dec_prefetch_mode),
)
register_codec(
    PrefetchRequestSpec,
    TypeCodec(
        tag="PrefetchRequestSpec",
        encode=_enc_prefetch_request_spec,
        decode=_dec_prefetch_request_spec,
    ),
)
register_codec(
    set,
    TypeCodec(tag="set", encode=_enc_set, decode=_dec_set),
)
