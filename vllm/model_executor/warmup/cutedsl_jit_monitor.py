# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Detect CuTeDSL JIT compiles after engine startup."""

from __future__ import annotations

import reprlib
import hashlib
import os
import sys
import traceback
from importlib import import_module
from typing import Any, Callable

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_CUTE_DSL_JIT_AFTER_STARTUP_CHECK_ENV = "VLLM_CUTEDSL_JIT_AFTER_STARTUP_CHECK"
_CUTE_DSL_LOG_JIT_KEYS_ENV = "VLLM_CUTEDSL_LOG_JIT_KEYS"
_CUTE_DSL_CACHE_MODULES = (
    "flash_attn.cute.cache_utils",
    "vllm.vllm_flash_attn.cute.cache_utils",
)
_VALID_ACTIONS = {"off", "warn", "error"}
_startup_complete = False
_patched_contains: dict[type, Callable[[Any, object], bool]] = {}


def install_cutedsl_jit_monitor() -> None:
    """Patch imported CuTeDSL cache classes to detect post-startup misses."""
    for module_name in _CUTE_DSL_CACHE_MODULES:
        module = _try_import_cache_module(module_name)
        if module is None:
            continue
        for class_name in ("JITCache", "JITPersistentCache"):
            cache_cls = getattr(module, class_name, None)
            if cache_cls is None or cache_cls in _patched_contains:
                continue
            original_contains = cache_cls.__contains__
            original_contains = getattr(
                original_contains,
                "_vllm_cutedsl_jit_monitor_original",
                original_contains,
            )
            _patched_contains[cache_cls] = original_contains

            def patched_contains(
                self: object,
                key: object,
                *,
                _original_contains: Callable[[Any, object], bool] = original_contains,
            ) -> bool:
                hit = _original_contains(self, key)
                _log_cutedsl_jit_key(self, key, hit)
                if not hit:
                    _handle_cutedsl_jit_miss_after_startup(self, key)
                return hit

            setattr(
                patched_contains,
                "_vllm_cutedsl_jit_monitor_original",
                original_contains,
            )
            cache_cls.__contains__ = patched_contains


def _try_import_cache_module(module_name: str) -> object | None:
    module = sys.modules.get(module_name)
    if module is not None:
        return module

    try:
        return import_module(module_name)
    except ImportError:
        return None


def mark_cutedsl_startup_complete() -> None:
    """Mark the point after which CuTeDSL JIT misses are unexpected."""
    global _startup_complete

    install_cutedsl_jit_monitor()
    _startup_complete = True


def _handle_cutedsl_jit_miss_after_startup(cache: object, key: object) -> None:
    if not _startup_complete:
        return

    action = _get_cutedsl_jit_after_startup_action()
    if action == "off":
        return

    key_repr = reprlib.repr(key)
    cache_path = getattr(cache, "cache_path", None)
    message = (
        "CuTeDSL JIT compile triggered after engine startup. "
        f"cache={type(cache).__name__}, cache_path={cache_path}, key={key_repr}. "
        "This usually means CuTeDSL warmup missed a serving shape."
    )
    stack = "".join(traceback.format_stack(limit=32)[:-2])

    if action == "error":
        raise RuntimeError(f"{message}\nStack trace:\n{stack}")
    logger.warning("%s\nStack trace:\n%s", message, stack)


def _log_cutedsl_jit_key(cache: object, key: object, hit: bool) -> None:
    if os.environ.get(_CUTE_DSL_LOG_JIT_KEYS_ENV) != "1":
        return

    key_repr = repr(key)
    key_hash = hashlib.sha256(key_repr.encode()).hexdigest()[:16]
    logger.info(
        "CUTEDSL_JIT_KEY startup_complete=%s hit=%s cache=%s "
        "cache_path=%s key_hash=%s key_fields=%s",
        _startup_complete,
        hit,
        type(cache).__name__,
        getattr(cache, "cache_path", None),
        key_hash,
        _describe_cutedsl_key(key),
    )


def _describe_cutedsl_key(key: object) -> object:
    if not isinstance(key, tuple):
        return reprlib.repr(key)

    fwd_fields = (
        "dtype",
        "head_dim",
        "head_dim_v",
        "qhead_per_kvhead",
        "causal",
        "score_mod_hash",
        "mask_mod_hash",
        "use_block_sparsity",
        "block_sparse_broadcast_pattern",
        "aux_tensor_metadata",
        "lse_is_none",
        "cu_seqlens_q_is_none",
        "cu_seqlens_k_is_none",
        "seqused_q_is_none",
        "seqused_k_is_none",
        "has_page_table",
        "has_window_size_left",
        "has_window_size_right",
        "has_learnable_sink",
        "has_q_descale",
        "has_k_descale",
        "has_v_descale",
        "block_sparse_cu_total_m_blocks_is_none",
        "block_sparse_cu_block_idx_offsets_is_none",
        "tile_m",
        "tile_n",
        "q_stage",
        "num_threads",
        "is_split_kv",
        "pack_gqa",
        "arch",
        "paged_kv_non_tma",
        "use_2cta_instrs",
        "q_subtile_factor",
        "mma_pv_is_rs",
        "intra_wg_overlap",
        "use_clc_scheduler",
        "has_qv",
        "gather_kv_length",
        "sparse_kv",
        "disable_sparse_kv_bitmask",
        "fa_log_level",
        "output_quant_key",
    )
    if len(key) == len(fwd_fields):
        return {
            field: _short_key_value(value)
            for field, value in zip(fwd_fields, key)
        }
    return {
        "len": len(key),
        "values": [_short_key_value(value) for value in key],
    }


def _short_key_value(value: object) -> object:
    if isinstance(value, (bool, int, float, str, type(None))):
        return value
    return reprlib.repr(value)


def _get_cutedsl_jit_after_startup_action() -> str:
    action = envs.VLLM_CUTEDSL_JIT_AFTER_STARTUP_CHECK
    if action in _VALID_ACTIONS:
        return action

    logger.warning(
        "Invalid %s=%r. Expected one of %s; using 'warn'.",
        _CUTE_DSL_JIT_AFTER_STARTUP_CHECK_ENV,
        action,
        sorted(_VALID_ACTIONS),
    )
    return "warn"


def _reset_cutedsl_jit_monitor_for_tests() -> None:
    global _startup_complete

    for cache_cls, original_contains in list(_patched_contains.items()):
        cache_cls.__contains__ = original_contains
    _patched_contains.clear()
    _startup_complete = False
