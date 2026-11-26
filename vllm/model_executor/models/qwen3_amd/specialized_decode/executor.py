# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3 RDNA dense specialized decode runtime."""

from __future__ import annotations

import ctypes

import torch

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

from . import backend
from .caches import gather_kv_bindings
from .planning import decide_step, matches_hidden_states
from .weights import _head_dim, build_layer_bindings, final_norm, rope_cache

log = init_logger(__name__)


class Qwen3DenseSpecializedDecode:
    def __init__(self, owner, config):
        self.owner = owner
        self.config = config
        self.lib = backend.load(config.lib_name)
        if self.lib is None:
            raise ValueError(
                "Qwen3 RDNA dense specialized decode library is missing for "
                f"{backend.arch()}"
            )
        self._check_lib_shape()
        self._handle = None
        self._keep: list[torch.Tensor] = []
        self._kv_ptr = None
        self._out = None
        self._off_reason = None
        self._warned = False
        self._steps = 0

    def _check_lib_shape(self) -> None:
        shape = backend.lib_shape(self.lib)
        model_config = self.owner.config
        expected_qk_norm = bool(shape.pop("qk_norm"))
        if expected_qk_norm != self.config.qk_norm:
            raise ValueError("Qwen3 RDNA dense specialized decode qk_norm mismatch")
        checks = {
            "hidden_size": model_config.hidden_size,
            "intermediate_size": model_config.intermediate_size,
            "num_attention_heads": model_config.num_attention_heads,
            "num_key_value_heads": model_config.num_key_value_heads,
            "head_dim": _head_dim(model_config),
        }
        for key, expected in shape.items():
            got = checks[key]
            if got != expected:
                raise ValueError(
                    "Qwen3 RDNA dense specialized decode expects "
                    f"{key}={expected}, got {got}"
                )
        if abs(model_config.rms_norm_eps - self.config.rms_norm_eps) > 1e-12:
            raise ValueError(
                "Qwen3 RDNA dense specialized decode expects "
                f"rms_norm_eps={self.config.rms_norm_eps}, got "
                f"{model_config.rms_norm_eps}"
            )
        if (
            self.owner.vllm_config.model_config.max_model_len
            > self.lib.mk_vllm_dense_max_ctx()
        ):
            raise ValueError(
                "Qwen3 RDNA dense specialized decode max_model_len exceeds "
                "kernel library support"
            )

    def _create(self) -> bool:
        model = self.owner.model
        config = self.owner.config
        layers = build_layer_bindings(self.owner, self.config.qk_norm, self._keep)
        norm = final_norm(self.owner, self._keep)
        rope = rope_cache(self.owner, self._keep)
        embed = model.embed_tokens.weight
        if embed.dtype != torch.bfloat16:
            self._off_reason = "embed_weight_dtype"
            return False
        if not embed.is_contiguous():
            embed = embed.contiguous()
            self._keep.append(embed)
        err = ctypes.create_string_buffer(512)
        handle = self.lib.mk_vllm_dense_create(
            len(model.layers),
            layers,
            embed.data_ptr(),
            norm.data_ptr(),
            rope.data_ptr(),
            rope.shape[0],
            err,
            len(err),
        )
        if not handle:
            self._off_reason = "create_failed:" + err.value.decode()
            return False
        self._handle = handle
        self._out = torch.empty(
            1, config.hidden_size, dtype=torch.bfloat16, device=embed.device
        )
        log.info(
            "Qwen3 RDNA dense specialized decode backend ready (%s)",
            backend.arch(),
        )
        return True

    def _bind(self, kv0: torch.Tensor) -> bool:
        bindings = gather_kv_bindings(
            self.owner.model,
            self.owner.config.num_key_value_heads,
            _head_dim(self.owner.config),
        )
        if bindings is None:
            self._off_reason = "kv_cache_layout"
            return False
        keys, values, block_size = bindings
        n = len(keys)
        rc = self.lib.mk_vllm_dense_bind(
            self._handle,
            (ctypes.c_void_p * n)(*keys),
            (ctypes.c_void_p * n)(*values),
            block_size,
        )
        if rc:
            self._off_reason = f"bind_failed:{rc}"
            return False
        self._kv_ptr = kv0.data_ptr()
        return True

    def _setup(self) -> bool:
        kv0 = self.owner.model.layers[0].self_attn.attn.kv_cache
        if not isinstance(kv0, torch.Tensor) or kv0.numel() == 0:
            self._off_reason = "kv_cache_unallocated"
            return False
        if self._handle is not None and self._kv_ptr == kv0.data_ptr():
            return True
        if torch.cuda.is_current_stream_capturing():
            self._off_reason = "graph_capture_before_bind"
            return False
        try:
            if self._handle is None and not self._create():
                return False
            return self._bind(kv0)
        except Exception as err:
            self._off_reason = f"setup_exception:{err!r}"
            return False

    def maybe_forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        metadata = get_forward_context().attn_metadata
        layer_name = self.owner.model.layers[0].self_attn.attn.layer_name
        decision = decide_step(
            input_ids=input_ids,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            metadata=metadata,
            layer_name=layer_name,
        )
        if not decision.use_specialized:
            return None
        if not self._setup():
            if not self._warned and self._off_reason not in (
                "kv_cache_unallocated",
                "graph_capture_before_bind",
            ):
                self._warned = True
                log.warning(
                    "Qwen3 RDNA dense specialized decode fell back to stock: %s",
                    self._off_reason,
                )
            return None
        assert input_ids.dtype == torch.int32 and positions.dtype == torch.int64
        rc = self.lib.mk_vllm_dense_launch(
            self._handle,
            input_ids.data_ptr(),
            positions.data_ptr(),
            decision.block_table.data_ptr(),
            self._out.data_ptr(),
            torch.cuda.current_stream().cuda_stream,
        )
        if rc:
            raise RuntimeError("Qwen3 RDNA dense specialized decode launch failed")
        if self._steps == 0:
            log.info(
                "Qwen3 RDNA dense specialized decode engaged (graph capture: %s)",
                torch.cuda.is_current_stream_capturing(),
            )
        self._steps += 1
        return self._out

    def check_hidden_states(self, hidden_states: torch.Tensor) -> None:
        if self._handle is not None and self.lib.mk_vllm_dense_status(self._handle):
            raise RuntimeError(
                "Qwen3 RDNA dense specialized decode grid barrier timed out"
            )
        if not matches_hidden_states(hidden_states, self._out):
            return
