# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only RWKV7 model."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from vllm import envs
from vllm.config import VllmConfig
from vllm.config.compilation import CompilationMode
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)

HEAD_SIZE = 64
DTYPE = torch.float16
CUDA_DEVICE = torch.device("cuda")
LOWRANK_SUFFIXES = (
    "att.w1",
    "att.w2",
    "att.a1",
    "att.a2",
    "att.g1",
    "att.g2",
    "att.v1",
    "att.v2",
)
LOWRANK_IN_ROWS_T = 7
LOWRANK_OUT_ROWS_T = 4
LOWRANK_FUSED_MIN_C = 1024
CMIX_NOFC_ROW20_MAX_T = 5
CMIX_NOFC_MAX_ROWS = 19
CMIX_NOFC_T512_MIN_ROWS = 8
LN1_TMIX_FUSE = True
CMIX_B1T1_NOFC = "b1t1_nofc"
CMIX_ROWS2_NOFC = "rows2_nofc"
CMIX_DENSE = "dense"


@dataclass(frozen=True)
class RWKV7ExecutionProfile:
    wkv_mode: str
    wkv_state_dtype: torch.dtype
    allow_fp16_accumulation: bool
    gemm_accumulation_policy: str


def resolve_execution_profile(wkv_mode: str) -> RWKV7ExecutionProfile:
    if wkv_mode == "fp32io16":
        return RWKV7ExecutionProfile(
            wkv_mode=wkv_mode,
            wkv_state_dtype=torch.float32,
            allow_fp16_accumulation=False,
            gemm_accumulation_policy="fp32",
        )
    if wkv_mode == "fp16":
        return RWKV7ExecutionProfile(
            wkv_mode=wkv_mode,
            wkv_state_dtype=torch.float16,
            allow_fp16_accumulation=True,
            gemm_accumulation_policy="fp16",
        )
    raise ValueError(
        f"VLLM_RWKV7_WKV_MODE={wkv_mode!r} is invalid for RWKV7. "
        "Expected one of: fp16, fp32io16."
    )


@dataclass(frozen=True)
class PathConfig:
    rows: int
    cmix_mode: str


def select_path(B: int, T: int) -> PathConfig:
    """All B/T dependent fast-path choices live here."""
    rows = B * T
    use_nofc = rows <= CMIX_NOFC_MAX_ROWS or (
        rows == 20 and CMIX_NOFC_ROW20_MAX_T >= T
    )
    cmix_mode = (
        CMIX_B1T1_NOFC
        if rows == 1
        else (CMIX_ROWS2_NOFC if use_nofc else CMIX_DENSE)
    )
    return PathConfig(rows=rows, cmix_mode=cmix_mode)


def is_lowrank_weight(key: str) -> bool:
    return key.endswith(LOWRANK_SUFFIXES)


def can_use_lowrank_fused(hidden_size: int, rows: int) -> bool:
    return hidden_size >= LOWRANK_FUSED_MIN_C and rows <= LOWRANK_IN_ROWS_T


def can_use_lowrank_out_fused(hidden_size: int, rows: int) -> bool:
    return hidden_size >= LOWRANK_FUSED_MIN_C and rows <= LOWRANK_OUT_ROWS_T


class RWKV7ForCausalLM(nn.Module):
    is_attention_free = True
    supports_pp = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        """Create the RWKV7 inference module; weights are loaded by vLLM."""
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.prefix = prefix
        self._validate_torch_compile_unsupported()

        hidden_size = int(getattr(self.config, "hidden_size", 0))
        vocab_size = int(getattr(self.config, "vocab_size", 0))
        head_size = int(getattr(self.config, "head_size", HEAD_SIZE))
        if hidden_size and head_size:
            num_attention_heads = hidden_size // head_size
        else:
            num_attention_heads = int(getattr(self.config, "num_attention_heads", 0))
            head_size = (
                hidden_size // num_attention_heads
                if hidden_size and num_attention_heads
                else HEAD_SIZE
            )
        num_hidden_layers = int(
            getattr(
                self.config,
                "num_hidden_layers",
                getattr(self.config, "n_layer", 0),
            )
        )

        self.hidden_size = hidden_size
        self.head_size = head_size
        self.num_attention_heads = num_attention_heads
        self.vocab_size = vocab_size
        self.z: dict[str, torch.Tensor] = {}
        self.raw_weight_names: set[str] | None = None
        self.total_num_layers = num_hidden_layers
        self.start_layer, self.end_layer = self._get_layer_range()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        if num_attention_heads % self.tp_size != 0:
            raise ValueError(
                "RWKV7 requires num_attention_heads "
                f"({num_attention_heads}) to be divisible "
                f"by tensor_parallel_size ({self.tp_size})."
            )
        self.tp_num_heads = num_attention_heads // self.tp_size
        self.tp_hidden_size = self.tp_num_heads * head_size
        self.vocab_size_padded = self._get_padded_vocab_size(vocab_size)
        self.execution_profile = resolve_execution_profile(envs.VLLM_RWKV7_WKV_MODE)
        self.wkv_mode = self.execution_profile.wkv_mode
        self.wkv_state_dtype = self.execution_profile.wkv_state_dtype
        self.allow_fp16_accumulation = self.execution_profile.allow_fp16_accumulation
        self.logits_processor = LogitsProcessor(vocab_size, logits_as_input=True)
        self.register_buffer("_dummy_param", torch.empty(0), persistent=False)

    def _get_layer_range(self) -> tuple[int, int]:
        parallel_config = getattr(self.vllm_config, "parallel_config", None)
        if parallel_config is None:
            return 0, self.total_num_layers
        get_indices = getattr(self.model_config, "get_layers_start_end_indices", None)
        if get_indices is not None:
            return get_indices(parallel_config)
        return 0, self.total_num_layers

    def _is_pp_first_rank(self) -> bool:
        return get_pp_group().is_first_rank

    def _is_pp_last_rank(self) -> bool:
        return get_pp_group().is_last_rank

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        hidden_size = self.hidden_size
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, hidden_size), dtype=DTYPE, device=device
                ),
                "v_first": torch.zeros(
                    (batch_size, hidden_size), dtype=DTYPE, device=device
                ),
            }
        )

    def _get_padded_vocab_size(self, vocab_size: int) -> int:
        tp_size = getattr(self, "tp_size", 1)
        if tp_size <= 1:
            return vocab_size
        return ((vocab_size + tp_size - 1) // tp_size) * tp_size

    def _tp_vocab_range(self, vocab_size: int | None = None) -> tuple[int, int, int]:
        vocab_size = self.vocab_size if vocab_size is None else vocab_size
        tp_size = getattr(self, "tp_size", 1)
        tp_rank = getattr(self, "tp_rank", 0)
        padded_vocab_size = self._get_padded_vocab_size(vocab_size)
        per_rank = padded_vocab_size // tp_size
        start = tp_rank * per_rank
        end = min(start + per_rank, vocab_size)
        return start, end, per_rank

    def _tp_slice(self, value: torch.Tensor, dim: int) -> torch.Tensor:
        tp_size = getattr(self, "tp_size", 1)
        if tp_size == 1:
            return value
        size = value.shape[dim]
        if size % tp_size != 0:
            raise ValueError(
                f"Cannot shard tensor dimension {dim} with size {size} "
                f"across tensor_parallel_size={tp_size}."
            )
        shard_size = size // tp_size
        start = getattr(self, "tp_rank", 0) * shard_size
        return value.narrow(dim, start, shard_size).contiguous()

    def _tp_hidden_slice(self, value: torch.Tensor, dim: int) -> torch.Tensor:
        tp_size = getattr(self, "tp_size", 1)
        if tp_size == 1:
            return value
        shard_size = getattr(self, "tp_hidden_size", value.shape[dim])
        start = getattr(self, "tp_rank", 0) * shard_size
        return value.narrow(dim, start, shard_size).contiguous()

    def _tp_vocab_slice(self, value: torch.Tensor) -> torch.Tensor:
        tp_size = getattr(self, "tp_size", 1)
        if tp_size == 1:
            return value
        start, end, per_rank = self._tp_vocab_range(value.shape[0])
        out = value.new_zeros((per_rank, *value.shape[1:]))
        if end > start:
            out[: end - start].copy_(value[start:end])
        return out.contiguous()

    def _shard_weight_for_tp(self, key: str, value: torch.Tensor) -> torch.Tensor:
        if getattr(self, "tp_size", 1) == 1:
            return value
        if key in ("emb.weight", "head.weight"):
            return self._tp_vocab_slice(value)
        parts = key.split(".")
        if len(parts) < 4 or parts[0] != "blocks":
            return value
        submodule = parts[2]
        name = ".".join(parts[3:])
        if submodule == "att":
            if name == "r_k":
                return self._tp_slice(value, 0)
            if name in {
                "ln_x.weight",
                "ln_x.bias",
                "k_k",
                "k_a",
                "w0",
                "a0",
                "v0",
            }:
                return self._tp_hidden_slice(value, 0)
            if name in {"receptance.weight", "key.weight", "value.weight"}:
                return self._tp_hidden_slice(value, 0)
            if name == "output.weight":
                return self._tp_hidden_slice(value, 1)
            if name in {"w2", "a2", "g2", "v2"}:
                return self._tp_hidden_slice(value, 1)
        elif submodule == "ffn":
            if name == "key.weight":
                return self._tp_slice(value, 0)
            if name == "value.weight":
                return self._tp_slice(value, 1)
        return value

    def _tp_all_reduce(self, value: torch.Tensor) -> torch.Tensor:
        if getattr(self, "tp_size", 1) == 1:
            return value
        return tensor_model_parallel_all_reduce(value)

    def _is_weight_needed_on_rank(self, key: str) -> bool:
        start_layer = getattr(self, "start_layer", 0)
        end_layer = getattr(
            self, "end_layer", getattr(self, "total_num_layers", 0)
        )
        total_layers = getattr(self, "total_num_layers", 0)
        if start_layer == 0 and end_layer >= total_layers:
            return True
        if key == "emb.weight" or key.startswith("blocks.0.ln0."):
            return start_layer == 0
        if key == "head.weight" or key.startswith("ln_out."):
            return end_layer >= total_layers
        parts = key.split(".")
        if len(parts) > 2 and parts[0] == "blocks":
            return start_layer <= int(parts[1]) < end_layer
        return True

    def _validate_torch_compile_unsupported(self) -> None:
        compilation_config = getattr(self.vllm_config, "compilation_config", None)
        if getattr(compilation_config, "mode", None) not in (
            None,
            CompilationMode.NONE,
        ):
            raise ValueError(
                "RWKV7 does not support torch.compile. Use non-compiled "
                "execution with CompilationMode.NONE."
            )

    @classmethod
    def get_model_state_cls(cls):
        from vllm.v1.worker.gpu.model_states.rwkv import RWKV7ModelState

        return RWKV7ModelState

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if hasattr(self, "_pending_weight_update"):
            pending = self._pending_weight_update
            loaded_names = set()
            for name, weight in weights:
                loaded_names.add(name)
                detached = weight.detach()
                pending[name] = (
                    detached.clone() if detached.is_cuda else detached.cpu().clone()
                )
            return loaded_names

        z = {name: weight.detach().cpu() for name, weight in weights}
        raw_weight_names = set(z.keys())
        self.raw_weight_names = raw_weight_names
        self._preprocess_weights(z)
        self._commit_preprocessed_weights(z)
        return raw_weight_names

    def start_weight_update(self) -> bool:
        """Handle checkpoint-format dense weight update chunks internally."""
        if hasattr(self, "_pending_weight_update"):
            raise RuntimeError("RWKV7 weight update is already active")
        self._pending_weight_update: dict[str, torch.Tensor] = {}
        return True

    def finish_weight_update(self) -> None:
        pending = getattr(self, "_pending_weight_update", None)
        if pending is None:
            raise RuntimeError("RWKV7 weight update is not active")
        try:
            if self.raw_weight_names is None:
                raise RuntimeError(
                    "RWKV7 online weight update requires a previous full "
                    "checkpoint load to establish raw weight names."
                )
            received = set(pending.keys())
            missing = self.raw_weight_names - received
            unexpected = received - self.raw_weight_names
            if missing or unexpected:
                raise ValueError(
                    "RWKV7 weight update key mismatch: "
                    f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
                )
            z = dict(pending)
            old_z = self.z
            try:
                self._preprocess_weights(z)
                self._commit_preprocessed_weights(
                    z,
                    reuse_existing_tensors=True,
                    existing_z=old_z,
                )
            except Exception:
                self.z = old_z
                raise
        finally:
            self.abort_weight_update()

    def abort_weight_update(self) -> None:
        if hasattr(self, "_pending_weight_update"):
            del self._pending_weight_update

    def get_parameter(self, target: str) -> nn.Parameter:
        if target == "_dummy_param":
            return super().get_parameter(target)
        raise NotImplementedError(
            "RWKV7 does not support direct kernel-format weight updates; "
            "use checkpoint-format dense weight update instead."
        )

    def _commit_preprocessed_weights(
        self,
        z: dict[str, torch.Tensor],
        *,
        reuse_existing_tensors: bool = False,
        existing_z: dict[str, torch.Tensor] | None = None,
    ) -> None:
        if reuse_existing_tensors and existing_z is not None:
            committed = existing_z
            for key in list(committed.keys()):
                if key not in z:
                    del committed[key]
            with torch.no_grad():
                for key, value in z.items():
                    old_value = committed.get(key)
                    if (
                        old_value is not None
                        and old_value.shape == value.shape
                        and old_value.dtype == value.dtype
                        and old_value.device == value.device
                    ):
                        old_value.copy_(value)
                    else:
                        committed[key] = value
            self.z = committed
        else:
            self.z = z
        torch.accelerator.synchronize()
        logger.info(
            "RWKV7 weights are ready L=%d C=%d H=%d N=%d V=%d",
            self.total_num_layers,
            self.hidden_size,
            self.num_attention_heads,
            self.head_size,
            self.vocab_size,
        )

    def _validate_raw_weight_shapes(self, z: dict[str, torch.Tensor]) -> None:
        r_k = z["blocks.0.att.r_k"].squeeze()
        emb = z["emb.weight"].squeeze()
        weight_heads, head_size = r_k.shape
        hidden_size = weight_heads * head_size
        vocab_size = emb.shape[0]
        max_layer = max(int(k.split(".")[1]) for k in z if k.startswith("blocks."))
        num_hidden_layers = max_layer + 1

        checks = (
            ("hidden_size", hidden_size),
            ("vocab_size", vocab_size),
            ("head_size", head_size),
            ("num_hidden_layers", num_hidden_layers),
        )
        for name, actual in checks:
            expected = getattr(self.config, name, None)
            if expected is not None and int(expected) != actual:
                raise ValueError(
                    f"RWKV7 config {name}={expected} does not match raw "
                    f"checkpoint {name}={actual}."
                )

    def _preprocess_weights(self, z: dict[str, torch.Tensor]) -> None:
        """Apply the albatross faster3a weight layout preprocessing."""
        self._validate_raw_weight_shapes(z)
        num_attention_heads, head_size = z["blocks.0.att.r_k"].shape
        hidden_size = num_attention_heads * head_size
        vocab_size = z["emb.weight"].shape[0]
        assert head_size == HEAD_SIZE
        self.num_attention_heads = num_attention_heads
        self.head_size = head_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        if num_attention_heads % getattr(self, "tp_size", 1) != 0:
            raise ValueError(
                "RWKV7 requires num_attention_heads "
                f"({num_attention_heads}) to be divisible "
                f"by tensor_parallel_size ({getattr(self, 'tp_size', 1)})."
            )
        max_layer = max(int(k.split(".")[1]) for k in z if k.startswith("blocks."))
        self.total_num_layers = max_layer + 1
        self.start_layer, self.end_layer = self._get_layer_range()
        self.tp_num_heads = num_attention_heads // getattr(self, "tp_size", 1)
        self.tp_hidden_size = self.tp_num_heads * head_size
        self.vocab_size_padded = self._get_padded_vocab_size(vocab_size)
        logger.info(
            "Detected RWKV7 model C=%d H=%d N=%d V=%d",
            hidden_size,
            num_attention_heads,
            head_size,
            vocab_size,
        )
        logger.info(
            "RWKV7 cmix no-fc path: rows<=%d row20_t<=%d",
            CMIX_NOFC_MAX_ROWS,
            CMIX_NOFC_ROW20_MAX_T,
        )

        emb_src = z["emb.weight"].squeeze()
        ln0_w_src = z["blocks.0.ln0.weight"].squeeze()
        ln0_b_src = z["blocks.0.ln0.bias"].squeeze()
        logger.info(
            "Preprocessing RWKV7 weights with profile=%s and GEMM accumulation=%s",
            self.execution_profile.wkv_mode,
            self.execution_profile.gemm_accumulation_policy,
        )
        for key in list(z.keys()):
            if not self._is_weight_needed_on_rank(key):
                del z[key]
                continue
            value = z[key].squeeze()
            value = self._shard_weight_for_tp(key, value)
            dev = CUDA_DEVICE
            is_lowrank = is_lowrank_weight(key)
            if not is_lowrank and (
                "key.weight" in key
                or "value.weight" in key
                or "receptance.weight" in key
                or "output.weight" in key
                or "head.weight" in key
            ):
                value = value.t()
            value = value.to(device=dev, dtype=DTYPE).contiguous()
            if key.endswith("att.r_k"):
                value = value.flatten().contiguous()
            if is_lowrank:
                z[key] = value
                z[key + ".t"] = value.t().contiguous()
            else:
                z[key] = value
        if self._is_weight_needed_on_rank("emb.weight"):
            emb_dev = CUDA_DEVICE
            ln0_w_bf16 = ln0_w_src.to(device=emb_dev).contiguous()
            ln0_b_bf16 = ln0_b_src.to(device=emb_dev).contiguous()
            vocab_start, vocab_end, vocab_per_rank = self._tp_vocab_range(vocab_size)
            emb = torch.zeros(
                (vocab_per_rank, hidden_size), dtype=DTYPE, device=emb_dev
            )
            if vocab_end > vocab_start:
                local = torch.ops.rwkv7_v3a_ops.emb_ln0_bf16_to_f16(
                    emb_src[vocab_start:vocab_end].to(device=emb_dev).contiguous(),
                    ln0_w_bf16,
                    ln0_b_bf16,
                )
                emb[: vocab_end - vocab_start].copy_(local)
            z["emb.weight"] = emb

    def zero_state(self, B: int) -> list[torch.Tensor]:
        """Create RWKV recurrent state tensors for a batch."""
        local_heads = getattr(self, "tp_num_heads", self.num_attention_heads)
        return [
            torch.zeros(
                (self.total_num_layers, 2, B, self.hidden_size),
                dtype=DTYPE,
                device="cuda",
            ),
            torch.zeros(
                (
                    self.total_num_layers,
                    B,
                    local_heads,
                    self.head_size,
                    self.head_size,
                ),
                dtype=self.wkv_state_dtype,
                device="cuda",
            ),
            torch.zeros((B,), dtype=torch.int32, device="cuda"),
        ]

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | list[torch.Tensor] | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        *,
        query_start_loc: torch.Tensor | None = None,
        idx_mapping: torch.Tensor | None = None,
        shift_state: torch.Tensor | None = None,
        wkv_state: torch.Tensor | None = None,
        elapsed: torch.Tensor | None = None,
        prefill_shift_state: torch.Tensor | None = None,
        prefill_wkv_state: torch.Tensor | None = None,
        prefill_elapsed: torch.Tensor | None = None,
        rwkv_decode_batch_size: int = 0,
        rwkv_decode_rows: list[int] | None = None,
        rwkv_decode_token_positions: torch.Tensor | list[int] | None = None,
        rwkv_prefill_token_ranges: list[tuple[int, int, int]] | None = None,
        rwkv_prefill_rows: list[int] | None = None,
        rwkv_prefill_groups: list[tuple[int, int, int, int, int, int]] | None = None,
        rwkv_prefill_query_start_loc: torch.Tensor | None = None,
        rwkv_prefill_slot_indices: torch.Tensor | None = None,
        rwkv_prefill_token_positions: torch.Tensor | None = None,
        rwkv_prefill_req_id: torch.Tensor | None = None,
        rwkv_prefill_max_t: int = 0,
        slot_indices: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        """Run RWKV7 from Model Runner V2 request-indexed state tensors."""
        if query_start_loc is None:
            raise RuntimeError(
                "RWKV7 requires Model Runner V2 request-indexed state inputs."
            )

        assert query_start_loc is not None
        assert shift_state is not None
        assert wkv_state is not None
        assert elapsed is not None

        is_first_pp_rank = getattr(self, "_is_pp_first_rank", lambda: True)()
        is_last_pp_rank = getattr(self, "_is_pp_last_rank", lambda: True)()
        if not (is_first_pp_rank and is_last_pp_rank):
            return self.forward_vllm_pp_stage(
                input_ids=input_ids,
                intermediate_tensors=intermediate_tensors,
                shift_state=shift_state,
                wkv_state=wkv_state,
                elapsed=elapsed,
                prefill_shift_state=prefill_shift_state,
                prefill_wkv_state=prefill_wkv_state,
                prefill_elapsed=prefill_elapsed,
                rwkv_decode_batch_size=rwkv_decode_batch_size,
                rwkv_decode_rows=rwkv_decode_rows,
                rwkv_decode_token_positions=rwkv_decode_token_positions,
                rwkv_prefill_token_ranges=rwkv_prefill_token_ranges,
                rwkv_prefill_rows=rwkv_prefill_rows,
                rwkv_prefill_groups=rwkv_prefill_groups,
                rwkv_prefill_query_start_loc=rwkv_prefill_query_start_loc,
                rwkv_prefill_slot_indices=rwkv_prefill_slot_indices,
                rwkv_prefill_token_positions=rwkv_prefill_token_positions,
                rwkv_prefill_req_id=rwkv_prefill_req_id,
                rwkv_prefill_max_t=rwkv_prefill_max_t,
                slot_indices=slot_indices,
                is_first_pp_rank=is_first_pp_rank,
                is_last_pp_rank=is_last_pp_rank,
            )

        assert input_ids is not None
        hidden_states = torch.empty(
            (input_ids.shape[0], self.hidden_size), dtype=DTYPE, device=CUDA_DEVICE
        )
        if (
            rwkv_decode_rows is not None
            or rwkv_prefill_token_ranges is not None
            or rwkv_prefill_groups is not None
        ):
            decode_rows = rwkv_decode_rows or []
            decode_positions = rwkv_decode_token_positions
            assert len(decode_rows) == self._decode_token_positions_length(
                decode_positions
            )
            if decode_rows:
                decode_batch_size = rwkv_decode_batch_size
                assert decode_batch_size > 0
                if slot_indices is not None:
                    decode_position_tensor = self._decode_token_positions_tensor(
                        decode_positions,
                        device=input_ids.device,
                    )
                    hidden_position_tensor = decode_position_tensor.to(
                        device=hidden_states.device
                    )
                    tokens = input_ids.index_select(0, decode_position_tensor).view(
                        decode_batch_size, 1
                    )
                    state = [shift_state, wkv_state, elapsed]
                    out = self.forward_tokens(
                        tokens,
                        state,
                        slot_indices=slot_indices[:decode_batch_size],
                    ).view(decode_batch_size, self.hidden_size)
                    hidden_states.index_copy_(0, hidden_position_tensor, out)
                else:
                    decode_position_list = self._decode_token_positions_list(
                        decode_positions
                    )
                    start, end = RWKV7ForCausalLM._contiguous_decode_token_range(
                        decode_batch_size, decode_rows, decode_position_list
                    )
                    tokens = input_ids[start:end].view(decode_batch_size, 1)
                    state = [
                        shift_state[:, :, :decode_batch_size, :],
                        wkv_state[:, :decode_batch_size, :, :, :],
                        elapsed[:decode_batch_size],
                    ]
                    out = self.forward_tokens(tokens, state).view(
                        decode_batch_size, self.hidden_size
                    )
                    hidden_states[start:end] = out.view(
                        decode_batch_size, self.hidden_size
                    )

            prefill_ranges = rwkv_prefill_token_ranges or []
            prefill_rows = rwkv_prefill_rows or []
            assert len(prefill_ranges) == len(prefill_rows)
            if prefill_shift_state is None:
                prefill_shift_state = shift_state
            if prefill_wkv_state is None:
                prefill_wkv_state = wkv_state
            if prefill_elapsed is None:
                prefill_elapsed = elapsed
            if (
                prefill_ranges
                and rwkv_prefill_query_start_loc is not None
                and rwkv_prefill_slot_indices is not None
                and rwkv_prefill_token_positions is not None
                and rwkv_prefill_req_id is not None
                and rwkv_prefill_max_t > 0
            ):
                input_position_tensor = rwkv_prefill_token_positions.to(
                    device=input_ids.device
                )
                hidden_position_tensor = rwkv_prefill_token_positions.to(
                    device=hidden_states.device
                )
                tokens = input_ids.index_select(0, input_position_tensor)
                state = [prefill_shift_state, prefill_wkv_state, prefill_elapsed]
                out = self.forward_varlen_hidden(
                    tokens,
                    state,
                    query_start_loc=rwkv_prefill_query_start_loc,
                    slot_indices=rwkv_prefill_slot_indices,
                    req_id=rwkv_prefill_req_id,
                    max_t=rwkv_prefill_max_t,
                )
                hidden_states.index_copy_(0, hidden_position_tensor, out)
                return hidden_states
            prefill_groups = rwkv_prefill_groups or []
            for (
                _batch_start,
                _batch_end,
                query_len,
                start,
                end,
                row_start,
            ) in prefill_groups:
                batch_size = (end - start) // query_len
                row_end = row_start + batch_size
                tokens = input_ids[start:end].view(batch_size, query_len)
                state = [
                    prefill_shift_state[:, :, row_start:row_end, :],
                    prefill_wkv_state[:, row_start:row_end, :, :, :],
                    prefill_elapsed[row_start:row_end],
                ]
                if query_len == 1:
                    out = self.forward_tokens(tokens, state)
                    hidden_states[start:end] = out.view(batch_size, self.hidden_size)
                else:
                    out = self.forward_all_hidden(tokens, state)
                    hidden_states[start:end] = out.view(
                        batch_size * query_len, self.hidden_size
                    )
            if prefill_groups:
                return hidden_states
            if prefill_ranges:
                raise RuntimeError(
                    "RWKV7 prefill requires grouped or varlen fast-path metadata."
                )
            return hidden_states

        raise RuntimeError(
            "RWKV7 requires decode, grouped-prefill, or varlen-prefill "
            "fast-path metadata."
        )

    def forward_vllm_pp_stage(
        self,
        *,
        input_ids: torch.Tensor | None,
        intermediate_tensors: IntermediateTensors | None,
        shift_state: torch.Tensor,
        wkv_state: torch.Tensor,
        elapsed: torch.Tensor,
        prefill_shift_state: torch.Tensor | None,
        prefill_wkv_state: torch.Tensor | None,
        prefill_elapsed: torch.Tensor | None,
        rwkv_decode_batch_size: int,
        rwkv_decode_rows: list[int] | None,
        rwkv_decode_token_positions: torch.Tensor | list[int] | None,
        rwkv_prefill_token_ranges: list[tuple[int, int, int]] | None,
        rwkv_prefill_rows: list[int] | None,
        rwkv_prefill_groups: list[tuple[int, int, int, int, int, int]] | None,
        rwkv_prefill_query_start_loc: torch.Tensor | None,
        rwkv_prefill_slot_indices: torch.Tensor | None,
        rwkv_prefill_token_positions: torch.Tensor | None,
        rwkv_prefill_req_id: torch.Tensor | None,
        rwkv_prefill_max_t: int,
        slot_indices: torch.Tensor | None,
        is_first_pp_rank: bool,
        is_last_pp_rank: bool,
    ) -> torch.Tensor | IntermediateTensors:
        if is_first_pp_rank:
            assert input_ids is not None
            total_tokens = input_ids.shape[0]
            incoming_hidden_states = None
            incoming_v_first = None
        else:
            assert intermediate_tensors is not None
            incoming_hidden_states = intermediate_tensors["hidden_states"].to(
                dtype=DTYPE
            )
            incoming_v_first = intermediate_tensors["v_first"].to(dtype=DTYPE)
            if incoming_v_first.shape[-1] == self.hidden_size:
                incoming_v_first = self._tp_hidden_slice(incoming_v_first, -1)
            total_tokens = incoming_hidden_states.shape[0]

        hidden_states = torch.empty(
            (total_tokens, self.hidden_size), dtype=DTYPE, device=CUDA_DEVICE
        )
        v_first_states = None
        if not is_last_pp_rank:
            v_first_states = torch.empty(
                (total_tokens, self.hidden_size),
                dtype=DTYPE,
                device=CUDA_DEVICE,
            )

        if (
            rwkv_decode_rows is not None
            or rwkv_prefill_token_ranges is not None
            or rwkv_prefill_groups is not None
        ):
            decode_rows = rwkv_decode_rows or []
            decode_positions = rwkv_decode_token_positions
            assert len(decode_rows) == self._decode_token_positions_length(
                decode_positions
            )
            if decode_rows:
                decode_batch_size = rwkv_decode_batch_size
                assert decode_batch_size > 0
                if slot_indices is not None:
                    decode_position_tensor = self._decode_token_positions_tensor(
                        decode_positions,
                        device=hidden_states.device,
                    )
                    if is_first_pp_rank:
                        assert input_ids is not None
                        input_position_tensor = decode_position_tensor.to(
                            device=input_ids.device
                        )
                        tokens = input_ids.index_select(0, input_position_tensor).view(
                            decode_batch_size, 1
                        )
                        x = self.embed(tokens)
                        group_v_first = None
                    else:
                        assert incoming_hidden_states is not None
                        assert incoming_v_first is not None
                        x = incoming_hidden_states.index_select(
                            0, decode_position_tensor
                        ).view(decode_batch_size, 1, self.hidden_size)
                        group_v_first = incoming_v_first.index_select(
                            0, decode_position_tensor
                        ).view(
                            decode_batch_size,
                            1,
                            getattr(self, "tp_hidden_size", self.hidden_size),
                        )
                    state = [shift_state, wkv_state, elapsed]
                    decode_slot_indices = slot_indices[:decode_batch_size]
                else:
                    decode_position_list = self._decode_token_positions_list(
                        decode_positions
                    )
                    start, end = RWKV7ForCausalLM._contiguous_decode_token_range(
                        decode_batch_size, decode_rows, decode_position_list
                    )
                    if is_first_pp_rank:
                        assert input_ids is not None
                        tokens = input_ids[start:end].view(decode_batch_size, 1)
                        x = self.embed(tokens)
                        group_v_first = None
                    else:
                        assert incoming_hidden_states is not None
                        assert incoming_v_first is not None
                        x = incoming_hidden_states[start:end].view(
                            decode_batch_size, 1, self.hidden_size
                        )
                        group_v_first = incoming_v_first[start:end].view(
                            decode_batch_size,
                            1,
                            getattr(self, "tp_hidden_size", self.hidden_size),
                        )
                    state = [
                        shift_state[:, :, :decode_batch_size, :],
                        wkv_state[:, :decode_batch_size, :, :, :],
                        elapsed[:decode_batch_size],
                    ]
                    decode_slot_indices = None
                    decode_position_tensor = None
                path = select_path(decode_batch_size, 1)
                forward_kwargs = {}
                if decode_slot_indices is not None:
                    forward_kwargs["slot_indices"] = decode_slot_indices
                out, out_v_first = self.forward_layer_range(
                    x,
                    state,
                    path,
                    v_first=group_v_first,
                    final=is_last_pp_rank,
                    all_logits=True,
                    last_indices=None,
                    **forward_kwargs,
                )
                out = out.view(decode_batch_size, self.hidden_size)
                if decode_position_tensor is None:
                    hidden_states[start:end] = out.view(
                        decode_batch_size, self.hidden_size
                    )
                else:
                    hidden_states.index_copy_(0, decode_position_tensor, out)
                if v_first_states is not None:
                    if out_v_first is None:
                        assert group_v_first is not None
                        out_v_first = group_v_first
                    if getattr(self, "tp_size", 1) > 1:
                        out_v_first = tensor_model_parallel_all_gather(out_v_first)
                    out_v_first = out_v_first.view(
                        decode_batch_size, self.hidden_size
                    )
                    if decode_position_tensor is None:
                        v_first_states[start:end] = out_v_first.view(
                            decode_batch_size, self.hidden_size
                        )
                    else:
                        v_first_states.index_copy_(
                            0, decode_position_tensor, out_v_first
                        )

            prefill_ranges = rwkv_prefill_token_ranges or []
            prefill_rows = rwkv_prefill_rows or []
            assert len(prefill_ranges) == len(prefill_rows)
            if prefill_shift_state is None:
                prefill_shift_state = shift_state
            if prefill_wkv_state is None:
                prefill_wkv_state = wkv_state
            if prefill_elapsed is None:
                prefill_elapsed = elapsed
            if (
                prefill_ranges
                and rwkv_prefill_query_start_loc is not None
                and rwkv_prefill_slot_indices is not None
                and rwkv_prefill_token_positions is not None
                and rwkv_prefill_req_id is not None
                and rwkv_prefill_max_t > 0
            ):
                hidden_position_tensor = rwkv_prefill_token_positions.to(
                    device=hidden_states.device
                )
                if is_first_pp_rank:
                    assert input_ids is not None
                    input_position_tensor = rwkv_prefill_token_positions.to(
                        device=input_ids.device
                    )
                    tokens = input_ids.index_select(0, input_position_tensor)
                    x = self.embed(tokens).view(tokens.numel(), self.hidden_size)
                    group_v_first = None
                else:
                    assert incoming_hidden_states is not None
                    assert incoming_v_first is not None
                    x = incoming_hidden_states.index_select(
                        0, hidden_position_tensor
                    ).view(-1, self.hidden_size)
                    group_v_first = incoming_v_first.index_select(
                        0, hidden_position_tensor
                    ).view(-1, getattr(self, "tp_hidden_size", self.hidden_size))
                state = [prefill_shift_state, prefill_wkv_state, prefill_elapsed]
                out, out_v_first = self.forward_varlen_layer_range(
                    x,
                    state,
                    query_start_loc=rwkv_prefill_query_start_loc,
                    slot_indices=rwkv_prefill_slot_indices,
                    req_id=rwkv_prefill_req_id,
                    max_t=rwkv_prefill_max_t,
                    v_first=group_v_first,
                    final=is_last_pp_rank,
                )
                hidden_states.index_copy_(0, hidden_position_tensor, out)
                if v_first_states is not None:
                    if out_v_first is None:
                        assert group_v_first is not None
                        out_v_first = group_v_first
                    if getattr(self, "tp_size", 1) > 1:
                        out_v_first = tensor_model_parallel_all_gather(out_v_first)
                    v_first_states.index_copy_(
                        0,
                        hidden_position_tensor,
                        out_v_first.view(-1, self.hidden_size),
                    )
                if not is_last_pp_rank:
                    assert v_first_states is not None
                    return IntermediateTensors(
                        {"hidden_states": hidden_states, "v_first": v_first_states}
                    )
                return hidden_states
            prefill_groups = rwkv_prefill_groups or []
            for (
                _batch_start,
                _batch_end,
                query_len,
                start,
                end,
                row_start,
            ) in prefill_groups:
                batch_size = (end - start) // query_len
                row_end = row_start + batch_size
                if is_first_pp_rank:
                    assert input_ids is not None
                    tokens = input_ids[start:end].view(batch_size, query_len)
                    x = self.embed(tokens)
                    group_v_first = None
                else:
                    assert incoming_hidden_states is not None
                    assert incoming_v_first is not None
                    x = incoming_hidden_states[start:end].view(
                        batch_size, query_len, self.hidden_size
                    )
                    group_v_first = incoming_v_first[start:end].view(
                        batch_size,
                        query_len,
                        getattr(self, "tp_hidden_size", self.hidden_size),
                    )
                state = [
                    prefill_shift_state[:, :, row_start:row_end, :],
                    prefill_wkv_state[:, row_start:row_end, :, :, :],
                    prefill_elapsed[row_start:row_end],
                ]
                path = select_path(batch_size, query_len)
                out, out_v_first = self.forward_layer_range(
                    x,
                    state,
                    path,
                    v_first=group_v_first,
                    final=is_last_pp_rank,
                    all_logits=True,
                    last_indices=None,
                )
                hidden_states[start:end] = out.view(
                    batch_size * query_len, self.hidden_size
                )
                if v_first_states is not None:
                    if out_v_first is None:
                        assert group_v_first is not None
                        out_v_first = group_v_first
                    if getattr(self, "tp_size", 1) > 1:
                        out_v_first = tensor_model_parallel_all_gather(out_v_first)
                    v_first_states[start:end] = out_v_first.view(
                        batch_size * query_len, self.hidden_size
                    )
            if prefill_groups:
                if not is_last_pp_rank:
                    assert v_first_states is not None
                    return IntermediateTensors(
                        {"hidden_states": hidden_states, "v_first": v_first_states}
                    )
                return hidden_states

            if prefill_ranges:
                raise RuntimeError(
                    "RWKV7 prefill requires grouped or varlen fast-path metadata."
                )

            if not is_last_pp_rank:
                assert v_first_states is not None
                return IntermediateTensors(
                    {"hidden_states": hidden_states, "v_first": v_first_states}
                )
            return hidden_states

        raise RuntimeError(
            "RWKV7 pipeline execution requires decode, grouped-prefill, or "
            "varlen-prefill fast-path metadata."
        )

    def forward_tokens(
        self,
        tokens: torch.Tensor,
        state: list[torch.Tensor],
        *,
        all_logits: bool = False,
        last_indices: torch.Tensor | None = None,
        slot_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        batch_size, token_count = tokens.shape
        x = self.embed(tokens)
        path = select_path(batch_size, token_count)
        if not all_logits and last_indices is None:
            return self.forward_from_x(x, state, path, slot_indices=slot_indices)
        return self.forward_from_x(
            x,
            state,
            path,
            all_logits=all_logits,
            last_indices=last_indices,
            slot_indices=slot_indices,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.linear(hidden_states, self.z["head.weight"])
        logits = self.logits_processor(None, logits)
        if logits is not None and getattr(self, "tp_size", 1) > 1:
            logits = tensor_model_parallel_all_gather(logits)
            if logits is not None:
                logits = logits[..., : self.vocab_size]
        return logits

    def compute_sampling_logits(
        self,
        hidden_states: torch.Tensor,
        logits_indices: torch.Tensor,
        input_batch: Any,
    ) -> torch.Tensor | None:
        num_reqs = getattr(input_batch, "num_reqs", None)
        num_draft_tokens = getattr(input_batch, "num_draft_tokens", None)
        num_scheduled_tokens = getattr(input_batch, "num_scheduled_tokens", None)
        if num_reqs is None or num_draft_tokens is None or num_scheduled_tokens is None:
            return None

        try:
            num_reqs = int(num_reqs)
            num_draft_tokens = int(num_draft_tokens)
        except (TypeError, ValueError):
            return None
        if num_reqs <= 0 or num_draft_tokens != 0:
            return None

        is_prefilling_np = getattr(input_batch, "is_prefilling_np", None)
        if is_prefilling_np is None:
            return None
        try:
            is_prefilling = is_prefilling_np[:num_reqs]
        except (TypeError, IndexError):
            return None
        if len(is_prefilling) != num_reqs:
            return None
        try:
            if bool(is_prefilling.any()):
                return None
        except (AttributeError, TypeError, RuntimeError):
            return None

        try:
            scheduled_tokens = num_scheduled_tokens[:num_reqs]
        except (TypeError, IndexError):
            return None
        if len(scheduled_tokens) != num_reqs:
            return None
        try:
            if bool((scheduled_tokens != 1).any()):
                return None
        except (AttributeError, TypeError, RuntimeError):
            return None

        if hidden_states.shape[0] < num_reqs:
            return None

        fast_path_metadata = getattr(
            input_batch,
            "rwkv_sampling_logits_contiguous",
            None,
        )
        if fast_path_metadata is None:
            if not isinstance(logits_indices, torch.Tensor):
                return None
            if (
                logits_indices.dim() != 1
                or logits_indices.numel() != num_reqs
                or logits_indices.dtype not in (torch.int32, torch.int64)
                or not logits_indices.is_contiguous()
            ):
                return None
            if logits_indices.is_cuda:
                return None
            if logits_indices.tolist() != list(range(num_reqs)):
                return None
        elif not bool(fast_path_metadata):
            return None

        return self.compute_logits(hidden_states[:num_reqs])

    @staticmethod
    def _decode_token_positions_length(
        decode_positions: torch.Tensor | list[int] | None,
    ) -> int:
        if decode_positions is None:
            return 0
        if isinstance(decode_positions, torch.Tensor):
            return int(decode_positions.numel())
        return len(decode_positions)

    @staticmethod
    def _decode_token_positions_tensor(
        decode_positions: torch.Tensor | list[int] | None,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if decode_positions is None:
            raise RuntimeError("RWKV7 decode token positions are missing")
        if isinstance(decode_positions, torch.Tensor):
            return decode_positions.to(device=device, dtype=torch.long)
        return torch.tensor(decode_positions, dtype=torch.long, device=device)

    @staticmethod
    def _decode_token_positions_list(
        decode_positions: torch.Tensor | list[int] | None,
    ) -> list[int]:
        if decode_positions is None:
            return []
        if isinstance(decode_positions, torch.Tensor):
            return decode_positions.tolist()
        return decode_positions

    @staticmethod
    def _contiguous_decode_token_range(
        decode_batch_size: int,
        decode_rows: list[int],
        decode_positions: list[int],
    ) -> tuple[int, int]:
        if decode_batch_size <= 0:
            raise RuntimeError("RWKV7 decode batch size must be positive")
        if len(decode_rows) != decode_batch_size:
            raise RuntimeError(
                "RWKV7 decode rows must match decode batch size "
                f"{decode_batch_size}; got {decode_rows}"
            )
        if len(decode_positions) != decode_batch_size:
            raise RuntimeError(
                "RWKV7 decode token positions must match decode batch size "
                f"{decode_batch_size}; got {decode_positions}"
            )
        for expected_row, row in enumerate(decode_rows):
            if row != expected_row:
                raise RuntimeError(
                    "RWKV7 decode rows must be contiguous prefix rows "
                    f"[0..{decode_batch_size - 1}]; got {decode_rows}"
                )
        start = decode_positions[0]
        end = start + decode_batch_size
        for offset, position in enumerate(decode_positions):
            if position != start + offset:
                raise RuntimeError(
                    "RWKV7 decode token positions must be a dense contiguous "
                    f"range; got {decode_positions}"
                )
        return start, end

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.device != self.z["emb.weight"].device:
            tokens = tokens.to(self.z["emb.weight"].device, non_blocking=True)
        if getattr(self, "tp_size", 1) == 1:
            return self.z["emb.weight"][tokens]
        vocab_start, vocab_end, vocab_per_rank = self._tp_vocab_range(
            self.vocab_size
        )
        mask = (tokens < vocab_start) | (tokens >= vocab_end)
        local = (tokens - vocab_start).clamp(min=0, max=vocab_per_rank - 1)
        out = self.z["emb.weight"][local]
        out.masked_fill_(mask.unsqueeze(-1), 0)
        return self._tp_all_reduce(out)

    def forward_from_x(
        self,
        x: torch.Tensor,
        state: list[torch.Tensor],
        path: PathConfig,
        all_logits: bool = False,
        last_indices=None,
        slot_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run RWKV7 from embedded input."""
        out, _ = self.forward_layer_range(
            x,
            state,
            path,
            v_first=None,
            final=True,
            all_logits=all_logits,
            last_indices=last_indices,
            slot_indices=slot_indices,
        )
        return out

    def forward_layer_range(
        self,
        x: torch.Tensor,
        state: list[torch.Tensor],
        path: PathConfig,
        *,
        v_first: torch.Tensor | None,
        final: bool,
        all_logits: bool,
        last_indices=None,
        slot_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        z = self.z
        B, T, _ = x.shape
        start_layer = getattr(self, "start_layer", 0)
        end_layer = getattr(self, "end_layer", self.total_num_layers)

        def advance_elapsed() -> None:
            if slot_indices is None:
                torch.ops.rwkv7_v3a_ops.advance_i32(state[2], T)
            else:
                torch.ops.rwkv7_v3a_ops.advance_i32_slots(state[2], slot_indices, T)

        if start_layer == 0 and v_first is None:
            v_first = x
        if start_layer >= end_layer:
            if final:
                x = self.ln(x, z["ln_out.weight"], z["ln_out.bias"])
                return x, v_first
            return x, v_first

        xx = self.ln(
            x,
            z[f"blocks.{start_layer}.ln1.weight"],
            z[f"blocks.{start_layer}.ln1.bias"],
        )
        pre_mix = None

        for layer in range(start_layer, end_layer):
            local_layer = layer - start_layer
            p = f"blocks.{layer}."
            layer_v_first = x if layer == 0 else v_first
            assert layer_v_first is not None
            xx, v_first = self.tmix(
                layer,
                xx,
                state[0][local_layer],
                state[1][local_layer],
                state[2],
                layer_v_first,
                p + "att.",
                path,
                pre_mix,
                slot_indices=slot_indices,
            )
            pre_mix = None
            if T == 1:
                if slot_indices is None:
                    x, mixed = torch.ops.rwkv7_v3a_ops.add_layer_norm_cmix_mix_f16(
                        x.contiguous(),
                        xx.contiguous(),
                        state[0][local_layer][1],
                        z[p + "ln2.weight"],
                        z[p + "ln2.bias"],
                        z[p + "ffn.x_k"],
                    )
                    cmix_path = path
                else:
                    (
                        x,
                        mixed,
                    ) = torch.ops.rwkv7_v3a_ops.add_layer_norm_cmix_mix_f16_slots(
                        x.contiguous(),
                        xx.contiguous(),
                        state[0][local_layer][1],
                        z[p + "ln2.weight"],
                        z[p + "ln2.bias"],
                        z[p + "ffn.x_k"],
                        slot_indices,
                    )
                    cmix_path = path
                xx = self.cmix_from_mixed(mixed, p + "ffn.", cmix_path)
            else:
                x, xx = self.add_ln(x, xx, z[p + "ln2.weight"], z[p + "ln2.bias"])
                xx = self.cmix(
                    xx,
                    state[0][local_layer],
                    p + "ffn.",
                    path,
                    slot_indices=slot_indices,
                )
            if layer + 1 < end_layer:
                p_next = f"blocks.{layer + 1}."
                if LN1_TMIX_FUSE and T == 1:
                    if slot_indices is None:
                        outs = torch.ops.rwkv7_v3a_ops.add_layer_norm_tmix_mix6_f16(
                            x.contiguous(),
                            xx.contiguous(),
                            state[0][local_layer + 1][0],
                            z[p_next + "ln1.weight"],
                            z[p_next + "ln1.bias"],
                            z[p_next + "att.x_r"],
                            z[p_next + "att.x_w"],
                            z[p_next + "att.x_k"],
                            z[p_next + "att.x_v"],
                            z[p_next + "att.x_a"],
                            z[p_next + "att.x_g"],
                        )
                    else:
                        outs = (
                            torch.ops.rwkv7_v3a_ops.add_layer_norm_tmix_mix6_f16_slots(
                                x.contiguous(),
                                xx.contiguous(),
                                state[0][local_layer + 1][0],
                                z[p_next + "ln1.weight"],
                                z[p_next + "ln1.bias"],
                                z[p_next + "att.x_r"],
                                z[p_next + "att.x_w"],
                                z[p_next + "att.x_k"],
                                z[p_next + "att.x_v"],
                                z[p_next + "att.x_a"],
                                z[p_next + "att.x_g"],
                                slot_indices,
                            )
                        )
                    x, pre_mix = outs[0], outs[1:]
                    xx = x
                else:
                    x, xx = self.add_ln(
                        x, xx, z[p_next + "ln1.weight"], z[p_next + "ln1.bias"]
                    )
            elif not final:
                x = self.add(x, xx)
                advance_elapsed()
                return x, v_first
            elif not all_logits:
                if last_indices is not None:
                    x = self.ln(self.add(x, xx), z["ln_out.weight"], z["ln_out.bias"])
                    x = x[torch.arange(B, device=x.device), last_indices].contiguous()
                else:
                    x = self.add_last_ln(x, xx, z["ln_out.weight"], z["ln_out.bias"])
                advance_elapsed()
                return x, v_first
            else:
                x = self.add(x, xx)

        x = self.ln(x, z["ln_out.weight"], z["ln_out.bias"])
        advance_elapsed()
        return x, v_first

    def ln(
        self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        return torch.ops.rwkv7_v3a_ops.layer_norm_f16(x.contiguous(), weight, bias)

    def forward_all_logits(
        self, tokens: torch.Tensor, state: list[torch.Tensor]
    ) -> torch.Tensor:
        return self.compute_logits(self.forward_tokens(tokens, state, all_logits=True))

    def forward_all_hidden(
        self, tokens: torch.Tensor, state: list[torch.Tensor]
    ) -> torch.Tensor:
        """Return hidden states for every input position."""
        return self.forward_tokens(tokens, state, all_logits=True)

    def forward_varlen_hidden(
        self,
        tokens: torch.Tensor,
        state: list[torch.Tensor],
        *,
        query_start_loc: torch.Tensor,
        slot_indices: torch.Tensor,
        req_id: torch.Tensor,
        max_t: int,
    ) -> torch.Tensor:
        tokens = tokens.reshape(-1)
        x = self.embed(tokens).view(tokens.numel(), self.hidden_size)
        out, _ = self.forward_varlen_layer_range(
            x,
            state,
            query_start_loc=query_start_loc,
            slot_indices=slot_indices,
            req_id=req_id,
            max_t=max_t,
            v_first=None,
            final=True,
        )
        return out

    def forward_varlen_layer_range(
        self,
        x: torch.Tensor,
        state: list[torch.Tensor],
        *,
        query_start_loc: torch.Tensor,
        slot_indices: torch.Tensor,
        req_id: torch.Tensor,
        max_t: int,
        v_first: torch.Tensor | None,
        final: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        z = self.z
        total_tokens = x.shape[0]
        path = PathConfig(total_tokens, CMIX_DENSE)
        start_layer = getattr(self, "start_layer", 0)
        end_layer = getattr(self, "end_layer", self.total_num_layers)

        def advance_elapsed() -> None:
            torch.ops.rwkv7_v3a_ops.advance_i32_varlen(
                state[2], query_start_loc, slot_indices
            )

        if start_layer == 0 and v_first is None:
            v_first = x
        if start_layer >= end_layer:
            if final:
                x = self.ln(x, z["ln_out.weight"], z["ln_out.bias"])
            return x, v_first

        xx = self.ln(
            x,
            z[f"blocks.{start_layer}.ln1.weight"],
            z[f"blocks.{start_layer}.ln1.bias"],
        )

        for layer in range(start_layer, end_layer):
            local_layer = layer - start_layer
            p = f"blocks.{layer}."
            layer_v_first = x if layer == 0 else v_first
            assert layer_v_first is not None
            xx, v_first = self.tmix_varlen(
                layer,
                xx,
                state[0][local_layer],
                state[1][local_layer],
                state[2],
                layer_v_first,
                p + "att.",
                path,
                query_start_loc=query_start_loc,
                slot_indices=slot_indices,
                req_id=req_id,
                max_t=max_t,
            )
            x, xx = self.add_ln(x, xx, z[p + "ln2.weight"], z[p + "ln2.bias"])
            xx = self.cmix_varlen(
                xx,
                state[0][local_layer],
                p + "ffn.",
                path,
                query_start_loc=query_start_loc,
                slot_indices=slot_indices,
                req_id=req_id,
            )
            if layer + 1 < end_layer:
                p_next = f"blocks.{layer + 1}."
                x, xx = self.add_ln(
                    x, xx, z[p_next + "ln1.weight"], z[p_next + "ln1.bias"]
                )
            elif not final:
                x = self.add(x, xx)
                advance_elapsed()
                return x, v_first
            else:
                x = self.ln(self.add(x, xx), z["ln_out.weight"], z["ln_out.bias"])
                advance_elapsed()
                return x, v_first

        raise AssertionError("unreachable RWKV7 varlen layer path")

    def forward_last_at(
        self,
        tokens: torch.Tensor,
        state: list[torch.Tensor],
        last_indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.compute_logits(
            self.forward_tokens(tokens, state, last_indices=last_indices)
        )

    def _project_tmix(
        self,
        layer: int,
        xr: torch.Tensor,
        xw: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        xa: torch.Tensor,
        xg: torch.Tensor,
        v_first: torch.Tensor,
        p: str,
        path: PathConfig,
        *,
        batch_size: int,
        time_steps: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Share the layout-independent time-mix projection pipeline."""
        z = self.z
        ops = torch.ops.rwkv7_fast_ops_fp16
        r = self.linear(xr, z[p + "receptance.weight"])
        k = self.linear(xk, z[p + "key.weight"])
        v = self.linear(xv, z[p + "value.weight"])
        local_c = r.shape[-1]
        local_h = local_c // self.head_size
        use_lowrank_in = can_use_lowrank_fused(self.hidden_size, path.rows)
        use_lowrank_out = can_use_lowrank_out_fused(self.hidden_size, path.rows)

        v1 = None
        if use_lowrank_in and use_lowrank_out and layer != 0:
            w1, a1, g1, v1 = torch.ops.rwkv7_v3a_ops.linear_wagv_rank_in_f16(
                xw.contiguous(),
                xa.contiguous(),
                xg.contiguous(),
                xv.contiguous(),
                z[p + "w1.t"],
                z[p + "a1.t"],
                z[p + "g1.t"],
                z[p + "v1.t"],
            )
        elif use_lowrank_in:
            w1, a1, g1 = torch.ops.rwkv7_v3a_ops.linear_wag_rank_in_f16(
                xw.contiguous(),
                xa.contiguous(),
                xg.contiguous(),
                z[p + "w1.t"],
                z[p + "a1.t"],
                z[p + "g1.t"],
            )
        else:
            w1 = self.linear_rank_in(
                xw, z.get(p + "w1"), z.get(p + "w1.t"), path.rows
            )
            a1 = self.linear_rank_in(
                xa, z.get(p + "a1"), z.get(p + "a1.t"), path.rows
            )
            g1 = self.linear_rank_in(
                xg, z.get(p + "g1"), z.get(p + "g1.t"), path.rows
            )

        v_done = False
        if use_lowrank_out and layer != 0 and v1 is not None:
            w, a, g, v = torch.ops.rwkv7_v3a_ops.linear_wagv_rank_out_f16(
                w1.contiguous(),
                a1.contiguous(),
                g1.contiguous(),
                v1.contiguous(),
                z[p + "w2.t"],
                z[p + "a2.t"],
                z[p + "g2.t"],
                z[p + "v2.t"],
                v.contiguous(),
                v_first.contiguous(),
                z[p + "v0"],
            )
            v_done = True
        elif use_lowrank_out:
            w, a, g = torch.ops.rwkv7_v3a_ops.linear_wag_rank_out_f16(
                w1.contiguous(),
                a1.contiguous(),
                g1.contiguous(),
                z[p + "w2.t"],
                z[p + "a2.t"],
                z[p + "g2.t"],
            )
        else:
            w = self.linear_rank_out_act(
                w1, z.get(p + "w2"), z.get(p + "w2.t"), path.rows, 1
            )
            a = self.linear_rank_out(
                a1, z.get(p + "a2"), z.get(p + "a2.t"), path.rows
            )
            g = self.linear_rank_out_act(
                g1, z.get(p + "g2"), z.get(p + "g2.t"), path.rows, 2
            )

        value_shape = (batch_size, time_steps, local_c)
        k3, neg_kk3, kka3 = ops.tmix_kk_a_gate(
            batch_size,
            time_steps,
            local_c,
            local_h,
            k.reshape(value_shape).contiguous(),
            z[p + "k_k"],
            z[p + "a0"],
            a.reshape(value_shape).contiguous(),
            z[p + "k_a"],
        )
        if k.dim() == 2:
            k = k3.reshape(-1, local_c)
            neg_kk = neg_kk3.reshape(-1, local_c)
            kka = kka3.reshape(-1, local_c)
        else:
            k, neg_kk, kka = k3, neg_kk3, kka3

        if layer == 0:
            v_first = v
        elif not v_done:
            if use_lowrank_out:
                if v1 is None:
                    v1 = self.linear_rank_in(
                        xv, z.get(p + "v1"), z.get(p + "v1.t"), path.rows
                    )
                v = torch.ops.rwkv7_v3a_ops.linear_t_vres_f16(
                    v1.contiguous(),
                    z[p + "v2.t"],
                    v.contiguous(),
                    v_first.contiguous(),
                    z[p + "v0"],
                )
            else:
                v12 = self.linear_rank_out(
                    self.linear_rank_in(
                        xv, z.get(p + "v1"), z.get(p + "v1.t"), path.rows
                    ),
                    z.get(p + "v2"),
                    z.get(p + "v2.t"),
                    path.rows,
                )
                v = ops.tmix_vres_gate(
                    batch_size,
                    time_steps,
                    local_c,
                    v.reshape(value_shape).contiguous(),
                    v_first.reshape(value_shape).contiguous(),
                    z[p + "v0"],
                    v12.reshape(value_shape).contiguous(),
                )
                if v.dim() == 3 and xv.dim() == 2:
                    v = v.reshape(-1, local_c)

        return r, w, k, v, neg_kk, kka, g, v_first

    def tmix_varlen(
        self,
        layer: int,
        x: torch.Tensor,
        shift_state: torch.Tensor,
        wkv_state: torch.Tensor,
        elapsed_t: torch.Tensor,
        v_first: torch.Tensor,
        p: str,
        path: PathConfig,
        *,
        query_start_loc: torch.Tensor,
        slot_indices: torch.Tensor,
        req_id: torch.Tensor,
        max_t: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.z
        ops = torch.ops.rwkv7_fast_ops_fp16
        B = int(slot_indices.numel())
        total_tokens = int(x.shape[0])
        xr, xw, xk, xv, xa, xg = ops.tmix_mix6_varlen(
            B,
            total_tokens,
            self.hidden_size,
            x.contiguous(),
            shift_state[0],
            slot_indices,
            z[p + "x_r"],
            z[p + "x_w"],
            z[p + "x_k"],
            z[p + "x_v"],
            z[p + "x_a"],
            z[p + "x_g"],
            query_start_loc,
            req_id,
        )

        r, w, k, v, neg_kk, kka, g, v_first = self._project_tmix(
            layer,
            xr,
            xw,
            xk,
            xv,
            xa,
            xg,
            v_first,
            p,
            path,
            batch_size=total_tokens,
            time_steps=1,
        )
        local_c = r.shape[-1]
        local_h = local_c // self.head_size

        y = torch.empty_like(r)
        if self.wkv_mode == "fp32io16":
            w_raw = ops.add_vec(local_c, w.contiguous(), z[p + "w0"])
            torch.ops.rwkv7_wkv_fp32_v2.forward_varlen(
                B,
                total_tokens,
                max_t,
                local_c,
                local_h,
                query_start_loc,
                slot_indices,
                wkv_state,
                r.contiguous(),
                w_raw.contiguous(),
                k.contiguous(),
                v.contiguous(),
                neg_kk.contiguous(),
                kka.contiguous(),
                y,
            )
        elif max_t <= 16:
            torch.ops.rwkv7_wkv_fp16_v2.wkv_seq_w0_varlen(
                B,
                total_tokens,
                max_t,
                local_c,
                local_h,
                query_start_loc,
                slot_indices,
                wkv_state,
                r.contiguous(),
                w.contiguous(),
                z[p + "w0"],
                k.contiguous(),
                v.contiguous(),
                neg_kk.contiguous(),
                kka.contiguous(),
                y,
                elapsed_t,
            )
        else:
            w_raw = ops.add_vec(local_c, w.contiguous(), z[p + "w0"])
            torch.ops.rwkv7_wkv_fp16_v2.wkv_seq_varlen(
                B,
                total_tokens,
                max_t,
                local_c,
                local_h,
                query_start_loc,
                slot_indices,
                wkv_state,
                r.contiguous(),
                w_raw.contiguous(),
                k.contiguous(),
                v.contiguous(),
                neg_kk.contiguous(),
                kka.contiguous(),
                y,
                elapsed_t,
            )
        y = ops.tmix_lnx_rkvres_xg(
            total_tokens,
            1,
            local_c,
            local_h,
            y.view(total_tokens, 1, local_c).contiguous(),
            r.view(total_tokens, 1, local_c).contiguous(),
            k.view(total_tokens, 1, local_c).contiguous(),
            v.view(total_tokens, 1, local_c).contiguous(),
            z[p + "r_k"],
            z[p + "ln_x.weight"],
            z[p + "ln_x.bias"],
            g.view(total_tokens, 1, local_c).contiguous(),
        ).view(total_tokens, local_c)
        out = self.linear(y, z[p + "output.weight"])
        return self._tp_all_reduce(out), v_first

    def cmix_varlen(
        self,
        x: torch.Tensor,
        shift_state: torch.Tensor,
        p: str,
        path: PathConfig,
        *,
        query_start_loc: torch.Tensor,
        slot_indices: torch.Tensor,
        req_id: torch.Tensor,
    ) -> torch.Tensor:
        ops = torch.ops.rwkv7_fast_ops_fp16
        total_tokens = int(x.shape[0])
        B = int(slot_indices.numel())
        mixed = ops.cmix_mix_varlen(
            B,
            total_tokens,
            self.hidden_size,
            x.contiguous(),
            shift_state[1],
            slot_indices,
            self.z[p + "x_k"],
            query_start_loc,
            req_id,
        )
        dense_path = PathConfig(path.rows, CMIX_DENSE)
        return self.cmix_from_mixed(
            mixed.view(total_tokens, 1, self.hidden_size), p, dense_path
        ).view(
            total_tokens, self.hidden_size
        )

    def tmix(
        self,
        layer: int,
        x: torch.Tensor,
        shift_state: torch.Tensor,
        wkv_state: torch.Tensor,
        elapsed_t: torch.Tensor,
        v_first: torch.Tensor,
        p: str,
        path: PathConfig,
        pre_mix=None,
        slot_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.z
        ops = torch.ops.rwkv7_fast_ops_fp16
        B, T, _ = x.shape
        if pre_mix is not None:
            xr, xw, xk, xv, xa, xg = pre_mix
        elif slot_indices is not None:
            xr, xw, xk, xv, xa, xg = ops.tmix_mix6_slot(
                B,
                T,
                self.hidden_size,
                x.contiguous(),
                shift_state[0],
                slot_indices,
                z[p + "x_r"],
                z[p + "x_w"],
                z[p + "x_k"],
                z[p + "x_v"],
                z[p + "x_a"],
                z[p + "x_g"],
            )
        else:
            xr, xw, xk, xv, xa, xg = ops.tmix_mix6(
                B,
                T,
                self.hidden_size,
                x.contiguous(),
                shift_state[0],
                z[p + "x_r"],
                z[p + "x_w"],
                z[p + "x_k"],
                z[p + "x_v"],
                z[p + "x_a"],
                z[p + "x_g"],
            )
        r, w, k, v, neg_kk, kka, g, v_first = self._project_tmix(
            layer,
            xr,
            xw,
            xk,
            xv,
            xa,
            xg,
            v_first,
            p,
            path,
            batch_size=B,
            time_steps=T,
        )
        local_c = r.shape[-1]
        local_h = local_c // self.head_size

        y = torch.empty_like(r)
        if self.wkv_mode == "fp32io16":
            w_raw = ops.add_vec(local_c, w.contiguous(), z[p + "w0"])
            if slot_indices is None:
                torch.ops.rwkv7_wkv_fp32_v2.forward(
                    B,
                    T,
                    local_c,
                    local_h,
                    wkv_state,
                    r.contiguous(),
                    w_raw.contiguous(),
                    k.contiguous(),
                    v.contiguous(),
                    neg_kk.contiguous(),
                    kka.contiguous(),
                    y,
                )
            else:
                torch.ops.rwkv7_wkv_fp32_v2.forward_slot(
                    B,
                    T,
                    local_c,
                    local_h,
                    wkv_state,
                    r.contiguous(),
                    w_raw.contiguous(),
                    k.contiguous(),
                    v.contiguous(),
                    neg_kk.contiguous(),
                    kka.contiguous(),
                    y,
                    slot_indices,
                )
        elif T <= 16:
            if slot_indices is None:
                torch.ops.rwkv7_wkv_fp16_v2.wkv_seq_w0(
                    B,
                    T,
                    local_c,
                    local_h,
                    wkv_state,
                    r.contiguous(),
                    w.contiguous(),
                    z[p + "w0"],
                    k.contiguous(),
                    v.contiguous(),
                    neg_kk.contiguous(),
                    kka.contiguous(),
                    y,
                    elapsed_t,
                )
            else:
                torch.ops.rwkv7_wkv_fp16_v2.wkv_seq_w0_slot(
                    B,
                    T,
                    local_c,
                    local_h,
                    wkv_state,
                    r.contiguous(),
                    w.contiguous(),
                    z[p + "w0"],
                    k.contiguous(),
                    v.contiguous(),
                    neg_kk.contiguous(),
                    kka.contiguous(),
                    y,
                    slot_indices,
                    elapsed_t,
                )
        else:
            w_raw = ops.add_vec(local_c, w.contiguous(), z[p + "w0"])
            if slot_indices is None:
                torch.ops.rwkv7_wkv_fp16_v2.wkv_seq(
                    B,
                    T,
                    local_c,
                    local_h,
                    wkv_state,
                    r.contiguous(),
                    w_raw.contiguous(),
                    k.contiguous(),
                    v.contiguous(),
                    neg_kk.contiguous(),
                    kka.contiguous(),
                    y,
                    elapsed_t,
                )
            else:
                torch.ops.rwkv7_wkv_fp16_v2.wkv_seq_slot(
                    B,
                    T,
                    local_c,
                    local_h,
                    wkv_state,
                    r.contiguous(),
                    w_raw.contiguous(),
                    k.contiguous(),
                    v.contiguous(),
                    neg_kk.contiguous(),
                    kka.contiguous(),
                    y,
                    slot_indices,
                    elapsed_t,
                )
        y = ops.tmix_lnx_rkvres_xg(
            B,
            T,
            local_c,
            local_h,
            y.contiguous(),
            r.contiguous(),
            k.contiguous(),
            v.contiguous(),
            z[p + "r_k"],
            z[p + "ln_x.weight"],
            z[p + "ln_x.bias"],
            g.contiguous(),
        )
        out = self.linear(y, z[p + "output.weight"])
        return self._tp_all_reduce(out), v_first

    def cmix(
        self,
        x: torch.Tensor,
        shift_state: torch.Tensor,
        p: str,
        path: PathConfig,
        *,
        slot_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = self.z
        ops = torch.ops.rwkv7_fast_ops_fp16
        B, T, _ = x.shape

        if slot_indices is not None:
            mixed = ops.cmix_mix_slot(
                B,
                T,
                self.hidden_size,
                x.contiguous(),
                shift_state[1],
                slot_indices,
                z[p + "x_k"],
            )
            return self.cmix_from_mixed(mixed, p, path)

        mixed = ops.cmix_mix(
            B, T, self.hidden_size, x.contiguous(), shift_state[1], z[p + "x_k"]
        )
        return self.cmix_from_mixed(mixed, p, path)

    def cmix_from_mixed(
        self, mixed: torch.Tensor, p: str, path: PathConfig
    ) -> torch.Tensor:
        z = self.z
        ops = torch.ops.rwkv7_fast_ops_fp16
        B, T, _ = mixed.shape
        hid = self.linear(mixed, z[p + "key.weight"])
        if path.cmix_mode == CMIX_B1T1_NOFC:
            return self._tp_all_reduce(
                ops.cmix_sparse_down_relu_one(
                    self.hidden_size,
                    z[p + "value.weight"].size(0),
                    hid.view(-1).contiguous(),
                    z[p + "value.weight"],
                )
            )
        if path.cmix_mode == CMIX_ROWS2_NOFC:
            F = z[p + "value.weight"].size(0)
            if (
                path.rows >= CMIX_NOFC_T512_MIN_ROWS
                and self.hidden_size % 512 == 0
                and F % 512 == 0
            ):
                return self._tp_all_reduce(
                    ops.cmix_sparse_down_relu_rows_t512(
                        B,
                        T,
                        self.hidden_size,
                        F,
                        hid.contiguous(),
                        z[p + "value.weight"],
                    )
                )
            return self._tp_all_reduce(
                ops.cmix_sparse_down_relu_rows(
                    B,
                    T,
                    self.hidden_size,
                    F,
                    hid.contiguous(),
                    z[p + "value.weight"],
                )
            )

        k = ops.relu_square(hid.contiguous())
        return self._tp_all_reduce(self.linear(k, z[p + "value.weight"]))

    def linear(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if x.numel() == x.size(-1) and weight.size(1) % 64 == 0:
            return torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk(x.contiguous(), weight)
        return torch.ops.rwkv7_v3a_ops.linear_f16(
            x.contiguous(), weight, self.allow_fp16_accumulation
        )

    def linear_rank_in(
        self, x: torch.Tensor, weight: torch.Tensor, weight_t: torch.Tensor, rows: int
    ) -> torch.Tensor:
        if rows <= LOWRANK_IN_ROWS_T:
            return torch.ops.rwkv7_v3a_ops.linear_t_f16(x.contiguous(), weight_t)
        return self.linear(x, weight)

    def linear_rank_out(
        self, x: torch.Tensor, weight: torch.Tensor, weight_t: torch.Tensor, rows: int
    ) -> torch.Tensor:
        if self.hidden_size >= LOWRANK_FUSED_MIN_C and rows <= LOWRANK_OUT_ROWS_T:
            return torch.ops.rwkv7_v3a_ops.linear_t_f16(x.contiguous(), weight_t)
        return self.linear(x, weight)

    def linear_rank_out_act(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_t: torch.Tensor,
        rows: int,
        act: int,
    ) -> torch.Tensor:
        if self.hidden_size >= LOWRANK_FUSED_MIN_C and rows <= LOWRANK_OUT_ROWS_T:
            return torch.ops.rwkv7_v3a_ops.linear_t_act_f16(
                x.contiguous(), weight_t, act
            )
        ops = torch.ops.rwkv7_fast_ops_fp16
        x = (
            ops.act_tanh(x.contiguous())
            if act == 1
            else ops.act_sigmoid(x.contiguous())
        )
        return self.linear(x, weight)

    def add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.ops.rwkv7_v3a_ops.add_f16(x.contiguous(), y.contiguous())

    def add_ln(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outs = torch.ops.rwkv7_v3a_ops.add_layer_norm_f16(
            x.contiguous(), residual.contiguous(), weight, bias
        )
        return outs[0], outs[1]

    def add_last_ln(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.rwkv7_v3a_ops.add_last_layer_norm_f16(
            x.contiguous(), residual.contiguous(), weight, bias
        )
