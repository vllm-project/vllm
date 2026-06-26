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
L, C, H, N, V = 0, 0, 0, HEAD_SIZE, 0
WKV_MODE = envs.VLLM_RWKV7_WKV_MODE
EMB_DEVICE = envs.VLLM_RWKV7_EMB_DEVICE
RKV_MODE = envs.VLLM_RWKV7_RKV_MODE
CMIX_SPARSE = envs.VLLM_RWKV7_CMIX_SPARSE
LOWRANK_WEIGHT = envs.VLLM_RWKV7_LOW_RANK_WEIGHT
ORIG_LINEAR_GROUPS = {"att_c2c", "ffn_key", "head"}
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
CMIX_B1T1_SPARSE = "b1t1_sparse"
CMIX_ROWS2_SPARSE = "rows2_sparse"
CMIX_B1T1_NOFC = "b1t1_nofc"
CMIX_ROWS2_NOFC = "rows2_nofc"
CMIX_DENSE = "dense"


def first_device() -> torch.device:
    return torch.device("cuda")


@dataclass(frozen=True)
class PathConfig:
    rows: int
    use_batched_rkv: bool
    cmix_mode: str


def select_path(B: int, T: int) -> PathConfig:
    """All B/T dependent fast-path choices live here."""
    rows = B * T
    if CMIX_SPARSE == "off":
        cmix_mode = CMIX_DENSE
    elif CMIX_SPARSE == "no-fc":
        use_nofc = rows <= CMIX_NOFC_MAX_ROWS or (
            rows == 20 and CMIX_NOFC_ROW20_MAX_T >= T
        )
        cmix_mode = (
            CMIX_B1T1_NOFC
            if rows == 1
            else (CMIX_ROWS2_NOFC if use_nofc else CMIX_DENSE)
        )
    elif rows == 1:
        cmix_mode = CMIX_B1T1_SPARSE
    elif rows == 2:
        cmix_mode = CMIX_ROWS2_NOFC
    else:
        cmix_mode = CMIX_DENSE
    if RKV_MODE == "auto":
        use_batched_rkv = (rows == 1) or (4 <= rows <= 64)
    elif RKV_MODE in ("on", "batched"):
        use_batched_rkv = True
    else:
        use_batched_rkv = False
    if use_orig_linear("att_c2c"):
        use_batched_rkv = False
    return PathConfig(rows=rows, use_batched_rkv=use_batched_rkv, cmix_mode=cmix_mode)


def use_orig_linear(group: str) -> bool:
    return group in ORIG_LINEAR_GROUPS


def is_lowrank_weight(key: str) -> bool:
    return key.endswith(LOWRANK_SUFFIXES)


def can_use_lowrank_fused(rows: int) -> bool:
    return C >= LOWRANK_FUSED_MIN_C and rows <= LOWRANK_IN_ROWS_T


def can_use_lowrank_out_fused(rows: int) -> bool:
    return C >= LOWRANK_FUSED_MIN_C and rows <= LOWRANK_OUT_ROWS_T


def is_att_c2c_weight(key: str) -> bool:
    return ".att." in key and key.endswith(
        ("receptance.weight", "key.weight", "value.weight", "output.weight")
    )


def is_orig_linear_weight(key: str) -> bool:
    return (
        (use_orig_linear("att_c2c") and is_att_c2c_weight(key))
        or (use_orig_linear("ffn_key") and ".ffn.key.weight" in key)
        or (use_orig_linear("head") and key == "head.weight")
    )


class RWKV7ForCausalLM(nn.Module):
    is_attention_free = True
    supports_pp = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        """Create the RWKV7 inference module; weights are loaded by vLLM."""
        global L, C, H, N, V
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.prefix = prefix
        self._validate_torch_compile_unsupported()

        C = int(getattr(self.config, "hidden_size", 0))
        V = int(getattr(self.config, "vocab_size", 0))
        N = int(getattr(self.config, "head_size", HEAD_SIZE))
        if C and N:
            H = C // N
        else:
            H = int(getattr(self.config, "num_attention_heads", 0))
            N = C // H if C and H else HEAD_SIZE
        L = int(
            getattr(
                self.config,
                "num_hidden_layers",
                getattr(self.config, "n_layer", 0),
            )
        )

        self.z: dict[str, torch.Tensor] = {}
        self.raw_weight_names: set[str] | None = None
        self.total_num_layers = L
        self.start_layer, self.end_layer = self._get_layer_range()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        if H % self.tp_size != 0:
            raise ValueError(
                f"RWKV7 requires num_attention_heads ({H}) to be divisible "
                f"by tensor_parallel_size ({self.tp_size})."
            )
        self.tp_num_heads = H // self.tp_size
        self.tp_hidden_size = self.tp_num_heads * N
        self.vocab_size = V
        self.vocab_size_padded = self._get_padded_vocab_size(V)
        self.wkv_mode = WKV_MODE
        self.emb_cpu = EMB_DEVICE == "cpu"
        self.emb_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self.logits_processor = LogitsProcessor(V, logits_as_input=True)
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
        hidden_size = int(getattr(self.config, "hidden_size", C))
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
        vocab_size = V if vocab_size is None else vocab_size
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
        end_layer = getattr(self, "end_layer", getattr(self, "total_num_layers", L))
        total_layers = getattr(self, "total_num_layers", L)
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
                pending[name] = weight.detach().cpu().clone()
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
            old_emb_cpu = self.emb_cpu
            old_emb_cache = self.emb_cache
            try:
                self._preprocess_weights(z)
                self._commit_preprocessed_weights(
                    z,
                    reuse_existing_tensors=True,
                    existing_z=old_z,
                )
            except Exception:
                self.z = old_z
                self.emb_cpu = old_emb_cpu
                self.emb_cache = old_emb_cache
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
        self.emb_cpu = EMB_DEVICE == "cpu"
        self.emb_cache = {}
        torch.accelerator.synchronize()
        logger.info("RWKV7 weights are ready L=%d C=%d H=%d N=%d V=%d", L, C, H, N, V)

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
        global L, C, H, N, V
        self._validate_raw_weight_shapes(z)
        H, N = z["blocks.0.att.r_k"].shape
        C, V = H * N, z["emb.weight"].shape[0]
        assert N == HEAD_SIZE
        if H % getattr(self, "tp_size", 1) != 0:
            raise ValueError(
                f"RWKV7 requires num_attention_heads ({H}) to be divisible "
                f"by tensor_parallel_size ({getattr(self, 'tp_size', 1)})."
            )
        max_layer = max(int(k.split(".")[1]) for k in z if k.startswith("blocks."))
        L = max_layer + 1
        self.total_num_layers = L
        self.start_layer, self.end_layer = self._get_layer_range()
        self.tp_num_heads = H // getattr(self, "tp_size", 1)
        self.tp_hidden_size = self.tp_num_heads * N
        self.vocab_size = V
        self.vocab_size_padded = self._get_padded_vocab_size(V)
        logger.info("Detected RWKV7 model C=%d H=%d N=%d V=%d", C, H, N, V)
        logger.info(
            "RWKV7 cmix no-fc path: rows<=%d row20_t<=%d",
            CMIX_NOFC_MAX_ROWS,
            CMIX_NOFC_ROW20_MAX_T,
        )

        emb_src = z["emb.weight"].squeeze()
        ln0_w_src = z["blocks.0.ln0.weight"].squeeze()
        ln0_b_src = z["blocks.0.ln0.bias"].squeeze()
        emb_cpu = emb_src if EMB_DEVICE == "cpu" else None
        logger.info("Preprocessing RWKV7 weights with emb=%s", EMB_DEVICE)
        for key in list(z.keys()):
            if not self._is_weight_needed_on_rank(key):
                del z[key]
                continue
            if key == "emb.weight" and emb_cpu is not None:
                continue
            value = z[key].squeeze()
            value = self._shard_weight_for_tp(key, value)
            dev = first_device()
            is_lowrank = is_lowrank_weight(key)
            if ".ffn.key.weight" in key and CMIX_SPARSE == "auto":
                z[key + ".fc"] = value.to(device=dev, dtype=DTYPE).contiguous()
            if not is_lowrank and (
                ("key.weight" in key and not is_orig_linear_weight(key))
                or ("value.weight" in key and not is_orig_linear_weight(key))
                or ("receptance.weight" in key and not is_orig_linear_weight(key))
                or ("output.weight" in key and not is_orig_linear_weight(key))
                or ("head.weight" in key and not is_orig_linear_weight(key))
            ):
                value = value.t()
            value = value.to(device=dev, dtype=DTYPE).contiguous()
            if key.endswith("att.r_k"):
                value = value.flatten().contiguous()
            if is_lowrank:
                if LOWRANK_WEIGHT in ("orig", "both"):
                    z[key] = value
                else:
                    del z[key]
                if LOWRANK_WEIGHT in ("transpose", "both"):
                    z[key + ".t"] = value.t().contiguous()
            else:
                z[key] = value
        if self._is_weight_needed_on_rank("emb.weight"):
            emb_dev = first_device()
            ln0_w_bf16 = ln0_w_src.to(device=emb_dev).contiguous()
            ln0_b_bf16 = ln0_b_src.to(device=emb_dev).contiguous()
            vocab_start, vocab_end, vocab_per_rank = self._tp_vocab_range(V)
            if emb_cpu is None:
                emb = torch.zeros((vocab_per_rank, C), dtype=DTYPE, device=emb_dev)
                if vocab_end > vocab_start:
                    local = torch.ops.rwkv7_v3a_ops.emb_ln0_bf16_to_f16(
                        emb_src[vocab_start:vocab_end].to(device=emb_dev).contiguous(),
                        ln0_w_bf16,
                        ln0_b_bf16,
                    )
                    emb[: vocab_end - vocab_start].copy_(local)
                z["emb.weight"] = emb
            else:
                emb = torch.zeros((vocab_per_rank, C), dtype=DTYPE, pin_memory=True)
                for start in range(vocab_start, vocab_end, 4096):
                    end = min(start + 4096, vocab_end)
                    chunk = emb_cpu[start:end].to(device=emb_dev).contiguous()
                    chunk = torch.ops.rwkv7_v3a_ops.emb_ln0_bf16_to_f16(
                        chunk, ln0_w_bf16, ln0_b_bf16
                    )
                    local_start = start - vocab_start
                    emb[local_start : local_start + end - start].copy_(chunk)
                z["emb.weight"] = emb
        if RKV_MODE != "off" and not use_orig_linear("att_c2c"):
            for layer in range(self.start_layer, self.end_layer):
                p = f"blocks.{layer}.att."
                z[p + "rkv.weight"] = torch.stack(
                    (
                        z[p + "receptance.weight"],
                        z[p + "key.weight"],
                        z[p + "value.weight"],
                    )
                ).contiguous()

    def zero_state(self, B: int) -> list[torch.Tensor]:
        """Create RWKV recurrent state tensors for a batch."""
        local_heads = getattr(self, "tp_num_heads", H)
        return [
            torch.zeros((L, 2, B, C), dtype=DTYPE, device="cuda"),
            torch.zeros(
                (L, B, local_heads, N, N),
                dtype=torch.float32 if WKV_MODE == "fp32io16" else DTYPE,
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
        prefill_idx_mapping: torch.Tensor | None = None,
        prefill_shift_state: torch.Tensor | None = None,
        prefill_wkv_state: torch.Tensor | None = None,
        prefill_elapsed: torch.Tensor | None = None,
        rwkv_decode_batch_size: int = 0,
        rwkv_decode_rows: list[int] | None = None,
        rwkv_decode_token_positions: list[int] | None = None,
        rwkv_prefill_token_ranges: list[tuple[int, int, int]] | None = None,
        rwkv_prefill_rows: list[int] | None = None,
        rwkv_prefill_groups: list[tuple[int, int, int, int, int, int]] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        """Run RWKV7 and return vLLM-compatible hidden states.

        The legacy albatross call style ``forward(tokens, state)`` is kept for
        local parity checks. vLLM passes flattened ``input_ids`` plus
        ``query_start_loc`` and request-indexed state tensors.
        """
        if isinstance(positions, list):
            return self.forward_tokens(input_ids, positions)
        if query_start_loc is None:
            assert input_ids is not None
            tokens = input_ids.view(1, -1)
            state = self.zero_state(1)
            if tokens.shape[1] == 1:
                return self.forward_tokens(tokens, state).view(1, C)
            return self.forward_all_hidden(tokens, state).view(-1, C)

        assert query_start_loc is not None
        assert idx_mapping is not None
        assert shift_state is not None
        assert wkv_state is not None
        assert elapsed is not None

        is_first_pp_rank = getattr(self, "_is_pp_first_rank", lambda: True)()
        is_last_pp_rank = getattr(self, "_is_pp_last_rank", lambda: True)()
        if not (is_first_pp_rank and is_last_pp_rank):
            return self.forward_vllm_pp_stage(
                input_ids=input_ids,
                intermediate_tensors=intermediate_tensors,
                query_start_loc=query_start_loc,
                idx_mapping=idx_mapping,
                shift_state=shift_state,
                wkv_state=wkv_state,
                elapsed=elapsed,
                prefill_idx_mapping=prefill_idx_mapping,
                prefill_shift_state=prefill_shift_state,
                prefill_wkv_state=prefill_wkv_state,
                prefill_elapsed=prefill_elapsed,
                rwkv_decode_batch_size=rwkv_decode_batch_size,
                rwkv_decode_rows=rwkv_decode_rows,
                rwkv_decode_token_positions=rwkv_decode_token_positions,
                rwkv_prefill_token_ranges=rwkv_prefill_token_ranges,
                rwkv_prefill_rows=rwkv_prefill_rows,
                rwkv_prefill_groups=rwkv_prefill_groups,
                is_first_pp_rank=is_first_pp_rank,
                is_last_pp_rank=is_last_pp_rank,
            )

        assert input_ids is not None
        hidden_states = torch.empty(
            (input_ids.shape[0], C), dtype=DTYPE, device=first_device()
        )
        if (
            rwkv_decode_rows is not None
            or rwkv_prefill_token_ranges is not None
            or rwkv_prefill_groups is not None
        ):
            decode_rows = rwkv_decode_rows or []
            decode_positions = rwkv_decode_token_positions or []
            assert len(decode_rows) == len(decode_positions)
            if decode_rows:
                decode_batch_size = rwkv_decode_batch_size
                assert decode_batch_size > 0
                start, end = RWKV7ForCausalLM._compact_decode_token_range(
                    decode_batch_size, decode_rows, decode_positions
                )
                tokens = input_ids[start:end].view(decode_batch_size, 1)
                state = [
                    shift_state[:, :, :decode_batch_size, :],
                    wkv_state[:, :decode_batch_size, :, :, :],
                    elapsed[:decode_batch_size],
                ]
                out = self.forward_tokens(tokens, state).view(decode_batch_size, C)
                hidden_states[start:end] = out.view(decode_batch_size, C)

            prefill_ranges = rwkv_prefill_token_ranges or []
            prefill_rows = rwkv_prefill_rows or []
            assert len(prefill_ranges) == len(prefill_rows)
            if prefill_shift_state is None:
                prefill_shift_state = shift_state
            if prefill_wkv_state is None:
                prefill_wkv_state = wkv_state
            if prefill_elapsed is None:
                prefill_elapsed = elapsed
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
                    hidden_states[start:end] = out.view(batch_size, C)
                else:
                    out = self.forward_all_hidden(tokens, state)
                    hidden_states[start:end] = out.view(batch_size * query_len, C)
            if prefill_groups:
                return hidden_states
            for (_batch_idx, start, end), row in zip(prefill_ranges, prefill_rows):
                query_len = end - start
                tokens = input_ids[start:end].view(1, query_len)
                state = [
                    prefill_shift_state[:, :, row : row + 1, :],
                    prefill_wkv_state[:, row : row + 1, :, :, :],
                    prefill_elapsed[row : row + 1],
                ]
                if query_len == 1:
                    out = self.forward_tokens(tokens, state)
                    hidden_states[start:end] = out.view(1, C)
                else:
                    out = self.forward_all_hidden(tokens, state)
                    hidden_states[start:end] = out.view(query_len, C)
            return hidden_states

        query_start_loc_cpu = query_start_loc.detach().cpu()
        idx_mapping_cpu = idx_mapping.detach().cpu()
        num_reqs = min(query_start_loc_cpu.numel() - 1, idx_mapping_cpu.numel())

        if prefill_idx_mapping is not None:
            prefill_idx_mapping_cpu = prefill_idx_mapping.detach().cpu()
        else:
            prefill_idx_mapping_cpu = torch.full((num_reqs,), -1, dtype=torch.int32)
        if prefill_shift_state is None:
            prefill_shift_state = shift_state
        if prefill_wkv_state is None:
            prefill_wkv_state = wkv_state
        if prefill_elapsed is None:
            prefill_elapsed = elapsed

        prefill_batch_indices: list[int] = []
        for batch_idx in range(num_reqs):
            start = int(query_start_loc_cpu[batch_idx].item())
            end = int(query_start_loc_cpu[batch_idx + 1].item())
            if end <= start:
                continue
            prefill_batch_indices.append(batch_idx)

        for batch_idx in prefill_batch_indices:
            start = int(query_start_loc_cpu[batch_idx].item())
            end = int(query_start_loc_cpu[batch_idx + 1].item())
            query_len = end - start
            tokens = input_ids[start:end].view(1, query_len)
            prefill_row = int(prefill_idx_mapping_cpu[batch_idx].item())
            if prefill_row >= 0:
                state_shift = prefill_shift_state
                state_wkv = prefill_wkv_state
                state_elapsed = prefill_elapsed
                row = prefill_row
            else:
                state_shift = shift_state
                state_wkv = wkv_state
                state_elapsed = elapsed
                row = int(idx_mapping_cpu[batch_idx].item())
            state = [
                state_shift[:, :, row : row + 1, :],
                state_wkv[:, row : row + 1, :, :, :],
                state_elapsed[row : row + 1],
            ]
            if query_len == 1:
                out = self.forward_tokens(tokens, state)
                hidden_states[start:end] = out.view(1, C)
            else:
                out = self.forward_all_hidden(tokens, state)
                hidden_states[start:end] = out.view(query_len, C)
        return hidden_states

    def forward_vllm_pp_stage(
        self,
        *,
        input_ids: torch.Tensor | None,
        intermediate_tensors: IntermediateTensors | None,
        query_start_loc: torch.Tensor,
        idx_mapping: torch.Tensor,
        shift_state: torch.Tensor,
        wkv_state: torch.Tensor,
        elapsed: torch.Tensor,
        prefill_idx_mapping: torch.Tensor | None,
        prefill_shift_state: torch.Tensor | None,
        prefill_wkv_state: torch.Tensor | None,
        prefill_elapsed: torch.Tensor | None,
        rwkv_decode_batch_size: int,
        rwkv_decode_rows: list[int] | None,
        rwkv_decode_token_positions: list[int] | None,
        rwkv_prefill_token_ranges: list[tuple[int, int, int]] | None,
        rwkv_prefill_rows: list[int] | None,
        rwkv_prefill_groups: list[tuple[int, int, int, int, int, int]] | None,
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
            if incoming_v_first.shape[-1] == C:
                incoming_v_first = self._tp_hidden_slice(incoming_v_first, -1)
            total_tokens = incoming_hidden_states.shape[0]

        hidden_states = torch.empty(
            (total_tokens, C), dtype=DTYPE, device=first_device()
        )
        v_first_states = None
        if not is_last_pp_rank:
            v_first_states = torch.empty(
                (total_tokens, C),
                dtype=DTYPE,
                device=first_device(),
            )

        if (
            rwkv_decode_rows is not None
            or rwkv_prefill_token_ranges is not None
            or rwkv_prefill_groups is not None
        ):
            decode_rows = rwkv_decode_rows or []
            decode_positions = rwkv_decode_token_positions or []
            assert len(decode_rows) == len(decode_positions)
            if decode_rows:
                decode_batch_size = rwkv_decode_batch_size
                assert decode_batch_size > 0
                start, end = RWKV7ForCausalLM._compact_decode_token_range(
                    decode_batch_size, decode_rows, decode_positions
                )
                if is_first_pp_rank:
                    assert input_ids is not None
                    tokens = input_ids[start:end].view(decode_batch_size, 1)
                    x = self.embed(tokens)
                    group_v_first = None
                else:
                    assert incoming_hidden_states is not None
                    assert incoming_v_first is not None
                    x = incoming_hidden_states[start:end].view(decode_batch_size, 1, C)
                    group_v_first = incoming_v_first[start:end].view(
                        decode_batch_size,
                        1,
                        getattr(self, "tp_hidden_size", C),
                    )
                state = [
                    shift_state[:, :, :decode_batch_size, :],
                    wkv_state[:, :decode_batch_size, :, :, :],
                    elapsed[:decode_batch_size],
                ]
                path = select_path(decode_batch_size, 1)
                out, out_v_first = self.forward_layer_range(
                    x,
                    state,
                    path,
                    v_first=group_v_first,
                    final=is_last_pp_rank,
                    all_logits=True,
                    last_indices=None,
                )
                out = out.view(decode_batch_size, C)
                hidden_states[start:end] = out.view(decode_batch_size, C)
                if v_first_states is not None:
                    if out_v_first is None:
                        assert group_v_first is not None
                        out_v_first = group_v_first
                    if getattr(self, "tp_size", 1) > 1:
                        out_v_first = tensor_model_parallel_all_gather(out_v_first)
                    out_v_first = out_v_first.view(decode_batch_size, C)
                    v_first_states[start:end] = out_v_first.view(decode_batch_size, C)

            prefill_ranges = rwkv_prefill_token_ranges or []
            prefill_rows = rwkv_prefill_rows or []
            assert len(prefill_ranges) == len(prefill_rows)
            if prefill_shift_state is None:
                prefill_shift_state = shift_state
            if prefill_wkv_state is None:
                prefill_wkv_state = wkv_state
            if prefill_elapsed is None:
                prefill_elapsed = elapsed
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
                    x = incoming_hidden_states[start:end].view(batch_size, query_len, C)
                    group_v_first = incoming_v_first[start:end].view(
                        batch_size,
                        query_len,
                        getattr(self, "tp_hidden_size", C),
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
                hidden_states[start:end] = out.view(batch_size * query_len, C)
                if v_first_states is not None:
                    if out_v_first is None:
                        assert group_v_first is not None
                        out_v_first = group_v_first
                    if getattr(self, "tp_size", 1) > 1:
                        out_v_first = tensor_model_parallel_all_gather(out_v_first)
                    v_first_states[start:end] = out_v_first.view(
                        batch_size * query_len, C
                    )
            if prefill_groups:
                if not is_last_pp_rank:
                    assert v_first_states is not None
                    return IntermediateTensors(
                        {"hidden_states": hidden_states, "v_first": v_first_states}
                    )
                return hidden_states

            for (_batch_idx, start, end), row in zip(prefill_ranges, prefill_rows):
                query_len = end - start
                if is_first_pp_rank:
                    assert input_ids is not None
                    tokens = input_ids[start:end].view(1, query_len)
                    x = self.embed(tokens)
                    group_v_first = None
                else:
                    assert incoming_hidden_states is not None
                    assert incoming_v_first is not None
                    x = incoming_hidden_states[start:end].view(1, query_len, C)
                    group_v_first = incoming_v_first[start:end].view(
                        1, query_len, getattr(self, "tp_hidden_size", C)
                    )
                state = [
                    prefill_shift_state[:, :, row : row + 1, :],
                    prefill_wkv_state[:, row : row + 1, :, :, :],
                    prefill_elapsed[row : row + 1],
                ]
                path = select_path(1, query_len)
                out, out_v_first = self.forward_layer_range(
                    x,
                    state,
                    path,
                    v_first=group_v_first,
                    final=is_last_pp_rank,
                    all_logits=True,
                    last_indices=None,
                )
                hidden_states[start:end] = out.view(query_len, C)
                if v_first_states is not None:
                    if out_v_first is None:
                        assert group_v_first is not None
                        out_v_first = group_v_first
                    if getattr(self, "tp_size", 1) > 1:
                        out_v_first = tensor_model_parallel_all_gather(out_v_first)
                    v_first_states[start:end] = out_v_first.view(query_len, C)

            if not is_last_pp_rank:
                assert v_first_states is not None
                return IntermediateTensors(
                    {"hidden_states": hidden_states, "v_first": v_first_states}
                )
            return hidden_states

        query_start_loc_cpu = query_start_loc.detach().cpu()
        idx_mapping_cpu = idx_mapping.detach().cpu()
        num_reqs = min(query_start_loc_cpu.numel() - 1, idx_mapping_cpu.numel())

        if prefill_idx_mapping is not None:
            prefill_idx_mapping_cpu = prefill_idx_mapping.detach().cpu()
        else:
            prefill_idx_mapping_cpu = torch.full((num_reqs,), -1, dtype=torch.int32)
        if prefill_shift_state is None:
            prefill_shift_state = shift_state
        if prefill_wkv_state is None:
            prefill_wkv_state = wkv_state
        if prefill_elapsed is None:
            prefill_elapsed = elapsed

        prefill_batch_indices: list[int] = []
        for batch_idx in range(num_reqs):
            start = int(query_start_loc_cpu[batch_idx].item())
            end = int(query_start_loc_cpu[batch_idx + 1].item())
            if end <= start:
                continue
            prefill_batch_indices.append(batch_idx)

        for batch_idx in prefill_batch_indices:
            start = int(query_start_loc_cpu[batch_idx].item())
            end = int(query_start_loc_cpu[batch_idx + 1].item())
            query_len = end - start
            if is_first_pp_rank:
                assert input_ids is not None
                tokens = input_ids[start:end].view(1, query_len)
                x = self.embed(tokens)
                group_v_first = None
            else:
                assert incoming_hidden_states is not None
                assert incoming_v_first is not None
                x = incoming_hidden_states[start:end].view(1, query_len, C)
                group_v_first = incoming_v_first[start:end].view(
                    1, query_len, getattr(self, "tp_hidden_size", C)
                )

            prefill_row = int(prefill_idx_mapping_cpu[batch_idx].item())
            if prefill_row >= 0:
                state_shift = prefill_shift_state
                state_wkv = prefill_wkv_state
                state_elapsed = prefill_elapsed
                row = prefill_row
            else:
                state_shift = shift_state
                state_wkv = wkv_state
                state_elapsed = elapsed
                row = int(idx_mapping_cpu[batch_idx].item())
            state = [
                state_shift[:, :, row : row + 1, :],
                state_wkv[:, row : row + 1, :, :, :],
                state_elapsed[row : row + 1],
            ]
            path = select_path(1, query_len)
            out, out_v_first = self.forward_layer_range(
                x,
                state,
                path,
                v_first=group_v_first,
                final=is_last_pp_rank,
                all_logits=True,
                last_indices=None,
            )
            hidden_states[start:end] = out.view(query_len, C)
            if v_first_states is not None:
                if out_v_first is None:
                    assert group_v_first is not None
                    out_v_first = group_v_first
                if getattr(self, "tp_size", 1) > 1:
                    out_v_first = tensor_model_parallel_all_gather(out_v_first)
                v_first_states[start:end] = out_v_first.view(query_len, C)

        if not is_last_pp_rank:
            assert v_first_states is not None
            return IntermediateTensors(
                {"hidden_states": hidden_states, "v_first": v_first_states}
            )
        return hidden_states

    def forward_tokens(
        self, tokens: torch.Tensor, state: list[torch.Tensor]
    ) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        B, T = tokens.shape
        path = select_path(B, T)
        x = self.embed(tokens)
        return self.forward_from_x(x, state, path)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.linear_head(hidden_states)
        logits = self.logits_processor(None, logits)
        if logits is not None and getattr(self, "tp_size", 1) > 1:
            logits = tensor_model_parallel_all_gather(logits)
            if logits is not None:
                logits = logits[..., : getattr(self, "vocab_size", V)]
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

        if not isinstance(logits_indices, torch.Tensor):
            return None
        if (
            logits_indices.dim() != 1
            or logits_indices.numel() != num_reqs
            or logits_indices.dtype not in (torch.int32, torch.int64)
            or not logits_indices.is_contiguous()
            or hidden_states.shape[0] < num_reqs
        ):
            return None

        expected_indices = torch.arange(
            num_reqs,
            dtype=logits_indices.dtype,
            device=logits_indices.device,
        )
        if not torch.equal(logits_indices, expected_indices):
            return None

        return self.compute_logits(hidden_states[:num_reqs])

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)

    def _get_cpu_embedding_cache(
        self, batch_size: int, query_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        host, dev = self.emb_cache.get((batch_size, query_len), (None, None))
        if host is None:
            host = torch.empty(
                (batch_size * query_len, C), dtype=DTYPE, pin_memory=True
            )
            dev = torch.empty(
                (batch_size, query_len, C), dtype=DTYPE, device=first_device()
            )
            self.emb_cache[(batch_size, query_len)] = (host, dev)
        return host, dev

    def _copy_cpu_embedding_to_device(
        self,
        tokens: torch.Tensor,
        host: torch.Tensor,
        dev: torch.Tensor,
        batch_size: int,
        query_len: int,
    ) -> torch.Tensor:
        flat = tokens.reshape(-1)
        if flat.device.type != "cpu":
            flat = flat.cpu()
        if getattr(self, "tp_size", 1) > 1:
            vocab_start, vocab_end, vocab_per_rank = self._tp_vocab_range(
                getattr(self, "vocab_size", V)
            )
            mask = (flat < vocab_start) | (flat >= vocab_end)
            local = (flat - vocab_start).clamp(min=0, max=vocab_per_rank - 1)
            torch.index_select(self.z["emb.weight"], 0, local, out=host)
            host[mask] = 0
        else:
            torch.index_select(self.z["emb.weight"], 0, flat, out=host)
        dev.copy_(host.view(batch_size, query_len, C), non_blocking=True)
        return dev

    @staticmethod
    @staticmethod
    def _compact_decode_token_range(
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
                    "RWKV7 decode rows must be compact prefix rows "
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

    def prepare_cudagraph_embedding(self, tokens: torch.Tensor) -> None:
        if not self.emb_cpu:
            return
        if tokens.dim() == 1:
            tokens = tokens.view(-1, 1)
        batch_size, query_len = tokens.shape
        host, dev = self._get_cpu_embedding_cache(batch_size, query_len)
        self._copy_cpu_embedding_to_device(tokens, host, dev, batch_size, query_len)

    def prepare_cudagraph_inputs(self, model_inputs: dict[str, Any]) -> None:
        input_ids = model_inputs.get("input_ids")
        decode_rows = model_inputs.get("rwkv_decode_rows")
        if input_ids is not None and decode_rows:
            decode_batch_size = int(model_inputs["rwkv_decode_batch_size"])
            decode_positions = model_inputs["rwkv_decode_token_positions"]
            start, end = self._compact_decode_token_range(
                decode_batch_size, decode_rows, decode_positions
            )
            tokens = input_ids[start:end].view(decode_batch_size, 1)
            self.prepare_cudagraph_embedding(tokens)
            return
        if input_ids is not None:
            self.prepare_cudagraph_embedding(input_ids)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.emb_cpu:
            if tokens.device != self.z["emb.weight"].device:
                tokens = tokens.to(self.z["emb.weight"].device, non_blocking=True)
            if getattr(self, "tp_size", 1) == 1:
                return self.z["emb.weight"][tokens]
            vocab_start, vocab_end, vocab_per_rank = self._tp_vocab_range(
                getattr(self, "vocab_size", V)
            )
            mask = (tokens < vocab_start) | (tokens >= vocab_end)
            local = (tokens - vocab_start).clamp(min=0, max=vocab_per_rank - 1)
            out = self.z["emb.weight"][local]
            out.masked_fill_(mask.unsqueeze(-1), 0)
            return self._tp_all_reduce(out)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        B, T = tokens.shape
        host, dev = self._get_cpu_embedding_cache(B, T)
        if torch.cuda.is_current_stream_capturing():
            return self._tp_all_reduce(dev)
        return self._tp_all_reduce(
            self._copy_cpu_embedding_to_device(tokens, host, dev, B, T)
        )

    def forward_from_x(
        self,
        x: torch.Tensor,
        state: list[torch.Tensor],
        path: PathConfig,
        all_logits: bool = False,
        last_indices=None,
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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        z = self.z
        B, T, _ = x.shape
        start_layer = getattr(self, "start_layer", 0)
        end_layer = getattr(self, "end_layer", L)
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
            )
            pre_mix = None
            if T == 1 and path.cmix_mode not in (CMIX_B1T1_SPARSE, CMIX_ROWS2_SPARSE):
                x, mixed = torch.ops.rwkv7_v3a_ops.add_layer_norm_cmix_mix_f16(
                    x.contiguous(),
                    xx.contiguous(),
                    state[0][local_layer][1],
                    z[p + "ln2.weight"],
                    z[p + "ln2.bias"],
                    z[p + "ffn.x_k"],
                )
                xx = self.cmix_from_mixed(mixed, p + "ffn.", path)
            else:
                x, xx = self.add_ln(x, xx, z[p + "ln2.weight"], z[p + "ln2.bias"])
                xx = self.cmix(xx, state[0][local_layer], p + "ffn.", path)
            if layer + 1 < end_layer:
                p_next = f"blocks.{layer + 1}."
                if LN1_TMIX_FUSE and B == 1 and T == 1:
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
                    x, pre_mix = outs[0], outs[1:]
                    xx = x
                else:
                    x, xx = self.add_ln(
                        x, xx, z[p_next + "ln1.weight"], z[p_next + "ln1.bias"]
                    )
            elif not final:
                x = self.add(x, xx)
                torch.ops.rwkv7_v3a_ops.advance_i32(state[2], T)
                return x, v_first
            elif not all_logits:
                if last_indices is not None:
                    x = self.ln(self.add(x, xx), z["ln_out.weight"], z["ln_out.bias"])
                    x = x[torch.arange(B, device=x.device), last_indices].contiguous()
                else:
                    x = self.add_last_ln(x, xx, z["ln_out.weight"], z["ln_out.bias"])
                torch.ops.rwkv7_v3a_ops.advance_i32(state[2], T)
                return x, v_first
            else:
                x = self.add(x, xx)

        x = self.ln(x, z["ln_out.weight"], z["ln_out.bias"])
        torch.ops.rwkv7_v3a_ops.advance_i32(state[2], T)
        return x, v_first

    def ln(
        self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        return torch.ops.rwkv7_v3a_ops.layer_norm_f16(x.contiguous(), weight, bias)

    def forward_all_logits(
        self, tokens: torch.Tensor, state: list[torch.Tensor]
    ) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        B, T = tokens.shape
        path = select_path(B, T)
        x = self.embed(tokens)
        hidden_states = self.forward_from_x(x, state, path, all_logits=True)
        return self.compute_logits(hidden_states)

    def forward_all_hidden(
        self, tokens: torch.Tensor, state: list[torch.Tensor]
    ) -> torch.Tensor:
        """Return hidden states for every input position."""
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        B, T = tokens.shape
        path = select_path(B, T)
        x = self.embed(tokens)
        return self.forward_from_x(x, state, path, all_logits=True)

    def forward_last_at(
        self,
        tokens: torch.Tensor,
        state: list[torch.Tensor],
        last_indices: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        B, T = tokens.shape
        path = select_path(B, T)
        x = self.embed(tokens)
        hidden_states = self.forward_from_x(x, state, path, last_indices=last_indices)
        return self.compute_logits(hidden_states)

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.z
        ops = torch.ops.rwkv7_fast_ops_fp16
        B, T, _ = x.shape
        if pre_mix is not None:
            xr, xw, xk, xv, xa, xg = pre_mix
        else:
            xr, xw, xk, xv, xa, xg = ops.tmix_mix6(
                B,
                T,
                C,
                x.contiguous(),
                shift_state[0],
                z[p + "x_r"],
                z[p + "x_w"],
                z[p + "x_k"],
                z[p + "x_v"],
                z[p + "x_a"],
                z[p + "x_g"],
            )
        if pre_mix is not None:
            if path.use_batched_rkv:
                flat = torch.stack(
                    (xr.reshape(-1, C), xk.reshape(-1, C), xv.reshape(-1, C))
                )
                rkv = torch.bmm(flat, z[p + "rkv.weight"])
                local_c = rkv.shape[-1]
                r, k, v = [t.view(B, T, local_c) for t in rkv.unbind(0)]
            else:
                r = self.linear_orig_layout(
                    xr, z[p + "receptance.weight"], path, "att_c2c"
                )
                k = self.linear_orig_layout(xk, z[p + "key.weight"], path, "att_c2c")
                v = self.linear_orig_layout(xv, z[p + "value.weight"], path, "att_c2c")
        else:
            if path.use_batched_rkv:
                flat = torch.stack(
                    (xr.reshape(-1, C), xk.reshape(-1, C), xv.reshape(-1, C))
                )
                rkv = torch.bmm(flat, z[p + "rkv.weight"])
                local_c = rkv.shape[-1]
                r, k, v = [t.view(B, T, local_c) for t in rkv.unbind(0)]
            else:
                r = self.linear_orig_layout(
                    xr, z[p + "receptance.weight"], path, "att_c2c"
                )
                k = self.linear_orig_layout(xk, z[p + "key.weight"], path, "att_c2c")
                v = self.linear_orig_layout(xv, z[p + "value.weight"], path, "att_c2c")
        local_c = r.shape[-1]
        local_h = local_c // N

        v1 = None
        if (
            LOWRANK_WEIGHT != "orig"
            and can_use_lowrank_fused(path.rows)
            and can_use_lowrank_out_fused(path.rows)
            and layer != 0
        ):
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
        elif LOWRANK_WEIGHT != "orig" and can_use_lowrank_fused(path.rows):
            w1, a1, g1 = torch.ops.rwkv7_v3a_ops.linear_wag_rank_in_f16(
                xw.contiguous(),
                xa.contiguous(),
                xg.contiguous(),
                z[p + "w1.t"],
                z[p + "a1.t"],
                z[p + "g1.t"],
            )
        else:
            w1 = self.linear_rank_in(xw, z.get(p + "w1"), z.get(p + "w1.t"), path.rows)
            a1 = self.linear_rank_in(xa, z.get(p + "a1"), z.get(p + "a1.t"), path.rows)
            g1 = self.linear_rank_in(xg, z.get(p + "g1"), z.get(p + "g1.t"), path.rows)
        v_done = False
        if (
            LOWRANK_WEIGHT != "orig"
            and can_use_lowrank_out_fused(path.rows)
            and layer != 0
            and v1 is not None
        ):
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
        elif LOWRANK_WEIGHT != "orig" and can_use_lowrank_out_fused(path.rows):
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
            a = self.linear_rank_out(a1, z.get(p + "a2"), z.get(p + "a2.t"), path.rows)
            g = self.linear_rank_out_act(
                g1, z.get(p + "g2"), z.get(p + "g2.t"), path.rows, 2
            )
        k, neg_kk, kka = ops.tmix_kk_a_gate(
            B,
            T,
            local_c,
            local_h,
            k.contiguous(),
            z[p + "k_k"],
            z[p + "a0"],
            a.contiguous(),
            z[p + "k_a"],
        )

        if layer == 0:
            v_first = v
        elif not v_done:
            if LOWRANK_WEIGHT != "orig" and can_use_lowrank_out_fused(path.rows):
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
                    B,
                    T,
                    local_c,
                    v.contiguous(),
                    v_first.contiguous(),
                    z[p + "v0"],
                    v12.contiguous(),
                )

        y = torch.empty_like(r)
        if WKV_MODE == "fp32io16":
            w_raw = ops.add_vec(local_c, w.contiguous(), z[p + "w0"])
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
        elif T <= 16:
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
            w_raw = ops.add_vec(local_c, w.contiguous(), z[p + "w0"])
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
        out = self.linear_orig_layout(y, z[p + "output.weight"], path, "att_c2c")
        return self._tp_all_reduce(out), v_first

    def cmix(
        self, x: torch.Tensor, shift_state: torch.Tensor, p: str, path: PathConfig
    ) -> torch.Tensor:
        z = self.z
        ops = torch.ops.rwkv7_fast_ops_fp16
        B, T, _ = x.shape

        if path.cmix_mode == CMIX_B1T1_SPARSE:
            return self._tp_all_reduce(
                ops.cmix_sparse_one(
                    C,
                    z[p + "key.weight.fc"].size(0),
                    x.contiguous(),
                    shift_state[1],
                    z[p + "x_k"],
                    z[p + "key.weight.fc"],
                    z[p + "value.weight"],
                )
            )
        if path.cmix_mode == CMIX_ROWS2_SPARSE:
            return self._tp_all_reduce(
                ops.cmix_sparse_rows(
                    B,
                    T,
                    C,
                    z[p + "key.weight.fc"].size(0),
                    x.contiguous(),
                    shift_state[1],
                    z[p + "x_k"],
                    z[p + "key.weight.fc"],
                    z[p + "value.weight"],
                )
            )

        mixed = ops.cmix_mix(B, T, C, x.contiguous(), shift_state[1], z[p + "x_k"])
        return self.cmix_from_mixed(mixed, p, path)

    def cmix_from_mixed(
        self, mixed: torch.Tensor, p: str, path: PathConfig
    ) -> torch.Tensor:
        z = self.z
        ops = torch.ops.rwkv7_fast_ops_fp16
        B, T, _ = mixed.shape
        hid = self.linear_orig_layout(mixed, z[p + "key.weight"], path, "ffn_key")
        if path.cmix_mode == CMIX_B1T1_NOFC:
            return self._tp_all_reduce(
                ops.cmix_sparse_down_relu_one(
                    C,
                    z[p + "value.weight"].size(0),
                    hid.view(-1).contiguous(),
                    z[p + "value.weight"],
                )
            )
        if path.cmix_mode == CMIX_ROWS2_NOFC:
            F = z[p + "value.weight"].size(0)
            if path.rows >= CMIX_NOFC_T512_MIN_ROWS and C % 512 == 0 and F % 512 == 0:
                return self._tp_all_reduce(
                    ops.cmix_sparse_down_relu_rows_t512(
                        B, T, C, F, hid.contiguous(), z[p + "value.weight"]
                    )
                )
            return self._tp_all_reduce(
                ops.cmix_sparse_down_relu_rows(
                    B, T, C, F, hid.contiguous(), z[p + "value.weight"]
                )
            )

        k = ops.relu_square(hid.contiguous())
        return self._tp_all_reduce(self.linear(k, z[p + "value.weight"]))

    def linear(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if x.numel() == x.size(-1) and weight.size(1) % 64 == 0:
            return torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk(x.contiguous(), weight)
        return torch.ops.rwkv7_v3a_ops.linear_f16(x.contiguous(), weight)

    def linear_head(self, x: torch.Tensor) -> torch.Tensor:
        z = self.z
        if not use_orig_linear("head"):
            return self.linear(x, z["head.weight"])
        rows = x.numel() // C
        return self.linear_orig_layout(
            x, z["head.weight"], PathConfig(rows, False, CMIX_DENSE), "head"
        )

    def linear_orig_layout(
        self, x: torch.Tensor, weight: torch.Tensor, path: PathConfig, group: str
    ) -> torch.Tensor:
        if not use_orig_linear(group):
            return self.linear(x, weight)
        if path.rows == 1:
            if group == "ffn_key":
                if C == 2560:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(
                        x.contiguous(), weight, 128, 2, True
                    )
                return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(
                    x.contiguous(), weight, 128, 2, C <= 1024
                )
            return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(
                x.contiguous(), weight, 128, 2, group != "att_c2c" or C < 2048
            )
        if path.rows == 2:
            if group == "att_c2c":
                return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(
                    x.contiguous(), weight, 64, 2, True
                )
            if group == "ffn_key":
                if C == 2560:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(
                        x.contiguous(), weight, 128, 2, False
                    )
                if C < 4096:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(
                        x.contiguous(), weight, 64, 2, True
                    )
                return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(
                    x.contiguous(), weight, 128, 2, False
                )
            if group == "head" and C == 2560:
                return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(
                    x.contiguous(), weight, 128, 2, False
                )
            return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(
                x.contiguous(), weight, 64, 2, True
            )
        if path.rows == 3:
            if group == "head":
                if C <= 2048:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(
                        x.contiguous(), weight
                    )
                if C == 2560:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(
                        x.contiguous(), weight
                    )
                return torch.ops.rwkv7_v3a_ops.linear_orig_rows_f16(
                    x.contiguous(), weight, 3, 2
                )
            if group == "ffn_key":
                if C <= 1024:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_cfg_f16(
                        x.contiguous(), weight, 64, 3, 4
                    )
                if C == 2048:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(
                        x.contiguous(), weight
                    )
                if C == 2560:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(
                        x.contiguous(), weight
                    )
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 0, 0
                )
            if group == "att_c2c":
                if C == 768:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_f16(
                        x.contiguous(), weight, 1, 2
                    )
                if C == 1024:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_f16(
                        x.contiguous(), weight, 2, 2
                    )
                if C == 2048:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_f16(
                        x.contiguous(), weight, 3, 4
                    )
                if C == 2560:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_f16(
                        x.contiguous(), weight, 3, 2
                    )
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 0, 2
                )
            return torch.ops.rwkv7_v3a_ops.linear_orig_rows_cfg_f16(
                x.contiguous(), weight, 64, 3, 4
            )
        if path.rows == 4:
            if group == "ffn_key":
                if C <= 1024:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_cfg_f16(
                        x.contiguous(), weight, 64, 2, 4
                    )
                if C == 2048:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(
                        x.contiguous(), weight
                    )
                if C == 2560:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(
                        x.contiguous(), weight
                    )
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 0, 0
                )
            if group == "att_c2c":
                if C <= 1024:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_f16(
                        x.contiguous(), weight, 2, 2
                    )
                if C == 2048:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_f16(
                        x.contiguous(), weight, 4, 2
                    )
                if C == 2560:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_f16(
                        x.contiguous(), weight, 4, 2
                    )
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 0, 2
                )
        if group == "head":
            if C == 768:
                if 192 <= path.rows < 256:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 128, 3
                    )
                if 96 <= path.rows < 160:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 0, 1
                    )
            if C == 1024:
                if 256 <= path.rows < 384:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(
                        x.contiguous(), weight
                    )
                if 192 <= path.rows < 256:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 0, 2
                    )
                if 96 <= path.rows < 160:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 32, 1
                    )
            if C == 2048:
                if 256 <= path.rows < 384:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 32, 0
                    )
                if 192 <= path.rows < 256:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 32, 6
                    )
                if 128 <= path.rows < 160:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 0, 1
                    )
                if 96 <= path.rows < 112:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 0, 0
                    )
            if C == 2560:
                if path.rows >= 256:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 32, 0
                    )
                if path.rows >= 192:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 0, 5
                    )
                if path.rows >= 160:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 32, 5
                    )
                if path.rows >= 128:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 0, 1
                    )
                if path.rows >= 96:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 32, 0
                    )
                if path.rows >= 80:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 0, 0
                    )
                if path.rows >= 72:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 32, 1
                    )
            if path.rows >= 1024:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 128, 0
                )
            if path.rows >= 512:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 0, 2
                )
            if path.rows >= 384:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 128, 2
                )
            if path.rows >= 256:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 0, 1
                )
            if path.rows >= 192:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 128, 0
                )
            if path.rows >= 160:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 32, 0
                )
            if path.rows >= 128:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 128, 0
                )
            if path.rows >= 112:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 32, 0
                )
            if path.rows >= 96:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 32, 1
                )
            if path.rows >= 80:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 32, 2
                )
            if path.rows >= 72:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 128, 2
                )
        if group == "att_c2c":
            if C == 2560 and 17 <= path.rows <= 20:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 0, 0
                )
            if C == 768:
                if 256 <= path.rows < 384:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 128, 1
                    )
                if 96 <= path.rows < 112:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 32, 3
                    )
            if C == 1024:
                if 256 <= path.rows < 384:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 128, 0
                    )
                if 96 <= path.rows < 112:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 32, 6
                    )
            if C == 2048:
                if 256 <= path.rows < 384:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 32, 3
                    )
                if 192 <= path.rows < 256:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 128, 0
                    )
                if 96 <= path.rows < 112:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 32, 4
                    )
            if C == 2560:
                if path.rows >= 256:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 0, 1
                    )
                if path.rows >= 160:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 0, 2
                    )
                if path.rows >= 128:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 128, 2
                    )
                if path.rows >= 112:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 128, 3
                    )
                if path.rows >= 96:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 32, 2
                    )
                if path.rows >= 72:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 128, 2
                    )
                if path.rows >= 5:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(
                        x.contiguous(), weight
                    )
            if path.rows >= 1024:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 32, 4
                )
            if path.rows >= 768:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 32, 0
                )
            if path.rows >= 512:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 32, 1
                )
            if path.rows >= 384:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 128, 2
                )
            if path.rows >= 256:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 32, 4
                )
            if path.rows >= 192:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 0, 0
                )
            if path.rows >= 160:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 128, 1
                )
            if path.rows >= 112:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight)
            if path.rows >= 96:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 0, 5
                )
            if path.rows >= 72:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 32, 0
                )
            if path.rows >= 48:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 32, 6
                )
            if path.rows >= 32:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 0, 0
                )
            if path.rows >= 24:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 0, 6
                )
            if path.rows >= 12:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 0, 0
                )
            if path.rows >= 5:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 0, 2
                )
        if group == "ffn_key":
            if C == 2560 and 17 <= path.rows <= 20:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 0, 0
                )
            if C == 768:
                if 256 <= path.rows < 384:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(
                        x.contiguous(), weight
                    )
                if 96 <= path.rows < 112:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(
                        x.contiguous(), weight
                    )
            if C == 1024:
                if 256 <= path.rows < 384:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 32, 2
                    )
                if 192 <= path.rows < 256:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 0, 0
                    )
                if 96 <= path.rows < 160:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 32, 2
                    )
            if C == 2048 and 128 <= path.rows < 160:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 0, 3
                )
            if C == 2560:
                if path.rows >= 192:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 32, 5
                    )
                if path.rows >= 160:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 0, 4
                    )
                if path.rows >= 128:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 32, 5
                    )
                if path.rows >= 112:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 128, 4
                    )
                if path.rows >= 96:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 128, 4
                    )
                if path.rows >= 80:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 0, 3
                    )
                if path.rows >= 72:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight, 32, 4
                    )
                if path.rows >= 3:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(
                        x.contiguous(), weight
                    )
            if path.rows >= 1024:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 0, 0
                )
            if path.rows >= 768:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 32, 1
                )
            if path.rows >= 512:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 128, 3
                )
            if path.rows >= 384:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 32, 0
                )
            if path.rows >= 256:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 128, 4
                )
            if path.rows >= 192:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 0, 1
                )
            if path.rows >= 160:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 0, 2
                )
            if path.rows >= 128:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 32, 0
                )
            if path.rows >= 112:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 32, 3
                )
            if path.rows >= 96:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 32, 1
                )
            if path.rows >= 72:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 128, 1
                )
            if path.rows >= 48:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 0, 1
                )
            if path.rows >= 12:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 0, 0
                )
            if path.rows in (5, 6):
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, 0, 1
                )
        return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight)

    def linear_rank_in(
        self, x: torch.Tensor, weight: torch.Tensor, weight_t: torch.Tensor, rows: int
    ) -> torch.Tensor:
        if weight_t is not None and rows <= LOWRANK_IN_ROWS_T:
            return torch.ops.rwkv7_v3a_ops.linear_t_f16(x.contiguous(), weight_t)
        return (
            self.linear_lowrank_orig(x, weight)
            if weight is not None
            else self.linear_t_orig(x, weight_t)
        )

    def linear_rank_out(
        self, x: torch.Tensor, weight: torch.Tensor, weight_t: torch.Tensor, rows: int
    ) -> torch.Tensor:
        if (
            weight_t is not None
            and C >= LOWRANK_FUSED_MIN_C
            and rows <= LOWRANK_OUT_ROWS_T
        ):
            return torch.ops.rwkv7_v3a_ops.linear_t_f16(x.contiguous(), weight_t)
        return (
            self.linear_lowrank_orig(x, weight)
            if weight is not None
            else self.linear_t_orig(x, weight_t)
        )

    def linear_rank_out_act(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_t: torch.Tensor,
        rows: int,
        act: int,
    ) -> torch.Tensor:
        if (
            weight_t is not None
            and C >= LOWRANK_FUSED_MIN_C
            and rows <= LOWRANK_OUT_ROWS_T
        ):
            return torch.ops.rwkv7_v3a_ops.linear_t_act_f16(
                x.contiguous(), weight_t, act
            )
        ops = torch.ops.rwkv7_fast_ops_fp16
        x = (
            ops.act_tanh(x.contiguous())
            if act == 1
            else ops.act_sigmoid(x.contiguous())
        )
        return (
            self.linear_lowrank_orig(x.contiguous(), weight)
            if weight is not None
            else self.linear_t_orig(x, weight_t)
        )

    def linear_lowrank_orig(
        self, x: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        return torch.ops.rwkv7_v3a_ops.linear_f16(x.contiguous(), weight)

    def linear_t_orig(self, x: torch.Tensor, weight_t: torch.Tensor) -> torch.Tensor:
        return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight_t)

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
