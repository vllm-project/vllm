# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
)

from .config import SING_PROBE_ATTN_MODEL_TYPE, ProbeConfig
from .loader import read_probe_config
from .token_probe import TokenProbe, TokenProbeForwardContext

if TYPE_CHECKING:
    from vllm.v1.worker.block_table import MultiGroupBlockTable


@dataclass(frozen=True)
class TokenProbeOutputCopy:
    scores: torch.Tensor
    copy_event: torch.cuda.Event

    def copy_to_cpu(self) -> torch.Tensor:
        scores_cpu = self.scores.to("cpu", non_blocking=True)
        self.copy_event.record()
        return scores_cpu


class TokenProbeRunner:
    def __init__(self, ckpt_path: str | None, output_rank: bool = True) -> None:
        self.enabled = ckpt_path is not None
        self.uses_kv_cache = bool(
            ckpt_path
            and ProbeConfig.from_dict(read_probe_config(ckpt_path)).model_type
            == SING_PROBE_ATTN_MODEL_TYPE
        )
        self.kv_cache_group_id: int | None = None
        self.probe: TokenProbe | None = None
        self.output_rank = output_rank

    def find_probe(self, model: nn.Module) -> TokenProbe:
        candidates = [model]
        seen: set[int] = set()
        while candidates:
            candidate = candidates.pop(0)
            if id(candidate) in seen:
                continue
            seen.add(id(candidate))
            probe = getattr(candidate, "token_probe", None)
            if isinstance(probe, TokenProbe):
                return probe
            for attr in ("model", "module", "language_model"):
                child = getattr(candidate, attr, None)
                if isinstance(child, nn.Module):
                    candidates.append(child)
        raise RuntimeError("token probe is enabled but was not found on the model")

    def bind_model(self, model: nn.Module) -> None:
        if not self.enabled or self.probe is not None:
            return
        self.probe = self.find_probe(model)
        if self.probe.uses_kv_cache != self.uses_kv_cache:
            raise RuntimeError("token probe checkpoint type changed after model load")

    def is_prefill_only(
        self,
        num_computed_tokens: np.ndarray,
        num_prompt_tokens: np.ndarray,
    ) -> bool:
        if not self.enabled or envs.VLLM_ENABLE_TOKEN_PROBE_PREFILL:
            return False
        for computed, prompt in zip(num_computed_tokens, num_prompt_tokens):
            if computed >= prompt:
                return False
        return True

    def model_kwargs(
        self,
        *,
        num_reqs: int,
        max_query_len: int,
        slot_mappings_by_group: dict[int, torch.Tensor] | None,
        block_tables: "MultiGroupBlockTable",
        query_start_loc: torch.Tensor,
        kv_cache_initialized: bool,
        is_prefill_only: bool,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {}
        score_enabled = self.output_rank and not is_prefill_only
        capture_enabled = self.output_rank and (score_enabled or self.uses_kv_cache)
        context: TokenProbeForwardContext = {
            "capture": capture_enabled,
            "score": score_enabled,
            "block_table": None,
            "slot_mapping": None,
            "query_start_loc": None,
            "max_query_len": max_query_len,
        }
        if not capture_enabled or not self.uses_kv_cache:
            return {"token_probe_context": context}
        if slot_mappings_by_group is None:
            if not kv_cache_initialized:
                return {"token_probe_context": context}
            raise RuntimeError("token probe requires an attention slot mapping")
        if self.kv_cache_group_id is None:
            raise RuntimeError(
                "token probe attention KV cache group is not initialized"
            )
        context["block_table"] = block_tables[self.kv_cache_group_id].get_device_tensor(
            num_reqs
        )
        context["slot_mapping"] = slot_mappings_by_group[self.kv_cache_group_id]
        context["query_start_loc"] = query_start_loc[: num_reqs + 1]
        return {"token_probe_context": context}

    def unpack_model_output(self, output: Any) -> tuple[Any, torch.Tensor | None]:
        if not self.enabled:
            return output, None
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("token probe scores are missing from model output")
        model_output, scores = output
        if scores is not None and not isinstance(scores, torch.Tensor):
            raise RuntimeError("token probe model output contains invalid scores")
        return model_output, scores

    @staticmethod
    def trim_scores(
        scores: torch.Tensor | None,
        num_tokens: int,
    ) -> torch.Tensor | None:
        if scores is None:
            return None
        return scores[:num_tokens].float()

    @classmethod
    def to_numpy(
        cls,
        scores: torch.Tensor | None,
        num_tokens: int,
    ) -> np.ndarray | None:
        trimmed = cls.trim_scores(scores, num_tokens)
        return None if trimmed is None else trimmed.cpu().numpy()

    def prepare_output_copy(
        self,
        scores: torch.Tensor | None,
        num_tokens: int,
    ) -> TokenProbeOutputCopy | None:
        scores = self.trim_scores(scores, num_tokens)
        if scores is None:
            return None
        assert self.probe is not None
        assert self.probe.output_copy_event is not None
        return TokenProbeOutputCopy(scores, self.probe.output_copy_event)

    def needs_paged_attention_inputs(self, kv_cache_initialized: bool) -> bool:
        return bool(
            self.enabled
            and self.output_rank
            and kv_cache_initialized
            and self.uses_kv_cache
        )

    def initialize_kv_cache(
        self,
        *,
        model: nn.Module,
        kv_cache_config: KVCacheConfig,
        block_tables: "MultiGroupBlockTable",
        async_output: bool,
    ) -> None:
        if not self.enabled:
            return
        self.bind_model(model)
        assert self.probe is not None
        if not self.output_rank:
            return
        if self.uses_kv_cache:
            self.kv_cache_group_id = next(
                (
                    group_id
                    for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
                    if isinstance(group.kv_cache_spec, AttentionSpec)
                    and not isinstance(group.kv_cache_spec, EncoderOnlyAttentionSpec)
                ),
                None,
            )
            if self.kv_cache_group_id is None:
                raise RuntimeError("token probe requires a causal attention KV cache")
            block_table = block_tables[self.kv_cache_group_id]
            self.probe.initialize_kv_cache(
                kv_cache_config.num_blocks * block_table.blocks_per_kv_block,
                block_table.block_size,
            )
        self.probe.initialize_runtime(async_output)
