# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unlimited-OCR-2 DASH with full decode and breakable prefill CUDA graphs."""

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.rswa_attention import RSWAAttention
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.unlimited_ocr2 import (
    DashRouter,
    dash_attention,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionMetadata,
)
from vllm.v1.attention.ops.unlimited_ocr2 import cache_dash_prefill, paged_dash_decode
from vllm.v1.worker.gpu.sample.logits_processor.interface import (
    LogitsContext,
    LogitsProcessor,
    LogitsProcRequestState,
)

TOKEN_ROLES_KEY = "unlimited_ocr2_token_roles"


class UnlimitedOCR2Attention(nn.Module):
    def __init__(
        self,
        vllm_config,
        config,
        hidden_size,
        num_heads,
        cache_config=None,
        quant_config=None,
        reduce_results=True,
        prefix="",
        **kwargs,
    ):
        # Keep main KV and one shared router/metadata head per TP rank.
        super().__init__()
        from vllm.distributed import get_tensor_model_parallel_world_size

        tp = get_tensor_model_parallel_world_size()
        self.num_heads = num_heads // tp
        self.num_kv_heads = max(1, config.num_key_value_heads // tp)
        self.head_dim = hidden_size // num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.max_model_len = vllm_config.model_config.max_model_len
        self.dash_backend = config.unlimited_ocr2_attention_backend
        self.sp = config.sparse_attention_config
        self.layer_idx = int(prefix.split(".")[-2])
        index_dim = int(self.sp["sparse_index_dim"])
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            num_heads,
            config.num_key_value_heads,
            bias=False,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            hidden_size,
            hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
        )
        self.dash_router = DashRouter(hidden_size, index_dim, config.rms_norm_eps)
        window = vllm_config.model_config.rswa_window
        attention_cls = RSWAAttention if window is not None else Attention
        attention_kwargs = {"rswa_window": window} if window is not None else {}
        self.attn = attention_cls(
            self.num_kv_heads + 1,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads + 1,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
            attn_backend=TritonAttentionBackend,
            **attention_kwargs,
        )

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], -1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view_as(k)
        iq, ik = self.dash_router(hidden_states)
        roles = get_forward_context().additional_kwargs.get(TOKEN_ROLES_KEY)
        result = torch.empty_like(q)
        self._attention(q, k, v, iq, ik, roles, result)
        return self.o_proj(result.flatten(1))[0]

    @eager_break_during_capture
    def _attention(self, q, k, v, iq, ik, roles, result):
        context = get_forward_context()
        all_metadata = context.attn_metadata
        if all_metadata is None:
            result.zero_()
            return
        assert isinstance(all_metadata, dict)
        metadata = all_metadata[self.attn.layer_name]
        assert isinstance(metadata, TritonAttentionMetadata)
        if (
            context.additional_kwargs.get("is_dummy_run", False)
            and metadata.max_query_len > 1
        ):
            result.zero_()
            return
        assert roles is not None
        assert isinstance(context.slot_mapping, dict)
        slots = context.slot_mapping[self.attn.layer_name]
        cache = self.attn.kv_cache
        if metadata.max_query_len == 1:
            result.copy_(
                paged_dash_decode(
                    q,
                    k,
                    v,
                    iq,
                    ik,
                    roles,
                    cache,
                    metadata.block_table,
                    metadata.seq_lens,
                    slots,
                    self.dash_router.head_cls,
                    self.sp,
                    self.layer_idx,
                    self.max_model_len,
                    metadata.rswa_prefix_lens,
                    backend=self.dash_backend,
                    cached_summaries=True,
                )
            )
        else:
            starts = metadata.query_start_loc.tolist()
            lengths = metadata.seq_lens.tolist()
            result.zero_()
            for request, length in enumerate(lengths):
                start, end = starts[request : request + 2]
                if start == end:
                    continue
                if end - start == 1:
                    result[start:end] = paged_dash_decode(
                        q[start:end],
                        k[start:end],
                        v[start:end],
                        iq[start:end],
                        ik[start:end],
                        roles[start:end],
                        cache,
                        metadata.block_table[request : request + 1],
                        metadata.seq_lens[request : request + 1],
                        slots[start:end],
                        self.dash_router.head_cls,
                        self.sp,
                        self.layer_idx,
                        self.max_model_len,
                        metadata.rswa_prefix_lens[request : request + 1]
                        if metadata.rswa_prefix_lens is not None
                        else None,
                        backend=self.dash_backend,
                        cached_summaries=True,
                    )
                else:
                    assert length == end - start, "DASH requires complete prefills"
                    result[start:end] = dash_attention(
                        q[start:end],
                        k[start:end],
                        v[start:end],
                        iq[start:end],
                        ik[start:end],
                        roles[start:end],
                        self.dash_router.head_cls,
                        self.sp,
                        self.layer_idx,
                        backend=self.dash_backend,
                    )
                    cache_dash_prefill(
                        k[start:end],
                        v[start:end],
                        ik[start:end],
                        roles[start:end],
                        slots[start:end].long(),
                        cache,
                        self.sp,
                        self.dash_router.head_cls,
                    )


DEFAULT_MAX_PAGE_TOKENS = 20000


def _force_token(logits: torch.Tensor, token_id: int) -> None:
    keep = logits[token_id].clone()
    logits[:] = -float("inf")
    logits[token_id] = keep


class PageCountGuard:
    """Per-request page-count state machine over the generated tokens.

    CPU-friendly core of the processor: token bookkeeping is plain Python
    ints/lists, only the one-row logits edit touches torch.
    """

    def __init__(
        self,
        page_token_id: int,
        eos_token_id: int,
        max_page_tokens: int = DEFAULT_MAX_PAGE_TOKENS,
    ):
        self.page_token_id = page_token_id
        self.eos_token_id = eos_token_id
        self.max_page_tokens = max_page_tokens
        self.expected_pages = -1
        self.marker_cum = 0
        self.page_len = 0
        self.scanned = 0

    def seed_expected_pages(self, prompt_ids: list[int]) -> None:
        """Count the prompt's page markers; called once the prompt is known."""
        self.expected_pages = prompt_ids.count(self.page_token_id)

    def step(self, output_ids: list[int], logits: torch.Tensor) -> None:
        """Apply the guard to one row of logits, in place."""
        if self.expected_pages <= 0:
            return
        n = len(output_ids)
        if n > self.scanned:
            tail = output_ids[self.scanned :]
            num_markers = tail.count(self.page_token_id)
            if num_markers:
                self.marker_cum += num_markers
                self.page_len = tail[::-1].index(self.page_token_id)
            else:
                self.page_len += len(tail)
            self.scanned = n

        if self.marker_cum > self.expected_pages:
            _force_token(logits, self.eos_token_id)
        elif 0 < self.max_page_tokens <= self.page_len:
            force = (
                self.page_token_id
                if self.marker_cum < self.expected_pages
                else self.eos_token_id
            )
            _force_token(logits, force)
        elif self.marker_cum < self.expected_pages:
            logits[self.eos_token_id] = -float("inf")


class PageCountGuardLogitsProcessor(LogitsProcessor):
    """Page-count guard for Unlimited-OCR-2 multi-page parsing.

    In the multi-page layout (``"<PAGE><image>" * n_pages + "Multi page
    parsing."``) the model must emit exactly one ``<PAGE>`` marker per prompt
    page marker. This request-level guard enforces that contract, mirroring
    the reference ``_uocr_page_count_guard``:

    * while fewer ``<PAGE>`` markers than prompt pages have been generated,
      EOS is masked so a page cannot be terminated early;
    * once the page count is reached EOS is allowed, and generating further
      ``<PAGE>`` markers forces EOS;
    * each page is capped at ``max_page_tokens`` generated tokens -- hitting
      the cap forces ``<PAGE>`` (next page) or EOS (last page);
    * prompts without a ``<PAGE>`` marker (single-page requests) pass
      through.

    Register it via the CLI::

        --logits-processors \\
          vllm.model_executor.models.unlimited_ocr2:PageCountGuardLogitsProcessor

    Per-request parameters are read from ``SamplingParams.extra_args``
    (``page_token_id``, ``eos_token_id``, ``max_page_tokens``); the token ids
    fall back to ``hf_config.sparse_attention_config.page_token_id`` and
    ``hf_config.eos_token_id`` when not given. The processor is inactive for
    requests (or v1 configs) where no page token id can be resolved.

    Per-slot state is keyed by the persistent request slot and the
    generated-token tail is re-read from the runner's committed token
    history each step.
    """

    def __init__(self, vllm_config: VllmConfig, req_states: LogitsProcRequestState):
        hf_config = vllm_config.model_config.hf_config
        sparse_config = getattr(hf_config, "sparse_attention_config", None)
        self._default_page_token_id = self._read(sparse_config, "page_token_id")
        eos_token_id = getattr(hf_config, "eos_token_id", None)
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0] if eos_token_id else None
        self._default_eos_token_id = eos_token_id

        self.all_token_ids = req_states.all_token_ids
        self.prompt_len_np = req_states.prompt_len.np
        self._guards: dict[int, PageCountGuard] = {}

    @staticmethod
    def _read(config: Any, key: str) -> Any:
        if config is None:
            return None
        if isinstance(config, Mapping):
            return config.get(key)
        return getattr(config, key, None)

    @staticmethod
    def _validate_token_id(name: str, value: Any):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"`{name}` has to be a non-negative integer, got {value}.")

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        extra_args = sampling_params.extra_args or {}
        for name in ("page_token_id", "eos_token_id"):
            if (value := extra_args.get(name)) is not None:
                cls._validate_token_id(name, value)
        value = extra_args.get("max_page_tokens")
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            raise ValueError(f"`max_page_tokens` has to be an integer, got {value}.")

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        self._guards.pop(req_idx, None)
        extra_args = sampling_params.extra_args or {}
        page_token_id = extra_args.get("page_token_id", self._default_page_token_id)
        if page_token_id is None:
            return False
        self._validate_token_id("page_token_id", page_token_id)

        eos_token_id = extra_args.get("eos_token_id", self._default_eos_token_id)
        if eos_token_id is None:
            raise ValueError(
                "`eos_token_id` is required for the page count guard but could "
                "not be resolved from `extra_args` or the model config."
            )
        self._validate_token_id("eos_token_id", eos_token_id)

        max_page_tokens = extra_args.get("max_page_tokens", DEFAULT_MAX_PAGE_TOKENS)
        if not isinstance(max_page_tokens, int) or isinstance(max_page_tokens, bool):
            raise ValueError(
                f"`max_page_tokens` has to be an integer, got {max_page_tokens}."
            )

        self._guards[req_idx] = PageCountGuard(
            page_token_id, eos_token_id, max_page_tokens
        )
        return True

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        if not self._guards:
            return logits
        # NOTE(eager): per-row host reads of the committed token history.
        # Rows are requests here; Unlimited-OCR-2 does not support
        # speculative decoding.
        for row, req in enumerate(ctx.idx_mapping_np):
            guard = self._guards.get(int(req))
            if guard is None:
                continue
            prompt_len = int(self.prompt_len_np[req])
            if guard.expected_pages < 0:
                # First apply for this request: the prompt is committed now,
                # so the page markers can be counted from the token history.
                prompt_ids = self.all_token_ids.gpu[req, :prompt_len].tolist()
                guard.seed_expected_pages(prompt_ids)
            if guard.expected_pages <= 0:
                continue
            total_len = int(ctx.seq_lens_upper_bound_np[row])
            output_ids = self.all_token_ids.gpu[req, prompt_len:total_len].tolist()
            guard.step(output_ids, logits[row])
        return logits
