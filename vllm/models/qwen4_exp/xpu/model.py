# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen4-Exp model (Qwen3.8-Flash-Next) for XPU.

Architecturally it is Qwen3.5-MoE (hybrid GDN linear-attention +
full-attention + MoE) extended with:

- **Hyper-connections** (``HyperConnection``): replace the per-layer
  RMSNorm + residual add. The state between blocks is ``hc`` parallel
  residual streams ``[T, hc, n_embd]``; each sub-block reads one mixed
  ``[T, n_embd]`` view and writes back through per-stream injection
  weights. The top-level ``hyper_connection_mixer`` acts as the final
  output norm (there is no separate RMSNorm).
- **PLE** n-gram embedding table, served on host from the checkpoint shards
  (see ``Qwen4ExpPLE``).
- **QSA** sparse attention on the indexed layers, via the CUDA
  ``Qwen4ExpQSAAttention`` when the config carries a QSA indexer
  (``indexer_n_heads``). Without one, those layers run dense.

The attention, GDN, and MoE sub-blocks are reused verbatim from
``qwen3_next`` / the GDN module.
"""

import re
from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn
from transformers import (
    Qwen4ExpConfig,
    Qwen4ExpTextConfig,
)

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateCopyFuncsByType,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextModel,
    Qwen3NextSparseMoeBlock,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    make_layers,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionBackend,
    PleShortConvAttentionMetadata,
)
from vllm.v1.kv_cache_interface import MambaSpec

from ..common.ops.ple import ple_conv, ple_gate, ple_ngram_ids

# nvidia.qsa imports nvidia.model, which imports nvidia.qsa: load the model first.
from ..nvidia.model import get_qsa_topk_buffer_width
from ..nvidia.qsa import Qwen4ExpQSAAttention

logger = init_logger(__name__)

# Transformers v5.18 renamed ``full_attention`` to ``indexed_attention`` in
# ``layer_types`` (some checkpoints spell it ``qwen_sparse_attention``), so the
# same checkpoint reports a different name per Transformers version. All three
# map to the same dense GQA block here -- these layers carry only
# ``self_attn.{q,k,v,o}_proj`` + ``{q,k}_norm`` and no indexer tensors.
# TODO: Delete the pre-5.18 spellings once Transformers 5.18.0 is the minimum.
_FULL_ATTENTION_LAYER_TYPES = (
    "full_attention",
    "qwen_sparse_attention",
    "indexed_attention",
)
_QSA_LAYER_TYPES = ("qwen_sparse_attention", "indexed_attention")


def _uses_qsa(config: Qwen4ExpTextConfig) -> bool:
    """Whether the indexed layers carry a QSA indexer (sparse attention)."""
    return getattr(config, "indexer_n_heads", None) is not None


# ``layers.<i>.ple.ple_embedding.ngram_embedding.<shard_j.weight|weight_scale>``
_PLE_TABLE_RE = re.compile(r"layers\.(\d+)\.ple\.ple_embedding\.ngram_embedding\.(.+)")
# ``layers.<i>.self_attn.indexer.*`` (QSA indexer weights)
_INDEXER_RE = re.compile(r"layers\.(\d+)\.self_attn\.indexer\.")


def _is_file_backed(tensor: torch.Tensor) -> bool | None:
    """Whether ``tensor``'s data lies in a file mapping (None if unknown).

    safetensors hands out zero-copy mmap views whose storage reports no
    filename, so look the data pointer up in ``/proc/self/maps`` instead.
    """
    ptr = tensor.data_ptr()
    try:
        with open("/proc/self/maps") as maps:
            for line in maps:
                fields = line.split()
                lo, hi = (int(x, 16) for x in fields[0].split("-"))
                if lo <= ptr < hi:
                    return len(fields) > 5 and fields[5].startswith("/")
    except OSError:
        return None
    return False


def _report_dense_qsa(config: Qwen4ExpTextConfig, max_model_len: int) -> None:
    """Report (once) that checkpoint QSA indexer weights are ignored.

    QSA attends to at most ``indexer_budget`` selected tokens, so dense
    attention is expected to match it only while a sequence fits the budget.
    """
    budget = getattr(config, "indexer_budget", None)
    if budget is not None and max_model_len > budget:
        logger.warning_once(
            "The checkpoint ships QSA indexer weights, but the XPU backend runs "
            "dense attention and ignores them. Dense attention is only expected "
            "to match the sparse model while a sequence fits the indexer budget "
            "(%d tokens); with max_model_len=%d, longer sequences attend to every "
            "token and may diverge from the reference. Use --max-model-len %d to "
            "stay within the budget.",
            budget,
            max_model_len,
            budget,
        )
    else:
        logger.info_once(
            "Ignoring the checkpoint's QSA indexer weights: the XPU backend runs "
            "dense attention, expected to match QSA within max_model_len=%d "
            "(indexer budget %s).",
            max_model_len,
            budget,
        )


class _WeightContainer(nn.Module):
    """Holds a single ``weight`` parameter (no bias, no forward).

    Used for the hyper-connection weights, which are plain BF16 tensors
    (excluded from quantization via ``re:.*hyper_connection.*``) loaded
    directly from the checkpoint. The container exists only so the weight
    sits at the exact ``<name>.weight`` path the checkpoint expects.
    """

    def __init__(self, shape: tuple[int, ...]):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(*shape))


class HyperConnection(nn.Module):
    """Hyper-connection mix/combine block.

    Replaces the standard per-layer RMSNorm + residual add. The state
    between blocks is ``hc`` parallel residual streams ``[T, hc, n_embd]``.

    ``mix`` reads one mixed ``[T, n_embd]`` view:
        grouped-RMSNorm (per stream) -> gamma (``hc_norm``) ->
        low-rank down -> ``silu(x / hc)`` -> up -> sigmoid gate ->
        ``x * gate`` -> mean-collapse the streams.
    It optionally returns ``[T, hc]`` scatter weights (``inject``).

    ``combine`` writes a block output back into the wide state:
        ``residual + block_out * 2 * sigmoid(inject / hc)``.
    The ``2 * sigmoid`` centres the scatter weights on 1, so an
    untrained injection matrix reproduces the plain residual add.

    ``has_inject=False`` is used for the top-level ``hyper_connection_mixer``
    (the final output norm), which has no ``block_inject_weight``.
    """

    def __init__(self, config: Qwen4ExpTextConfig, has_inject: bool = True):
        super().__init__()
        self.hc = config.hc_count
        self.n_embd = config.hidden_size
        self.hc_dim = self.hc * self.n_embd
        self.lowrank = config.hc_lowrank
        self.eps = config.rms_norm_eps
        self.has_inject = has_inject

        self.hc_norm = _WeightContainer([self.hc_dim])
        self.input_mix_weight_down = _WeightContainer([self.lowrank, self.hc_dim])
        self.input_mix_weight_up = _WeightContainer([self.hc_dim, self.lowrank])
        if has_inject:
            self.block_inject_weight = _WeightContainer([self.hc, self.hc_dim])

    def _grouped_rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, hc, n_embd]; normalize each [n_embd] stream independently
        # (reduces over the last dim), matching ggml_rms_norm over ne[0].
        var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps).to(x.dtype)

    def mix(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        # x: [T, hc, n_embd]
        xn = self._grouped_rms_norm(x)
        xn_flat = xn.reshape(xn.shape[0], self.hc_dim)
        # The checkpoint stores the gamma DEVIATION w (mean ~ -0.06); the true
        # RMSNorm gamma is (1 + w). llama.cpp folds this in the GGUF converter
        # ("Gammas were folded to (1 + w) by the converter"), so it uses 1 + w.
        # Compute in fp32 to avoid bf16 precision loss adding 1.0 to small w.
        gamma = (1.0 + self.hc_norm.weight.float()).to(xn_flat.dtype)
        xn_flat = xn_flat * gamma
        lo = torch.nn.functional.linear(xn_flat, self.input_mix_weight_down.weight)
        lo = torch.nn.functional.silu(lo / self.hc)
        gate = torch.sigmoid(
            torch.nn.functional.linear(lo, self.input_mix_weight_up.weight)
        )
        gated = (xn_flat * gate).reshape(xn.shape[0], self.hc, self.n_embd)
        mixed = gated.mean(dim=1)  # collapse streams by their mean -> [T, n_embd]
        inject = None
        if self.has_inject:
            inject = torch.nn.functional.linear(
                xn_flat, self.block_inject_weight.weight
            )  # [T, hc]
        return mixed, inject

    def combine(
        self,
        residual: torch.Tensor,
        block_out: torch.Tensor,
        inject: torch.Tensor,
    ) -> torch.Tensor:
        # residual: [T, hc, n_embd]; block_out: [T, n_embd]; inject: [T, hc]
        w = 2.0 * torch.sigmoid(inject / self.hc)  # [T, hc]
        w = w.unsqueeze(-1)  # [T, hc, 1]
        b = block_out.unsqueeze(1)  # [T, 1, n_embd]
        return residual + b * w


class PLEEmbedding(nn.Module):
    """PLE n-gram hash constants (int64). The embedding table itself is not a
    param; ``Qwen4ExpPLE`` serves it from the checkpoint shards.
    """

    def __init__(self, ngram: int, n_heads: int):
        super().__init__()
        self.register_parameter(
            "layer_multipliers",
            nn.Parameter(torch.empty(ngram, dtype=torch.int64), requires_grad=False),
        )
        self.register_parameter(
            "ngram_heads_offsets",
            nn.Parameter(torch.empty(n_heads, dtype=torch.int64), requires_grad=False),
        )
        self.register_parameter(
            "ngram_heads_vocab_sizes",
            nn.Parameter(torch.empty(n_heads, dtype=torch.int64), requires_grad=False),
        )


class _PLETableFinalizer(QuantizeMethodBase):
    """Runs ``Qwen4ExpPLE.finalize_table`` from vLLM's post-load pass.

    ``Qwen4ExpModel.load_weights`` can run several times, because the loader
    groups the checkpoint stream by consecutive name prefix, so the n-gram
    table can only be validated once every weight has been seen. The CUDA/ROCm
    PLE embedding binds its table from the same hook.
    """

    requires_device_loading = False
    supports_pre_processed_weights = True

    def create_weights(self, layer: nn.Module, *args, **kwargs) -> None:
        raise NotImplementedError

    def apply(self, layer: nn.Module, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        layer.finalize_table()


class Qwen4ExpPLE(nn.Module, MambaBase):
    """Per-Layer Embedding (n-gram table) for Qwen4-Exp, layer 1 only.

    The n-gram table (320M rows: ~95 GiB in BF16, ~48 GiB in FP8) never
    becomes a device param. The checkpoint's ``ngram_embedding.shard_*``
    tensors (BF16, or FP8 with a global ``weight_scale``) are kept as the
    loader's zero-copy mmap views of the checkpoint files, so the table lives
    in the page cache (shared across TP ranks); rows are gathered on host and
    only those move to the device.

    The rest follows the CUDA/ROCm PLE layer and runs the shared kernels:
    16 row ids per token are hashed on device from the step's tokens and the
    request's preceding tokens (``ngram_context``, rebuilt every step from the
    token history by ``Qwen4ExpModelState``); key/value proj -> grouped
    RMSNorm gate -> dilated causal conv (kernel 4, dilation 3) -> added to the
    wide residual ``[T, hc, n_embd]``. The conv history is a vLLM-managed
    short-conv Mamba state, so prefix caching and spec-decode rollback cover
    it like the GDN states.
    """

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.hc = config.hc_count
        self.n_embd = config.hidden_size
        self.hc_dim = self.hc * self.n_embd
        self.eps = config.rms_norm_eps
        self.ngram = config.ngram_size
        self.per_gram = config.heads_per_ngram
        self.n_heads = (self.ngram - 1) * self.per_gram
        self.head_dim = config.ple_embed_dim // self.n_heads
        self.kern = config.ple_conv_kernel_size
        self.dil = self.ngram
        self.hist = (self.kern - 1) * self.dil
        self.eos = int(config.eos_token_id)

        self.prefix = prefix
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.num_spec_tokens = vllm_config.num_speculative_tokens
        self.kv_cache = (torch.tensor([]),)
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        # BF16 weights (unquantized in the checkpoints, like HC).
        self.key_proj = _WeightContainer([self.hc_dim, self.n_embd])
        self.value_proj = _WeightContainer([self.n_embd, self.n_embd])
        self.conv1d = _WeightContainer([self.hc_dim, 1, self.kern])
        self.norm_key = _WeightContainer([self.hc_dim])
        self.norm_query = _WeightContainer([self.hc_dim])
        self.norm_conv = _WeightContainer([self.hc_dim])
        self.ple_embedding = PLEEmbedding(self.ngram, self.n_heads)

        # Checkpoint n-gram table shards (collected at load time, validated by
        # the post-load hook). Dummy loads ship no shards; their lookups
        # return zeros.
        self.quant_method = _PLETableFinalizer()
        self._dummy_load = vllm_config.load_config.load_format == "dummy"
        self._ckpt_shards: dict[int, torch.Tensor] = {}
        self._ckpt_scale: torch.Tensor | None = None
        self._ckpt_dtype: torch.dtype | None = None
        self._shard_starts: torch.Tensor | None = None

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.SHORT_CONV

    @property
    def is_kv_cache_tp_replicated(self) -> bool:
        return True

    def get_attn_backend(self) -> type[PleShortConvAttentionBackend]:
        return PleShortConvAttentionBackend

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=1,
            intermediate_size=self.hc_dim,
            conv_kernel=self.hist + 1,
            num_spec=self.num_spec_tokens,
        )

    def accept_checkpoint_tensor(self, name: str, tensor: torch.Tensor) -> None:
        """Keep one ``ngram_embedding`` checkpoint tensor (``shard_<i>.weight``
        or ``weight_scale``). Shards are held as the loader's views, which are
        mmap'd from the checkpoint files, so no host copy is made."""
        if name == "weight_scale":
            self._ckpt_scale = tensor.detach().to(torch.float32).reshape(-1)
            return
        match = re.fullmatch(r"shard_(\d+)\.weight", name)
        if match is None:
            raise ValueError(f"Unexpected PLE n-gram embedding tensor {name!r}")
        index = int(match.group(1))
        if index in self._ckpt_shards:
            raise ValueError(f"Duplicate PLE n-gram embedding shard {index}")
        if tensor.dim() != 2 or tensor.shape[1] != self.head_dim:
            raise ValueError(
                f"PLE shard {index} has shape {tuple(tensor.shape)}; expected "
                f"[rows, {self.head_dim}]"
            )
        if self._ckpt_dtype is None:
            self._ckpt_dtype = tensor.dtype
        elif tensor.dtype != self._ckpt_dtype:
            raise ValueError(
                f"PLE shard {index} dtype {tensor.dtype} differs from "
                f"{self._ckpt_dtype}"
            )
        self._ckpt_shards[index] = tensor.detach()

    def finalize_table(self) -> None:
        """Validate the n-gram table once all weights are loaded."""
        if not self._ckpt_shards and self._dummy_load:
            return
        if not self._ckpt_shards:
            raise ValueError(
                "The checkpoint ships no PLE n-gram table (`ngram_embedding.shard_*`)"
            )
        indices = sorted(self._ckpt_shards)
        if indices != list(range(len(indices))):
            raise ValueError(
                f"PLE n-gram embedding shards are not contiguous: {indices}"
            )
        rows = torch.tensor([self._ckpt_shards[i].shape[0] for i in indices])
        self._shard_starts = torch.cat((rows.new_zeros(1), rows.cumsum(0)[:-1]))
        total_rows = int(rows.sum())
        offsets = self.ple_embedding.ngram_heads_offsets.cpu()
        vocab_sizes = self.ple_embedding.ngram_heads_vocab_sizes.cpu()
        needed = int((offsets + vocab_sizes).max())
        if total_rows < needed:
            raise ValueError(
                f"PLE n-gram shards hold {total_rows} rows but the n-gram heads "
                f"address {needed}"
            )
        if self._ckpt_dtype == torch.float8_e4m3fn:
            if self._ckpt_scale is None or self._ckpt_scale.numel() != 1:
                raise ValueError("FP8 PLE checkpoint is missing its global scale")
        elif self._ckpt_dtype != torch.bfloat16:
            raise ValueError(f"Unsupported PLE n-gram table dtype {self._ckpt_dtype}")
        table_gib = (
            total_rows * self.head_dim * self._ckpt_shards[0].element_size() / 2**30
        )
        if _is_file_backed(self._ckpt_shards[0]) is False:
            logger.warning(
                "PLE n-gram table shards are not file-backed; the %.1f GiB "
                "table is resident in host memory.",
                table_gib,
            )
        logger.info(
            "PLE n-gram table: %d shards, %d rows, %s, %.1f GiB, served from "
            "the checkpoint files.",
            len(indices),
            total_rows,
            self._ckpt_dtype,
            table_gib,
        )

    def _gather_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """Host-side gather of n-gram rows -> ``[N, head_dim]`` BF16 on CPU."""
        if self._shard_starts is None and self._dummy_load:
            return torch.zeros(rows.numel(), self.head_dim, dtype=torch.bfloat16)
        starts = self._shard_starts
        if rows.numel() and int(rows.min()) < 0:
            raise IndexError("PLE row id must be non-negative")
        shard = torch.searchsorted(starts, rows, right=True) - 1
        width = self.head_dim * self._ckpt_shards[0].element_size()
        out = torch.empty(rows.numel(), width, dtype=torch.uint8)
        # Gather raw bytes: indexing works for every storage dtype (incl. FP8).
        for index in shard.unique().tolist():
            sel = (shard == index).nonzero().squeeze(1)
            local = rows[sel] - starts[index]
            src = self._ckpt_shards[index]
            if int(local.max()) >= src.shape[0]:
                raise IndexError(f"PLE row id out of range for shard {index}")
            out[sel] = src.view(torch.uint8)[local]
        emb = out.view(self._ckpt_dtype)
        if self._ckpt_dtype == torch.float8_e4m3fn:
            # Same dequantization as the CUDA/ROCm FP8 PLE embedding.
            emb = emb.to(torch.bfloat16) * self._ckpt_scale.to(torch.bfloat16)
        return emb

    def lookup(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        """N-gram embeddings ``[T, n_heads * head_dim]`` (BF16, on device).

        Row ids are hashed on device, then the rows are gathered from the
        host-side table and moved to the device.
        """
        emb = self.ple_embedding
        rows = ple_ngram_ids(
            input_ids=input_ids.reshape(-1),
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
            layer_multipliers=emb.layer_multipliers,
            ngram_heads_vocab_sizes=emb.ngram_heads_vocab_sizes,
            ngram_heads_offsets=emb.ngram_heads_offsets,
            eos_token_id=self.eos,
            heads_per_ngram=self.per_gram,
        )
        out = self._gather_rows(rows.reshape(-1).cpu())
        return out.reshape(rows.shape[0], -1).to(input_ids.device)

    def short_conv(
        self,
        inputs: torch.Tensor,
        residual: torch.Tensor,
        outer_residual: torch.Tensor,
    ) -> None:
        """Add ``silu(dilated_conv(inputs)) + outer_residual`` to ``residual``
        in place and update the conv state (same dispatch as CUDA/ROCm)."""
        attn_metadata = get_forward_context().attn_metadata
        metadata = (
            attn_metadata.get(self.prefix) if isinstance(attn_metadata, dict) else None
        )
        if metadata is None:
            # Profiling and dummy runs carry no metadata for this layer.
            residual.add_(outer_residual)
            return
        assert isinstance(metadata, PleShortConvAttentionMetadata)
        conv_state = self.kv_cache[0]
        # Canonicalize both cache layouts to [slot, channel, window].
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        conv_state = conv_state[..., -(self.hist + self.num_spec_tokens) :]
        conv_weights = self.conv1d.weight.squeeze(1).to(inputs.dtype)

        n = metadata.num_actual_tokens
        inputs, residual, outer_residual = inputs[:n], residual[:n], outer_residual[:n]
        conv = dict(conv_state=conv_state, conv_weights=conv_weights, dilation=self.dil)
        has_spec = metadata.spec_sequence_masks is not None
        has_prefill = metadata.num_prefills > 0
        has_decode = metadata.num_decodes > 0
        spec_token_indices = non_spec_token_indices = None
        if has_spec and (has_prefill or has_decode):
            spec_token_indices = metadata.spec_token_indx
            non_spec_token_indices = metadata.non_spec_token_indx

        if has_spec:
            ple_conv(
                inputs,
                residual,
                state_indices=metadata.spec_state_indices_tensor[
                    : metadata.num_spec_decodes
                ],
                outer_residual=outer_residual,
                mode="spec",
                query_start_loc=metadata.spec_query_start_loc,
                num_accepted_tokens=metadata.num_accepted_tokens,
                spec_query_len=metadata.spec_query_len,
                token_indices=spec_token_indices,
                **conv,
            )
        if not (has_prefill or has_decode):
            return

        state_indices = metadata.state_indices_tensor
        if not has_prefill:
            num_decode_rows = (
                non_spec_token_indices.numel()
                if non_spec_token_indices is not None
                else inputs.shape[0]
            )
            ple_conv(
                inputs,
                residual,
                state_indices=state_indices[:num_decode_rows],
                outer_residual=outer_residual,
                mode="decode",
                has_initial_states=metadata.has_initial_states_d,
                token_indices=non_spec_token_indices,
                **conv,
            )
            return

        # Decodes come first, then prefills.
        num_d, num_dt = metadata.num_decodes, metadata.num_decode_tokens
        state_d, state_p = torch.split(
            state_indices, [num_d, metadata.num_prefills], dim=0
        )
        if non_spec_token_indices is None:
            split = [num_dt, metadata.num_prefill_tokens]
            inputs_d, inputs_p = torch.split(inputs, split, dim=0)
            residual_d, residual_p = torch.split(residual, split, dim=0)
            outer_d, outer_p = torch.split(outer_residual, split, dim=0)
            tokens_d = tokens_p = None
        else:
            inputs_d = inputs_p = inputs
            residual_d = residual_p = residual
            outer_d = outer_p = outer_residual
            tokens_d, tokens_p = torch.split(
                non_spec_token_indices, [num_dt, metadata.num_prefill_tokens], dim=0
            )
        if has_decode:
            ple_conv(
                inputs_d,
                residual_d,
                state_indices=state_d,
                outer_residual=outer_d,
                mode="decode",
                has_initial_states=metadata.has_initial_states_d,
                token_indices=tokens_d,
                **conv,
            )
        ple_conv(
            inputs_p,
            residual_p,
            state_indices=state_p,
            outer_residual=outer_p,
            mode="prefill",
            query_start_loc=metadata.query_start_loc_p,
            has_initial_states=metadata.has_initial_states_p,
            token_indices=tokens_p,
            **conv,
        )

    def forward(
        self,
        residual: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        # residual: [T, hc, n_embd] (wide state). Returns the updated wide state.
        T = residual.shape[0]
        hidden = residual.reshape(T, self.hc_dim).contiguous()
        emb = torch.empty(
            T, self.n_heads * self.head_dim, dtype=hidden.dtype, device=hidden.device
        )
        torch.ops.vllm.qwen4_exp_xpu_ple_lookup(
            input_ids, query_start_loc, ngram_context, emb, self.prefix
        )
        key = emb @ self.key_proj.weight.t()  # [T, hc_dim]
        value = emb @ self.value_proj.weight.t()  # [T, n_embd]
        gated, conv_input = torch.ops.vllm.qwen4_exp_xpu_ple_gate(
            key,
            value,
            hidden,
            self.norm_key.weight,
            self.norm_query.weight,
            self.norm_conv.weight,
            self.eps,
        )
        # gated += silu(conv(conv_input)) + hidden, in place.
        torch.ops.vllm.qwen4_exp_ple_short_conv(conv_input, gated, hidden, self.prefix)
        return gated.view(T, self.hc, self.n_embd)


def _ple_layer(layer_name: str) -> Qwen4ExpPLE:
    layer = get_forward_context().no_compile_layers[layer_name]
    if not isinstance(layer, Qwen4ExpPLE):
        raise TypeError(f"{layer_name} is not a Qwen4Exp PLE layer")
    return layer


def qwen4_exp_xpu_ple_lookup(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    # Host-side table gather: keep it out of the compiled graph.
    layer = _ple_layer(layer_name)
    output.copy_(layer.lookup(input_ids, query_start_loc, ngram_context))


def qwen4_exp_ple_short_conv(
    inputs: torch.Tensor,
    residual: torch.Tensor,
    outer_residual: torch.Tensor,
    layer_name: str,
) -> None:
    _ple_layer(layer_name).short_conv(inputs, residual, outer_residual)


def _ple_gate_fake(
    key: torch.Tensor,
    value: torch.Tensor,
    hidden: torch.Tensor,
    norm_key_w: torch.Tensor,
    norm_query_w: torch.Tensor,
    norm_conv_w: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(hidden), torch.empty_like(hidden)


direct_register_custom_op(
    op_name="qwen4_exp_xpu_ple_lookup",
    op_func=qwen4_exp_xpu_ple_lookup,
    mutates_args=["output"],
)
direct_register_custom_op(
    op_name="qwen4_exp_xpu_ple_gate",
    op_func=ple_gate,
    mutates_args=[],
    fake_impl=_ple_gate_fake,
)
# Same name as the CUDA/ROCm op: CompilationConfig lists it as a graph-
# splitting op (the conv-state routing reads per-step request metadata).
direct_register_custom_op(
    op_name="qwen4_exp_ple_short_conv",
    op_func=qwen4_exp_ple_short_conv,
    mutates_args=["residual"],
)


class Qwen4ExpDecoderLayer(nn.Module):
    """A single Qwen4-Exp decoder layer.

    Replaces the standard ``input_layernorm`` / ``post_attention_layernorm``
    + residual add with two :class:`HyperConnection` blocks (one before the
    attention sub-block, one before the MLP sub-block). The wide state
    ``[T, hc, n_embd]`` is threaded through the layer.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        self.hc = config.hc_count

        # PLE (n-gram table) runs at the configured layer BEFORE the attention
        # hyper-connection mix, matching llama.cpp's forward loop. The config
        # ``ple_layer_ids`` is 1-based (e.g. [2]); the checkpoint stores PLE at
        # 0-based index 1 (``layers.1.ple.*``), so convert to 0-based here.
        self.ple = None
        ple_layer_indices = {v - 1 for v in config.ple_layer_ids}
        if self.layer_idx in ple_layer_indices:
            self.ple = Qwen4ExpPLE(config, vllm_config, prefix=f"{prefix}.ple")

        # Hyper-connections replace the per-layer RMSNorm + residual add.
        self.attn_hyper_connection = HyperConnection(config, has_inject=True)
        self.mlp_hyper_connection = HyperConnection(config, has_inject=True)

        if self.layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
                reduce_results=True,
            )
        elif self.layer_type in _QSA_LAYER_TYPES and _uses_qsa(config):
            if topk_indices_buffer is None:
                raise ValueError("QSA layers require a shared top-k buffer")
            self.self_attn = Qwen4ExpQSAAttention(
                vllm_config=vllm_config,
                config=config,
                layer_id=self.layer_idx,
                topk_indices_buffer=topk_indices_buffer,
                quant_config=quant_config,
                reduce_results=True,
                prefix=f"{prefix}.self_attn",
            )
        elif self.layer_type in _FULL_ATTENTION_LAYER_TYPES:
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                reduce_results=True,
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        # Every layer uses the sparse MoE block (512 experts, top-10,
        # shared expert). In the FP8 checkpoint only the routed experts are
        # quantized; the shared expert stays BF16.
        self.mlp = Qwen3NextSparseMoeBlock(
            vllm_config=vllm_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor = None,
        input_ids: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        ngram_context: torch.Tensor | None = None,
        **kwargs: object,
    ):
        # hidden_states: [T, n_embd] (only consumed when residual is None)
        # residual: [T, hc, n_embd] (the wide hyper-connection state)
        if residual is None:
            # First layer: broadcast the embedding to `hc` identical streams.
            residual = hidden_states.unsqueeze(1).expand(-1, self.hc, -1).contiguous()

        # PLE (n-gram table) at the configured layer, before the attention mix.
        if self.ple is not None:
            if input_ids is None or query_start_loc is None or ngram_context is None:
                raise RuntimeError("PLE inputs were not prepared")
            residual = self.ple(residual, input_ids, query_start_loc, ngram_context)

        # Attention sub-block.
        cur, inject = self.attn_hyper_connection.mix(residual)
        if self.layer_type == "linear_attention":
            cur = self.linear_attn(hidden_states=cur)
        else:
            cur = self.self_attn(hidden_states=cur, positions=positions)
        residual = self.attn_hyper_connection.combine(residual, cur, inject)

        # MLP sub-block.
        cur, inject = self.mlp_hyper_connection.mix(residual)
        cur = self.mlp(cur)
        residual = self.mlp_hyper_connection.combine(residual, cur, inject)

        # Return the mean-collapsed view (only used if the next layer has
        # residual=None) + the wide state.
        return residual.mean(dim=1), residual


class Qwen4ExpMixtureOfExperts(MixtureOfExperts):
    """MoE metadata mixin for Qwen4-Exp (matches ``Qwen4ExpDecoderLayer``)."""

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_moe_parameters(self):
        self.moe_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, Qwen4ExpDecoderLayer) and isinstance(
                layer.mlp, Qwen3NextSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError("No Qwen4Exp layer found in the model.layers.")

        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is (3, seq_len) when mrope is enabled, else (seq_len,).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "query_start_loc": 0,
        "ngram_context": 0,
    }
)
class Qwen4ExpModel(Qwen3NextModel):
    # The GDN in_proj checkpoints are stored separately (Qwen3.5 layout);
    # fuse them on top of the Qwen3-Next QKV / gate_up mapping.
    hf_to_vllm_mapper = Qwen3NextModel.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_stacked={
            ".in_proj_qkv": (".in_proj_qkvz", (0, 1, 2)),
            ".in_proj_z": (".in_proj_qkvz", 3),
            ".in_proj_b": (".in_proj_ba", 0),
            ".in_proj_a": (".in_proj_ba", 1),
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3NextModel, self).__init__()

        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.vocab_size = config.vocab_size
        self.hc = config.hc_count
        self.max_model_len = vllm_config.model_config.max_model_len

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        # One top-k buffer shared by all QSA layers. It lives on the model, not in a
        # local, so a draft model can share it.
        self.topk_indices_buffer: torch.Tensor | None = None
        if _uses_qsa(config) and any(t in _QSA_LAYER_TYPES for t in config.layer_types):
            self.topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                get_qsa_topk_buffer_width(config),
                dtype=torch.int32,
            )

        def get_layer(prefix: str):
            return Qwen4ExpDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )

        # Custom intermediate-tensors factory: the residual is the wide
        # hyper-connection state [T, hc, n_embd].
        def _make_empty_intermediate_tensors(num_tokens, dtype, device):
            return IntermediateTensors(
                {
                    "hidden_states": torch.empty(
                        num_tokens,
                        config.hidden_size,
                        dtype=dtype,
                        device=device,
                    ),
                    "residual": torch.empty(
                        num_tokens,
                        config.hc_count,
                        config.hidden_size,
                        dtype=dtype,
                        device=device,
                    ),
                }
            )

        self.make_empty_intermediate_tensors = _make_empty_intermediate_tensors

        # The top-level hyper-connection mixer acts as the final output norm
        # (there is no separate RMSNorm).
        if get_pp_group().is_last_rank:
            self.hyper_connection_mixer = HyperConnection(config, has_inject=False)
        else:
            self.hyper_connection_mixer = PPMissingLayer()

        # MTP draft model consumes the target's wide hyper-connection
        # residual state [T, hc, n_embd] (flattened to [T, hc * n_embd]).
        # Allocate a buffer when speculative decoding (MTP) is enabled so the
        # drafter can read the pre-mixer residual after each target step.
        spec_config = vllm_config.speculative_config
        needs_mtp_hidden = spec_config is not None and (
            spec_config.use_eagle() or spec_config.uses_draft_model()
        )
        if get_pp_group().is_last_rank and needs_mtp_hidden:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.hc * config.hidden_size,
                dtype=vllm_config.model_config.dtype,
                device=current_platform.device_type,
            )
        else:
            self._mtp_hidden_buffer = None

        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        ngram_context: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        # query_start_loc / ngram_context: per-request PLE n-gram inputs from
        # Qwen4ExpModelState.
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                input_ids=input_ids,
                query_start_loc=query_start_loc,
                ngram_context=ngram_context,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        # Expose the wide residual state to the MTP draft model (flattened
        # [T, hc * n_embd]). Valid after each target step.
        if getattr(self, "_mtp_hidden_buffer", None) is not None:
            _nt = residual.shape[0]
            self._mtp_hidden_buffer[:_nt].copy_(residual.flatten(1))

        # Final output norm = the top-level hyper-connection mixer.
        hidden_states, _ = self.hyper_connection_mixer.mix(residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The PLE n-gram table is not a param: its shards (and FP8 scale) are
        # handed to the owning Qwen4ExpPLE, which serves rows from them on
        # host and validates them in its post-load hook (this method can run
        # several times, once per run of consecutive checkpoint names).
        # Indexer tensors of layers that run dense (no QSA indexer in the
        # config, or a layer type that is not indexed) are dropped and
        # reported.
        saw_indexer = False

        def route(weights: Iterable[tuple[str, torch.Tensor]]):
            nonlocal saw_indexer
            for name, tensor in weights:
                match = _INDEXER_RE.match(name)
                if match is not None and not isinstance(
                    getattr(self.layers[int(match.group(1))], "self_attn", None),
                    Qwen4ExpQSAAttention,
                ):
                    saw_indexer = True
                    continue
                match = _PLE_TABLE_RE.fullmatch(name)
                if match is not None:
                    ple = getattr(self.layers[int(match.group(1))], "ple", None)
                    if ple is not None:
                        ple.accept_checkpoint_tensor(match.group(2), tensor)
                    continue
                yield name, tensor

        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(route(weights), mapper=self.hf_to_vllm_mapper)
        if saw_indexer:
            _report_dense_qsa(self.config, self.max_model_len)
        return loaded


class Qwen4ExpForCausalLM(
    nn.Module,
    HasInnerState,
    IsHybrid,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
    Qwen4ExpMixtureOfExperts,
):
    # The checkpoint is the Qwen3-VL wrapper layout: the language model sits
    # under ``model.language_model.`` and the (unused) vision tower under
    # ``model.visual.``. Strip the ``language_model`` prefix and skip the
    # vision tower + MTP head.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "model.",
        }
    )

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        # GDN fused projections (separate in the checkpoint).
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config

        scheduler_config = vllm_config.scheduler_config
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = Qwen4ExpModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters.
        self.set_moe_parameters()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            query_start_loc=kwargs.get("query_start_loc"),
            ngram_context=kwargs.get("ngram_context"),
        )
        return hidden_states

    @staticmethod
    def get_model_state_cls():
        from .model_state import Qwen4ExpModelState

        return Qwen4ExpModelState

    @classmethod
    def get_ple_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_ple_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, ...], ...]:
        hf_config = vllm_config.model_config.hf_text_config
        conv_state_len = (hf_config.ple_conv_kernel_size - 1) * hf_config.ngram_size
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=1,
            intermediate_size=hf_config.hidden_size * hf_config.hc_count,
            conv_kernel=conv_state_len + 1,
            num_spec=vllm_config.num_speculative_tokens,
        )

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    @classmethod
    def get_mamba_state_copy_funcs(
        cls,
        mamba_types: set[MambaAttentionBackendEnum],
    ) -> MambaStateCopyFuncsByType:
        copy_funcs_by_type = {
            MambaAttentionBackendEnum.GDN_ATTN: cls.get_mamba_state_copy_func(),
            MambaAttentionBackendEnum.SHORT_CONV: (
                MambaStateCopyFuncCalculator.short_conv_state_copy_func()
            ),
        }
        missing_types = mamba_types - copy_funcs_by_type.keys()
        assert not missing_types, f"missing state copy funcs for {missing_types}"
        return {t: copy_funcs_by_type[t] for t in mamba_types}

    @classmethod
    def get_mamba_specs_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[MambaSpec, ...]:
        """All Mamba state layouts: the GDN layers and the PLE short conv."""
        return (
            MambaSpec(
                shapes=cls.get_mamba_state_shape_from_config(vllm_config),
                dtypes=cls.get_mamba_state_dtype_from_config(vllm_config),
                block_size=-1,
            ),
            MambaSpec(
                shapes=cls.get_ple_mamba_state_shape_from_config(vllm_config),
                dtypes=cls.get_ple_mamba_state_dtype_from_config(vllm_config),
                block_size=-1,
                tp_replicated=True,
            ),
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        """Wide hyper-connection residual buffer [T, hc * n_embd] for the MTP
        draft model. Populated by ``Qwen4ExpModel.forward``; valid after each
        target step. Returns None when MTP is not enabled."""
        return getattr(self.model, "_mtp_hidden_buffer", None)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        """M-RoPE positions for the text-only model.

        The checkpoint's ``text_config`` ships ``mrope_section`` /
        ``mrope_interleaved`` (inherited from the Qwen3.5 vision lineage), so
        vLLM treats this as an M-RoPE model and the model runner calls this
        method. For pure text there are no multimodal features, so all three
        M-RoPE axes (T/H/W) receive the same sequential position and the
        position delta is zero. This mirrors the text-only branch of
        ``Qwen2VLForConditionalGeneration.get_mrope_input_positions``.
        """
        del mm_features  # text-only: no multimodal grids
        seq_len = len(input_tokens)
        llm_positions = (
            torch.arange(seq_len, dtype=torch.int64).unsqueeze(0).repeat(3, 1)
        )
        mrope_position_delta = 0
        return llm_positions, mrope_position_delta

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # PLE table shards and QSA indexer tensors are handled by
        # ``Qwen4ExpModel.load_weights``; the MTP head and vision tower are not
        # part of the text-only model.
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_prefixes=["mtp.", "model.visual."],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class Qwen4ExpProcessingInfo(Qwen3VLProcessingInfo):
    """Multimodal processing info for Qwen4-Exp.

    Reuses the Qwen3-VL processor (``Qwen3VLProcessor``) and only overrides
    the HF config class so the processor resolves the ``qwen4_exp`` config.
    """

    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen4ExpConfig)


class Qwen4ExpMoeMixtureOfExperts(MixtureOfExperts):
    """MoE metadata mixin for the Qwen4-Exp multimodal wrapper.

    Mirrors :class:`Qwen4ExpMixtureOfExperts` but resolves the decoder layers
    through ``self.language_model.model`` (the wrapper layout, where the text
    model is nested under ``language_model``).
    """

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.language_model.model.layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_moe_parameters(self):
        self.moe_layers = []
        example_moe = None
        for layer in self.language_model.model.layers:
            if isinstance(layer, Qwen4ExpDecoderLayer) and isinstance(
                layer.mlp, Qwen3NextSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError(
                "No Qwen4Exp layer found in the language_model.model.layers."
            )

        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen4ExpProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen4ExpForConditionalGeneration(
    Qwen3VLForConditionalGeneration,
    IsHybrid,
    Qwen4ExpMoeMixtureOfExperts,
):
    """Multimodal (vision + text) wrapper for Qwen4-Exp.

    Wraps the Qwen3-VL vision tower (``Qwen3_VisionTransformer``) around the
    :class:`Qwen4ExpForCausalLM` text model. The checkpoint's vision config is
    byte-identical to the Qwen3-VL ViT (27 blocks, hidden 1152, patch 16,
    spatial-merge 2, out 2560, no deepstack), so the Qwen3-VL vision tower,
    processor, and MRoPE wiring are reused verbatim.
    """

    supports_multimodal_pruning = True
    is_3d_moe_weight: bool = True

    packed_modules_mapping = Qwen3VLForConditionalGeneration.packed_modules_mapping | {
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # protocols have no __init__, so use nn.Module.__init__ directly.
        nn.Module.__init__(self)
        config: Qwen4ExpConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self._init_video_pruning(multimodal_config)
        self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)

        # Deepstack is unused by this checkpoint (``deepstack_visual_indexes``
        # is empty); guard on a non-empty list so the deepstack code paths are
        # skipped.
        self.use_deepstack = (
            hasattr(config.vision_config, "deepstack_visual_indexes")
            and len(config.vision_config.deepstack_visual_indexes) > 0
        )
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen4ExpForCausalLM(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters (expert parallelism / EPLB).
        self.set_moe_parameters()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        # Override the base forward: the Qwen4-Exp text model's forward does
        # not accept ``deepstack_input_embeds`` (this checkpoint has no
        # deepstack), so drop that kwarg.
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            query_start_loc=kwargs.get("query_start_loc"),
            ngram_context=kwargs.get("ngram_context"),
        )
        return hidden_states

    @staticmethod
    def get_model_state_cls():
        return Qwen4ExpForCausalLM.get_model_state_cls()

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        """Forward the MTP hook to the inner text model.

        ``_target_feeds_hc_residual()`` in the speculator probes the
        *top-level* architecture class, which for this checkpoint is the
        VL wrapper. Without this delegation the hook is invisible and the
        drafter is fed the collapsed hidden state instead of the wide
        hyper-connection residual. Mirrors the amd/nvidia implementations.
        """
        return self.language_model.get_mtp_target_hidden_states()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Load the vision tower (``model.visual.*`` -> ``visual.*``) and the
        # text model (``model.language_model.*`` -> ``language_model.model.*``)
        # via the inherited Qwen3-VL weight mapper. Skip the MTP drafter head
        # (``mtp.*``); PLE table shards and QSA indexer tensors are handled by
        # ``Qwen4ExpModel.load_weights``.
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_prefixes=["mtp."],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        return Qwen4ExpForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return Qwen4ExpForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return Qwen4ExpForCausalLM.get_mamba_state_copy_func()

    @classmethod
    def get_mamba_state_copy_funcs(
        cls,
        mamba_types: set[MambaAttentionBackendEnum],
    ) -> MambaStateCopyFuncsByType:
        return Qwen4ExpForCausalLM.get_mamba_state_copy_funcs(mamba_types)

    @classmethod
    def get_mamba_specs_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[MambaSpec, ...]:
        return Qwen4ExpForCausalLM.get_mamba_specs_from_config(vllm_config)


__all__ = [
    "HyperConnection",
    "Qwen4ExpDecoderLayer",
    "Qwen4ExpMixtureOfExperts",
    "Qwen4ExpMoeMixtureOfExperts",
    "Qwen4ExpModel",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpProcessingInfo",
    "Qwen4ExpForConditionalGeneration",
]
