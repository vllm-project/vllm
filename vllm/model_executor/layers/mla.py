# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch

import vllm.envs as envs
from vllm.config import CacheConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.utils.torch_utils import current_stream, direct_register_custom_op

logger = init_logger(__name__)

# Token threshold for multi-stream indexer overlap.
# Disables multi-stream for batches > 1024 to avoid SM contention.
_INDEXER_STREAM_TOKEN_THRESHOLD = 1024


@dataclass
class MLAModules:
    """Modules used in MLA."""

    kv_a_layernorm: torch.nn.Module
    kv_b_proj: torch.nn.Module
    rotary_emb: torch.nn.Module
    o_proj: torch.nn.Module
    fused_qkv_a_proj: torch.nn.Module | None
    kv_a_proj_with_mqa: torch.nn.Module | None
    q_a_layernorm: torch.nn.Module | None
    q_b_proj: torch.nn.Module | None
    q_proj: torch.nn.Module | None
    indexer: torch.nn.Module | None
    is_sparse: bool
    topk_indices_buffer: torch.Tensor | None
    indexer_rotary_emb: torch.nn.Module | None = None
    alt_stream: torch.cuda.Stream | None = None


class _WkForkModule(torch.nn.Module):
    """Compiled module for wk_weights_proj+k_norm on alt_stream.

    Wraps the indexer's fused wk_weights_proj and k_norm into a single
    compilation unit.  When compiled with torch.compile the operations
    benefit from Inductor optimizations:
      - wk_weights_proj:  single fused GEMM for wk + weights_proj
      - k_norm:  operator fusion with surrounding ops

    The compiled module is called inside the mla_wk_fork custom op,
    which runs it on alt_stream concurrent with QKV-A on the main
    stream.

    Returns a concatenated ``[k, raw_weights]`` tensor; the join
    caller splits it back using known ``wk_dim`` and ``weights_dim``.

    Sub-modules are stored via ``object.__setattr__`` so they do NOT
    appear in ``_modules`` / ``state_dict()``.  This prevents:
      1. Duplicate parameter entries (they are shared with Indexer).
      2. State-dict key mismatches during weight loading.
      3. ``isinstance`` false-positives when tests use MagicMock.
    """

    def __init__(self, wk_weights_proj, k_norm, head_dim):
        super().__init__()
        object.__setattr__(self, "wk_weights_proj", wk_weights_proj)
        object.__setattr__(self, "k_norm", k_norm)
        object.__setattr__(self, "head_dim", head_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        kw, _ = self.wk_weights_proj(hidden_states)
        k = kw[:, : self.head_dim]
        raw_weights = kw[:, self.head_dim :]
        k = self.k_norm(k)
        return torch.cat([k, raw_weights], dim=-1)


# ---- Multi-Stream wk_weights_proj Overlap Custom Ops ----
#
# Two minimal custom ops overlap wk_weights_proj+k_norm with QKV-A:
#   mla_wk_fork: launches a COMPILED wk_weights_proj+k_norm module on
#                alt_stream (concurrent with QKV-A on main)
#   mla_wk_join: waits for alt_stream, returns pre-computed
#                [k | raw_weights] concatenated tensor
#
# CRITICAL DESIGN PRINCIPLES:
#   1. ALL indexer GEMMs (wq_b) and q_b_proj MUST stay inside the
#      main torch.compile graph.
#   2. Fork operations MUST ALSO be compiled — running them eagerly
#      loses operator fusion and kernel selection overhead.
#   3. The fix: a SEPARATELY torch.compile'd _WkForkModule wraps
#      wk_weights_proj+k_norm.  The compiled module is called inside
#      the fork custom op on alt_stream.
#
# WHY wk_weights_proj+k_norm:
#   The fused wk_weights_proj GEMM depends ONLY on hidden_states (the
#   layer input).  It can start at the VERY BEGINNING of the forward
#   pass, concurrent with the QKV-A GEMM on the main stream.
#
#   Alt stream (compiled):  wk_weights_proj fused GEMM + k_norm
#   Main stream (compiled): QKV-A + Q-A LN + Q-B proj + kv preprocess
#                           + RoPE
#   Alt < Main → fork is completely hidden!
#
# The indexer call stays INLINE in forward() (traced by torch.compile).
# Indexer.forward() receives pre-computed k via precomputed_k and raw
# weights via precomputed_weights, skipping its own wk_weights_proj
# and k_norm.  The remaining indexer GEMM (wq_b) and
# sparse_attn_indexer stay in the compiled graph.
#
# Pattern EXTENDS MoE shared expert streaming (default_moe_runner.py):
#   1. Register the layer in static_forward_context during __init__
#   2. Custom ops retrieve the layer by name from forward_context
#   3. Stream fork/join happens inside the custom ops (opaque)
#   4. Fake implementations provide output shape for symbolic execution
#   5. NOT in _attention_ops — opaque nodes inside compiled region
#   6. tags=(torch.Tag.needs_fixed_stride_order,) prevents Inductor
#      stride conversion overhead
#
# DIFFERENCE from MoE: the MoE shared expert runs EAGERLY inside its
# custom op.  Here we add a SEPARATE torch.compile unit (_WkForkModule)
# for the fork operations.  This is a novel extension; a graceful
# fallback to eager is included in case torch.compile fails.
#
# Fork/Join symmetry:
#   The fork sets wrapper._wk_forked = True when multi-stream is used.
#   The join checks this flag to decide whether to wait_stream.  This
#   ensures fork and join ALWAYS agree on whether multi-stream is active.


def _mla_wk_fork(
    hidden_states: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    """Launch compiled wk_weights_proj+k_norm on alt_stream.

    Returns a clone of hidden_states to establish a data dependency
    while obeying PyTorch's custom-op contract: outputs MUST NOT alias
    inputs.  Returning the input directly caused undefined behaviour in
    the Inductor buffer-assignment pass (incorrect buffer reuse in the
    generated code) and triggered a CUDA-graph capture error.

    Stores the concatenated [k, raw_weights] result in
    ``wrapper._fork_result`` for the join op.

    The fork calls ``wrapper._compiled_fork_ops`` — a separately
    torch.compile'd _WkForkModule — so that the operations benefit
    from Inductor optimisations (operator fusion, kernel selection)
    even when running on the alt_stream.
    """
    wrapper = get_forward_context().no_compile_layers[layer_name]
    indexer = wrapper.indexer

    if indexer is None or not wrapper.is_sparse:
        wrapper._wk_forked = False
        # Clone to satisfy the custom-op no-alias contract.
        # The fake impl returns torch.empty_like (new tensor),
        # so the real impl must also return a non-aliasing tensor.
        return hidden_states.clone()

    use_multi_stream = (
        wrapper.alt_stream is not None
        and not envs.VLLM_DISABLE_INDEXER_STREAM
        and hidden_states.shape[0] <= _INDEXER_STREAM_TOKEN_THRESHOLD
    )

    fork_ops = wrapper._compiled_fork_ops

    if use_multi_stream:
        main_stream = current_stream()
        alt_stream = wrapper.alt_stream

        # Prevent GC from freeing hidden_states while alt_stream reads it.
        hidden_states.record_stream(alt_stream)

        # alt_stream waits for hidden_states to be ready on main.
        alt_stream.wait_stream(main_stream)

        # Launch compiled wk_weights_proj+k_norm on alt_stream
        # (concurrent with QKV-A on main).
        with torch.cuda.stream(alt_stream):
            wrapper._fork_result = fork_ops(hidden_states)

        wrapper._wk_forked = True
    else:
        # Sequential: run compiled fork ops on main stream.
        wrapper._fork_result = fork_ops(hidden_states)
        wrapper._wk_forked = False

    # Clone to satisfy the custom-op no-alias contract.
    # The clone is a lightweight memcpy (e.g. ~14 KB for decode
    # batch_size=1 with hidden_size=7168 in bf16).  Both streams
    # read the original hidden_states concurrently; the clone
    # provides a separate buffer for downstream compiled code
    # (QKV-A on main stream) so the Inductor's buffer-liveness
    # analysis stays correct.
    return hidden_states.clone()


def _mla_wk_fork_fake(
    hidden_states: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="mla_wk_fork",
    op_func=_mla_wk_fork,
    mutates_args=[],
    fake_impl=_mla_wk_fork_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def _mla_wk_join(
    hidden_states: torch.Tensor,
    layer_name: str,
    join_dim: int,
) -> torch.Tensor:
    """Get pre-computed [k, raw_weights], waiting for alt_stream if needed.

    Returns the concatenated tensor stored by ``_mla_wk_fork``.
    Shape: ``[num_tokens, join_dim]`` where ``join_dim = wk_dim + weights_dim``.
    Only waits if the fork op set ``wrapper._wk_forked = True``,
    ensuring symmetric fork/join behaviour.
    """
    wrapper = get_forward_context().no_compile_layers[layer_name]

    # Check the flag set by fork — guarantees fork/join symmetry.
    if getattr(wrapper, "_wk_forked", False):
        main_stream = current_stream()
        main_stream.wait_stream(wrapper.alt_stream)
        wrapper._wk_forked = False

    # Return the concatenated [k, raw_weights] produced by the fork.
    # The caller splits using known wk_dim and weights_dim.
    return wrapper._fork_result


def _mla_wk_join_fake(
    hidden_states: torch.Tensor,
    layer_name: str,
    join_dim: int,
) -> torch.Tensor:
    return torch.empty(
        hidden_states.shape[0],
        join_dim,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )


direct_register_custom_op(
    op_name="mla_wk_join",
    op_func=_mla_wk_join,
    mutates_args=[],
    fake_impl=_mla_wk_join_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


# --8<-- [start:multi_head_latent_attention]
@PluggableLayer.register("multi_head_latent_attention")
class MultiHeadLatentAttentionWrapper(PluggableLayer):
    """Pluggable MLA layer which allows OOT backends to add
    custom implementations of the outer MLA layer (including rope & o_proj).
    Note that currently oot platforms can still use CustomOp.register_oot to
    replace MLA layer entirely, although we use PluggableLayer to register
    this layer now.

    This class takes positions and hidden_states as input.
    The input tensors can either contain prefill tokens or decode tokens.
    The class does the following:

    1. MLA Preprocess.
    2. Perform multi-head attention to prefill tokens and
       multi-query attention to decode tokens separately.
    3. Return the output tensor.
    """

    # --8<-- [end:multi_head_latent_attention]

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        mla_modules: MLAModules,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.fused_qkv_a_proj = mla_modules.fused_qkv_a_proj
        self.kv_a_proj_with_mqa = mla_modules.kv_a_proj_with_mqa
        self.q_a_layernorm = mla_modules.q_a_layernorm
        self.q_b_proj = mla_modules.q_b_proj
        self.q_proj = mla_modules.q_proj
        self.kv_a_layernorm = mla_modules.kv_a_layernorm
        self.kv_b_proj = mla_modules.kv_b_proj
        self.rotary_emb = mla_modules.rotary_emb
        self.o_proj = mla_modules.o_proj
        self.indexer = mla_modules.indexer
        self.indexer_rope_emb = mla_modules.indexer_rotary_emb
        self.is_sparse = mla_modules.is_sparse
        self.alt_stream = mla_modules.alt_stream
        # Flag for symmetric fork/join. Set by _mla_wk_fork, checked by
        # _mla_wk_join. Ensures join only waits when fork actually
        # launched work on alt_stream.
        self._wk_forked = False

        if self.indexer is not None:
            assert hasattr(self.indexer, "topk_tokens")
            self.topk_tokens = self.indexer.topk_tokens
            self.topk_indices_buffer = mla_modules.topk_indices_buffer
            # Store dimensions for the fork/join custom ops.
            # wk_dim: output dimension of indexer wk (head_dim=128)
            # weights_dim: output dimension of indexer weights_proj (n_head=64)
            # join_dim: total concatenated dim returned by mla_wk_join
            self.wk_dim = self.indexer.head_dim
            self.weights_dim = self.indexer.n_head
            self.join_dim = self.wk_dim + self.weights_dim

            # Compile wk_weights_proj+k_norm as a SEPARATE torch.compile
            # unit.  The compiled module runs on alt_stream inside the
            # mla_wk_fork custom op, concurrent with QKV-A on main.
            # Uses object.__setattr__ to avoid registering as a sub-module
            # (prevents state_dict / weight-loading duplication).
            #
            # NOTE: This EXTENDS the MoE shared-expert streaming pattern
            # (default_moe_runner.py) — the MoE pattern runs shared experts
            # EAGERLY, while we add a separate torch.compile unit for the
            # fork ops.  Graceful fallback to eager if compilation fails.
            _fork_mod = _WkForkModule(
                self.indexer.wk_weights_proj,
                self.indexer.k_norm,
                self.indexer.head_dim,
            )
            try:
                _compiled = torch.compile(_fork_mod, dynamic=True)
            except Exception:
                logger.warning(
                    "Failed to compile MLA fork ops for layer %s, "
                    "falling back to eager execution.",
                    prefix,
                )
                _compiled = _fork_mod
            object.__setattr__(
                self,
                "_compiled_fork_ops",
                _compiled,
            )

        self.mla_attn = MLAAttention(
            num_heads=self.num_heads,
            scale=scale,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            kv_b_proj=self.kv_b_proj,
            use_sparse=self.is_sparse,
            indexer=self.indexer,
        )

        self.prefix = prefix

        # Register in static_forward_context so the fork/join custom ops
        # (mla_wk_fork, mla_wk_join) can retrieve this wrapper.
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_c = None
        kv_lora = None

        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None, (
                "fused_qkv_a_proj is required when q_lora_rank is not None"
            )
            assert self.q_a_layernorm is not None, (
                "q_a_layernorm is required when q_lora_rank is not None"
            )
            assert self.q_b_proj is not None, (
                "q_b_proj is required when q_lora_rank is not None"
            )

            # Fork: launch wk_weights_proj+k_norm on alt_stream,
            # concurrent with QKV-A. Opaque to torch.compile.
            # Fused GEMM hidden behind QKV-A+Q-A LN+Q-B on main.
            # All other GEMMs stay INSIDE torch.compile scope.
            hidden_states = torch.ops.vllm.mla_wk_fork(
                hidden_states,
                self.prefix,
            )

            # QKV-A GEMM on main stream — COMPILED, concurrent with wk.
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_c = self.q_a_layernorm(q_c)

            # q_b_proj on main stream — INSIDE torch.compile scope.
            q = self.q_b_proj(q_c)[0]
        else:
            assert self.kv_a_proj_with_mqa is not None, (
                "kv_a_proj_with_mqa is required when q_lora_rank is None"
            )
            assert self.q_proj is not None, (
                "q_proj is required when q_lora_rank is None"
            )
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = self.q_proj(hidden_states)[0]

        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_heads, self.qk_head_dim)
        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        if self.rotary_emb is not None:
            q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                positions, q[..., self.qk_nope_head_dim :], k_pe
            )

        # Join wk_weights_proj + run indexer INLINE (COMPILED on main).
        # The indexer GEMM (wq_b) stays in torch.compile scope.
        # wk_weights_proj+k_norm run on alt_stream (hidden behind QKV-A).
        # sparse_attn_indexer remains a PIECEWISE split point (as original).
        if self.indexer is not None and self.is_sparse:
            k_weights = torch.ops.vllm.mla_wk_join(
                hidden_states,
                self.prefix,
                self.join_dim,
            )
            # Split the concatenated join result into k and raw_weights.
            k_pre, weights_pre = k_weights.split(
                [self.wk_dim, self.weights_dim],
                dim=-1,
            )
            self.indexer(
                hidden_states,
                q_c,
                positions,
                self.indexer_rope_emb,
                precomputed_k=k_pre,
                precomputed_weights=weights_pre,
            )

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        attn_out = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(
                hidden_states.shape[0],
                self.num_heads * self.v_head_dim,
            ),
        )

        return self.o_proj(attn_out)[0]
