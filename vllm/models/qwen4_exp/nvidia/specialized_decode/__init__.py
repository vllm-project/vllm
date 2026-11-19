# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp specialized decode on NVIDIA GPUs: a decode executor over the
stock backbone, enabled by
`additional_config={"decode_backend": "latency_specialized"}`.
`additional_config["decode_metadata_check_steps"]` (default 0) checks the
first that many replays of every uniform decode shape's attention metadata
against the stock build, and `additional_config["decode_metadata_overlap"]`
(default true) runs those replays on a side stream that the forward graph
waits for before its first read of the metadata (`runtime/decode_metadata.py`);
false replays them on the runner's stream before the forward graph.

The executor runs the target model's FULL-graph pure-decode steps, and the MTP
draft model's FULL-graph draft passes of them start with a fused front
(`runtime/draft_front.py`); every other step runs the stock forward. Their
later steps issue cross-rank collectives, so the deployment check below is
collective over the TP group and every per-step decision reads host state that
all TP ranks share.

At TP4 the target's logits are gathered by `logits/gather.py` whenever the
setting is on, whatever the speculative config. Only the draft passes' token
choice depends on it: they choose their tokens with `logits/choice.py` (for
up to `min(max_num_seqs, MAX_DECODE_REQUESTS)` requests) only when the
speculative config sets `"use_local_argmax_reduction": true` (Model Runner V2
then asks the draft model for its tokens instead of its logits); without it
the token choice is neither built nor warmed up, and they keep the stock
choice (a logits all-gather and `greedy_argmax`), which the check below logs
once at startup.

`additional_config["draft_head_fp8_prefilter"]` (default false) builds, at
TP1, the exact FP8-prefiltered draft head (`logits/draft_head.py`): draft
passes of up to 8 rows find their greedy tokens from an FP8 copy of the LM
head and rescore only the entries that could win with the BF16 head, with the
same tokens and about 28 to 35% sooner on GB300. The FP8 copy and its
statistics take 0.65 GB, which memory profiling counts, so the KV cache
shrinks by as much, and only draft passes of at most 8 rows (decode steps of
at most 8 requests) use it: turn it on for TP1 deployments that run at low
concurrency. It needs `latency_specialized`, tensor parallel size 1,
`use_local_argmax_reduction` and a drafter that shares the target's BF16
head (no `draft_lm_head_quantization`), and raises at startup otherwise.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.config import CompilationMode, CUDAGraphMode, KVTransferConfig, VllmConfig
from vllm.config.kernel import NATIVE_MEGA_MOE_BACKENDS
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.platforms import current_platform

from ...common.hyperconnection import is_hc_sequence_parallel_enabled

logger = init_logger(__name__)

# Requests of the largest decode step the executor runs, by tensor parallel
# size: the highest concurrency it serves at each.
MAX_DECODE_REQUESTS = {1: 64, 4: 64}

# The hyper-connection shape the read kernels (``hc/``) are built for:
# Qwen3.8-Flash-Next's four streams of 2560 values and low rank 320.
HC_STREAMS = 4
HC_HIDDEN = 2560
HC_LOW_RANK = 320


def max_decode_rows(tensor_parallel_size: int, tokens_per_request: int) -> int:
    """Rows of the largest decode step the executor runs at TP size
    `tensor_parallel_size` (1 or 4): `MAX_DECODE_REQUESTS` requests of
    `tokens_per_request` rows (the MTP draft tokens plus one), so 256 at
    both with three draft tokens."""
    return MAX_DECODE_REQUESTS[tensor_parallel_size] * tokens_per_request


@dataclass(frozen=True)
class DecodeConfig:
    """The specialized decode settings of one deployment."""

    # Rows of one request in a decode step: the MTP draft tokens plus one.
    tokens_per_request: int
    # Rows of the largest decode step the executor runs (max_decode_rows).
    max_rows: int
    # Replays of every uniform decode shape's attention metadata checked
    # against the stock build.
    metadata_check_steps: int = 0
    # Overlap those replays with the forward graph (decode_metadata_overlap).
    metadata_overlap: bool = True
    # Build the FP8-prefiltered draft head (draft_head_fp8_prefilter).
    draft_head_fp8: bool = False


def _metadata_check_steps(extra: object) -> int:
    """`additional_config["decode_metadata_check_steps"]`: a non-negative
    integer, 0 when unset."""
    steps = (
        extra.get("decode_metadata_check_steps", 0) if isinstance(extra, dict) else 0
    )
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("decode_metadata_check_steps must be a non-negative integer")
    return steps


def _metadata_overlap(extra: object) -> bool:
    """`additional_config["decode_metadata_overlap"]`: a boolean, true when
    unset."""
    value = (
        extra.get("decode_metadata_overlap", True) if isinstance(extra, dict) else True
    )
    if not isinstance(value, bool):
        raise ValueError("decode_metadata_overlap must be true or false")
    return value


def _draft_head_fp8(extra: object) -> bool:
    """`additional_config["draft_head_fp8_prefilter"]`: a boolean, false when
    unset."""
    value = (
        extra.get("draft_head_fp8_prefilter", False)
        if isinstance(extra, dict)
        else False
    )
    if not isinstance(value, bool):
        raise ValueError("draft_head_fp8_prefilter must be true or false")
    return value


def _draft_head_unsupported(vllm_config: VllmConfig) -> list[str]:
    """Every requirement of `draft_head_fp8_prefilter` beyond the specialized
    decode's that this deployment misses (config only)."""
    spec = vllm_config.speculative_config
    assert spec is not None
    requirements = {
        "tensor parallel size 1": vllm_config.parallel_config.tensor_parallel_size == 1,
        '"use_local_argmax_reduction": true in --speculative-config': (
            spec.use_local_argmax_reduction
        ),
        "a drafter that shares the target's BF16 LM head (no "
        "draft_lm_head_quantization)": spec.draft_lm_head_quantization is None,
    }
    return [need for need, ok in requirements.items() if not ok]


# The in-tree offload specs whose GPU-CPU worker (``CPUOffloadingWorker``)
# submits its copies on a stream that waits for the compute stream.
_CPU_OFFLOAD_SPECS = ("CPUOffloadingSpec", "TieringOffloadingSpec")


def _kv_transfer_supported(kv_transfer: KVTransferConfig | None) -> bool:
    """Whether a KV transfer config is absent or vLLM's in-tree CPU offload.

    `--kv-offloading-size` (native backend) sets the in-tree
    `OffloadingConnector`: its per-layer hooks do nothing, and it moves whole
    blocks only between steps (`start_load_kv` and `wait_for_save`) on a
    stream that waits for the compute stream, so after the GDN commit of
    `postprocess_state`. Any other connector, an external module of the same
    name, or an out-of-tree offload spec may hook the stock layer forward,
    which the executor replaces, or order its copies differently.

    Args:
        kv_transfer: The engine's `kv_transfer_config`.

    """
    if kv_transfer is None:
        return True
    extra = kv_transfer.kv_connector_extra_config
    return (
        kv_transfer.kv_connector == "OffloadingConnector"
        and kv_transfer.kv_connector_module_path is None
        and extra.get("spec_name", "CPUOffloadingSpec") in _CPU_OFFLOAD_SPECS
    )


def unsupported(vllm_config: VllmConfig) -> list[str]:
    """Every requirement of the specialized decode this deployment misses."""
    parallel = vllm_config.parallel_config
    spec = vllm_config.speculative_config
    compilation = vllm_config.compilation_config
    cudagraph_mode = compilation.cudagraph_mode
    text = vllm_config.model_config.hf_text_config
    hc_shape = (text.hc_count, text.hidden_size, text.hc_lowrank)
    requirements = {
        "a GB300 (SM103) GPU": current_platform.is_device_capability(103),
        # The read kernels are built for one hyper-connection shape.
        f"hyper-connections of {HC_STREAMS} streams of {HC_HIDDEN} values with "
        f"low rank {HC_LOW_RANK} (Qwen3.8-Flash-Next)": (
            hc_shape == (HC_STREAMS, HC_HIDDEN, HC_LOW_RANK)
        ),
        # Expert parallelism over the TP ranks is allowed.
        "tensor parallel size 1 or 4 without pipeline, data or context "
        "parallelism or micro-batching": (
            parallel.tensor_parallel_size in MAX_DECODE_REQUESTS
            and parallel.pipeline_parallel_size == 1
            and parallel.data_parallel_size == 1
            and parallel.decode_context_parallel_size == 1
            and parallel.prefill_context_parallel_size == 1
            and not parallel.use_ubatching
        ),
        # The executor and the draft front serve MTP speculation.
        "MTP speculative decoding": spec is not None and spec.method == "mtp",
        "FULL CUDA graphs for decode batches": (
            cudagraph_mode is not None
            and cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
        ),
        # The executor runs in forward; a compiled forward would trace it.
        "no torch.compile of the model": (
            compilation.mode != CompilationMode.STOCK_TORCH_COMPILE
        ),
        # The executor will bind weight addresses at build.
        "no sleep mode": not vllm_config.model_config.enable_sleep_mode,
        "no runtime weight updates": vllm_config.weight_transfer_config is None,
        # The CPU offload's in-tree connector is allowed (_kv_transfer_supported).
        "no LoRA, batch invariance or KV transfer but the in-tree CPU offload": (
            vllm_config.lora_config is None
            and _kv_transfer_supported(vllm_config.kv_transfer_config)
            and not envs.VLLM_BATCH_INVARIANT
        ),
        # Every TP rank runs every token through the layers.
        "no hyper-connection or MoE sequence parallelism": (
            not is_hc_sequence_parallel_enabled(vllm_config)
            and not parallel.use_sequence_parallel_moe
        ),
        # The boundary kernel reads the sparse MoE block's experts and gate.
        "routed experts not on DeepGEMM MegaMoE": (
            vllm_config.kernel_config.moe_backend not in NATIVE_MEGA_MOE_BACKENDS
        ),
    }
    return [need for need, ok in requirements.items() if not ok]


def specialized_decode_config(vllm_config: VllmConfig) -> DecodeConfig | None:
    """The specialized decode settings, or None for the stock decode.

    `additional_config["decode_backend"]` is `auto` (the default: the stock
    decode) or `latency_specialized`; any other value raises, and so does a
    `decode_metadata_check_steps` that is not a non-negative integer or is
    set without `latency_specialized`, a `decode_metadata_overlap` that is
    not a boolean or is false without `latency_specialized`, and a
    `draft_head_fp8_prefilter`
    that is not a boolean or is set on a deployment it does not serve (module
    docstring). Collective over the TP group when requested: every rank
    raises if any rank cannot run the specialized decode.
    """
    extra = vllm_config.additional_config
    backend = extra.get("decode_backend", "auto") if isinstance(extra, dict) else "auto"
    check_steps = _metadata_check_steps(extra)
    overlap = _metadata_overlap(extra)
    draft_head_fp8 = _draft_head_fp8(extra)
    if backend == "auto":
        if check_steps:
            raise ValueError(
                "decode_metadata_check_steps needs decode_backend=latency_specialized"
            )
        if not overlap:
            raise ValueError(
                "decode_metadata_overlap needs decode_backend=latency_specialized"
            )
        if draft_head_fp8:
            raise ValueError(
                "draft_head_fp8_prefilter needs decode_backend=latency_specialized"
            )
        return None
    if backend != "latency_specialized":
        raise ValueError("Supported decode_backend values are auto/latency_specialized")
    missing = unsupported(vllm_config)
    supported = torch.tensor([0 if missing else 1], dtype=torch.int32)
    # Ranks that disagree would issue different collectives and hang, so they
    # all raise if any one misses a requirement.
    dist.all_reduce(supported, op=dist.ReduceOp.MIN, group=get_tp_group().cpu_group)
    if not supported.item():
        reason = "; ".join(missing) or "another tensor parallel rank to support it"
        raise ValueError(f"decode_backend=latency_specialized needs {reason}.")
    spec = vllm_config.speculative_config
    assert spec is not None and spec.num_speculative_tokens is not None
    if draft_head_fp8:
        missing_head = _draft_head_unsupported(vllm_config)
        if missing_head:
            raise ValueError(
                f"draft_head_fp8_prefilter needs {'; '.join(missing_head)}."
            )
    tokens_per_request = spec.num_speculative_tokens + 1
    max_rows = max_decode_rows(
        vllm_config.parallel_config.tensor_parallel_size, tokens_per_request
    )
    logger.info_once(
        "Qwen4Exp specialized decode on, for FULL-graph decode steps of up to %d rows.",
        max_rows,
    )
    if not spec.use_local_argmax_reduction:
        logger.info_once(
            "Qwen4Exp specialized decode: use_local_argmax_reduction is off, so "
            "the MTP draft passes choose their tokens with the stock path (a "
            "logits all-gather and greedy_argmax). Set "
            '"use_local_argmax_reduction": true in --speculative-config for '
            "the specialized token choice. The target's logits all-gather at "
            "TP4 does not depend on it."
        )
    return DecodeConfig(
        tokens_per_request=tokens_per_request,
        max_rows=max_rows,
        metadata_check_steps=check_steps,
        metadata_overlap=overlap,
        draft_head_fp8=draft_head_fp8,
    )
