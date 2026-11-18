# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm import envs
from vllm.config.model import PROCESSED_LOGPROBS_MODES, LogprobsMode
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import HAS_TRITON
from vllm.utils.flashinfer import has_flashinfer

if HAS_TRITON:
    from vllm.v1.sample.ops.topk_topp_triton import (
        _topk_topp,
        _topp_split_mask,
        _topp_split_stats,
        _topp_split_step,
        apply_top_k_top_p_triton,
    )

logger = init_logger(__name__)


def register_top_k_top_p_warmups() -> None:
    """Register every native accelerator sampling kernel used at runtime."""
    if HAS_TRITON and not current_platform.is_cpu():
        _topk_topp.register_warmup()
        if current_platform.is_cuda_alike():
            _topp_split_stats.register_warmup()
            _topp_split_step.register_warmup()
            _topp_split_mask.register_warmup()


def _flashinfer_jit_unsupported_reason(capability: DeviceCapability) -> str | None:
    """Return why FlashInfer JIT codegen cannot target the current GPU, or
    None if it can.

    FlashInfer swallows arch-detection errors when building its compilation
    context (e.g. SM 12.x with a CUDA toolkit older than 12.9), leaving an
    empty target-arch set that makes every JIT spec fail with a misleading
    "requires sm75 or higher" error at first use — killing the engine during
    startup profiling (https://github.com/vllm-project/vllm/issues/42393).
    """
    try:
        from flashinfer.jit.core import check_cuda_arch
    except ImportError:
        return None
    try:
        check_cuda_arch()
        return None
    except RuntimeError as e:
        reason = str(e)
    try:
        # Re-derive the real error that FlashInfer swallowed during arch
        # detection, e.g. "SM 12.x requires CUDA >= 12.9".
        from flashinfer.compilation_context import CompilationContext

        CompilationContext._normalize_cuda_arch(capability.major, capability.minor)
    except RuntimeError as e:
        reason = str(e)
    except Exception:
        # FlashInfer internals changed; keep the original error message.
        pass
    return reason


def flashinfer_sampler_supported() -> bool:
    """Decide whether FlashInfer's top-p/top-k sampler can be used.

    Returns False (with appropriate logging) when `VLLM_USE_FLASHINFER_SAMPLER`
    is 0, when the platform isn't CUDA, when the GPU's compute capability is
    unsupported, when the GPU has 16 or fewer SMs, or when FlashInfer cannot
    JIT-compile for the current GPU/CUDA toolchain. Raises `RuntimeError` if
    the user explicitly opted in via the env var but FlashInfer is unavailable.

    Assumes flashinfer is installed, as guaranteed by `requirements/cuda.txt`;
    otherwise importing the FlashInfer backend below raises `ImportError`.

    Note: callers must additionally ensure `logprobs_mode` doesn't require
    post-top-k/top-p logits/logprobs for any request whose logprobs will be
    returned in this step, since FlashInfer doesn't expose those.
    """
    if not current_platform.is_cuda():
        return False
    if not envs.VLLM_USE_FLASHINFER_SAMPLER:
        logger.info_once(
            "FlashInfer top-p/top-k sampling disabled via "
            "VLLM_USE_FLASHINFER_SAMPLER=0."
        )
        return False
    from vllm.v1.attention.backends.flashinfer import FlashInferBackend

    capability = current_platform.get_device_capability()
    assert capability is not None
    unsupported_reason: str | None = None
    if not has_flashinfer():
        unsupported_reason = "FlashInfer kernels are disabled"
    elif not FlashInferBackend.supports_compute_capability(capability):
        unsupported_reason = (
            f"unsupported compute capability {capability.as_version_str()}"
        )
    elif (
        num_sms := current_platform.num_compute_units(
            torch.accelerator.current_device_index()
        )
    ) <= 16:
        # FlashInfer 0.7+ rejects multi-CTA top-k masking on <=16 SMs because
        # its cross-CTA software barrier cannot guarantee forward progress.
        unsupported_reason = (
            f"top-k masking requires more than 16 SMs; device has {num_sms}"
        )
    else:
        unsupported_reason = _flashinfer_jit_unsupported_reason(capability)

    if unsupported_reason is None:
        logger.info_once("Using FlashInfer for top-p & top-k sampling.", scope="global")
        return True
    if envs.is_set("VLLM_USE_FLASHINFER_SAMPLER"):
        raise RuntimeError(
            f"FlashInfer top-p/top-k sampling unavailable: {unsupported_reason}. "
            "Unset VLLM_USE_FLASHINFER_SAMPLER=1."
        )
    logger.warning_once(
        "FlashInfer top-p/top-k sampling unavailable: %s; falling back. "
        "Set VLLM_USE_FLASHINFER_SAMPLER=0 to silence.",
        unsupported_reason,
    )
    return False


def xpu_sampler_supported() -> bool:
    """Decide whether the fused XPU top-k/top-p sampler kernel can be used.

    Returns False (with appropriate logging) when the platform isn't XPU, when
    `VLLM_XPU_USE_SAMPLER_KERNEL` is 0.

    Note: callers must additionally ensure no request needs a per-request seed
    or greedy sampling, since the kernel draws from the device's default
    generator and always samples randomly.
    """
    if not current_platform.is_xpu():
        return False
    if not envs.VLLM_XPU_USE_SAMPLER_KERNEL:
        logger.info_once(
            "Fused XPU top-p/top-k sampling disabled via VLLM_XPU_USE_SAMPLER_KERNEL=0."
        )
        return False

    logger.info_once("Using the fused XPU kernel for top-p & top-k sampling.")
    return True


def xpu_sample(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    logprobs_mode: LogprobsMode = "raw_logprobs",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sample from the logits using the fused XPU top-k/top-p kernel.

    Statistically equivalent to top-k/top-p filtering followed by weighted random
    sampling, but avoids sorting the vocab and never materializes the probability
    tensor. The noise comes from the
    device's default generator, so per-request generators aren't supported.

    Returns the sampled token ids and, for processed logprobs modes, the
    post-top-k/top-p logits (or logprobs); otherwise None.
    """
    logits = logits.to(dtype=torch.float32).contiguous()
    sampled = torch.empty(logits.shape[0], dtype=torch.int64, device=logits.device)
    logits_to_return = (
        torch.empty_like(logits) if logprobs_mode in PROCESSED_LOGPROBS_MODES else None
    )

    generator = torch.xpu.default_generators[logits.device.index]
    state = generator.get_state()
    seed, offset = state.view(torch.int64).tolist()
    seeds = torch.tensor([seed, offset], dtype=torch.int64, device="cpu")
    # The XPU kernel expects k as int64 (Long), but the input batch
    # stores top_k as int32. Cast here to avoid dtype mismatch.
    if k is not None:
        k = k.to(torch.int64)
    torch.ops.vllm.xpu_topk_topp_sampler(
        sampled, logits_to_return, logits, k, p, logprobs_mode, seeds
    )
    # The custom XPU sampler kernel consumes RNG values internally, so advance
    # the default generator's offset to keep future draws deterministic.
    # pytorch: offset must be multiple of 4
    offset = (offset + logits.numel() + 3) // 4 * 4
    state.view(torch.int64)[1] = offset
    generator.set_state(state)
    return sampled, logits_to_return


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    has_topp_only_rows: bool = True,
) -> torch.Tensor:
    """Apply top-k and/or top-p masking to logits.

    has_topp_only_rows must be True if any row has p < 1 and
    k >= logits.shape[-1] (no active top-k).
    """
    if p is None and k is None:
        return logits

    if HAS_TRITON:
        return apply_top_k_top_p_triton(
            logits, k, p, has_topp_only_rows=has_topp_only_rows
        )

    is_cpu = current_platform.is_cpu()
    return apply_top_k_top_p_pytorch(logits, k, p, allow_cpu_sync=is_cpu)


def apply_top_k_top_p_pytorch(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    allow_cpu_sync: bool = False,
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits.

    If a top-p is used, this function will sort the logits tensor,
    which can be slow for large batches.

    The logits tensor may be updated in-place.
    """
    if p is None:
        if k is None:
            return logits

        if allow_cpu_sync:
            # Avoid sorting vocab for top-k only case.
            return apply_top_k_only(logits, k)

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)


def apply_top_k_only(logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Apply top-k mask to the logits.

    This implementation doesn't involve sorting the entire vocab.
    Note however that it involves a GPU->CPU sync which can be detrimental for
    async scheduling performance.

    The logits tensor may be updated in-place.
    """
    no_top_k_mask = k == logits.shape[1]
    # Set non-top-k rows to 1 so that we can gather.
    k = k.masked_fill(no_top_k_mask, 1)
    max_top_k = k.max()
    # topk.values tensor has shape [batch_size, max_top_k].
    # Convert top k to 0-based index in range [0, max_top_k).
    k_index = k.sub_(1).unsqueeze(1)
    top_k_mask = logits.topk(max_top_k, dim=1).values.gather(1, k_index.long())
    # Handle non-topk rows.
    top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
    return logits.masked_fill_(logits < top_k_mask, -float("inf"))


def flashinfer_sample(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    generators: dict[int, torch.Generator] = {},  # noqa
) -> torch.Tensor:
    """Sample from the logits using FlashInfer.

    Statistically, this function is equivalent to top-k/top-p filtering followed by
    weighted random sampling. However, this function is faster because it avoids
    sorting the logits tensor via rejection sampling.
    """
    import flashinfer

    assert not (k is None and p is None)
    if k is None:
        # Top-p only.
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        next_token_ids = flashinfer.sampling.top_p_sampling_from_probs(
            probs, p, deterministic=True
        )
    elif p is None:
        # Top-k only.
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        next_token_ids = flashinfer.sampling.top_k_sampling_from_probs(
            probs, k, deterministic=True
        )
    else:
        # Both top-k and top-p.
        next_token_ids = flashinfer.sampling.top_k_top_p_sampling_from_logits(
            logits, k, p, deterministic=True
        )

    return next_token_ids.view(-1)
