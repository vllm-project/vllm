# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up sparse-MLA Triton metadata kernels."""

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

_DEEPSEEK_V4_SPARSE_MLA_BACKENDS = frozenset(
    {
        "FLASHMLA_SPARSE_DSV4",
        "FLASHINFER_MLA_SPARSE_DSV4",
        "ROCM_FLASHMLA_SPARSE_DSV4",
        "DEEPSEEK_SPARSE_SWA",
    }
)
_GENERIC_SPARSE_MLA_BACKENDS = frozenset(
    {
        "FLASHMLA_SPARSE",
        "FLASHINFER_MLA_SPARSE",
        "FLASHINFER_MLA_SPARSE_SM120",
    }
)

_SPARSE_PREFILL_METADATA_NUM_PREFILLS = (1, 2, 4, 8)
_SPARSE_PREFILL_METADATA_NUM_DECODES = (0, 1, 2)
_DSV4_PREFILL_CHUNK_METADATA_COMPRESS_RATIOS = (4, 128)
_PREFILL_CHUNK_METADATA_SEQ_LEN_MULTIPLIERS = (2, 3)
_PREFILL_CHUNK_METADATA_QUERY_SLICE_OFFSETS = (
    # query_slice_start offset, query_slice_stop offset
    (0, 0),
    (0, -1),
    (1, 0),
    (1, -1),
)
_COMBINE_TOPK_SWA_INPUT_VARIANTS = (
    # offset_topk, offset_query_and_seq, offset_gather
    (False, False, False),
    (False, True, False),
    (True, True, True),
)
_DSV4_COMBINE_TOPK_SWA_WARMUP_CASES = (
    # compress_ratio, topk, topk_width, N
    (1, 0, 512, 512),
    (4, 512, 512, 512 * 4),
    # DSv4-Pro C4A traffic uses top-k 1024 with N=1024.
    (4, 1024, 1024, 1024),
    (128, 8192, 8192, 8192 * 128),
    # Real C128A traffic also specializes N=1 in one call path.
    (128, 8192, 8192, 1),
)


def _clamp_warmup_tokens(num_tokens: int, max_tokens: int) -> int:
    return max(0, min(num_tokens, max_tokens))


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _hf_config_int(runner: "GPUModelRunner", name: str, default: int) -> int:
    model_config = getattr(runner.vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    return int(getattr(hf_config, name, default) or default)


def _attention_backend_name(backend: object) -> str | None:
    get_name = getattr(backend, "get_name", None)
    if get_name is None:
        return None
    try:
        return get_name()
    except NotImplementedError:
        return None


def _has_attention_backend(
    runner: "GPUModelRunner",
    backend_names: frozenset[str],
) -> bool:
    for groups in getattr(runner, "attn_groups", []) or ():
        for group in groups:
            name = _attention_backend_name(getattr(group, "backend", None))
            if name in backend_names:
                return True
    return False


def _warm_sparse_swa_prefill_metadata_kernel(
    device: torch.device,
    window_size: int,
    prefill_tokens: int,
) -> None:
    from vllm.v1.attention.backends.mla.sparse_swa import (
        _COMPUTE_PREFILL_METADATA_KERNEL,
    )

    for num_prefills in _SPARSE_PREFILL_METADATA_NUM_PREFILLS:
        for num_decodes in _SPARSE_PREFILL_METADATA_NUM_DECODES:
            query_lens = [1] * num_decodes
            query_lens += [prefill_tokens] * num_prefills
            query_start_locs = [0]
            for query_len in query_lens:
                query_start_locs.append(query_start_locs[-1] + query_len)
            query_start_loc = torch.tensor(
                query_start_locs,
                dtype=torch.int32,
                device=device,
            )
            seq_lens = torch.tensor(
                [1] * num_decodes + [window_size + q for q in query_lens[num_decodes:]],
                dtype=torch.int32,
                device=device,
            )
            prefill_gather_lens = torch.empty(
                num_prefills, dtype=torch.int32, device=device
            )
            _COMPUTE_PREFILL_METADATA_KERNEL(
                prefill_gather_lens_ptr=prefill_gather_lens,
                seq_lens_ptr=seq_lens,
                query_start_loc_ptr=query_start_loc,
                num_prefills=num_prefills,
                num_decodes=num_decodes,
                window_size=window_size,
            )


def _warm_prefill_chunk_metadata_kernel(
    device: torch.device,
    compress_ratio: int,
    query_slice_start: int,
    query_slice_stop: int,
    query_len: int,
) -> None:
    from vllm.v1.attention.backends.mla.indexer import build_prefill_chunk_metadata

    num_reqs = 2
    query_start_loc_cpu = torch.arange(
        0, (num_reqs + 1) * query_len, query_len, dtype=torch.int32
    )
    query_start_loc = query_start_loc_cpu.to(device=device)

    uncompressed_seq_lens_cpu = torch.tensor(
        [
            compress_ratio * multiplier + query_len
            for multiplier in _PREFILL_CHUNK_METADATA_SEQ_LEN_MULTIPLIERS
        ],
        dtype=torch.int32,
    )
    compressed_seq_lens_cpu = uncompressed_seq_lens_cpu // compress_ratio
    uncompressed_seq_lens = uncompressed_seq_lens_cpu.to(device=device)
    compressed_seq_lens = compressed_seq_lens_cpu.to(device=device)
    block_table = torch.zeros(
        (num_reqs, int(compressed_seq_lens_cpu.max().item())),
        dtype=torch.int32,
        device=device,
    )

    build_prefill_chunk_metadata(
        0,
        num_reqs,
        query_start_loc,
        query_start_loc_cpu,
        uncompressed_seq_lens,
        compressed_seq_lens,
        compressed_seq_lens_cpu,
        block_table,
        compress_ratio,
        query_slice=slice(query_slice_start, query_slice_stop),
    )


def _warm_combine_topk_swa_indices_kernel(
    device: torch.device,
    num_tokens: int,
    window_size: int,
    compress_ratio: int,
    topk: int,
    topk_width: int,
    n: int,
) -> None:
    from vllm.models.deepseek_v4.common.ops.cache_utils import combine_topk_swa_indices

    if num_tokens <= 0:
        return

    def _make_topk_indices(*, offset: bool) -> torch.Tensor:
        if offset:
            topk_storage = torch.full(
                (num_tokens * topk_width + 1,),
                -1,
                dtype=torch.int32,
                device=device,
            )
            topk_indices = topk_storage[1:].reshape(num_tokens, topk_width)
        else:
            topk_indices = torch.full(
                (num_tokens, topk_width), -1, dtype=torch.int32, device=device
            )
        if topk > 0:
            topk_indices.copy_(
                torch.arange(num_tokens * topk_width, dtype=torch.int32, device=device)
                .reshape(num_tokens, topk_width)
                .remainder(topk_width)
            )
        return topk_indices

    query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    seq_lens = torch.tensor(
        [window_size + num_tokens], dtype=torch.int32, device=device
    )
    gather_lens = torch.tensor(
        [min(window_size + num_tokens, window_size + num_tokens - 1)],
        dtype=torch.int32,
        device=device,
    )
    offset_query_start_loc = torch.empty(3, dtype=torch.int32, device=device)[1:]
    offset_query_start_loc.copy_(query_start_loc)
    offset_seq_lens = torch.empty(2, dtype=torch.int32, device=device)[1:]
    offset_seq_lens.copy_(seq_lens)
    offset_gather_lens = torch.empty(2, dtype=torch.int32, device=device)[1:]
    offset_gather_lens.copy_(gather_lens)

    for (
        offset_topk,
        offset_query_and_seq,
        offset_gather,
    ) in _COMBINE_TOPK_SWA_INPUT_VARIANTS:
        warmup_topk_indices = _make_topk_indices(offset=offset_topk)
        warmup_query_start_loc = (
            offset_query_start_loc if offset_query_and_seq else query_start_loc
        )
        warmup_seq_lens = offset_seq_lens if offset_query_and_seq else seq_lens
        warmup_gather_lens = offset_gather_lens if offset_gather else gather_lens
        n_values = (n,) if n == 1 else (n, n + 1)
        for m in (window_size + num_tokens, topk_width):
            for n_value in n_values:
                combine_topk_swa_indices(
                    warmup_topk_indices,
                    warmup_query_start_loc,
                    warmup_seq_lens,
                    warmup_gather_lens,
                    window_size,
                    compress_ratio,
                    topk,
                    M=m,
                    N=n_value,
                )


@torch.inference_mode()
def sparse_mla_triton_warmup(
    runner: "GPUModelRunner",
    num_tokens: int,
    *,
    compress_ratios: tuple[int, ...],
    combine_topk_swa_cases: tuple[tuple[int, int, int, int], ...] = (),
) -> None:
    from vllm.v1.attention.backends.mla.indexer import (
        BUILD_PREFILL_CHUNK_METADATA_KERNEL,
    )

    device = getattr(runner, "device", torch.device("cuda"))
    window_size = _hf_config_int(runner, "sliding_window", 128)

    _warm_sparse_swa_prefill_metadata_kernel(device, window_size, num_tokens)
    query_slice_bounds = tuple(
        (start, 2 * num_tokens + stop)
        for start, stop in _PREFILL_CHUNK_METADATA_QUERY_SLICE_OFFSETS
    )
    compile_keys = tuple(
        dict.fromkeys(
            BUILD_PREFILL_CHUNK_METADATA_KERNEL.compile_key(
                {
                    "query_slice_start": query_slice_start,
                    "query_slice_stop": query_slice_stop,
                    "BLOCK_SIZE": 1024,
                    "COMPRESS_RATIO": compress_ratio,
                }
            )
            for compress_ratio in compress_ratios
            for query_slice_start, query_slice_stop in query_slice_bounds
        )
    )
    for key in compile_keys:
        _warm_prefill_chunk_metadata_kernel(
            device,
            key.COMPRESS_RATIO,
            key.query_slice_start,
            key.query_slice_stop,
            num_tokens,
        )
    for compress_ratio, topk, topk_width, n in combine_topk_swa_cases:
        _warm_combine_topk_swa_indices_kernel(
            device,
            num_tokens,
            window_size,
            compress_ratio,
            topk,
            topk_width,
            n,
        )


def deepseek_v4_sparse_triton_warmup(
    runner: "GPUModelRunner",
    num_tokens: int,
) -> None:
    sparse_mla_triton_warmup(
        runner,
        num_tokens,
        compress_ratios=_DSV4_PREFILL_CHUNK_METADATA_COMPRESS_RATIOS,
        combine_topk_swa_cases=_DSV4_COMBINE_TOPK_SWA_WARMUP_CASES,
    )


def sparse_mla_triton_warmup_if_needed(worker: "Worker") -> None:
    runner = worker.model_runner
    if runner.is_pooling_model:
        return

    max_tokens = worker.scheduler_config.max_num_batched_tokens
    num_tokens = _clamp_warmup_tokens(8, max_tokens)
    if num_tokens <= 0:
        return

    try:
        if _has_attention_backend(runner, _DEEPSEEK_V4_SPARSE_MLA_BACKENDS):
            deepseek_v4_sparse_triton_warmup(runner, num_tokens)
        elif _has_attention_backend(runner, _GENERIC_SPARSE_MLA_BACKENDS):
            sparse_mla_triton_warmup(
                runner,
                num_tokens,
                compress_ratios=(1,),
            )
    except Exception:
        logger.warning("Skipping sparse MLA Triton warmup.", exc_info=True)
