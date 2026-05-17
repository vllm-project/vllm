# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import regex as re
import torch

from vllm.logger import init_logger
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "silu_mul_fp8 Helion kernel requires helion to be installed. "
        "Install it with: pip install helion"
    )


logger = init_logger(__name__)


def generate_mla_decode_kv_split_inputs() -> dict[str, tuple[Any, ...]]:
    intermediate_sizes = [2048, 2880, 4096, 8192, 11008, 14336]

    # Use the same num_tokens values as vLLM's default cudagraph capture sizes.
    # See vllm/config/vllm.py _set_cudagraph_sizes() for the canonical formula.
    num_tokens_list = [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 513, 16))

    inputs = {}
    for num_tokens in num_tokens_list:
        for intermediate_size in intermediate_sizes:
            input_tensor = torch.randn(
                num_tokens,
                2 * intermediate_size,
                device="cuda",
                dtype=torch.bfloat16,
            )
            scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)

            config_key = f"intermediate_{intermediate_size}_numtokens_{num_tokens}"
            inputs[config_key] = (input_tensor, scale)

    return inputs


def pick_mla_decode_kv_split_config(
    args: tuple[Any, ...], config_keys: list[str]
) -> str | None:
    """Pick the best pre-tuned config for the given input shape.

    Selection strategy:
      1. Find the closest intermediate_size among available configs
         (exact match preferred).
      2. Among the num_tokens values tuned for that intermediate_size, pick
         the smallest num_tokens >= the input's num_tokens. If the input is
         larger than all available num_tokens, fall back to the largest.

    Config keys must be "default" or follow the format
    "intermediate_{int}_numtokens_{int}".
    """
    if not config_keys:
        return None

    input_tensor, _scale = args
    intermediate_size = input_tensor.shape[-1] // 2
    num_tokens = input_tensor.view(-1, input_tensor.shape[-1]).shape[0]
    configs: dict[int, list[int]] = {}
    for key in config_keys:
        if key == "default":
            continue
        match = re.fullmatch(r"intermediate_(\d+)_numtokens_(\d+)", key)
        if not match:
            raise ValueError(
                f"Malformed config key '{key}', "
                f"expected format 'intermediate_{{int}}_numtokens_{{int}}'"
            )
        isize_str, ntokens_str = match.groups()
        configs.setdefault(int(isize_str), []).append(int(ntokens_str))

    if not configs:
        return "default" if "default" in config_keys else None

    best_isize = min(configs, key=lambda s: abs(s - intermediate_size))
    available_ntokens = sorted(configs[best_isize])
    best_ntokens = next(
        (n for n in available_ntokens if n >= num_tokens), available_ntokens[-1]
    )

    return f"intermediate_{best_isize}_numtokens_{best_ntokens}"


def mla_decode_kv_split(
    q_absorbed: torch.Tensor,
    latent_kv: torch.Tensor,
    kv_scale: torch.Tensor,
    Req_to_Tokens: torch.Tensor,
    NUM_KV_SPLITS: torch.Tensor,
    PAGE_SIZE: int,
):
    pass
