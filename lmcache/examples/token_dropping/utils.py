# SPDX-License-Identifier: Apache-2.0
"""Helpers for the token dropping example."""

# Standard
import json
import re

# Third Party
from transformers import AutoConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
import httpx
import torch

# First Party
import lmcache.sdk.request as lmc_request


def _rotate_half_neox(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs using GPT-NeoX/Llama-style half layout."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs using interleaved layout."""
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def _resolve_rope_theta(config: AutoConfig, rope_scaling: dict | None) -> float:
    """Return the RoPE base frequency (theta) for the model config."""
    theta = None
    if rope_scaling is not None:
        theta = rope_scaling.get("rope_theta")
    if theta is None:
        theta = getattr(config, "rope_theta", None)
    if theta is None:
        raise ValueError("could not resolve rope_theta from config or rope_scaling")
    return float(theta)


def _rope_cos_sin(
    *,
    config: AutoConfig,
    positions: torch.Tensor,
    rotary_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Build RoPE cos/sin for the model config, including HF rope_scaling."""
    rope_scaling = getattr(config, "rope_scaling", None)
    rope_type = None
    if rope_scaling is not None:
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))

    if rope_scaling is None or rope_type == "default":
        rope_theta = _resolve_rope_theta(config, rope_scaling)
        inv_freq = 1.0 / (
            rope_theta
            ** (
                torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32)
                / rotary_dim
            )
        )
        attention_scaling = 1.0
    else:
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
        if rope_type is None:
            raise ValueError(f"rope_scaling is missing rope_type/type: {rope_scaling}")

        if rope_type == "default":
            rope_theta = _resolve_rope_theta(config, rope_scaling)
            inv_freq = 1.0 / (
                rope_theta
                ** (
                    torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32)
                    / rotary_dim
                )
            )
            attention_scaling = 1.0
        else:
            inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](
                config=config,
                device=device,
                seq_len=int(positions.max().item()) + 1,
            )

    inv_freq = inv_freq[: rotary_dim // 2].to(device=device, dtype=torch.float32)
    positions = positions.to(device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)

    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=dtype)
    sin = emb.sin().to(dtype=dtype)
    return cos[:, None, :], sin[:, None, :], float(attention_scaling)


def _rerotate_k_cache(
    *,
    k_flat: torch.Tensor,
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    config: AutoConfig,
    head_size: int,
    is_neox_style: bool,
) -> torch.Tensor:
    """Move key cache RoPE from old positions to new positions."""
    num_tokens = k_flat.shape[0]
    k = k_flat.view(num_tokens, -1, head_size)

    partial_factor = getattr(config, "partial_rotary_factor", 1.0)
    rotary_dim = int(head_size * partial_factor)
    if rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim must be even, got {rotary_dim}")

    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]

    old_cos, old_sin, old_scale = _rope_cos_sin(
        config=config,
        positions=old_positions,
        rotary_dim=rotary_dim,
        device=k.device,
        dtype=k.dtype,
    )
    new_cos, new_sin, new_scale = _rope_cos_sin(
        config=config,
        positions=new_positions,
        rotary_dim=rotary_dim,
        device=k.device,
        dtype=k.dtype,
    )

    rotate_half = _rotate_half_neox if is_neox_style else _rotate_half_interleaved

    # Detach old RoPE: inverse rotation
    k_unscaled = k_rot / old_scale
    k_plain = (k_unscaled * old_cos) - (rotate_half(k_unscaled) * old_sin)

    # Attach new RoPE.
    k_new = new_scale * ((k_plain * new_cos) + (rotate_half(k_plain) * new_sin))

    return torch.cat((k_new, k_pass), dim=-1).reshape_as(k_flat)


def rerotate_k_cache(
    kv_tensor: torch.Tensor,
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    model_config: AutoConfig,
) -> torch.Tensor:
    """Move key cache RoPE from old positions to new positions.
    Args:
        kv_tensor: The KV cache tensor of shape [2, L, T, D].
        old_positions: The old token positions.
        new_positions: The new token positions.
        model_config: The model configuration.
    Returns:
        The rerotated KV cache tensor.
    Raises:
        ValueError: If the input tensor shape or dimensions are invalid.
    """
    if kv_tensor.ndim != 4:
        raise ValueError(f"kv_tensor must be 4D [2, L, T, D], got {kv_tensor.ndim}D")

    head_size = getattr(
        model_config,
        "head_dim",
        model_config.hidden_size // model_config.num_attention_heads,
    )
    if kv_tensor.shape[3] % head_size != 0:
        raise ValueError(
            f"hidden dim {kv_tensor.shape[3]} not divisible by head_size {head_size}"
        )

    is_neox_style = True
    for layer in range(kv_tensor.shape[1]):
        kv_tensor[0, layer] = _rerotate_k_cache(
            k_flat=kv_tensor[0, layer],
            old_positions=old_positions,
            new_positions=new_positions,
            config=model_config,
            head_size=head_size,
            is_neox_style=is_neox_style,
        )
    return kv_tensor


def _extract_token_id(choice: dict) -> int:
    """Pull the generated token id from a streaming choice.

    Requires the vLLM server to be started with --return-tokens-as-token-ids,
    so logprobs token strings look like "token_id:123".
    """
    logprobs = choice.get("logprobs") or {}
    toks = logprobs.get("tokens") or []
    if not toks:
        return -1
    last = toks[-1]
    return int(last.split(":")[-1]) if ":" in last else -1


def extract_answer_choice(response: str) -> str:
    """Extract the predicted choice letter from a LongBench-v2 response.

    Args:
        response: The model's generated text.

    Returns:
        The predicted letter ("A", "B", "C" or "D").
        Empty string if none was found.
    """
    cleaned = response.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", cleaned)
    if match is None:
        match = re.search(r"The correct answer is ([A-D])", cleaned)
    return match.group(1) if match is not None else ""


def score_answers(
    ground_truth: list[str], responses: list[str]
) -> tuple[int, int, list[bool]]:
    """Score generated responses against ground-truth choice letters.

    Args:
        ground_truth: list of ground-truth letters.
        responses: generated answers for each sample.

    Returns:
        A tuple ``(correct, total, per_example)``.
        ``correct`` is the number of samples with correct answers,
        ``total`` is the total number of samples, and
        ``per_example`` is a list of boolean for each sample's correctness.

    Raises:
        ValueError: If the two lists have different lengths.
    """
    if len(ground_truth) != len(responses):
        raise ValueError(f"length mismatch: {len(ground_truth)} and {len(responses)}")
    per_example = [
        extract_answer_choice(resp) == truth
        for truth, resp in zip(ground_truth, responses, strict=False)
    ]
    return sum(per_example), len(per_example), per_example


def make_post_completion(
    vllm_url: str, model_name: str, timeout: float
) -> lmc_request.PostCompletion:
    """Build a streaming post_completion callable for lmcache.sdk.request.

    Args:
        vllm_url: The URL of the vLLM server.
        model_name: The name of the model.
        timeout: Timeout in seconds for HTTP requests.

    Returns:
        A streaming post_completion callable.
    """

    def post_completion(prompt_token_ids, sampling_params, cache_salt):
        payload = {
            **sampling_params,
            "model": model_name,
            "prompt": list(prompt_token_ids),
            "stream": True,
            "logprobs": 0,
        }
        if cache_salt:
            payload["cache_salt"] = cache_salt
        timeout_http = httpx.Timeout(
            connect=timeout, read=None, write=timeout, pool=timeout
        )
        with httpx.stream(
            "POST",
            f"{vllm_url.rstrip('/')}/v1/completions",
            json=payload,
            timeout=timeout_http,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    break
                body = json.loads(line.removeprefix("data: "))
                choices = body.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                yield lmc_request.TokenEvent(
                    token_id=_extract_token_id(choice),
                    text=choice.get("text", ""),
                )

    return post_completion
