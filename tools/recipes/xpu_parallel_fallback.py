# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conservative XPU layout fallback; estimates never imply validated startup.

Only shrink TP/DP for single-node, PP=1 layouts. The model's actual weight
artifacts (not parameter-count guesses) and attention-head constraints are
required. Unknown sizes/layouts are rejected rather than silently accepted.
"""

from __future__ import annotations

import math
from typing import Any

import regex as re
from model_constraints import load_head_constraints
from recipe_validation_status import RecipeSkipError

GIB = 1024**3


def _positive(config: dict[str, Any], key: str) -> int:
    value = config.get(key, 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Invalid recipe {key}={value!r}")
    return value


def _weight_bytes(config: dict[str, Any]) -> tuple[int, str]:
    """Read concrete safetensors file sizes without downloading model weights.

    Model-file metadata is read through the configured HF revision. Only an
    unambiguous standard weight set is accepted. Quantization is reflected in
    actual file sizes rather than a guessed bytes/parameter ratio.
    """
    from vllm.transformers_utils.repo_utils import hf_api

    model = config.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("Model ID is missing")
    revision = config.get("revision")
    info = hf_api().model_info(model, revision=revision, files_metadata=True)
    siblings = {f.rfilename: f.size for f in info.siblings or []}
    files = [name for name in siblings if name.endswith(".safetensors")]
    # Ignore non-weight auxiliary files, but reject alternate weight sets.
    if files == ["model.safetensors"]:
        selected = files
    else:
        indexed = [
            name
            for name in files
            if re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", name)
        ]
        if len(indexed) != len(files) or not indexed:
            raise ValueError(
                "Cannot identify an unambiguous standard safetensors weight set"
            )
        counts = {int(re.search(r"-of-(\d{5})", name).group(1)) for name in indexed}
        if len(counts) != 1 or next(iter(counts)) != len(indexed):
            raise ValueError("Incomplete or ambiguous safetensors shards")
        selected = indexed
    sizes = [siblings[name] for name in selected]
    if any(not isinstance(n, int) or n <= 0 for n in sizes):
        raise ValueError(
            "Hub file metadata does not contain reliable safetensors sizes"
        )
    return sum(sizes), "Hugging Face safetensors file metadata"


def _kv_bytes(config: dict[str, Any], tp: int) -> int:
    """Estimate at least one max-length request's uncompressed KV capacity."""
    from vllm.transformers_utils.config import get_config

    hf = get_config(
        config["model"],
        trust_remote_code=config.get("trust-remote-code", False),
        revision=config.get("revision"),
        code_revision=config.get("code-revision"),
        config_format=config.get("config-format", "auto"),
    )
    if getattr(hf, "vision_config", None) is not None:
        raise ValueError("Multimodal KV/runtime footprint is not currently estimated")
    text = hf.get_text_config()
    layers = getattr(text, "num_hidden_layers", None)
    heads = getattr(text, "num_attention_heads", None)
    kv_heads = getattr(text, "num_key_value_heads", heads)
    hidden = getattr(text, "hidden_size", None)
    maximum = config.get("max-model-len") or getattr(
        text, "max_position_embeddings", None
    )
    if not all(
        isinstance(x, int) and x > 0 for x in (layers, heads, kv_heads, hidden, maximum)
    ):
        raise ValueError("Model context limit and standard decoder dimensions required")
    if hidden % heads:
        raise ValueError("Unsupported attention head dimension")
    if config.get("kv-cache-dtype", "auto") not in ("auto", "bfloat16", "float16"):
        raise ValueError("Unsupported KV-cache dtype for conservative fit estimate")
    # vLLM partitions KV heads unless they are fewer than TP, when replicated.
    local_kv_heads = math.ceil(kv_heads / tp) if kv_heads >= tp else 1
    # At least 2 bytes/element; include 25% padding for cache block allocation.
    return math.ceil(
        2 * layers * local_kv_heads * (hidden // heads) * maximum * 2 * 1.25
    )


def select_xpu_fallback(config: dict[str, Any], hardware: Any) -> dict[str, Any]:
    """Choose a feasible smaller single-node layout, or SKIP with evidence.

    Feasibility is intentionally conservative and remains ESTIMATED. The vLLM
    server startup and test request remain the final correctness checks.
    """
    tp = _positive(config, "tensor-parallel-size")
    dp = _positive(config, "data-parallel-size")
    pp = _positive(config, "pipeline-parallel-size")
    if (
        pp != 1
        or any(
            config.get(k, 1) != 1
            for k in ("prefill-context-parallel-size", "decode-context-parallel-size")
        )
        or config.get("enable-expert-parallel")
    ):
        raise RecipeSkipError(
            "XPU_FALLBACK_UNSUPPORTED_LAYOUT",
            "Only TP/DP without PP, context or expert parallelism can be auto-scaled",
        )
    visible = hardware.gpu_count
    if visible < 1:
        raise RecipeSkipError("INSUFFICIENT_XPU_DEVICES", "No visible XPU devices")
    if tp * dp <= visible:
        return {
            "changed": False,
            "recipe": {"tp": tp, "dp": dp, "pp": pp},
            "effective": {"tp": tp, "dp": dp, "pp": pp},
            "visible_gpus": visible,
            "fit_status": "recipe_unmodified",
        }
    if any("b70" not in d.name.lower() for d in hardware.devices):
        raise RecipeSkipError(
            "XPU_FALLBACK_UNSUPPORTED_HARDWARE",
            "Auto fit requires all visible GPUs to be Arc Pro B70",
        )
    if config.get("quantization") in ("bitsandbytes", "gguf") or config.get(
        "load-format"
    ) in ("gguf", "bitsandbytes"):
        raise RecipeSkipError(
            "XPU_FALLBACK_UNKNOWN_MEMORY",
            "Cannot reliably estimate the requested weight loading format",
        )
    try:
        constraints = load_head_constraints(config)
        weight_bytes, provenance = _weight_bytes(config)
    except Exception as exc:
        raise RecipeSkipError(
            "XPU_FALLBACK_UNKNOWN_MEMORY", f"Cannot verify a smaller TP: {exc}"
        ) from exc
    usable = min(
        min(
            d.free_memory_bytes - int(d.total_memory_bytes * 0.05),
            int(
                d.total_memory_bytes * float(config.get("gpu-memory-utilization", 0.90))
            ),
        )
        for d in hardware.devices
    )
    if usable <= 0:
        raise RecipeSkipError(
            "XPU_FALLBACK_INSUFFICIENT_MEMORY",
            "No available XPU memory after reserve",
        )
    evaluated = []
    # DP first: preserve original TP when feasible; then shrink TP by powers of two.
    candidates = [n for n in (8, 4, 2, 1) if n <= tp and n <= visible]
    for candidate_tp in candidates:
        if not constraints or not all(c.supports(candidate_tp) for c in constraints):
            evaluated.append({"tp": candidate_tp, "reason": "head_constraints"})
            continue
        try:
            kv_bytes = _kv_bytes(config, candidate_tp)
        except Exception as exc:
            raise RecipeSkipError(
                "XPU_FALLBACK_UNKNOWN_MEMORY", f"Cannot estimate KV cache: {exc}"
            ) from exc
        # 25% replication/loading margin and 3 GiB fixed runtime/activation reserve.
        required = math.ceil(weight_bytes * 1.25 / candidate_tp) + kv_bytes + 3 * GIB
        evaluated.append(
            {
                "tp": candidate_tp,
                "estimated_per_gpu_bytes": required,
                "usable_per_gpu_bytes": usable,
                "fits": required <= usable,
            }
        )
        if required > usable:
            continue
        candidate_dp = min(dp, visible // candidate_tp)
        if candidate_dp < 1:
            continue
        return {
            "changed": (candidate_tp, candidate_dp) != (tp, dp),
            "recipe": {"tp": tp, "dp": dp, "pp": pp},
            "effective": {"tp": candidate_tp, "dp": candidate_dp, "pp": pp},
            "visible_gpus": visible,
            "fit_status": "estimated_requires_startup_validation",
            "weight_bytes": weight_bytes,
            "weight_source": provenance,
            "estimated_per_gpu_bytes": required,
            "usable_per_gpu_bytes": usable,
            "candidates": evaluated,
        }
    raise RecipeSkipError(
        "XPU_FALLBACK_INSUFFICIENT_MEMORY",
        "No smaller supported TP/DP layout passed conservative VRAM checks",
        recipe_parallelism={"tp": tp, "dp": dp, "pp": pp},
        visible_gpus=visible,
        candidates=evaluated,
    )
