# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any


def validate_engine_args(
    engine_args: Any,
    model_config: Any,
    speculative_config: Any | None,
) -> None:
    if engine_args.probe_ckpt is None:
        return

    if "BailingMoeV3ForCausalLM" not in model_config.architectures:
        raise ValueError(
            "--probe-ckpt currently supports only Bailing V3 MoE "
            "(BailingMoeV3ForCausalLM); got "
            f"architectures={model_config.architectures!r}."
        )
    if engine_args.pipeline_parallel_size != 1:
        raise ValueError("--probe-ckpt does not support pipeline parallelism")
    if (
        engine_args.prefill_context_parallel_size != 1
        or engine_args.decode_context_parallel_size != 1
    ):
        raise ValueError("--probe-ckpt does not support context parallelism")
    if engine_args.enable_dbo:
        raise ValueError("--probe-ckpt does not support dual batch overlap")

    if speculative_config is not None:
        target_paths = {model_config.model, model_config.model_weights}
        target_paths.discard("")
        draft_path = speculative_config.model
        same_checkpoint = draft_path is None or draft_path in target_paths
        if draft_path is not None and any(os.path.exists(p) for p in target_paths):
            same_checkpoint = any(
                os.path.realpath(draft_path) == os.path.realpath(p)
                for p in target_paths
                if os.path.exists(p)
            )
        if speculative_config.method != "mtp" or not same_checkpoint:
            raise ValueError(
                "--probe-ckpt supports speculative decoding only with "
                "bundled MTP from the target checkpoint; got "
                f"method={speculative_config.method!r}, "
                f"draft_model={draft_path!r}."
            )
