# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import torch

from vllm.config import VllmConfig
from vllm.v1.spec_decode.registry import get_speculative_method

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.states import RequestState


def init_speculator(
    vllm_config: VllmConfig,
    device: torch.device,
    req_states: "RequestState",
):
    """Build the speculator for this config."""
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    registered = get_speculative_method(speculative_config.method)
    if registered is not None:
        from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

        cls = registered.resolve_speculator_cls()
        if not issubclass(cls, BaseSpeculator):
            raise TypeError(
                f"Registered speculator {registered.speculator_cls!r} must "
                "inherit BaseSpeculator"
            )
        factory = cast(Callable[[VllmConfig, torch.device], BaseSpeculator], cls)
        return factory(vllm_config, device)
    if speculative_config.method == "extract_hidden_states":
        from vllm.v1.worker.gpu.spec_decode.extract_hidden_states import (
            ExtractHiddenStatesSpeculator,
        )

        return ExtractHiddenStatesSpeculator(vllm_config, device)
    elif speculative_config.method == "dflash":
        if "LiLiCorrDraftModel" in speculative_config.draft_model_config.architectures:
            from vllm.v1.worker.gpu.spec_decode.lilicorr.speculator import (
                LiLiCorrSpeculator,
            )

            return LiLiCorrSpeculator(vllm_config, device)
        if {"DFlash2DraftModel", "DFlash2BeamDraftModel"}.intersection(
            speculative_config.draft_model_config.architectures
        ):
            from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
                DFlash2Speculator,
            )

            return DFlash2Speculator(vllm_config, device)
        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
            DFlashSpeculator,
        )

        return DFlashSpeculator(vllm_config, device)
    elif speculative_config.method == "dspark":
        from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (
            DSparkSpeculator,
        )

        return DSparkSpeculator(vllm_config, device)
    elif speculative_config.use_gemma4_mtp():
        from vllm.v1.worker.gpu.spec_decode.gemma4.speculator import (
            Gemma4Speculator,
        )

        return Gemma4Speculator(vllm_config, device)
    elif speculative_config.use_multi_module_mtp():
        from vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator import (
            MultiModuleMTPSpeculator,
        )

        return MultiModuleMTPSpeculator(vllm_config, device)
    elif speculative_config.method == "mtp":
        from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

        return MTPSpeculator(vllm_config, device)
    elif speculative_config.use_eagle():
        from vllm.v1.worker.gpu.spec_decode.eagle.speculator import (
            EagleSpeculator,
        )

        return EagleSpeculator(vllm_config, device)
    elif speculative_config.uses_draft_model():
        from vllm.v1.worker.gpu.spec_decode.standalone_ar.speculator import (
            StandaloneARSpeculator,
        )

        return StandaloneARSpeculator(vllm_config, device)
    elif speculative_config.use_ngram():
        from vllm.v1.worker.gpu.spec_decode.ngram.speculator import (
            NgramGPUSpeculator,
        )

        return NgramGPUSpeculator(vllm_config, device, req_states)
    else:
        raise NotImplementedError(f"{speculative_config.method} is not supported yet.")
