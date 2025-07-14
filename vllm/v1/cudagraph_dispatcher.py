# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Callable, NamedTuple, Optional

from vllm.compilation.cuda_graph import CUDAGraphOptions, CUDAGraphWrapper
from vllm.config import (CompilationLevel, CUDAGraphMode,
                         CUDAGraphRuntimeStyle, VllmConfig)
from vllm.logger import init_logger

logger = init_logger(__name__)


class DispatchKey(NamedTuple):
    """
    Key for dispatching cudagraphs.
    """
    cudagraph_runtime_style: CUDAGraphRuntimeStyle
    # Be aware that is_pure_decode should be default None
    # for both piecewise cudagraphs and no cudagraphs.
    is_pure_decode: Optional[bool] = None


class CudagraphDispatcher:
    """
    Runtime cudagraph dispatcher to switch between multiple cudagraphs.
    """

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.cudagraph_mode = self.compilation_config.cudagraph_mode

        self.model: Callable = None  # type: ignore
        # we lazy initialize self.model once the model loading of
        # runner have been done.

        # Dict to store cudagraph candidates for runtime dispatching.
        self.cudagraph_candidates: dict[DispatchKey, Any] = {}

        # Verify if correctly piecewise compilation for attention.
        piecewise_compilation = not vllm_config.model_config.enforce_eager\
            and self.compilation_config.level == CompilationLevel.PIECEWISE
        self.piecewise_attn_compilation = piecewise_compilation and\
            all(op in self.compilation_config.splitting_ops for op in [
            "vllm.unified_attention", "vllm.unified_attention_with_output"])

    def after_load_model(self, model: Callable):
        # add original model to cudagraph_candidates for profile run.
        assert model is not None, "model should not be None"
        self.model = model
        self.cudagraph_candidates.update(
            {DispatchKey(CUDAGraphRuntimeStyle.NONE): self.model})
        logger.debug("Cudagraph candidates for NONE style initialized")

    def maybe_initialize_cudagraph(self, create_mixed_batch_full_cg: bool):
        assert self.model is not None, (
            "No model have been assigned to cudagraph dispatcher")
        # This should be called only after attention backend is initialized.

        if self.compilation_config.level == CompilationLevel.PIECEWISE\
                and len(self.compilation_config.splitting_ops)>0:
            self.cudagraph_candidates.update(
                {DispatchKey(CUDAGraphRuntimeStyle.PIECEWISE): self.model})
            logger.debug("Piecewise cudagraph initialized")

        if self.compilation_config.cudagraph_mode == CUDAGraphMode.FULL:
            # create full cudagraph for mix prefill-decode/general batches
            if create_mixed_batch_full_cg:
                self.cudagraph_candidates.update({
                    DispatchKey(CUDAGraphRuntimeStyle.FULL, False):
                    CUDAGraphWrapper(self.model,
                                     self.vllm_config,
                                     runtime_style=CUDAGraphRuntimeStyle.FULL,
                                     cudagraph_options=CUDAGraphOptions(
                                         usage_str="full/mixed"))
                })
                logger.debug("Full cudagraph for mixed batches initialized")
            # always create full cudagraph for pure decode batches if speparate
            # attention routine.
            if self.compilation_config.separate_attention_routine:
                self.cudagraph_candidates.update({
                    DispatchKey(CUDAGraphRuntimeStyle.FULL, True):
                    CUDAGraphWrapper(self.model,
                                     self.vllm_config,
                                     runtime_style=CUDAGraphRuntimeStyle.FULL,
                                     cudagraph_options=CUDAGraphOptions(
                                         usage_str="full/pure-decode"))
                })
                logger.debug(
                    "Full cudagraph for pure decode batches initialized")

    def get_cudagraph_runtime_style(
            self, attn_cuda_graphs: bool) -> CUDAGraphRuntimeStyle:  # noqa

        if self.cudagraph_mode == CUDAGraphMode.NONE:
            return CUDAGraphRuntimeStyle.NONE

        if self.cudagraph_mode == CUDAGraphMode.PIECEWISE:
            # safe to direct return as we have already checked
            # CUDAGraphMode.PIECEWISE is compatible only when
            # enable vllm compilation.
            return CUDAGraphRuntimeStyle.PIECEWISE

        # Otherwise, for modes that enable full cudagraph.
        assert self.cudagraph_mode == CUDAGraphMode.FULL
        # If attention backend supports full cudagraphs for current batch,
        # run with full cudagraphs.
        if attn_cuda_graphs:
            return CUDAGraphRuntimeStyle.FULL

        # Fall back to piecewise cudagraphs if possible
        if self.piecewise_attn_compilation:
            return CUDAGraphRuntimeStyle.PIECEWISE

        # Otherwise, fall back to running entirely without cudagraphs
        return CUDAGraphRuntimeStyle.NONE

    def dispatch(self, cudagraph_runtime_style: CUDAGraphRuntimeStyle,
                 is_pure_decode: bool) -> Any:
        assert self.model is not None, ("No model have been assigned"
                                        "to cudagraph dispatcher")
        # if no cudagraph candidates, just skip dispatching.
        if not self.cudagraph_candidates:
            logger.warning_once("cudagraphs are not initialized."
                                " No cudagraph will be used.")
            return self.model

        # select between no cudagraph and piecewise cudagraph
        if cudagraph_runtime_style in [
                CUDAGraphRuntimeStyle.NONE, CUDAGraphRuntimeStyle.PIECEWISE
        ]:
            dispatchkey = DispatchKey(cudagraph_runtime_style)
            selected_model = self.cudagraph_candidates.get(dispatchkey, None)
        else:
            # for full cudagraph, select between mixed batches
            # or pure decode batches
            decode_case = self.compilation_config.separate_attention_routine\
                and is_pure_decode
            dispatchkey = DispatchKey(cudagraph_runtime_style, decode_case)
            selected_model = self.cudagraph_candidates.get(dispatchkey, None)
        assert selected_model is not None, (
            f"cudagraph_candidates is not correctly initialized for key: "
            f"{dispatchkey}")
        return selected_model
