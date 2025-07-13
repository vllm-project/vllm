# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Callable

from vllm.compilation.cuda_graph import CUDAGraphOptions, CUDAGraphWrapper
from vllm.config import (CompilationLevel, CUDAGraphMode,
                         CUDAGraphRuntimeStyle, VllmConfig)
from vllm.logger import init_logger

logger = init_logger(__name__)

# constant for pure decode
DECODE_BOOLEN = True


class CudagraphDispatcher:
    """
    Runtime cudagraph dispatcher to switch between multiple cudagraphs.
    """

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.cudagraph_mode = self.compilation_config.cudagraph_mode
        self.no_compilation = self.compilation_config.level != \
            CompilationLevel.PIECEWISE or \
            vllm_config.model_config.enforce_eager

        self.model: Callable = None  # type: ignore
        # we lazy initialize self.model after model loading of model
        # runner have been done.

        # Dict to store cudagraph candidates for runtime dispatching.
        self.cudagraph_candidates: dict[tuple, Any] = {}

    def after_load_model(self, model: Callable):
        # add original model to cudagraph_candidates for profile run.
        assert model is not None, "model should not be None"
        self.model = model
        self.cudagraph_candidates.update({
            (CUDAGraphRuntimeStyle.NONE, ):
            self.model
        })

    def maybe_initialize_cudagraph(self, create_mixed_batch_full_cg: bool):
        assert self.model is not None, (
            "No model have been assigned to cudagraph dispatcher")
        # This should be called only after attention backend is initialized.

        if self.compilation_config.level == CompilationLevel.PIECEWISE\
                and len(self.compilation_config.splitting_ops)>0:
            self.cudagraph_candidates.update({
                (CUDAGraphRuntimeStyle.PIECEWISE, ):
                self.model
            })
            logger.debug("Piecewise cudagraph initialized")

        if self.compilation_config.cudagraph_mode == CUDAGraphMode.FULL:
            # create full cudagraph for mix prefill-decode/general batches
            if create_mixed_batch_full_cg:
                self.cudagraph_candidates.update({
                    (CUDAGraphRuntimeStyle.FULL, not DECODE_BOOLEN):
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
                    (CUDAGraphRuntimeStyle.FULL, DECODE_BOOLEN):
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

        # Some attention backends only support CUDA Graphs in pure decode.
        # If attention doesn't support CUDA Graphs for this batch, we skip them,
        # and turn back to the piecewise CUDA graphs.
        cudagraph_runtime_style = CUDAGraphRuntimeStyle.FULL if\
              attn_cuda_graphs else CUDAGraphRuntimeStyle.PIECEWISE

        # PIECEWISE would fall back to NONE if no compilation
        if cudagraph_runtime_style == CUDAGraphRuntimeStyle.PIECEWISE and \
                self.no_compilation:
            cudagraph_runtime_style = CUDAGraphRuntimeStyle.NONE

        #TODO: can we optimize above logic?
        return cudagraph_runtime_style

    def dispatch(self, cudagraph_runtime_style: CUDAGraphRuntimeStyle,
                 is_pure_decode: bool) -> Any:
        assert self.model is not None, ("No model have been assigned"
                                        "to cudagraph dispatcher")
        # if no cudagraph candidates,
        # just skip cudagraph dispatching.
        if not self.cudagraph_candidates:
            logger.warning_once("cudagraphs are not initialized."
                                " No cudagraph will be used.")
            return self.model

        # select between no cudagraph and piecewise cudagraph
        if cudagraph_runtime_style in [
                CUDAGraphRuntimeStyle.NONE, CUDAGraphRuntimeStyle.PIECEWISE
        ]:
            selected_model = self.cudagraph_candidates.get(
                (cudagraph_runtime_style, ), None)
            assert selected_model is not None, (
                "cudagraph_candidates is not"
                " correctly initialized for key: "
                f"({cudagraph_runtime_style}, ).")
        else:
            # for full cudagraph, select between general batches
            # or pure decode batches
            decode_case = (DECODE_BOOLEN,) if self.compilation_config.\
                separate_attention_routine and is_pure_decode \
                else (not DECODE_BOOLEN,)
            tuple_key = (cudagraph_runtime_style, ) + decode_case
            selected_model = self.cudagraph_candidates.get(tuple_key, None)
            assert selected_model is not None, (
                "cudagraph_candidates is not"
                " correctly initialized for key: "
                f"({cudagraph_runtime_style}, "
                f"{is_pure_decode}).")
        return selected_model
