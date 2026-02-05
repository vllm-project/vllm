# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, cast

import torch
import torch.nn as nn
from tqdm import tqdm

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.parallel_state import is_global_first_rank
from vllm.forward_context import (
    BatchDescriptor,
    set_forward_context,
)
from vllm.logger import init_logger
from vllm.multimodal import BatchedTensorInputs
from vllm.multimodal.processing import BaseDummyInputsBuilder
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

logger = init_logger(__name__)


class MMEncoderCudagraphManager:
    def __init__(
        self,
        vllm_config: VllmConfig,
        dummy_input_builder: BaseDummyInputsBuilder[Any],
    ):
        self.vllm_config = vllm_config
        self.dispatcher = CudagraphDispatcher(self.vllm_config)
        self.dummy_input_builder = dummy_input_builder

        # Check if using data parallel mode for ViT
        self.is_vit_dp_mode = self._check_vit_dp_mode(vllm_config)

    def _check_vit_dp_mode(self, vllm_config: VllmConfig) -> bool:
        """Check if ViT is running in data parallel mode."""
        mm_config = getattr(vllm_config.model_config, "multimodal_config", None)
        if mm_config is None:
            return False

        mm_encoder_tp_mode = mm_config.mm_encoder_tp_mode
        tp_size = vllm_config.parallel_config.tensor_parallel_size

        return mm_encoder_tp_mode == "data" and tp_size > 1

    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None:
        """Initialize cudagraph dispatcher keys for MM Encoder.

        MM Encoder only supports PIECEWISE cudagraphs.
        """
        if cudagraph_mode.mixed_mode() in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]:
            mm_cudagraph_mode = CUDAGraphMode.PIECEWISE
        else:
            mm_cudagraph_mode = CUDAGraphMode.NONE

        max_capture_size = (
            self.vllm_config.compilation_config.max_mm_encoder_cudagraph_capture_size
        )

        capture_sizes = (
            self.vllm_config.compilation_config.mm_encoder_cudagraph_capture_sizes
        )

        self.dispatcher.initialize_cudagraph_keys(
            mm_cudagraph_mode,
            capture_sizes=capture_sizes,
            max_capture_size=max_capture_size,
            enable_lora=False,
        )

    def dispatch_and_pad_mm_input(
        self,
        mm_kwargs_group: BatchedTensorInputs,
    ) -> tuple[CUDAGraphMode, BatchDescriptor | None, int, BatchedTensorInputs]:
        pixel_values = cast(torch.Tensor, mm_kwargs_group["pixel_values"])
        num_tokens = pixel_values.shape[0]

        image_grid_thw = mm_kwargs_group["image_grid_thw"]
        if isinstance(image_grid_thw, torch.Tensor):
            original_num_imgs = image_grid_thw.shape[0]
        else:
            original_num_imgs = len(image_grid_thw)

        # Dispatch to get the target padded size
        cudagraph_runtime_mode, batch_descriptor = self.dispatcher.dispatch(
            num_tokens=num_tokens,
        )
        target_num_tokens = batch_descriptor.num_tokens

        # Pad if necessary
        if target_num_tokens > num_tokens:
            # Pad pixel_values
            padding_size = target_num_tokens - num_tokens
            padding_mm_inputs = self.dummy_input_builder.get_dummy_mm_encoder_input(
                padding_size,
            )

            mm_kwargs_group["pixel_values"] = torch.cat(
                [pixel_values, padding_mm_inputs["pixel_values"]], dim=0
            )

            padding_image_grid_thw = padding_mm_inputs["image_grid_thw"]
            if isinstance(image_grid_thw, torch.Tensor):
                mm_kwargs_group["image_grid_thw"] = torch.cat(
                    [image_grid_thw, padding_image_grid_thw], dim=0
                )
            else:
                mm_kwargs_group["image_grid_thw"] = (
                    image_grid_thw + padding_image_grid_thw.tolist()
                )

        return (
            cudagraph_runtime_mode,
            batch_descriptor,
            original_num_imgs,
            mm_kwargs_group,
        )

    def capture_graph(
        self,
        num_tokens: int,
        model: nn.Module,
        cudagraph_mode: CUDAGraphMode,
    ) -> None:
        dummy_mm_inputs = self.dummy_input_builder.get_dummy_mm_encoder_input(
            num_tokens
        )

        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)

        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_mode,
            batch_descriptor=batch_descriptor,
        ):
            model.embed_multimodal(**dummy_mm_inputs)

    @torch.inference_mode()
    def capture(
        self,
        model: nn.Module,
        batch_descs: "list[BatchDescriptor]",
        cudagraph_mode: CUDAGraphMode,
    ) -> None:
        self.vllm_config.in_mm_encoder_tracing = True

        if is_global_first_rank():
            batch_descriptors: Any = tqdm(
                batch_descs,
                disable=not self.vllm_config.load_config.use_tqdm_on_load,
                desc="Capturing MM_Encoder CUDA graphs (PIECEWISE)",
            )
        else:
            batch_descriptors = batch_descs

        for batch_desc in batch_descriptors:
            capture_size = batch_desc.num_tokens
            self.capture_graph(
                capture_size,
                model=model,
                cudagraph_mode=cudagraph_mode,
            )

        self.vllm_config.in_mm_encoder_tracing = False
