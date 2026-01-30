# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import weakref
from functools import partial

import pytest
import torch

from vllm import LLM
from vllm.config import CompilationConfig, CUDAGraphMode
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.forward_context import set_forward_context
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.executor.multiproc_executor import MultiprocExecutor


def _worker_embed_multimodal(
    worker, vllm_config, cudagraph_runtime_mode, batch_descriptor, multi_modal_data
):
    """Helper function to run multimodal embedding on a worker.
    This function sets up the necessary forward context for tensor-parallel (TP)
    execution and then calls the model's `embed_multimodal` method.
    Note: For data-parallel (DP) mode, the forward context is typically
          created and managed within the
          vision.py:run_dp_sharded_mrope_vision_model(), which would override the
          context set here.
    Args:
        worker: The worker instance containing the model runner.
        vllm_config: The vLLM engine configuration.
        cudagraph_runtime_mode: The runtime mode for CUDA graph execution.
        batch_descriptor: An object describing the current batch.
        multi_modal_data: A dictionary of keyword arguments to be passed to
            the model's `embed_multimodal` method.
    Returns:
        The output from the model's `embed_multimodal` method.
    """

    # Access model via worker.model_runner.model
    # Note: Accessing internal attributes. Assuming V1 worker structure.
    model = worker.model_runner.model

    # Move multi_modal_data to the model's device
    target_device = next(model.parameters()).device
    multi_modal_data = {
        k: v.to(target_device) if isinstance(v, torch.Tensor) else v
        for k, v in multi_modal_data.items()
    }

    with (
        set_forward_context(
            None,
            vllm_config=vllm_config,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=batch_descriptor,
        ),
        torch.inference_mode(),
    ):
        ans = model.embed_multimodal(**multi_modal_data)
        torch.cuda.synchronize()
        return ans


# Format: (model_name, tp_size, mm_encoder_tp_mode)
TEST_CONFIGS = [
    ("Qwen/Qwen2.5-VL-3B-Instruct", 1, "weights"),
    ("Qwen/Qwen3-VL-4B-Instruct", 1, "weights"),
    # TP/DP modes with 2 GPUs
    ("Qwen/Qwen2.5-VL-3B-Instruct", 2, "data"),
    ("Qwen/Qwen2.5-VL-3B-Instruct", 2, "weights"),
    ("Qwen/Qwen3-VL-4B-Instruct", 2, "data"),
    ("Qwen/Qwen3-VL-4B-Instruct", 2, "weights"),
]


@pytest.fixture(
    params=TEST_CONFIGS, ids=lambda x: f"{x[0].split('/')[-1]}-tp{x[1]}-{x[2]}"
)
def llm(request):
    model_name, tp_size, mm_mode = request.param

    if torch.cuda.device_count() < tp_size:
        pytest.skip(f"Not enough GPUs for tp_size={tp_size}")

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # Common configuration
    common_args = {
        "model": model_name,
        "trust_remote_code": True,
        "max_model_len": 4096,
        "max_num_seqs": 16,
        "gpu_memory_utilization": 0.2,
        "tensor_parallel_size": tp_size,
        "mm_encoder_tp_mode": mm_mode,
    }

    # Initialize LLM with ViT CUDA graph enabled (piecewise)
    # We only need one LLM instance. For eager execution, we will force
    # cudagraph_runtime_mode=NONE at runtime.
    llm_instance = None
    try:
        llm_instance = LLM(
            **common_args,
            compilation_config=CompilationConfig(
                cudagraph_mode="PIECEWISE",
                compile_mm_encoder=True,
                mm_encoder_cudagraph_capture_sizes=[64, 128, 256],
            ),
        )
        print(f"LLM initialized for {model_name} tp={tp_size} mode={mm_mode}")
        yield weakref.proxy(llm_instance)
    finally:
        print("Cleaning up LLM after testing.")
        if llm_instance:
            # Ensure model executor and workers are properly shut down
            # llm_instance.llm_engine is vllm.v1.engine.llm_engine.LLMEngine
            # which has engine_core (InprocClient).
            if hasattr(llm_instance.llm_engine, "engine_core"):
                llm_instance.llm_engine.engine_core.shutdown()
            del llm_instance

        # Clean up distributed environment
        cleanup_dist_env_and_memory()


class TestQwenVLCUDAGraph:
    def _run_embed_multimodal(
        self, llm, multi_modal_data, num_patches, force_eager=False
    ):
        """Runs the multimodal embedding process, potentially with CUDA graphs.
        This method manually constructs a CudagraphDispatcher because accessing the
        one within the GPU model runner is difficult. It then dispatches based on
        the number of image patches to determine the appropriate CUDA graph or
        eager mode for execution. The actual embedding is performed on the
        worker(s) via an RPC call.
        Args:
            llm: The LLM object containing the model engine and configuration.
            multi_modal_data: A dictionary containing the multimodal data to be
                processed.
            num_patches: The number of image patches, used to determine the
                number of tokens for the dispatcher.
            force_eager: If True, forces the execution to run in eager mode,
                bypassing CUDA graphs.
        Returns:
            The outputs from the multimodal embedding process executed on the
            worker.
        """
        vllm_config = llm.llm_engine.vllm_config

        dispatcher = CudagraphDispatcher(vllm_config)
        dispatcher.initialize_cudagraph_keys(
            cudagraph_mode=vllm_config.compilation_config.cudagraph_mode,
            uniform_decode_query_len=1,
        )

        # Dispatch to get runtime mode and batch descriptor
        cudagraph_runtime_mode, batch_descriptor = dispatcher.dispatch(
            num_tokens=num_patches,
            uniform_decode=False,
            has_lora=False,
            is_mm_encoder=True,
        )

        model_executor = llm.llm_engine.model_executor

        rpc_kwargs = {}
        # Use collective_rpc to execute on driver worker (rank 0)
        if isinstance(model_executor, MultiprocExecutor):
            rpc_kwargs["unique_reply_rank"] = 0
        # If force_eager is True, override the runtime mode to NONE
        if force_eager:
            cudagraph_runtime_mode = CUDAGraphMode.NONE
        else:
            multi_modal_data["cudagraph_dispatcher"] = dispatcher
        outputs = model_executor.collective_rpc(
            partial(
                _worker_embed_multimodal,
                vllm_config=vllm_config,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=batch_descriptor,
                multi_modal_data=multi_modal_data,
            ),
            **rpc_kwargs,
        )

        if isinstance(outputs, list) and len(outputs) == 1:
            outputs = outputs[0]
        return outputs

    def test_vit_cudagraph_consistency(self, llm):
        print("Starting test for ViT CUDA graph consistency.")

        model_name = llm.llm_engine.vllm_config.model_config.model
        # Qwen3-VL uses patch_size=16, temporal_patch_size=2 -> 16*16*3*2 = 1536
        # Qwen2.5-VL uses patch_size=14, temporal_patch_size=2 -> 14*14*3*2 = 1176
        input_dim = 1536 if "Qwen3-VL" in model_name else 1176

        num_patches = 64
        for num_imgs in [1, 2, 4]:
            image_grid_thw = torch.tensor(
                [[1, 2, num_patches // 2]] * num_imgs, dtype=torch.long, device="cpu"
            )
            pixel_values = torch.rand(
                (num_patches * num_imgs, input_dim), dtype=torch.bfloat16, device="cpu"
            )

            multi_modal_data = {
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            }
            print(
                "Running inference with single LLM (Piecewise vs Eager via context)."
                "num_imgs:",
                num_imgs,
            )

            # Run with Piecewise CUDA Graph
            piecewise_outputs = self._run_embed_multimodal(
                llm, multi_modal_data, num_patches * num_imgs, force_eager=False
            )

            # Run with Eager Mode (simulated by setting runtime mode to NONE)
            eager_outputs = self._run_embed_multimodal(
                llm, multi_modal_data, num_patches * num_imgs, force_eager=True
            )

            if isinstance(piecewise_outputs, torch.Tensor):
                assert torch.allclose(
                    piecewise_outputs, eager_outputs, atol=1e-3, rtol=1e-5
                ), (
                    f"num_imgs: {num_imgs}. Piecewise and Eager outputs do not match. "
                    "Max abs diff: "
                    f"{torch.max(torch.abs(piecewise_outputs - eager_outputs))}. "
                    "Max rel diff: "
                    f"{
                        torch.max(
                            torch.abs(piecewise_outputs - eager_outputs)
                            / (torch.abs(eager_outputs) + 1e-8)
                        )
                    }"
                )
            elif isinstance(piecewise_outputs, tuple):
                assert isinstance(eager_outputs, tuple), (
                    "Output types mismatch, piecewise is tuple but eager is not."
                )
                assert len(piecewise_outputs) == len(eager_outputs), (
                    "Output tuple lengths mismatch."
                )
                for i, (p_out, e_out) in enumerate(
                    zip(piecewise_outputs, eager_outputs)
                ):
                    assert torch.allclose(p_out, e_out, atol=1e-3, rtol=1e-5), (
                        f"num_imgs: {num_imgs}. "
                        f"Tuple element {i} does not match. "
                        "Max abs diff: "
                        f"{torch.max(torch.abs(p_out - e_out))}. "
                        "Max rel diff: "
                        f"{
                            torch.max(
                                torch.abs(p_out - e_out) / (torch.abs(e_out) + 1e-8)
                            )
                        }"
                    )
            else:
                raise TypeError(f"Unsupported output type: {type(piecewise_outputs)}")
