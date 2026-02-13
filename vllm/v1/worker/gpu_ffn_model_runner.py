# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
import tqdm

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.afd_transfer.afd_connector.factory import AFDConnectorFactory
from vllm.distributed.afd_transfer.afd_connector.metadata import AFDConnectorMetadata
from vllm.distributed.communication_op import tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_world_group,
    graph_capture,
    is_global_first_rank
)
from vllm.forward_context import (
    DPMetadata,
    set_forward_context,
    get_forward_context,
)
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_loader
from vllm.utils.mem_utils import DeviceMemoryProfiler, GiB_bytes
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.compilation.monitor import set_cudagraph_capturing_enabled

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec

logger = init_logger(__name__)


class GPUFFNModelRunner(LoRAModelRunnerMixin):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = device
        self.dtype = self.model_config.dtype
        self.load_config = vllm_config.load_config

        self.afd_config = vllm_config.afd_config
        if not self.afd_config or not self.afd_config.is_ffn_server:
            raise ValueError(
                "AFD config must be provided with afd_role='ffn' for FFN server"
            )

        self._counter = 0

        # Initialize torch.profile for performance monitoring
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1000, warmup=1, active=10, repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                "./profiler_logs/ffn"
            ),
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
        )

        # Initialize CUDA graph support
        self.use_cuda_graph = not self.model_config.enforce_eager

        # self.cudagraph_batch_sizes sorts in ascending order.
        # The batch sizes in the config are in descending order.
        self.cudagraph_batch_sizes = list(
            reversed(self.vllm_config.compilation_config.cudagraph_capture_sizes)
        )

        # Storage for captured graphs, keyed by dp_metadata graph key
        # {dp_metadata_graph_key: {"graph": CUDAGraph, ...}}
        self._cuda_graphs: dict[tuple, dict] = {}
        self._graph_memory_pool = None
        self.dummy_run_call_cnt = 0

        assert self.afd_config.is_ffn_server
        self.connector = AFDConnectorFactory.create_connector(
            get_world_group().rank, get_world_group().local_rank, self.vllm_config
        )

        if getattr(self.model_config.hf_config, "text_config", None) is not None:
            self.num_layers = self.model_config.hf_config.text_config.num_hidden_layers
        else:
            self.num_layers = self.model_config.hf_config.num_hidden_layers
        
        self._execute_model_count = 0

    def get_model(self) -> nn.Module:
        return self.model

    def initialize_afd_connector(self) -> None:
        self.connector.init_afd_connector()

    def load_model(self, **kwargs) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)
        with DeviceMemoryProfiler() as m:  # noqa: SIM117
            time_before_load = time.perf_counter()
            model_loader = get_model_loader(self.load_config)

            if not hasattr(self, "model"):
                logger.info("Loading model from scratch...")
                self.model = model_loader.load_model(
                    vllm_config=self.vllm_config, model_config=self.model_config
                )
            else:
                logger.info("Model was already initialized. Loading weights inplace...")
                model_loader.load_weights(self.model, model_config=self.model_config)
            time_after_load = time.perf_counter()
        self.model_memory_usage = m.consumed_memory
        logger.info(
            "Model loading took %.4f GiB and %.6f seconds",
            self.model_memory_usage / GiB_bytes,
            time_after_load - time_before_load,
        )

        logger.info("AFD FFN Model loaded successfully")

    @staticmethod
    def _make_graph_key(dp_metadata_list: dict) -> tuple:
        """Extract a hashable key from dp_metadata_list for CUDA graph lookup.

        The key is a tuple of (stage_idx, tuple(num_tokens_across_dp_cpu))
        for each stage, sorted by stage_idx.
        """
        return tuple(
            (stage_idx, tuple(meta.num_tokens_across_dp_cpu.tolist()))
            for stage_idx, meta in sorted(dp_metadata_list.items())
        )

    def _ffn_forward(self,
                     dp_metadata_list: dict | None = None,
                     is_graph_capturing: bool = False):
        num_ubatches = len(dp_metadata_list) if dp_metadata_list else 1
        rank_ffn_output = None

        assert dp_metadata_list is not None
        self.connector.update_state_from_dp_metadata(
            dp_metadata_list, is_graph_capturing=is_graph_capturing
        )
        
        # TODO(jcz): process first_k_dense_replace
        # for layer_idx in range(self.first_k_dense_replace,self.num_layers):
        for layer_idx in range(0, self.num_layers):
            for ubatch_idx in range(num_ubatches):
                hidden_states, recv_metadata = self.connector.recv_attn_output(ubatch_idx=ubatch_idx)
                dp_metadata = dp_metadata_list.get(
                    recv_metadata.stage_idx, None
                )
                if recv_metadata is not None and recv_metadata.recv_handle_list is not None:
                    for work in recv_metadata.recv_handle_list:
                        work.wait()
                # Fallback to eager mode
                with set_forward_context(
                    attn_metadata=None, vllm_config=self.vllm_config
                ):
                    get_forward_context().dp_metadata = dp_metadata
                    rank_ffn_output = self._execute_eager_mode(
                        hidden_states, layer_idx
                    )

                recv_metadata.recv_handle_list = None
                self.connector.send_ffn_output(rank_ffn_output, recv_metadata)
        self._execute_model_count += 1
        return rank_ffn_output

    
    @torch.inference_mode()
    def execute_model(
        self, 
        scheduler_output=None, 
        intermediate_tensors=None, 
        dp_metadata_list: dict | None = None,
        is_graph_capturing: bool = False
    ):
        """Execute FFN computation for a single request"""
        self.profiler.step()
        try:
            if self.use_cuda_graph and dp_metadata_list is not None:
                graph_key = self._make_graph_key(dp_metadata_list)
                cuda_graph_info = self._cuda_graphs.get(graph_key)
                if cuda_graph_info is not None:
                    cuda_graph_info["graph"].replay()
                    logger.info(f"ffn replay cudagraph for key {graph_key}")
                else:
                    logger.warning(
                        f"No CUDA graph found for key {graph_key}, "
                        f"falling back to eager mode")
                    self._ffn_forward(
                        dp_metadata_list=dp_metadata_list,
                        is_graph_capturing=is_graph_capturing,
                    )
            else:
                logger.info(f"ffn_forward, dp_metadata_list is {dp_metadata_list}")
                self._ffn_forward(
                    dp_metadata_list=dp_metadata_list,
                    is_graph_capturing=is_graph_capturing,
                )
        except Exception as e:
            raise ValueError(f"Error computing FFN: {e}") from e
        return None  # FFN server doesn't return ModelRunnerOutput

    def _execute_eager_mode(
        self,
        hidden_states: torch.Tensor,
        current_layer_idx: int,
    ):
        """Execute FFN computation in eager mode (fallback)."""
        # Step the profiler for performance monitoring

        # Handle TP case: all-gather tensors from all TP ranks
        tp_world_size = get_tensor_model_parallel_world_size()
        if tp_world_size > 1:
            # All-gather hidden states from all TP ranks
            gathered_hidden_states = tensor_model_parallel_all_gather(
                hidden_states, dim=0
            )
            ffn_output = self.model.compute_ffn_output(
                gathered_hidden_states, current_layer_idx
            )
            # Extract the output corresponding to current rank
            start_idx = hidden_states.shape[0] * get_tensor_model_parallel_rank()
            end_idx = start_idx + hidden_states.shape[0]
            rank_ffn_output = ffn_output[start_idx:end_idx, :]
        else:
            # Single TP case
            rank_ffn_output = self.model.compute_ffn_output(
                hidden_states, current_layer_idx
            )

        return rank_ffn_output

    # Methods required for interface compatibility with GPUModelRunner
    def profile_run(self) -> None:
        """FFN servers don't need profiling."""
        pass

    def get_kv_cache_spec(self) -> dict[str, "KVCacheSpec"]:
        """FFN servers don't use KV cache."""
        return {}

    def initialize_kv_cache(self, kv_cache_config: "KVCacheConfig") -> None:
        """FFN servers don't use KV cache."""
        pass

    def _dummy_run(
        self,
        cudagraph_runtime_mode: CUDAGraphMode,
        dp_metadata_list: dict,
        is_attn_graph_capturing: bool,
    ):
        assert cudagraph_runtime_mode in {
            CUDAGraphMode.NONE,
            CUDAGraphMode.FULL,
        }

        if cudagraph_runtime_mode == CUDAGraphMode.FULL:
            if self._graph_memory_pool is None:
                self._graph_memory_pool = torch.cuda.graph_pool_handle()
            graph_key = self._make_graph_key(dp_metadata_list)
            cudagraph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cudagraph, pool=self._graph_memory_pool):
                output = self._ffn_forward(
                    dp_metadata_list=dp_metadata_list,
                    is_graph_capturing=is_attn_graph_capturing,
                )
            self._cuda_graphs[graph_key] = {
                "graph": cudagraph,
                "input_hidden_states": output,
                "output": output,
            }
            logger.info(f"Captured CUDA graph for key {graph_key}")
        else:
            self._ffn_forward(
                dp_metadata_list=dp_metadata_list,
                is_graph_capturing=is_attn_graph_capturing,
            )
        self.dummy_run_call_cnt += 1

    def capture_model(
        self,
        dp_metadata_list: Optional[dict] = None,
    ) -> int:
        """Capture CUDA graphs for FFN operations.
        When dp_metadata_list and is_graph_capturing are provided (e.g. from attn
        side), use them without recv so that no extra recv is needed.
        """
        if not self.use_cuda_graph:
            logger.warning("Skipping CUDA graph capture.")
            return 0

        logger.info("Starting CUDA graph capture for FFN operations...")
        start_time = time.perf_counter()
        start_free_gpu_memory = torch.cuda.mem_get_info()[0]

        # Create memory pool for graphs
        if self._graph_memory_pool is None:
            self._graph_memory_pool = torch.cuda.graph_pool_handle()

        # Capture graphs for each layer and different batch sizes
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        set_cudagraph_capturing_enabled(True)
        with graph_capture(device=self.device):
            self._capture_graphs(
                cudagraph_runtime_mode=CUDAGraphMode.FULL,
                dp_metadata_list=dp_metadata_list)
        set_cudagraph_capturing_enabled(False)

        end_time = time.perf_counter()
        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory

        logger.info(
            "FFN CUDA graph capturing finished in %.0f secs, took %.2f GiB",
            elapsed_time,
            cuda_graph_size / (1 << 30),
        )
        return cuda_graph_size
    
    def _capture_graphs(
        self,
        cudagraph_runtime_mode: CUDAGraphMode,
        dp_metadata_list: dict,
    ):
        assert cudagraph_runtime_mode == CUDAGraphMode.FULL
        self._dummy_run(
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            dp_metadata_list=dp_metadata_list,
            is_attn_graph_capturing=True,
        )

    def _run_ffn_computation(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        capture_mode: bool = False,
    ):
        """Run FFN computation for graph capture or replay."""
        tp_world_size = get_tensor_model_parallel_world_size()
        if tp_world_size > 1:
            # Handle TP case: all-gather tensors from all TP ranks
            gathered_hidden_states = tensor_model_parallel_all_gather(
                hidden_states, dim=0
            )
            ffn_output = self.model.compute_ffn_output(
                gathered_hidden_states, layer_idx
            )

            # Extract the output corresponding to current rank
            start_idx = hidden_states.shape[0] * get_tensor_model_parallel_rank()
            end_idx = start_idx + hidden_states.shape[0]
            rank_ffn_output = ffn_output[start_idx:end_idx, :]
        else:
            # Single TP case
            rank_ffn_output = self.model.compute_ffn_output(
                hidden_states, layer_idx
            )

        return rank_ffn_output

    def _dummy_sampler_run(self, hidden_states: torch.Tensor) -> None:
        """FFN servers don't use samplers."""
        pass

    def update_config(self, overrides: dict[str, Any]) -> None:
        """Update configuration for FFN model runner."""
        allowed_config_names = {"load_config", "model_config"}
        for config_name, config_overrides in overrides.items():
            assert config_name in allowed_config_names, (
                f"Config `{config_name}` not supported. "
                f"Allowed configs: {allowed_config_names}"
            )
            config = getattr(self, config_name)
            from vllm.config import update_config

            new_config = update_config(config, config_overrides)
            setattr(self, config_name, new_config)

    def reload_weights(self) -> None:
        """Reload model weights for FFN model runner."""
        assert getattr(self, "model", None) is not None, (
            "Cannot reload weights before model is loaded."
        )
        model_loader = get_model_loader(self.load_config)
        logger.info("Reloading weights inplace...")
        model = self.get_model()
        model_loader.load_weights(model, model_config=self.model_config)

    @property
    def lora_config(self):
        """FFN servers don't support LoRA."""
        return None

    @property
    def is_pooling_model(self) -> bool:
        """FFN servers are not pooling models."""
        return False

    def _dummy_pooler_run(self, hidden_states: torch.Tensor):
        """FFN servers don't have poolers."""
        pass

    def get_supported_tasks(self):
        """Get supported tasks for FFN model runner."""
        return []

    def _get_num_input_tokens(self, num_scheduled_tokens: int) -> int:
        """Get number of input tokens for FFN model runner."""
        return num_scheduled_tokens

    def take_draft_token_ids(self, **kwargs):
        """FFN servers don't support draft tokens."""
        pass

    @property
    def eplb_state(self):
        """FFN servers don't have EPLB state."""
        return None

    def ensure_kv_transfer_shutdown(self):
        """FFN servers don't need KV transfer shutdown."""
        pass

    def save_tensorized_model(
        self,
        tensorizer_config: "TensorizerConfig",
    ) -> None:
        """FFN servers don't support tensorized model saving."""
        raise NotImplementedError("FFN servers don't support tensorized model saving")
