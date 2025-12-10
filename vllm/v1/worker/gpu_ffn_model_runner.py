# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed.afd_transfer.afd_connector.factory import AFDConnectorFactory
from vllm.distributed.afd_transfer.afd_connector.metadata import AFDConnectorMetadata
from vllm.distributed.communication_op import tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_world_group,
    graph_capture,
)
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_loader
from vllm.utils.mem_utils import DeviceMemoryProfiler, GiB_bytes
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

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
                wait=6000 * 27 + 4000 * 27 * 2, warmup=1, active=30, repeat=1
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

        # Storage for captured graphs
        self._cuda_graphs: dict[
            tuple[int, int], torch.cuda.CUDAGraph
        ] = {}  # {(layer_idx, num_tokens): CUDAGraph}
        self._graph_memory_pool = None

        assert self.afd_config.is_ffn_server
        self.connector = AFDConnectorFactory.create_connector(
            get_world_group().rank, get_world_group().local_rank, self.vllm_config
        )

        if getattr(self.model_config.hf_config, "text_config", None) is not None:
            self.num_layers = self.model_config.hf_config.text_config.num_hidden_layers
        else:
            self.num_layers = self.model_config.hf_config.num_hidden_layers

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

    def _get_current_layer_idx(self) -> int:
        return (self._counter // self.afd_config.num_afd_stages) % self.num_layers

    @torch.inference_mode()
    def execute_model(self, scheduler_output=None, intermediate_tensors=None):
        """Execute FFN computation for a single request"""
        # scheduler_output and intermediate_tensors are unused in FFN server
        # mode
        self.profiler.step()

        try:
            hidden_states, recv_metadata = self.connector.recv_attn_output()
            current_layer_idx = recv_metadata.layer_idx
            logger.info(
                f"layer {current_layer_idx} moe recv hidden states type:{type(hidden_states)}, shape:{hidden_states.shape}"
            )
            num_tokens = hidden_states.shape[0]
            if recv_metadata is not None and recv_metadata.recv_handle_list is not None:
                for work in recv_metadata.recv_handle_list:
                    work.wait()
            # Try to use CUDA graph if available
            cuda_graph_info = self._find_cuda_graph(current_layer_idx, num_tokens)
            if cuda_graph_info is not None:
                # Use captured CUDA graph for computation
                with set_forward_context(
                    attn_metadata=None, vllm_config=self.vllm_config
                ):
                    rank_ffn_output = self._execute_with_cuda_graph(
                        hidden_states, cuda_graph_info
                    )
            else:
                # Fallback to eager mode
                with set_forward_context(
                    attn_metadata=None, vllm_config=self.vllm_config
                ):
                    rank_ffn_output = self._execute_eager_mode(
                        hidden_states, current_layer_idx
                    )

            recv_metadata.recv_handle_list = None
            self.connector.send_ffn_output(rank_ffn_output, recv_metadata)
        except Exception as e:
            raise ValueError(f"Error computing FFN: {e}") from e
        finally:
            self._counter += 1
            if self._counter == self.num_layers * self.afd_config.num_afd_stages:
                self._counter = 0
        return None  # FFN server doesn't return ModelRunnerOutput

    def _execute_with_cuda_graph(
        self, hidden_states: torch.Tensor, cuda_graph_info: dict
    ):
        """Execute FFN computation using captured CUDA graph."""
        graph = cuda_graph_info["graph"]
        input_tensor = cuda_graph_info["input_hidden_states"]
        output_tensor = cuda_graph_info["output"]

        # Copy input data to graph's input tensor
        # Handle padding if necessary
        actual_tokens = hidden_states.shape[0]
        graph_tokens = input_tensor.shape[0]

        if actual_tokens <= graph_tokens:
            # Copy actual data and pad with zeros if needed
            input_tensor[:actual_tokens].copy_(hidden_states)
            if actual_tokens < graph_tokens:
                input_tensor[actual_tokens:].zero_()
        else:
            raise ValueError(
                f"Input size {actual_tokens} exceeds graph capacity {graph_tokens}"
            )

        # Replay the captured graph
        graph.replay()

        # Return only the actual output (without padding)
        return output_tensor[:actual_tokens].clone()

    def _execute_eager_mode(
        self,
        hidden_states: torch.Tensor,
        current_layer_idx: int,
        recv_metadata: AFDConnectorMetadata = None,
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
                current_layer_idx, gathered_hidden_states
            )
            # Extract the output corresponding to current rank
            start_idx = hidden_states.shape[0] * get_tensor_model_parallel_rank()
            end_idx = start_idx + hidden_states.shape[0]
            rank_ffn_output = ffn_output[start_idx:end_idx, :]
        else:
            # Single TP case
            rank_ffn_output = self.model.compute_ffn_output(
                current_layer_idx, hidden_states
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

    def _dummy_run(self, num_tokens: int = 1, **kwargs) -> torch.Tensor:
        """FFN servers don't need dummy runs."""
        # Return a dummy tensor for interface compatibility
        return torch.zeros(
            num_tokens,
            self.model_config.hf_config.hidden_size,
            dtype=self.dtype,
            device=self.device,
        )

    def capture_model(self) -> int:
        """Capture CUDA graphs for FFN operations."""
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
        with graph_capture(device=self.device):
            for layer_idx in range(self.num_layers):
                for num_tokens in reversed(self.cudagraph_batch_sizes):
                    with set_forward_context(
                        attn_metadata=None, vllm_config=self.vllm_config
                    ):
                        self._capture_graph_for_layer_and_size(layer_idx, num_tokens)

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

    def _capture_graph_for_layer_and_size(self, layer_idx: int, num_tokens: int):
        """Capture CUDA graph for specific layer and number of tokens."""
        # Create dummy hidden states
        dummy_hidden_states = torch.randn(
            num_tokens,
            self.model_config.hf_config.hidden_size,
            dtype=self.dtype,
            device=self.device,
        )

        # Warm up the operations for this specific layer
        for _ in range(self.vllm_config.compilation_config.cudagraph_num_of_warmups):
            self._run_ffn_computation(
                dummy_hidden_states, layer_idx=layer_idx, capture_mode=True
            )

        # Create and capture the graph
        graph = torch.cuda.CUDAGraph()

        # Start graph capture
        with torch.cuda.graph(graph, pool=self._graph_memory_pool):
            output = self._run_ffn_computation(
                dummy_hidden_states, layer_idx=layer_idx, capture_mode=True
            )

        # Store the captured graph with layer and token count as key
        self._cuda_graphs[(layer_idx, num_tokens)] = {
            "graph": graph,
            "input_hidden_states": dummy_hidden_states,
            "output": output,
        }

        logger.debug(
            "Captured CUDA graph for layer %s with %s tokens", layer_idx, num_tokens
        )

    def _run_ffn_computation(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int | None = None,
        capture_mode: bool = False,
    ):
        """Run FFN computation for graph capture or replay."""
        if layer_idx is None:
            current_layer_idx = self._get_current_layer_idx() if not capture_mode else 0
        else:
            current_layer_idx = layer_idx

        tp_world_size = get_tensor_model_parallel_world_size()
        if tp_world_size > 1:
            # Handle TP case: all-gather tensors from all TP ranks
            gathered_hidden_states = tensor_model_parallel_all_gather(
                hidden_states, dim=0
            )
            ffn_output = self.model.compute_ffn_output(
                current_layer_idx, gathered_hidden_states
            )

            # Extract the output corresponding to current rank
            start_idx = hidden_states.shape[0] * get_tensor_model_parallel_rank()
            end_idx = start_idx + hidden_states.shape[0]
            rank_ffn_output = ffn_output[start_idx:end_idx, :]
        else:
            # Single TP case
            rank_ffn_output = self.model.compute_ffn_output(
                current_layer_idx, hidden_states
            )

        return rank_ffn_output

    def _find_cuda_graph(self, layer_idx: int, num_tokens: int):
        """Find the smallest graph that can handle the given layer and
        number of tokens."""
        if not self.use_cuda_graph:
            return None

        # Find the minimum capture size that can handle num_tokens for this
        # layer
        for capture_size in self.cudagraph_batch_sizes:
            if num_tokens <= capture_size:
                return self._cuda_graphs.get((layer_idx, capture_size))
        return None

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
