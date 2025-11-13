# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Parallel context for user-defined models.

This module provides a simple interface for user models to access vLLM's
parallel configuration (tensor parallelism, pipeline parallelism, etc.)
without needing to understand vLLM's internal distributed state management.

Example usage:
    ```python
    def build_my_model(vllm_config, parallel_context):
        # Access parallel information
        tp_size = parallel_context.get_tensor_parallel_world_size()
        tp_rank = parallel_context.get_tensor_parallel_rank()

        # Use to configure model's parallel layers
        model = MyModel(config=vllm_config.hf_config, tp_size=tp_size)
        return model
    ```
"""

from dataclasses import dataclass

from vllm.distributed import parallel_state


@dataclass
class ParallelContext:
    """
    Context object providing parallel configuration information to user models.

    This class wraps vLLM's internal parallel state and provides a clean
    interface for user models to access parallelism settings.

    Attributes:
        tensor_model_parallel_size: Number of GPUs used for tensor parallelism
        pipeline_model_parallel_size: Number of GPUs used for pipeline parallelism
        data_parallel_size: Number of data parallel replicas
    """

    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    data_parallel_size: int = 1

    def get_tensor_parallel_rank(self) -> int:
        """
        Get the rank of this worker within the tensor parallel group.

        Returns:
            Rank in range [0, tensor_model_parallel_size), or 0 if not initialized
        """
        try:
            return parallel_state.get_tensor_model_parallel_rank()
        except (AssertionError, AttributeError):
            # Parallel state not initialized (e.g., in tests or single-GPU mode)
            return 0

    def get_tensor_parallel_world_size(self) -> int:
        """
        Get the size of the tensor parallel group (number of GPUs for TP).

        Returns:
            Number of GPUs used for tensor parallelism
        """
        return self.tensor_model_parallel_size

    def get_pipeline_parallel_rank(self) -> int:
        """
        Get the rank of this worker within the pipeline parallel group.

        Returns:
            Rank in range [0, pipeline_model_parallel_size), or 0 if not initialized
        """
        try:
            return parallel_state.get_pipeline_model_parallel_rank()
        except (AssertionError, AttributeError):
            # Parallel state not initialized (e.g., in tests or single-GPU mode)
            return 0

    def get_pipeline_parallel_world_size(self) -> int:
        """
        Get the size of the pipeline parallel group (number of stages).

        Returns:
            Number of pipeline stages
        """
        return self.pipeline_model_parallel_size

    def get_data_parallel_size(self) -> int:
        """
        Get the number of data parallel replicas.

        Returns:
            Number of data parallel replicas
        """
        return self.data_parallel_size

    def get_tp_process_group(self):
        """
        Get the tensor parallel process group (PyTorch ProcessGroup).

        This is the actual torch.distributed.ProcessGroup object that can be
        passed to external parallelism libraries like Megatron-LM.

        Returns:
            torch.distributed.ProcessGroup for tensor parallelism, or None if
            not initialized

        Example:
            ```python
            def build_model(vllm_config, parallel_context):
                tp_group = parallel_context.get_tp_process_group()

                # Pass to Megatron layers
                from megatron.core.tensor_parallel import ColumnParallelLinear

                layer = ColumnParallelLinear(
                    hidden_size,
                    output_size,
                    tp_group=tp_group,  # Clean!
                )
            ```
        """
        try:
            tp_coordinator = parallel_state.get_tp_group()
            return tp_coordinator.device_group
        except (AssertionError, AttributeError):
            # Parallel state not initialized
            return None

    def get_pp_process_group(self):
        """
        Get the pipeline parallel process group (PyTorch ProcessGroup).

        Returns:
            torch.distributed.ProcessGroup for pipeline parallelism, or None
            if not initialized
        """
        try:
            pp_coordinator = parallel_state.get_pp_group()
            return pp_coordinator.device_group
        except (AssertionError, AttributeError):
            # Parallel state not initialized
            return None

    @staticmethod
    def from_config(parallel_config) -> "ParallelContext":
        """
        Create a ParallelContext from vLLM's ParallelConfig.

        Args:
            parallel_config: vLLM's ParallelConfig object

        Returns:
            ParallelContext instance
        """
        from vllm.config import ParallelConfig

        if not isinstance(parallel_config, ParallelConfig):
            raise TypeError(f"Expected ParallelConfig, got {type(parallel_config)}")

        return ParallelContext(
            tensor_model_parallel_size=parallel_config.tensor_parallel_size,
            pipeline_model_parallel_size=parallel_config.pipeline_parallel_size,
            data_parallel_size=parallel_config.data_parallel_size,
        )


__all__ = ["ParallelContext"]
