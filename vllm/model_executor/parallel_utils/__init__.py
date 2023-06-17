import vllm.model_executor.parallel_utils.parallel_state
import vllm.model_executor.parallel_utils.tensor_parallel

# Alias parallel_state as mpu, its legacy name
mpu = parallel_state

__all__ = [
    "parallel_state",
    "tensor_parallel",
]
