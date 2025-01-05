from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.models.llama import LlamaModel

NUMBER_OF_GOOD_PASSES = 10

class ModelForwardError(Exception):
    pass

class EvilLlamaModel(LlamaModel):
    """Evil Llama Class For Simulating Model Issue."""

    def __init__(self, *args, **kwargs):
        super.__init__(*args, **kwargs)
        self.good_passes = 0

    def forward(self, *args, **kwargs):
        """Raise an after N iterations"""

        if (self.good_passes == NUMBER_OF_GOOD_PASSES and
            get_tensor_model_parallel_rank() == 0):
            raise ModelForwardError(
                "Simulated illegal memory access on rank 0!")
        self.good_passes += 1
        return self.forward(*args, **kwargs)

    


