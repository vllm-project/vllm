from transformers.models.idefics2.configuration_idefics2 import (
    Idefics2VisionConfig)
from transformers.models.llama.configuration_llama import LlamaConfig


class AriaVisionConfig(Idefics2VisionConfig):
    model_type = "aria_vision_model"


class AriaMoELMConfig(LlamaConfig):
    """
    Configuration class for AriaMoE language model.

    This class extends the LlamaConfig to include additional parameters specific
    to the Mixture of Experts (MoE) architecture.
    """

    model_type = "aria_moe_lm"

    def __init__(
        self,
        moe_intermediate_size: int = 4096,
        moe_num_experts: int = 8,
        moe_topk: int = 2,
        moe_num_shared_experts: int = 2,
        **kwargs,
    ):
        """
        Initialize the AriaMoELMConfig.

        Args:
            moe_intermediate_size (int): The intermediate size for MoE layers.
                Default is 4096.
            moe_num_experts (int): The number of experts in the MoE layer.
                Default is 8.
            moe_topk (int): The number of top experts to route to for each 
                token. Default is 2.
            moe_num_shared_experts (int): The number of shared experts. Default
                is 2. 
            **kwargs: Additional keyword arguments to be passed to the parent
                LlamaConfig.
        """
        super().__init__(**kwargs)
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_num_experts = moe_num_experts
        self.moe_topk = moe_topk
        self.moe_num_shared_experts = moe_num_shared_experts
