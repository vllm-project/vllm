from typing import TYPE_CHECKING, List, Optional, Tuple
import torch

from transformers import XLMRobertaModel as XLMRobertaModelHF
from transformers import XLMRobertaConfig

from vllm.model_executor.input_metadata import InputMetadata
from vllm.config import LoRAConfig
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    UnquantizedLinearMethod,
)

KVCache = Tuple[torch.Tensor, torch.Tensor]

# Faster model init with empty rather than random weights.
def skip(*args, **kwargs):
    pass

torch.nn.init.kaiming_uniform_ = skip
torch.nn.init.uniform_ = skip
torch.nn.init.normal_ = skip

class XLMRobertaModel(torch.nn.Module):
    def __init__(
        self,
        config: XLMRobertaConfig,
        linear_method: Optional[LinearMethodBase] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        if linear_method is not None and not isinstance(linear_method, UnquantizedLinearMethod):
            raise NotImplementedError(
                "XLMRobertaModel does not currently support quantization. "
                "Please raise an issue if you would like this feature."
            )
        if lora_config is not None:
            raise NotImplementedError(
                "XLMRobertaModel does not currently support deployments with "
                "LORA Adapters. Please raise an issue if you would like this feature."
            )
        
        self.config = config
        self.linear_method = linear_method
        
        # TODO (Milestone 2): We will implement XLM-RoBERTa with vllm 
        #   layer primitives to not rely on tranformers implementation.
        hf_model = XLMRobertaModelHF(self.config)
        self.embeddings = hf_model.embeddings
        self.encoder = hf_model.encoder
        self.pooler = hf_model.pooler

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, load_format, revision):
            param = params_dict[name]
            default_weight_loader(param, loaded_weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:

        # FIXME: convert the vllm input format into the hf input format
        #   [ Handle batched and non-batched cases ]
        # see https://github.com/huggingface/transformers/blob/v4.39.1/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py#L830C9-L850C90
        
        embedding_output = self.embeddings(
            # ...
        )
        encoder_outputs = self.encoder(
            # ...
        )
        sequence_output = encoder_outputs[0]
        return self.pooler(sequence_output)
