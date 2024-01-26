from typing import List, Optional

import torch

from vllm.lora.lora import LoRALayerWeights, PackedLoRALayerWeights


class DummyLoRAManager:

    def __init__(self):
        super().__init__()
        self._loras = {}

    def set_module_lora(self, module_name: str, lora: LoRALayerWeights):
        self._loras[module_name] = lora

    def get_module_lora(self, module_name: str) -> Optional[LoRALayerWeights]:
        return self._loras.get(module_name, None)

    def init_random_lora(self,
                         module_name: str,
                         weight: torch.Tensor,
                         rank: int = 8,
                         generate_embeddings_tensor: int = 0):
        lora = LoRALayerWeights(
            module_name,
            rank=rank,
            lora_alpha=1,
            lora_a=torch.rand([weight.shape[1], rank],
                              dtype=weight.dtype,
                              device="cuda"),
            lora_b=torch.rand([rank, weight.shape[0]],
                              dtype=weight.dtype,
                              device="cuda"),
        )
        if generate_embeddings_tensor:
            lora.embeddings_tensor = torch.rand(5,
                                                generate_embeddings_tensor,
                                                dtype=weight.dtype,
                                                device="cuda")
        self.set_module_lora(module_name, lora)

        return lora

    def init_lora(self,
                  module_name: str,
                  input_dim: int,
                  output_dim: int,
                  rank=8,
                  noop=False,
                  embeddings_tensor=None):
        lora = LoRALayerWeights(
            module_name,
            rank=rank,
            lora_alpha=1,
            lora_a=torch.rand([input_dim, rank], device="cuda"),
            lora_b=torch.rand([rank, output_dim], device="cuda"),
            embeddings_tensor=embeddings_tensor,
        )
        self.set_module_lora(module_name, lora)
        return lora

    def reset_lora(self):
        self._loras = {}

    def init_packed_lora(
        self,
        module_name: str,
        input_dim: int,
        output_dims: List[int],
        noop_lora_index: List[int] = None,
        rank=8,
    ):
        base_loras = []
        noop_lora_index = set(noop_lora_index or [])

        for i, out_dim in enumerate(output_dims):
            base_lora = self.init_lora(
                module_name + "_000_" + str(i),
                input_dim,
                out_dim,
                rank=rank,
                noop=i in noop_lora_index,
            )
            base_loras.append(base_lora)
        packed_lora = PackedLoRALayerWeights.pack(base_loras)
        self.set_module_lora(module_name, packed_lora)
        return packed_lora
