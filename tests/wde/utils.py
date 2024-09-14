
from typing import TypeVar, List, Optional, Dict, Any
import gc
import torch
import torch.nn as nn
from vllm.wde.entrypoints.llm import LLM
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, is_cpu
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding, BatchFeature

_T = TypeVar("_T", nn.Module, torch.Tensor, BatchEncoding, BatchFeature)


class VllmRunner:
    def __init__(
        self,
        model_name: str,
        max_num_seqs: int = 4,
        tokenizer_name: Optional[str] = None,
        dtype: str = "half",
        scheduling: str = "sync"
    ) -> None:
        self.model = LLM(
            model=model_name,
            tokenizer=tokenizer_name,
            trust_remote_code=True,
            max_num_seqs=max_num_seqs,
            dtype=dtype,
            scheduling=scheduling)

    def encode(self, prompts: List[str]) -> List[List[float]]:
        req_outputs = self.model.encode(prompts)
        outputs = []
        for req_output in req_outputs:
            embedding = req_output.outputs
            outputs.append(embedding)
        return outputs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        cleanup()


class HfRunner:
    def wrap_device(self, input: _T) -> _T:
        if not is_cpu():
            # Check if the input is already on the GPU
            if hasattr(input, 'device') and input.device.type == "cuda":
                return input  # Already on GPU, no need to move
            return input.to("cuda")
        else:
            # Check if the input is already on the CPU
            if hasattr(input, 'device') and input.device.type == "cpu":
                return input  # Already on CPU, no need to move
            return input.to("cpu")

    def __init__(
        self,
        model_name: str,
        dtype: str = "half",
        *,
        model_kwargs: Optional[Dict[str, Any]] = None,
        auto_cls=AutoModelForCausalLM
    ) -> None:
        torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[dtype]

        self.model_name = model_name

        model_kwargs = model_kwargs if model_kwargs is not None else {}

        self.model = self.wrap_device(
            auto_cls.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                **model_kwargs,
            ))

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

    @torch.inference_mode
    def encode(self, prompts: List[str]) -> List[List[torch.Tensor]]:
        encoded_input = self.tokenizer(prompts,
                                       padding=True,
                                       truncation=True,
                                       return_tensors='pt').to("cuda")

        logits = self.model(**encoded_input).logits
        seq_len = encoded_input.attention_mask.sum(axis=1)

        logits_list = []
        for e, s in zip(logits, seq_len):
            logits_list.append(e[:s])
        return logits_list

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        cleanup()


def cleanup():
    gc.collect()
    if not is_cpu():
        torch.cuda.empty_cache()