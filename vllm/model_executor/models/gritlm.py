import re
from typing import List, Optional, Union

import torch
from torch import nn
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

from vllm.attention import AttentionMetadata
from vllm.attention.backends.xformers import XFormersImpl
from vllm.config import ModelConfig, VllmConfig
from vllm.inputs import (INPUT_REGISTRY, DecoderOnlyInputs, InputContext,
                         token_inputs)
from vllm.logger import init_logger
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.pooling_metadata import (PoolingMetadata,
                                                  PoolingTensors)
from vllm.multimodal.utils import cached_get_tokenizer
from vllm.pooling_params import PoolingParams
from vllm.sequence import (EmbeddingSequenceGroupOutput, IntermediateTensors,
                           PoolerOutput)

logger = init_logger(__name__)


class GritLMPooler(nn.Module):

    def __init__(
        self,
        model_config: ModelConfig,
    ):
        super().__init__()

        self.model_config = model_config

    def _get_instruction_lens(
            self, device: torch.device,
            pooling_metadata: PoolingMetadata) -> torch.Tensor:
        """
        Compute the number of tokens of each instruction using the tokenizer.
        """
        self.tokenizer = cached_get_tokenizer(
            self.model_config.tokenizer,
            tokenizer_mode=self.model_config.tokenizer_mode,
            tokenizer_revision=self.model_config.tokenizer_revision,
            trust_remote_code=self.model_config.trust_remote_code,
            truncation_side="left",
        )

        def query_instruction_missing(pooling_params: PoolingParams) -> bool:
            return (pooling_params is None
                    or pooling_params.additional_data is None
                    or "instruction_seq" not in pooling_params.additional_data)

        for seq_group in pooling_metadata.seq_groups:
            if query_instruction_missing(seq_group[1]):
                logger.warning(
                    "Query instruction not found in prompt,"
                    "thus using empty string instead. GritLM requires "
                    "query instruction in prompt.")

        instruction_lens = torch.tensor(
            [
                len(
                    self.tokenizer(
                        ("" if query_instruction_missing(seq_group[1]) else
                         seq_group[1].additional_data["instruction_seq"]),
                        padding=False,
                        truncation=True,
                        add_special_tokens=True,
                    )["input_ids"])
                for seq_group in pooling_metadata.seq_groups
            ],
            device=device,
        )

        return instruction_lens

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        """
        Pool the hidden states by summing the embeddings of
        non-instruction tokens.
        """
        instruction_lens = self._get_instruction_lens(
            device=hidden_states.device, pooling_metadata=pooling_metadata)

        prompt_lens = PoolingTensors.from_pooling_metadata(
            pooling_metadata, hidden_states.device).prompt_lens

        mask = torch.zeros_like(hidden_states, dtype=torch.bool)

        start_idx = 0
        for prompt_len, instruction_len in zip(prompt_lens, instruction_lens):
            end_idx = start_idx + prompt_len
            mask[start_idx + instruction_len:end_idx] = True
            start_idx = end_idx

        masked_hidden_states = hidden_states.masked_fill(~mask, 0.0)

        sum_embeddings = torch.zeros(len(prompt_lens),
                                     hidden_states.size(1),
                                     device=hidden_states.device)

        start_idx = 0
        for i, prompt_len in enumerate(prompt_lens):
            end_idx = start_idx + prompt_len
            sum_embeddings[i] = masked_hidden_states[start_idx:end_idx].sum(
                dim=0)
            start_idx = end_idx

        num_non_instruction_tokens = prompt_lens - instruction_lens
        mean_embeddings = sum_embeddings / num_non_instruction_tokens.unsqueeze(
            1)

        pooled_data = nn.functional.normalize(mean_embeddings, p=2, dim=1)

        pooled_outputs = [
            EmbeddingSequenceGroupOutput(data.tolist()) for data in pooled_data
        ]

        return PoolerOutput(outputs=pooled_outputs)


def input_processor_for_gritlm(ctx: InputContext, inputs: DecoderOnlyInputs):
    """
    Extracts query instruction from prompt and adds it to token inputs.
    """
    model_config = ctx.model_config
    tokenizer = cached_get_tokenizer(model_config.tokenizer)

    prompt = inputs.get("prompt", None)
    instruction = ""

    if prompt is None and "prompt_token_ids" in inputs:
        prompt = tokenizer.decode(inputs["prompt_token_ids"])

    if prompt is not None:
        match_instruction = re.match(r"(<s> )?(<\|user\|>\n.*\n<\|embed\|>\n)",
                                     prompt)
        match_empty_instruction = re.match(r"(<s> )?(<\|embed\|>\n)", prompt)

        if match_instruction and match_instruction.group(2):
            instruction = match_instruction.group(2)
        elif match_empty_instruction:
            instruction = match_empty_instruction.group(2)
        else:
            logger.warning("Query instruction not found in prompt,"
                           "thus using empty string instead. GritLM requires "
                           "query instruction in prompt.")

    return token_inputs(
        prompt_token_ids=inputs["prompt_token_ids"],
        prompt=prompt,
        instruction_seq=instruction,
    )


@INPUT_REGISTRY.register_input_processor(input_processor_for_gritlm)
class GritLM(LlamaForCausalLM):
    """This class implements the embedding model for parasail-ai/GritLM-7B-vllm.

    The class inherits from LlamaForCausalLM and provides a custom pooling
    layer.

    The task "embedding" must be specified in the server arguments.

    The main difference between the pooling layer in GritLM and the one in
    LlamaForCausalLM is that GritLM ignores the query instruction in the prompt
    when pooling the hidden states.

    Instructions can be passed to the model in two ways:
    1. By prepending the instruction to the prompt. The instruction should be
    in the format "<|user|>\n<instruction>\n<|embed|>\n".
    2. By passing the instruction as additional data in the pooling parameters
    (e.g. extra_body of client.embeddings.create).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix, **kwargs)

        self._pooler = GritLMPooler(model_config=vllm_config.model_config)

        assert isinstance(
            self.model.layers[0].self_attn.attn.impl,
            XFormersImpl), "GritLM is only supported by XFormers backend, "
        "which can be forced by VLLM_ATTENTION_BACKEND=XFORMERS"

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # Change attention to non-causal.
        assert attn_metadata.prefill_metadata.attn_bias is None
        attn_metadata.prefill_metadata.attn_bias = [
            BlockDiagonalMask.from_seqlens(attn_metadata.seq_lens)
        ]

        return super().forward(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            **kwargs,
        )

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, pooling_metadata)
