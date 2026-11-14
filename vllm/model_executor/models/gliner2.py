# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLiNER2.5 Decide's DeBERTa encoder and classification head."""

from collections.abc import Iterable, Set

import torch
from torch import Tensor, nn
from transformers import DebertaV2Config, DebertaV2Model

from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler.abstract import Pooler
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata


class GLiNER2DecidePooler(Pooler):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.ReLU(),
            nn.Linear(2 * hidden_size, 1),
        )

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"classify"}

    def forward(
        self, hidden_states: Tensor, pooling_metadata: PoolingMetadata
    ) -> list[Tensor]:
        outputs = []
        offset = 0
        for length, params in zip(
            pooling_metadata.prompt_lens.tolist(),
            pooling_metadata.pooling_params,
            strict=True,
        ):
            row = hidden_states[offset : offset + length]
            # The runner calls the pooler with a synthetic prompt at startup.
            if params.extra_kwargs is None:
                outputs.append(row.new_zeros(1))
            else:
                labels = params.extra_kwargs.get("label_positions")
                if not labels:
                    raise ValueError("GLiNER2.5 Decide requires label_positions")
                outputs.append(self.classifier(row[labels]).flatten())
            offset += length
        return outputs


class Gliner2ForSchemaExtraction(nn.Module):
    """Serve the English GLiNER2.5 Decide checkpoint for structured decisions.

    This uses Transformers' DeBERTa v2 implementation, which works on CPU and
    does not depend on custom CUDA kernels. Other GLiNER2 extraction heads are
    intentionally not supported.
    """

    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        if config.architecture != "span" or config.counting_layer != "count_lstm":
            raise ValueError("Only the GLiNER2.5 Decide span checkpoint is supported")
        encoder_config = DebertaV2Config(**config.encoder_config)
        encoder_config.hidden_dropout_prob = 0.0
        encoder_config.attention_probs_dropout_prob = 0.0
        self.encoder = DebertaV2Model(encoder_config)
        self.pooler = GLiNER2DecidePooler(encoder_config.hidden_size)
        self.pad_token_id = encoder_config.pad_token_id or 0

    def embed_input_ids(self, input_ids: Tensor) -> Tensor:
        return self.encoder.embeddings.word_embeddings(input_ids)

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        intermediate_tensors=None,
        inputs_embeds=None,
    ) -> Tensor:
        flat = input_ids.flatten()
        pos = positions.flatten()
        # Positions restart at zero for each sequence; the runner can append
        # zero-position padding after the last real sequence.
        nonzero = pos.nonzero().flatten()
        real = int(nonzero[-1]) + 1 if nonzero.numel() else flat.numel()
        starts = (pos[:real] == 0).nonzero().flatten().tolist()
        if len(starts) == 1:
            packed = self.encoder(input_ids=flat[:real][None], return_dict=False)[0][0]
        else:
            lengths = [b - a for a, b in zip(starts, [*starts[1:], real])]
            rows = (pos[:real] == 0).cumsum(0) - 1
            cols = pos[:real]
            ids = flat.new_full((len(lengths), max(lengths)), self.pad_token_id)
            ids[rows, cols] = flat[:real]
            mask = torch.zeros_like(ids)
            mask[rows, cols] = 1
            hidden = self.encoder(
                input_ids=ids, attention_mask=mask, return_dict=False
            )[0]
            packed = hidden[rows, cols]
        if packed.shape[0] == flat.shape[0]:
            return packed
        return nn.functional.pad(packed, (0, 0, 0, flat.shape[0] - packed.shape[0]))

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded = set()
        for name, tensor in weights:
            if name.startswith("classifier."):
                name = "pooler." + name
            param = params.get(name)
            if param is None:
                continue  # Span/count extraction heads are unused by Decide.
            if param.shape != tensor.shape:
                raise ValueError(f"GLiNER2 weight shape mismatch for {name}")
            param.data.copy_(tensor)
            loaded.add(name)
        missing = params.keys() - loaded
        if missing:
            raise ValueError(f"Missing GLiNER2 Decide weights: {sorted(missing)}")
        return loaded
