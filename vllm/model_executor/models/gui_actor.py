# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GUI-Actor: a pointer head over the visual patches of a Qwen2.5-VL backbone.

GUI-Actor neither regresses coordinates nor answers by generating text. The
prompt carries a click action whose coordinate list holds the
``<|pointer_start|><|pointer_pad|><|pointer_end|>`` placeholder triple, and the
answer is the attention of the ``<|pointer_pad|>`` hidden state over the image
patches. Placeholder mode is a single prefill: the answer does not depend on a
generated token.

The head is ported from Microsoft's GUI-Actor (MIT License, Copyright (c) 2025
Microsoft), the same arrangement ``clef_joint_schema`` has with
Cloudflare/clef-flash, so these checkpoints serve with no dependency on that
repository. Two asymmetries in the ported head are deliberate and must not be
"fixed":

- The encoder side reads the *embed* layer's hidden state -- the vision features
  already scattered into the placeholders, exactly what the language model
  consumed -- while the decoder side reads the *last* layer's hidden state at
  ``<|pointer_pad|>``. They are not interchangeable: the head compares "what the
  LM decided to click" against "what the pixels look like".
- The head's output is a softmax over patches, and training supervised it
  against a distribution (uniform over the in-ground-truth-box patches), not a
  class index. That is what makes the inference-time connected-component decode
  in ``decisions/spatial.py`` work at all.

The pooler returns that raw per-patch distribution for one question. The answer
arithmetic (region centres, box edges, confidence) lives in
``vllm/entrypoints/openai/decisions/spatial.py`` instead, both because the region
decode runs on the CPU and because the raw logits are what let the confidence be
recomputed at any temperature without re-running the model.
"""

from typing import Any

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler import Pooler, PoolingParamsUpdate
from vllm.model_executor.layers.pooler.tokwise.methods import AllPool
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tasks import PoolingTask
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata

from .interfaces_base import default_pooling_type
from .qwen2_5_vl import (
    Qwen2_5_VLDummyInputsBuilder,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLMultiModalProcessor,
    Qwen2_5_VLProcessingInfo,
)
from .utils import WeightsMapper

POINTER_HEAD_WEIGHTS = "multi_patch_pointer_head."

#: The five special tokens a GUI-Actor checkpoint ships. The head reads the two
#: pointer ones; a checkpoint without them cannot ground at all.
POINTER_TOKENS = (
    "<|recipient|>",
    "<|diff_marker|>",
    "<|pointer_start|>",
    "<|pointer_end|>",
    "<|pointer_pad|>",
)


class PointerHead(nn.Module):
    """Scores every visual patch against every ``<|pointer_pad|>`` query.

    Shape-agnostic in ``n_enc`` (the number of image patches), which is why a
    single checkpoint transfers across screenshot resolutions.
    """

    def __init__(
        self,
        d_model: int,
        projection_dim: int,
        num_attention_heads: int = 8,
        dropout_rate: float = 0.1,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Post-hoc calibration scalar, read from ``config.pointer_temperature``
        # and applied by the *caller* to the confidence it derives -- never
        # inside ``forward()``. The returned distribution feeds a threshold +
        # BFS region post-process, so rescaling it here would move the predicted
        # point; dividing after the region is fixed is what keeps every existing
        # output bit-identical.
        #
        # A plain attribute on purpose: a buffer would enter ``state_dict()``
        # and break strict loading of checkpoints saved before this feature
        # existed, and a parameter would join ``named_parameters()``.
        self.temperature = float(temperature)

        # No extra normalisation on the inputs: the backbone already emits
        # RMSNorm'd hidden states.
        self.projection_enc = nn.Sequential(
            nn.Linear(d_model, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, d_model),
        )
        self.projection_dec = nn.Sequential(
            nn.Linear(d_model, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, d_model),
        )

        # The vision tower's output carries no text context, so the patches
        # interact with each other once before being matched against the query.
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_attention_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self, hidden_state_enc: torch.Tensor, hidden_state_dec: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(attn_weights, patch_logits)``.

        ``attn_weights`` is *always* the raw softmax (temperature 1). The fitted
        temperature is applied by the caller to the confidence scalar only,
        after the predicted region has been extracted from these weights.
        """
        enc_input = hidden_state_enc.unsqueeze(0)
        attn_output, _ = self.self_attention(
            query=enc_input, key=enc_input, value=enc_input, need_weights=False
        )
        enc_ctx = self.layer_norm(enc_input + self.dropout(attn_output)).squeeze(0)

        proj_enc = self.projection_enc(enc_ctx)  # [n_enc, d_model]
        proj_dec = self.projection_dec(hidden_state_dec)  # [n_dec, d_model]

        # Scaling by sqrt(d_model) keeps the logits well-conditioned for any
        # n_enc.
        scaling = self.d_model**0.5
        patch_logits = torch.matmul(proj_dec, proj_enc.transpose(0, 1)) / scaling
        attn_weights = torch.softmax(patch_logits, dim=-1)
        return attn_weights, patch_logits


def pointer_temperature(config: Any) -> float:
    """The head's fitted calibration scalar from ``config.json``.

    Like the ``pointer_*_token_id`` fields it is a plain extra key, so it
    survives ``save_pretrained`` / ``from_pretrained`` without a custom config
    class.

    ``GuiActorForConditionalGenerationConfig`` validates and normalizes it
    before the model is built, so this only reads the value back; validating it
    again here would let the model and the API process disagree about what a
    malformed config means.
    """
    temperature = getattr(config, "pointer_temperature", 1.0)
    return 1.0 if temperature is None else float(temperature)


class GuiActorPooler(Pooler):
    """Runs the pointer head on each finished prompt and returns the per-patch
    distribution of each question's ``<|pointer_pad|>``.

    A request marks its prompt through ``PoolingParams.extra_kwargs["pointer"]``
    with the token ids that delimit the image (``vision_start_token_id``,
    ``vision_end_token_id``) and its pointer (``pointer_token_id``). The pooler
    locates those tokens in the prompt itself rather than trusting a caller's
    offsets, so an expanded image -- which grows the prompt by one pad token
    per patch -- cannot shift the answer.

    It never raises, because it runs in the engine core: a prompt without a
    valid schema, such as a profiling dummy, gets an empty output.
    """

    def __init__(self, head: PointerHead, model: nn.Module) -> None:
        super().__init__()
        self.pooling = AllPool()
        self.head = head
        # The model that captures its embed-layer hidden states for the head's
        # encoder side. Held in a list so ``nn.Module``'s attribute assignment
        # does not register it as a submodule, which would put the whole model
        # in the pooler's ``state_dict()``.
        self.model = [model]

    def get_supported_tasks(self) -> set[PoolingTask]:
        return {"classify"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        pooled = self.pooling(hidden_states, pooling_metadata)
        prompt_token_ids = pooling_metadata.get_prompt_token_ids_cpu()
        # ``captured_hidden_states`` holds the whole batch's flattened embed
        # layer, so a prompt's positions are offset by the tokens scheduled
        # before it. The prompt is one step -- chunked prefill is off for this
        # model -- so its slice in the capture lines up with ``token_ids``.
        cursor = pooling_metadata.get_pooling_cursor()
        offsets = [0]
        for length in cursor.num_scheduled_tokens_cpu.tolist():
            offsets.append(offsets[-1] + length)

        capture = self.model[0].captured_hidden_states
        outputs: list[torch.Tensor | None] = []
        for i, (hidden, token_ids, params) in enumerate(
            zip(pooled, prompt_token_ids, pooling_metadata.pooling_params)
        ):
            if hidden is None:
                outputs.append(None)
                continue
            pointer = (params.extra_kwargs or {}).get("pointer")
            if not isinstance(pointer, dict) or capture is None:
                outputs.append(hidden.new_empty(0, dtype=torch.float32))
                continue
            outputs.append(
                self._patch_logits(
                    hidden.float(), token_ids, capture, pointer, offsets[i]
                )
            )
        return outputs

    def _patch_logits(
        self,
        hidden: torch.Tensor,
        token_ids: torch.Tensor,
        capture: torch.Tensor,
        pointer: dict[str, Any],
        offset: int,
    ) -> torch.Tensor:
        """The ``<|pointer_pad|>`` of this prompt, scored against the image's
        patches. Returns an empty tensor for a prompt that carries neither, such
        as a profiling dummy."""
        device = hidden.device
        # ``token_ids`` arrives as a tensor on this path and as a list on the
        # dummy one.
        ids = async_tensor_h2d(torch.as_tensor(token_ids, dtype=torch.long), device)

        pad = torch.nonzero(ids == int(pointer["pointer_token_id"])).reshape(-1)
        # The image's visual tokens sit between its start and end markers.
        start = torch.nonzero(ids == int(pointer["vision_start_token_id"])).reshape(-1)
        end = torch.nonzero(ids == int(pointer["vision_end_token_id"])).reshape(-1)
        if pad.numel() == 0 or start.numel() == 0 or end.numel() == 0:
            return hidden.new_empty(0, dtype=torch.float32)

        # The image's positions in the batch's capture.
        image = torch.arange(
            offset + int(start[-1]) + 1, offset + int(end[-1]), device=device
        )
        if image.numel() == 0 or int(image[-1]) >= capture.shape[0]:
            return hidden.new_empty(0, dtype=torch.float32)

        # The head's own weights decide the dtype; promoting either side to
        # float32 here would make the projection's matmul reject the other.
        dtype = self.head.projection_enc[0].weight.dtype
        encoder = capture.index_select(0, image).to(dtype)
        # One pointer per prompt, like the reference's single click action.
        decoder = hidden.index_select(0, pad[:1]).to(dtype)
        _, patch_logits = self.head(encoder, decoder)
        return patch_logits[0].float()


@default_pooling_type(tok_pooling_type="ALL")
@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5_VLMultiModalProcessor,
    info=Qwen2_5_VLProcessingInfo,
    dummy_inputs=Qwen2_5_VLDummyInputsBuilder,
)
class GuiActorForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """Qwen2.5-VL with the GUI-Actor pointer head, served as a ``classify``
    model.

    Only Qwen2.5-VL backbones are supported: the head is square (the vision
    tower's ``out_hidden_size`` equals the language model's ``hidden_size``),
    which holds for Qwen2.5-VL and does not for every later backbone.
    """

    is_pooling_model = True

    hf_to_vllm_mapper = Qwen2_5_VLForConditionalGeneration.hf_to_vllm_mapper | (
        # The head's tensors are named for the reference class, not for this
        # one; the base's mappings carry the backbone and lm_head.
        WeightsMapper(orig_to_new_prefix={"multi_patch_pointer_head.": "pooler.head."})
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        # The class is only reachable through the qwen2_5_vl convertor, but the
        # head's square projection needs this backbone's widths, so say so
        # explicitly rather than failing later with a shape error.
        model_type = getattr(config, "model_type", None)
        if model_type != "qwen2_5_vl":
            raise ValueError(
                "GUI-Actor is served on Qwen2.5-VL backbones only in this "
                f"release, but this checkpoint is {model_type!r}"
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        hidden_size = config.text_config.hidden_size
        # A checkpoint whose tokenizer lacks the pointer tokens cannot ground
        # at all: the click action's placeholder would encode as plain text and
        # the head would score a random token. Fail at load, as the reference's
        # `_require_pointer_tokens` does, rather than serve random grounding
        # scores.
        vocab = cached_tokenizer_from_config(vllm_config.model_config).get_vocab()
        if missing := [token for token in POINTER_TOKENS if token not in vocab]:
            raise ValueError(
                f"{vllm_config.model_config.model} is missing GUI-Actor's "
                f"special tokens {missing}; it does not look like a GUI-Actor "
                "checkpoint"
            )
        # The embed-layer hidden states the head's encoder side reads, written
        # by ``embed_input_ids`` on the step that consumes it. A plain
        # attribute, so it stays out of ``state_dict()``.
        self.captured_hidden_states: torch.Tensor | None = None
        self.pooler = GuiActorPooler(
            PointerHead(
                hidden_size,
                hidden_size,
                temperature=pointer_temperature(config),
            ),
            self,
        )

    def embed_input_ids(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        inputs_embeds = super().embed_input_ids(*args, **kwargs)
        # The head's encoder side reads exactly what the language model
        # consumes: the vision features scattered into the placeholders. It is
        # copied, not referenced: the caller writes the result into a
        # pre-allocated ``inputs_embeds`` buffer it reuses every step, so the
        # buffer would already hold another step's tokens by pooling time.
        self.captured_hidden_states = inputs_embeds.detach().clone()
        return inputs_embeds
