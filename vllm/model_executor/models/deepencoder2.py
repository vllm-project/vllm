# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# adapted from
# https://github.com/deepseek-ai/DeepSeek-OCR-2/blob/main/DeepSeek-OCR2-master/DeepSeek-OCR2-vllm/deepencoderv2/qwen2_d2e.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import transformers


class CustomQwen2Decoder(nn.Module):
    """
    Qwen2 visual encoder
    non-causal attention + causal attention
    token_type_ids ：0=non-causal, 1=causal
    """

    def __init__(
        self,
        decoder_layer: int = 24,
        max_position_embeddings: int = 131072,
        hidden_dimension: int = 896,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
        intermediate_size: int = 4864,
        vocab_size: int = 151936,
        attn_implementation: str = "sdpa",  # ⭐
        rms_norm_eps: float = 1e-06,
        rope_theta: float = 1000000.0,
        attention_dropout: float = 0.0,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
    ):
        super().__init__()

        # load
        Qwen2Model = transformers.models.qwen2.modeling_qwen2.Qwen2Model
        Qwen2Config = transformers.Qwen2Config

        # config
        config = Qwen2Config(
            hidden_size=hidden_dimension,
            num_hidden_layers=decoder_layer,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            vocab_size=vocab_size,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            attention_dropout=attention_dropout,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            _attn_implementation=attn_implementation,  # ⭐
        )

        #
        self.model = self._create_custom_model(Qwen2Model, config)

        del self.model.embed_tokens

    def _create_custom_model(self, Qwen2Model, config):
        """Qwen2Model"""

        class CustomQwen2ModelInner(Qwen2Model):
            def __init__(self, config):
                super().__init__(config)
                # Detect transformers version by checking the forward method source
                # New version uses create_causal_mask function and doesn't call _update_causal_mask
                # Old version calls self._update_causal_mask in forward
                import inspect
                try:
                    source = inspect.getsource(Qwen2Model.forward)
                    # Check for new version patterns (uses create_causal_mask function)
                    has_new_pattern = ('create_causal_mask' in source and
                                      'causal_mask_mapping' in source)
                    # Check for old version patterns (calls self._update_causal_mask)
                    has_old_pattern = 'self._update_causal_mask' in source

                    # New version: has new pattern and no old pattern
                    # Old version: has old pattern
                    self._is_new_version = has_new_pattern and not has_old_pattern
                except Exception:
                    # If we can't get source, assume old version (safer default)
                    self._is_new_version = False

            def forward(
                self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                token_type_ids=None,  # ⭐
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                cache_position=None,
            ):
                # Save token_type_ids for custom mask creation
                self._current_token_type_ids = token_type_ids

                # Check if this is the new version by inspecting if the method will be called
                if self._is_new_version:
                    # NEW VERSION: Create custom mask dict and pass it to bypass default mask creation
                    if inputs_embeds is None:
                        inputs_embeds = self.embed_tokens(input_ids)

                    custom_causal_mask = self._create_custom_causal_mask_for_new_version(
                        attention_mask=attention_mask,
                        inputs_embeds=inputs_embeds,
                        cache_position=cache_position,
                        past_key_values=past_key_values,
                        token_type_ids=token_type_ids,
                    )

                    # Pass the custom mask as a dict to bypass the new version's mask creation
                    causal_mask_mapping = {
                        "full_attention": custom_causal_mask,
                        "sliding_attention": custom_causal_mask,
                    }

                    outputs = super().forward(
                        input_ids=input_ids,
                        attention_mask=causal_mask_mapping,  # Pass dict for new version
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                        cache_position=cache_position,
                    )
                else:
                    # OLD VERSION: Pass original attention_mask, _update_causal_mask will be called
                    outputs = super().forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,  # Pass original mask for old version
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                        cache_position=cache_position,
                    )

                return outputs

            def _create_custom_causal_mask_for_new_version(
                self,
                attention_mask,
                inputs_embeds,
                cache_position,
                past_key_values,
                token_type_ids,
            ):
                """
                Create custom causal mask for new transformers version.
                This replaces the old _update_causal_mask method.
                """
                dtype, device = inputs_embeds.dtype, inputs_embeds.device
                min_dtype = torch.finfo(dtype).min
                batch_size, sequence_length = (
                    inputs_embeds.shape[0],
                    inputs_embeds.shape[1],
                )

                # Create custom attention mask based on token_type_ids
                causal_mask = self._create_custom_4d_mask(
                    sequence_length=sequence_length,
                    dtype=dtype,
                    device=device,
                    batch_size=batch_size,
                    token_type_ids=token_type_ids,
                )

                # Apply padding mask if provided (and if it's a tensor, not a dict)
                if attention_mask is not None and not isinstance(attention_mask, dict):
                    if attention_mask.dim() == 2:
                        padding_mask = attention_mask[:, None, None, :].to(dtype=dtype)
                        padding_mask = (1.0 - padding_mask) * min_dtype
                        causal_mask = causal_mask + padding_mask

                return causal_mask

            def _update_causal_mask(
                self,
                attention_mask,
                input_tensor,
                cache_position,
                past_key_values,
                output_attentions,
            ):
                """
                Legacy method for old transformers version compatibility.
                Kept for backward compatibility with old transformers versions.
                """
                # Safety check: if attention_mask is a dict, we're in the wrong code path
                # This shouldn't happen if version detection is correct, but just in case
                if isinstance(attention_mask, dict):
                    # Extract the actual mask from dict (new version behavior leaked into old version)
                    # Return the full_attention mask as fallback
                    return attention_mask.get('full_attention', None)

                token_type_ids = self._current_token_type_ids
                return self._create_custom_causal_mask_for_new_version(
                    attention_mask=attention_mask,
                    inputs_embeds=input_tensor,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    token_type_ids=token_type_ids,
                )

            def _create_custom_4d_mask(
                self,
                sequence_length,
                dtype,
                device,
                batch_size,
                token_type_ids,
            ):
                min_dtype = torch.finfo(dtype).min

                masks = []
                for b in range(batch_size):
                    mask = torch.full(
                        (sequence_length, sequence_length),
                        fill_value=min_dtype,
                        dtype=dtype,
                        device=device,
                    )

                    type_ids = token_type_ids[b]

                    image_positions = (type_ids == 0).nonzero(as_tuple=True)[0]
                    text_positions = (type_ids == 1).nonzero(as_tuple=True)[0]

                    # non-casual
                    if len(image_positions) > 0:
                        mask[image_positions[:, None], image_positions] = 0.0

                    # causal
                    for i, text_pos in enumerate(text_positions):
                        if len(image_positions) > 0:
                            mask[text_pos, image_positions] = 0.0
                        mask[text_pos, text_positions[: i + 1]] = 0.0

                    masks.append(mask)

                mask = torch.stack(masks, dim=0).unsqueeze(1)
                return mask

        return CustomQwen2ModelInner(config)

    def forward(self, inputs_embeds, token_type_ids, attention_mask=None, **kwargs):
        """
        Args:
            inputs_embeds: [batch_size, seq_len, hidden_dim]
            token_type_ids: [batch_size, seq_len], 0=non-causal, 1=causal
            attention_mask: [batch_size, seq_len], optional
        """
        return self.model(
            inputs_embeds=inputs_embeds,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            **kwargs,
        )


class Qwen2Decoder2Encoder(nn.Module):
    """
    Decoder based on Multilingual BART
    Set the initial weights and configuration with a pretrained multilingual BART model,
    and modify the detailed configurations as a Nougat decoder
    """

    def __init__(
        self,
        decoder_layer: int,
        hidden_dimension: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
    ):
        super().__init__()

        self.model = CustomQwen2Decoder(
            decoder_layer=decoder_layer,
            hidden_dimension=hidden_dimension,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            attn_implementation="sdpa",
        )
        self.query_768 = nn.Embedding(144, hidden_dimension)
        self.query_1024 = nn.Embedding(256, hidden_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)

        bs, n_query, _ = x.shape

        if n_query == 144:
            param_img = self.query_768.weight
        elif n_query == 256:
            param_img = self.query_1024.weight

        batch_query_imgs = param_img.unsqueeze(0).expand(
            bs, -1, -1
        )  # (batch_size, num_queries, hidden_size)

        x_combined = torch.cat([x, batch_query_imgs], dim=1)

        token_type_ids = torch.cat(
            [
                torch.zeros(bs, n_query, dtype=torch.long),
                torch.ones(bs, n_query, dtype=torch.long),
            ],
            dim=1,
        )

        y = self.model(x_combined, token_type_ids)[0]

        y = y[:, n_query:, :]  # causal flow query

        return y


def build_qwen2_decoder_as_encoder(
    decoder_layer=24,
    hidden_dimension=896,
    num_attention_heads=14,
    num_key_value_heads=2,
    intermediate_size=4864,
):
    decoder_as_encoder = Qwen2Decoder2Encoder(
        decoder_layer=decoder_layer,
        hidden_dimension=hidden_dimension,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
    )

    return decoder_as_encoder
