# Multi-Modal Support

This document walks you through the steps to extend a basic model so that it accepts [multi-modal inputs](../../features/multimodal_inputs.md).

## 1. Update the base vLLM model

It is assumed that you have already implemented the model in vLLM according to [these steps](basic.md).
Further update the model as follows:

- Implement [get_placeholder_str][vllm.model_executor.models.interfaces.SupportsMultiModal.get_placeholder_str] to define the placeholder string which is used to represent the multi-modal item in the text prompt. This should be consistent with the chat template of the model.

    ??? code

        ```python
        class YourModelForImage2Seq(nn.Module):
            ...

            @classmethod
            def get_placeholder_str(cls, modality: str, i: int) -> str | None:
                if modality.startswith("image"):
                    return "<image>"

                raise ValueError("Only image modality is supported")
        ```

- Inside `__init__` method, initialize the language components of the model inside [_mark_language_model][vllm.model_executor.models.interfaces.SupportsMultiModal._mark_language_model], and the multimodal components of the model inside [_mark_tower_model][vllm.model_executor.models.interfaces.SupportsMultiModal._mark_tower_model], e.g.:

    ```python
        def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
            super().__init__()

            config = vllm_config.model_config.hf_config

            with self._mark_tower_model(vllm_config, "image"):
                self.vision_encoder = ...
                self.multi_modal_projector = ...

            with self._mark_language_model(vllm_config):
                self.language_model = init_vllm_registered_model(
                    vllm_config=vllm_config,
                    hf_config=config.text_config,
                    prefix=maybe_prefix(prefix, "language_model"),
                )
    ```

- Remove the embedding part from the [forward][torch.nn.Module.forward] method:
    - Move the multi-modal embedding to [embed_multimodal][vllm.model_executor.models.interfaces.SupportsMultiModal.embed_multimodal].
    - The text embedding and embedding merge are handled automatically by a default implementation of [embed_input_ids][vllm.model_executor.models.interfaces.SupportsMultiModal.embed_input_ids]. It does not need to be overridden in most cases.

    ```diff
      def forward(
          self,
          input_ids: torch.Tensor | None,
    -     pixel_values: torch.Tensor,
          positions: torch.Tensor,
          intermediate_tensors: IntermediateTensors | None = None,
          inputs_embeds: torch.Tensor | None = None,
      ) -> torch.Tensor:
    -     if inputs_embeds is None:
    -         inputs_embeds = self.get_input_embeddings()(input_ids)
    -
    -     if pixel_values is not None:
    -         image_features = self.get_image_features(
    -             pixel_values=pixel_values,
    -         )
    -         special_image_mask = self.get_placeholder_mask(
    -             input_ids,
    -             inputs_embeds=inputs_embeds,
    -             image_features=image_features,
    -         )
    -         inputs_embeds = inputs_embeds.masked_scatter(
    -             special_image_mask,
    -             image_features,
    -         )

           hidden_states = self.language_model(
               input_ids,
               positions,
               intermediate_tensors,
               inputs_embeds=inputs_embeds,
           )
         ...
  
    +  def embed_multimodal(
    +      self,
    +      pixel_values: torch.Tensor,
    +  ) -> MultiModalEmbeddings | None:
    +      return self.get_image_features(
    +          pixel_values=pixel_values,
    +      )
    ```

    Below we provide a boilerplate of a typical implementation pattern of [embed_multimodal][vllm.model_executor.models.interfaces.SupportsMultiModal.embed_multimodal], but feel free to adjust it to your own needs.

    ```python
    def _process_image_input(self, image_input: YourModelImageInputs) -> torch.Tensor:
        image_features = self.vision_encoder(image_input)
        return self.multi_modal_projector(image_features)

    def embed_multimodal(
        self,
        **kwargs: object,
    ) -> MultiModalEmbeddings | None:
        # Validate the multimodal input keyword arguments
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None

        # Run multimodal inputs through encoder and projector
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings
    ```

!!! important
    The returned `multimodal_embeddings` must be either a **3D [torch.Tensor][]** of shape `(num_items, feature_size, hidden_size)`, or a **list / tuple of 2D [torch.Tensor][]'s** of shape `(feature_size, hidden_size)`, so that `multimodal_embeddings[i]` retrieves the embeddings generated from the `i`-th multimodal data item (e.g, image) of the request.

!!! note
    By default, vLLM merges the multimodal embeddings into text embeddings depending on the information of their locations defined in
    [PlaceholderRange][vllm.multimodal.inputs.PlaceholderRange] from input processing.
    This logic can be found at [embed_input_ids][vllm.model_executor.models.interfaces.SupportsMultiModal.embed_input_ids].

    You may override this method if additional logic is required for your model when merging embeddings.

- Once the above steps are done, update the model class with the [SupportsMultiModal][vllm.model_executor.models.interfaces.SupportsMultiModal] interface.

  ```diff
  + from vllm.model_executor.models.interfaces import SupportsMultiModal

  - class YourModelForImage2Seq(nn.Module):
  + class YourModelForImage2Seq(nn.Module, SupportsMultiModal):
  ```

!!! note
    The model class does not have to be named `*ForCausalLM`.
    Check out [the HuggingFace Transformers documentation](https://huggingface.co/docs/transformers/model_doc/auto#multimodal) for some examples.

## 2. Specify processing information

Next, create a subclass of [BaseProcessingInfo][vllm.multimodal.processing.BaseProcessingInfo]
to provide basic information related to HF processing.

### Maximum number of input items

You need to override the abstract method [get_supported_mm_limits][vllm.multimodal.processing.BaseProcessingInfo.get_supported_mm_limits]
to return the maximum number of input items for each modality supported by the model.

For example, if the model supports any number of images but only one video per prompt:

```python
def get_supported_mm_limits(self) -> Mapping[str, int | None]:
    return {"image": None, "video": 1}
```

## 3. Specify dummy inputs

Then, inherit [BaseDummyInputsBuilder][vllm.multimodal.processing.BaseDummyInputsBuilder] to construct dummy inputs for
HF processing. The processed outputs are also used for memory profiling.

Override the abstract methods [get_dummy_text][vllm.multimodal.processing.BaseDummyInputsBuilder.get_dummy_text] and [get_dummy_mm_data][vllm.multimodal.processing.BaseDummyInputsBuilder.get_dummy_mm_data] to construct dummy inputs. These dummy inputs should result in the worst-case memory usage of the model so that vLLM can reserve the correct amount of memory for it.

Assuming that the memory usage increases with the number of tokens, the dummy inputs can be constructed to maximize the number of output embeddings, which is the same number as placeholder feature tokens.

=== "Basic example: LLaVA"

    Looking at the code of HF's `LlavaForConditionalGeneration`:

    ??? code

        ```python
        # https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/llava/modeling_llava.py#L530-L544
        n_image_tokens = (input_ids == self.config.image_token_index).sum().item()
        n_image_features = image_features.shape[0] * image_features.shape[1]

        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        special_image_mask = (
            (input_ids == self.config.image_token_index)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
        ```

    The number of placeholder feature tokens per image is `image_features.shape[1]`.
    `image_features` is calculated inside the `get_image_features` method:

    ??? code

        ```python
        # https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/llava/modeling_llava.py#L290-L300
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)

        selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}")
        image_features = self.multi_modal_projector(selected_image_feature)
        return image_features
        ```

    We can infer that `image_features.shape[1]` is based on `image_outputs.hidden_states.shape[1]` from the vision tower
    (`CLIPVisionModel` for the [`llava-hf/llava-1.5-7b-hf`](https://huggingface.co/llava-hf/llava-1.5-7b-hf) model).
    Moreover, we only need the sequence length (the second dimension of the tensor) to get `image_features.shape[1]`.
    The sequence length is determined by the initial hidden states in `CLIPVisionTransformer` since the attention
    mechanism doesn't change the sequence length of the output hidden states.

    ```python
    # https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/clip/modeling_clip.py#L1094-L1102
    hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
    hidden_states = self.pre_layrnorm(hidden_states)

    encoder_outputs = self.encoder(
        inputs_embeds=hidden_states,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    ```

    To find the sequence length, we turn to the code of `CLIPVisionEmbeddings`:

    ??? code

        ```python
        # https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/clip/modeling_clip.py#L247-L257
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings
        ```

    We can infer that `embeddings.shape[1] == self.num_positions`, where

    ```python
    # https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/clip/modeling_clip.py#L195-L196
    self.num_patches = (self.image_size // self.patch_size) ** 2
    self.num_positions = self.num_patches + 1
    ```

    Overall, the number of placeholder feature tokens for an image can be calculated as:

    ??? code

        ```python
        def get_num_image_tokens(
            self,
            *,
            image_width: int,
            image_height: int,
        ) -> int:
            hf_config = self.get_hf_config()
            hf_processor = self.get_hf_processor()

            image_size = hf_config.vision_config.image_size
            patch_size = hf_config.vision_config.patch_size

            num_image_tokens = (image_size // patch_size) ** 2 + 1
            if hf_processor.vision_feature_select_strategy == "default":
                num_image_tokens -= 1

            return num_image_tokens
        ```

    Notice that the number of image tokens doesn't depend on the image width and height.
    We can simply use a dummy `image_size` to calculate the multimodal profiling data:

    ??? code

        ```python
        # NOTE: In actuality, this is usually implemented as part of the
        # model's subclass of `BaseProcessingInfo`, but we show it as is
        # here for simplicity.
        def get_image_size_with_most_features(self) -> ImageSize:
            hf_config = self.get_hf_config()
            width = height = hf_config.image_size
            return ImageSize(width=width, height=height)

        def get_dummy_mm_data(
            self,
            seq_len: int,
            mm_counts: Mapping[str, int],
            mm_options: Mapping[str, BaseDummyOptions],
        ) -> MultiModalDataDict:
            num_images = mm_counts.get("image", 0)

            target_width, target_height = \
                self.info.get_image_size_with_most_features()

            image_overrides = mm_options.get("image")

            return {
                "image": self._get_dummy_images(
                    width=target_width,
                    height=target_height,
                    num_images=num_images,
                    overrides=image_overrides,
                )
            }
        ```

    For the text, we simply expand the multimodal image token from the model config to match the desired number of images.

    ```python
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        processor = self.info.get_hf_processor()
        image_token = processor.image_token

        return image_token * num_images
    ```

=== "No input placeholders: PaliGemma"

    Unlike LLaVA, PaliGemma's HF processor does not expect image placeholder
    tokens in the input prompt; the placeholder feature tokens are instead
    inserted afterwards (see [Prompt updates](#prompt-updates)). So the dummy
    prompt text is empty regardless of the number of images:

    ```python
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""
    ```

    PaliGemma resizes every image to a square of `vision_config.image_size`, so
    the number of placeholder feature tokens per image is fixed at
    `(image_size // patch_size) ** 2`. This is computed by the SigLIP vision
    encoder that PaliGemma uses:

    ??? code

        ```python
        # vllm/model_executor/models/siglip.py
        class SiglipEncoderInfo(VisionEncoderInfo[SiglipVisionConfig]):
            def get_num_image_tokens(
                self,
                *,
                image_width: int,
                image_height: int,
            ) -> int:
                return self.get_patch_grid_length() ** 2

            def get_patch_grid_length(self) -> int:
                image_size, patch_size = self.get_image_size(), self.get_patch_size()
                return image_size // patch_size
        ```

    Since the number of image tokens doesn't depend on the input image dimensions,
    we can simply use a dummy image of the model's expected input size for the
    multimodal profiling data:

    ??? code

        ```python
        def get_dummy_mm_data(
            self,
            seq_len: int,
            mm_counts: Mapping[str, int],
            mm_options: Mapping[str, BaseDummyOptions],
        ) -> MultiModalDataDict:
            hf_config = self.info.get_hf_config()
            vision_config = hf_config.vision_config
            max_image_size = vision_config.image_size

            num_images = mm_counts.get("image", 0)

            image_overrides = mm_options.get("image")

            return {
                "image": self._get_dummy_images(
                    width=max_image_size,
                    height=max_image_size,
                    num_images=num_images,
                    overrides=image_overrides,
                )
            }
        ```

## 4. Specify processing details

Afterwards, create a subclass of [BaseMultiModalProcessor][vllm.multimodal.processing.BaseMultiModalProcessor]
to fill in the missing details about HF processing.

!!! info
    [Multi-Modal Data Processing](../../design/mm_processing.md)

### Multi-modal fields

Override [_get_mm_fields_config][vllm.multimodal.processing.BaseMultiModalProcessor._get_mm_fields_config] to
return a schema of the tensors outputted by the HF processor that are related to the input multi-modal items.

=== "Basic example: LLaVA"

    The output of `CLIPImageProcessor` is a simple tensor with shape
    `(num_images, num_channels, image_height, image_width)`:


    ```python
    # https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/clip/image_processing_clip.py#L339-L345
    images = [
        to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
        for image in all_images
    ]

    data = {"pixel_values": images}
    return BatchFeature(data=data, tensor_type=return_tensors)
    ```

    So, we override [_get_mm_fields_config][vllm.multimodal.processing.BaseMultiModalProcessor._get_mm_fields_config] as follows:

    ```python
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
        )
    ```

    !!! note
        Our [actual code](../../../vllm/model_executor/models/llava.py) additionally supports
        pre-computed image embeddings, which can be passed to be model via the `image_embeds` argument.

=== "With postprocessing: Mistral3"

    The `pixel_values` output of Mistral3's HF processor pads every image in the
    batch to a common size, so that they can be stacked into a single tensor.

    To use [MultiModalFieldConfig.batched][vllm.multimodal.inputs.MultiModalFieldConfig.batched]
    like in LLaVA, each image's features must be independent of the others (which
    is also required for prefix caching to work correctly). So, we un-pad each image
    back to its own size by overriding
    [BaseMultiModalProcessor._call_hf_processor][vllm.multimodal.processing.BaseMultiModalProcessor._call_hf_processor]:

    ??? code

        ```python
        def _call_hf_processor(
            self,
            prompt: str,
            mm_data: Mapping[str, object],
            mm_kwargs: Mapping[str, object],
            tok_kwargs: Mapping[str, object],
        ) -> BatchFeature:
            processed_outputs = super()._call_hf_processor(
                prompt=prompt,
                mm_data=mm_data,
                mm_kwargs=mm_kwargs,
                tok_kwargs=tok_kwargs,
            )

            pixel_values = processed_outputs.get("pixel_values")
            if pixel_values is not None:
                # Avoid padding since we need the output for each image to be
                # independent of other images for the cache to work correctly
                image_sizes = processed_outputs["image_sizes"]
                assert len(pixel_values) == len(image_sizes)

                processed_outputs["pixel_values"] = [
                    p[:, :h, :w] for p, (h, w) in zip(pixel_values, image_sizes)
                ]

            return processed_outputs
        ```

    !!! note
        The `_call_hf_processor` method specifies both `mm_kwargs` and `tok_kwargs` for
        processing. `mm_kwargs` is used to both initialize and call the huggingface
        processor, whereas `tok_kwargs` is only used to call the huggingface processor.

    Since `pixel_values` is now a list with one tensor per image, we can override
    [_get_mm_fields_config][vllm.multimodal.processing.BaseMultiModalProcessor._get_mm_fields_config] as follows:

    ```python
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )
    ```

    !!! note
        See our [actual code](../../../vllm/model_executor/models/mistral3.py) for the full implementation.

### Prompt updates

Override [_get_prompt_updates][vllm.multimodal.processing.BaseMultiModalProcessor._get_prompt_updates] to
return a list of [PromptUpdate][vllm.multimodal.processing.PromptUpdate] instances.

Each [PromptUpdate][vllm.multimodal.processing.PromptUpdate] instance specifies an update operation
(e.g.: insertion, replacement) performed by the HF processor.

=== "Basic example: LLaVA"

    Looking at HF's `LlavaProcessor`:

    ```python
    # https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/llava/processing_llava.py#L167-L170
    prompt_strings = []
    for sample in text:
        sample = sample.replace(self.image_token, self.image_token * num_image_tokens)
        prompt_strings.append(sample)
    ```

    It simply repeats each input `image_token` a number of times equal to the number of placeholder feature tokens (`num_image_tokens`).
    Based on this, we override [_get_prompt_updates][vllm.multimodal.processing.BaseMultiModalProcessor._get_prompt_updates] as follows:

    ??? code

        ```python
        def _get_prompt_updates(
            self,
            mm_items: MultiModalDataItems,
            hf_processor_mm_kwargs: Mapping[str, object],
            out_mm_kwargs: MultiModalKwargsItems,
        ) -> Sequence[PromptUpdate]:
            hf_config = self.info.get_hf_config()
            image_token_id = hf_config.image_token_index

            def get_replacement(item_idx: int):
                images = mm_items.get_items("image", ImageProcessorItems)

                image_size = images.get_image_size(item_idx)
                num_image_tokens = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                )

                return [image_token_id] * num_image_tokens

            return [
                PromptReplacement(
                    modality="image",
                    target=[image_token_id],
                    replacement=get_replacement,
                ),
            ]
        ```

=== "Handling additional tokens: PaliGemma"

    PaliGemma's HF processor inserts, after the prompt's leading `<bos>` token, a
    run of image tokens followed by a second `<bos>` token that marks the start of
    the text prompt. We start by building the run of image tokens, one per
    placeholder feature token:

    ??? code

        ```python
        def get_insertion(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems)
            )

            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:
                image_size = images.get_image_size(item_idx)
                num_image_tokens = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                )

            image_tokens = [image_token_id] * num_image_tokens
            ...
        ```

    The trailing `<bos>` token is an additional token that must **not** receive a
    vision embedding. To assign the vision embeddings to only the image tokens,
    instead of returning the token ids directly you can return an instance of
    [PromptUpdateDetails][vllm.multimodal.processing.PromptUpdateDetails] and mark
    the embedding tokens with `embed_token_id`:

    ??? code

        ```python
        return PromptUpdateDetails.select_token_id(
            image_tokens + [bos_token_id],
            embed_token_id=image_token_id,
        )
        ```

    Putting it together, we override [_get_prompt_updates][vllm.multimodal.processing.BaseMultiModalProcessor._get_prompt_updates].
    Since these tokens are inserted (rather than replacing an existing placeholder)
    after the prompt's leading `<bos>`, we use [PromptInsertion][vllm.multimodal.processing.PromptInsertion]
    with a prefix target:

    ??? code

        ```python
        def _get_prompt_updates(
            self,
            mm_items: MultiModalDataItems,
            hf_processor_mm_kwargs: Mapping[str, object],
            out_mm_kwargs: MultiModalKwargsItems,
        ) -> Sequence[PromptUpdate]:
            hf_config = self.info.get_hf_config()
            image_token_id = hf_config.image_token_index

            tokenizer = self.info.get_tokenizer()

            bos_token_id = tokenizer.bos_token_id
            assert isinstance(bos_token_id, int)

            def get_insertion(item_idx: int):
                images = mm_items.get_items(
                    "image", (ImageEmbeddingItems, ImageProcessorItems)
                )

                if isinstance(images, ImageEmbeddingItems):
                    num_image_tokens = images.get_feature_size(item_idx)
                else:
                    image_size = images.get_image_size(item_idx)
                    num_image_tokens = self.info.get_num_image_tokens(
                        image_width=image_size.width,
                        image_height=image_size.height,
                    )

                image_tokens = [image_token_id] * num_image_tokens

                return PromptUpdateDetails.select_token_id(
                    image_tokens + [bos_token_id],
                    embed_token_id=image_token_id,
                )

            return [
                PromptInsertion(
                    modality="image",
                    target=PromptIndexTargets.prefix(
                        [bos_token_id] if tokenizer.add_bos_token else []
                    ),
                    insertion=get_insertion,
                )
            ]
        ```

## 5. Register processor-related classes

After you have defined [BaseProcessingInfo][vllm.multimodal.processing.BaseProcessingInfo] (Step 2),
[BaseDummyInputsBuilder][vllm.multimodal.processing.BaseDummyInputsBuilder] (Step 3),
and [BaseMultiModalProcessor][vllm.multimodal.processing.BaseMultiModalProcessor] (Step 4),
decorate the model class with [MULTIMODAL_REGISTRY.register_processor][vllm.multimodal.registry.MultiModalRegistry.register_processor]
to register them to the multi-modal registry:

```diff
  from vllm.model_executor.models.interfaces import SupportsMultiModal
+ from vllm.multimodal import MULTIMODAL_REGISTRY

+ @MULTIMODAL_REGISTRY.register_processor(
+     YourMultiModalProcessor,
+     info=YourProcessingInfo,
+     dummy_inputs=YourDummyInputsBuilder,
+ )
  class YourModelForImage2Seq(nn.Module, SupportsMultiModal):
```

## Notes

### Inserting feature tokens without replacement

Some HF processors directly insert feature tokens without replacing anything in the original prompt. In that case, you can use [PromptInsertion][vllm.multimodal.processing.PromptInsertion] instead of [PromptReplacement][vllm.multimodal.processing.PromptReplacement] inside [_get_prompt_updates][vllm.multimodal.processing.BaseMultiModalProcessor._get_prompt_updates].

Examples:

- BLIP-2 (insert at start of prompt): [vllm/model_executor/models/blip2.py](../../../vllm/model_executor/models/blip2.py)
- Molmo (insert after `<|endoftext|>` token): [vllm/model_executor/models/molmo.py](../../../vllm/model_executor/models/molmo.py)

### Handling prompt updates unrelated to multi-modal data

[_get_prompt_updates][vllm.multimodal.processing.BaseMultiModalProcessor._get_prompt_updates] assumes that each application of prompt update corresponds to one multi-modal item. If the HF processor performs additional processing regardless of how many multi-modal items there are, you should override [_apply_hf_processor_tokens_only][vllm.multimodal.processing.BaseMultiModalProcessor._apply_hf_processor_tokens_only] so that the processed token inputs are consistent with the result of applying the HF processor on text inputs. This is because token inputs bypass the HF processor according to [our design](../../design/mm_processing.md).

Examples:

- Chameleon (appends `sep_token`): [vllm/model_executor/models/chameleon.py](../../../vllm/model_executor/models/chameleon.py)
- Molmo2 (prepends `bos_token`): [vllm/model_executor/models/molmo2.py](../../../vllm/model_executor/models/molmo2.py)
- Molmo (applies chat template which is not defined elsewhere): [vllm/model_executor/models/molmo.py](../../../vllm/model_executor/models/molmo.py)

### Custom HF processor

Some models don't define an HF processor class on HF Hub. In that case, you can define a custom HF processor that has the same call signature as HF processors and pass it to [_call_hf_processor][vllm.multimodal.processing.BaseMultiModalProcessor._call_hf_processor].

Examples:

- DeepSeek-VL2: [vllm/model_executor/models/deepseek_vl2.py](../../../vllm/model_executor/models/deepseek_vl2.py)
- InternVL: [vllm/model_executor/models/internvl.py](../../../vllm/model_executor/models/internvl.py)
