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
            mm_options: Mapping[str, BaseDummyOptions] | None = None,
        ) -> MultiModalDataDict:
            num_images = mm_counts.get("image", 0)

            target_width, target_height = \
                self.info.get_image_size_with_most_features()

            image_overrides = mm_options.get("image") if mm_options else None

            return {
                "image":
                self._get_dummy_images(width=target_width,
                                    height=target_height,
                                    num_images=num_images,
                                    overrides=image_overrides)
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

=== "No input placeholders: Fuyu"

    Looking at the code of HF's `FuyuForCausalLM`:

    ??? code

        ```python
        # https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/fuyu/modeling_fuyu.py#L311-L322
        if image_patches is not None and past_key_values is None:
            patch_embeddings = [
                self.vision_embed_tokens(patch.to(self.vision_embed_tokens.weight.dtype))
                .squeeze(0)
                .to(inputs_embeds.device)
                for patch in image_patches
            ]
            inputs_embeds = self.gather_continuous_embeddings(
                word_embeddings=inputs_embeds,
                continuous_embeddings=patch_embeddings,
                image_patch_input_indices=image_patches_indices,
            )
        ```

    The number of placeholder feature tokens for the `i`th item in the batch is `patch_embeddings[i].shape[0]`,
    which is the same as `image_patches[i].shape[0]`, i.e. `num_total_patches`.

    Unlike LLaVA, Fuyu does not define the number of patches inside the modeling file. Where can we get more information?
    Considering that the model input comes from the output of `FuyuProcessor`, let's **look at the preprocessing files**.

    The image outputs are obtained by calling `FuyuImageProcessor.preprocess` and then
    `FuyuImageProcessor.preprocess_with_tokenizer_info` inside `FuyuProcessor`.

    In `FuyuImageProcessor.preprocess`, the images are resized and padded to the target `FuyuImageProcessor.size`,
    returning the dimensions after resizing (but before padding) as metadata.

    ??? code

        ```python
        # https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/fuyu/processing_fuyu.py#L541-L544
        image_encoding = self.image_processor.preprocess(images, **output_kwargs["images_kwargs"])
        batch_images = image_encoding["images"]
        image_unpadded_heights = image_encoding["image_unpadded_heights"]
        image_unpadded_widths = image_encoding["image_unpadded_widths"]

        # https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/fuyu/image_processing_fuyu.py#L480-L
        if do_resize:
            batch_images = [
                [self.resize(image, size=size, input_data_format=input_data_format) for image in images]
                for images in batch_images
            ]

        image_sizes = [get_image_size(images[0], channel_dim=input_data_format) for images in batch_images]
        image_unpadded_heights = [[image_size[0]] for image_size in image_sizes]
        image_unpadded_widths = [[image_size[1]] for image_size in image_sizes]

        if do_pad:
            batch_images = [
                [
                    self.pad_image(
                        image,
                        size=size,
                        mode=padding_mode,
                        constant_values=padding_value,
                        input_data_format=input_data_format,
                    )
                    for image in images
                ]
                for images in batch_images
            ]
        ```

    In `FuyuImageProcessor.preprocess_with_tokenizer_info`, the images are split into patches based on this metadata:

    ??? code

        ```python
        # https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/fuyu/processing_fuyu.py#L417-L425
        model_image_input = self.image_processor.preprocess_with_tokenizer_info(
            image_input=tensor_batch_images,
            image_present=image_present,
            image_unpadded_h=image_unpadded_heights,
            image_unpadded_w=image_unpadded_widths,
            image_placeholder_id=image_placeholder_id,
            image_newline_id=image_newline_id,
            variable_sized=True,
        )

        # https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/fuyu/image_processing_fuyu.py#L638-L658
        image_height, image_width = image.shape[1], image.shape[2]
        if variable_sized:  # variable_sized=True
            new_h = min(
                image_height,
                math.ceil(image_unpadded_h[batch_index, subseq_index] / patch_height) * patch_height,
            )
            new_w = min(
                image_width,
                math.ceil(image_unpadded_w[batch_index, subseq_index] / patch_width) * patch_width,
            )
            image = image[:, :new_h, :new_w]
            image_height, image_width = new_h, new_w

        num_patches = self.get_num_patches(image_height=image_height, image_width=image_width)
        tensor_of_image_ids = torch.full(
            [num_patches], image_placeholder_id, dtype=torch.int32, device=image_input.device
        )
        patches = self.patchify_image(image=image.unsqueeze(0)).squeeze(0)
        assert num_patches == patches.shape[0]
        ```

    The number of patches is in turn defined by `FuyuImageProcessor.get_num_patches`:

    ??? code

        ```python
        # https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/fuyu/image_processing_fuyu.py#L552-L562
        patch_size = patch_size if patch_size is not None else self.patch_size
        patch_height, patch_width = self.patch_size["height"], self.patch_size["width"]

        if image_height % patch_height != 0:
            raise ValueError(f"{image_height=} must be divisible by {patch_height}")
        if image_width % patch_width != 0:
            raise ValueError(f"{image_width=} must be divisible by {patch_width}")

        num_patches_per_dim_h = image_height // patch_height
        num_patches_per_dim_w = image_width // patch_width
        num_patches = num_patches_per_dim_h * num_patches_per_dim_w
        ```

    These image patches correspond to placeholder tokens (`|SPEAKER|`). So, we just need to maximize the number of image patches. Since input images are first resized
    to fit within `image_processor.size`, we can maximize the number of image patches by inputting an image with size equal to `image_processor.size`.

    ```python
    def get_image_size_with_most_features(self) -> ImageSize:
        image_processor = self.get_image_processor()
        return ImageSize(
            width=image_processor.size["width"],
            height=image_processor.size["height"],
        )
    ```

    Fuyu does not expect image placeholders in the inputs to HF processor, so
    the dummy prompt text is empty regardless of the number of images.

    ```python
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""
    ```

    For the multimodal image profiling data, the logic is very similar to LLaVA:

    ??? code

        ```python
        def get_dummy_mm_data(
            self,
            seq_len: int,
            mm_counts: Mapping[str, int],
            mm_options: Optional[Mapping[str, BaseDummyOptions]] = None,
        ) -> MultiModalDataDict:
            target_width, target_height = \
                self.info.get_image_size_with_most_features()
            num_images = mm_counts.get("image", 0)

            image_overrides = mm_options.get("image") if mm_options else None

            return {
                "image":
                self._get_dummy_images(
                    width=target_width,
                    height=target_height,
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

=== "With postprocessing: Fuyu"

    The `image_patches` output of `FuyuImageProcessor.preprocess_with_tokenizer_info` concatenates
    the patches from each image belonging to an item in the batch:

    ```python
    # https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/fuyu/image_processing_fuyu.py#L673-L679
            image_input_ids.append(tensor_of_image_ids)
            image_patches.append(patches)
        else:
            image_input_ids.append(torch.tensor([], dtype=torch.int32, device=image_input.device))

    batch_image_input_ids.append(image_input_ids)
    batch_image_patches.append(image_patches)
    ```

    The shape of `image_patches` outputted by `FuyuImageProcessor` is therefore
    `(1, num_images, num_patches, patch_width * patch_height * num_channels)`.

    In order to support the use of
    [MultiModalFieldConfig.batched][vllm.multimodal.inputs.MultiModalFieldConfig.batched]
    like in LLaVA, we remove the extra batch dimension by overriding
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

            image_patches = processed_outputs.get("image_patches")
            if image_patches is not None:
                images = mm_data["images"]
                assert isinstance(images, list)

                # Original output: (1, num_images, Pn, Px * Py * C)
                # New output: (num_images, Pn, Px * Py * C)
                assert (isinstance(image_patches, list)
                        and len(image_patches) == 1)
                assert (isinstance(image_patches[0], torch.Tensor)
                        and len(image_patches[0]) == len(images))

                processed_outputs["image_patches"] = image_patches[0]

            return processed_outputs
        ```

    !!! note
        Our [actual code](../../../vllm/model_executor/models/fuyu.py) has special handling
        for text-only inputs to prevent unnecessary warnings from HF processor.

    !!! note
        The `_call_hf_processor` method specifies both `mm_kwargs` and `tok_kwargs` for
        processing. `mm_kwargs` is used to both initialize and call the huggingface
        processor, whereas `tok_kwargs` is only used to call the huggingface processor.

    This lets us override [_get_mm_fields_config][vllm.multimodal.processing.BaseMultiModalProcessor._get_mm_fields_config] as follows:

    ```python
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(image_patches=MultiModalFieldConfig.batched("image"))
    ```

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

=== "Handling additional tokens: Fuyu"

    Recall the layout of feature tokens from Step 2:

    ```
    |SPEAKER||SPEAKER|...|SPEAKER||NEWLINE|
    |SPEAKER||SPEAKER|...|SPEAKER||NEWLINE|
    ...
    |SPEAKER||SPEAKER|...|SPEAKER||NEWLINE|
    ```

    We define a helper function to return `ncols` and `nrows` directly:

    ??? code

        ```python
        def get_image_feature_grid_size(
            self,
            *,
            image_width: int,
            image_height: int,
        ) -> tuple[int, int]:
            image_processor = self.get_image_processor()
            target_width = image_processor.size["width"]
            target_height = image_processor.size["height"]
            patch_width = image_processor.patch_size["width"]
            patch_height = image_processor.patch_size["height"]

            if not (image_width <= target_width and image_height <= target_height):
                height_scale_factor = target_height / image_height
                width_scale_factor = target_width / image_width
                optimal_scale_factor = min(height_scale_factor, width_scale_factor)

                image_height = int(image_height * optimal_scale_factor)
                image_width = int(image_width * optimal_scale_factor)

            ncols = math.ceil(image_width / patch_width)
            nrows = math.ceil(image_height / patch_height)
            return ncols, nrows
        ```

    Based on this, we can initially define our replacement tokens as:

    ??? code

        ```python
        def get_replacement(item_idx: int):
            images = mm_items.get_items("image", ImageProcessorItems)
            image_size = images.get_image_size(item_idx)

            ncols, nrows = self.info.get_image_feature_grid_size(
                image_width=image_size.width,
                image_height=image_size.height,
            )

            # `_IMAGE_TOKEN_ID` corresponds to `|SPEAKER|`
            # `_NEWLINE_TOKEN_ID` corresponds to `|NEWLINE|`
            return ([_IMAGE_TOKEN_ID] * ncols + [_NEWLINE_TOKEN_ID]) * nrows
        ```

    However, this is not entirely correct. After `FuyuImageProcessor.preprocess_with_tokenizer_info` is called,
    a BOS token (`<s>`) is also added to the promopt:

    ??? code

        ```python
        # https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/fuyu/processing_fuyu.py#L417-L435
        model_image_input = self.image_processor.preprocess_with_tokenizer_info(
            image_input=tensor_batch_images,
            image_present=image_present,
            image_unpadded_h=image_unpadded_heights,
            image_unpadded_w=image_unpadded_widths,
            image_placeholder_id=image_placeholder_id,
            image_newline_id=image_newline_id,
            variable_sized=True,
        )
        prompt_tokens, prompts_length = _tokenize_prompts_with_image_and_batch(
            tokenizer=self.tokenizer,
            prompts=prompts,
            scale_factors=scale_factors,
            max_tokens_to_generate=self.max_tokens_to_generate,
            max_position_embeddings=self.max_position_embeddings,
            add_BOS=True,
            add_beginning_of_answer_token=True,
        )
        ```

    To assign the vision embeddings to only the image tokens, instead of a string
    you can return an instance of [PromptUpdateDetails][vllm.multimodal.processing.PromptUpdateDetails]:

    ??? code

        ```python
        hf_config = self.info.get_hf_config()
        bos_token_id = hf_config.bos_token_id  # `<s>`
        assert isinstance(bos_token_id, int)

        def get_replacement_fuyu(item_idx: int):
            images = mm_items.get_items("image", ImageProcessorItems)
            image_size = images.get_image_size(item_idx)

            ncols, nrows = self.info.get_image_feature_grid_size(
                image_width=image_size.width,
                image_height=image_size.height,
            )
            image_tokens = ([_IMAGE_TOKEN_ID] * ncols + [_NEWLINE_TOKEN_ID]) * nrows

            return PromptUpdateDetails.select_token_id(
                image_tokens + [bos_token_id],
                embed_token_id=_IMAGE_TOKEN_ID,
            )
        ```

    Finally, noticing that the HF processor removes the `|ENDOFTEXT|` token from the tokenized prompt,
    we can search for it to conduct the replacement at the start of the string:

    ??? code

        ```python
        def _get_prompt_updates(
            self,
            mm_items: MultiModalDataItems,
            hf_processor_mm_kwargs: Mapping[str, object],
            out_mm_kwargs: MultiModalKwargsItems,
        ) -> Sequence[PromptUpdate]:
            hf_config = self.info.get_hf_config()
            bos_token_id = hf_config.bos_token_id
            assert isinstance(bos_token_id, int)

            tokenizer = self.info.get_tokenizer()
            eot_token_id = tokenizer.bos_token_id
            assert isinstance(eot_token_id, int)

            def get_replacement_fuyu(item_idx: int):
                images = mm_items.get_items("image", ImageProcessorItems)
                image_size = images.get_image_size(item_idx)

                ncols, nrows = self.info.get_image_feature_grid_size(
                    image_width=image_size.width,
                    image_height=image_size.height,
                )
                image_tokens = ([_IMAGE_TOKEN_ID] * ncols + [_NEWLINE_TOKEN_ID]) * nrows

                return PromptUpdateDetails.select_token_id(
                    image_tokens + [bos_token_id],
                    embed_token_id=_IMAGE_TOKEN_ID,
                )

            return [
                PromptReplacement(
                    modality="image",
                    target=[eot_token_id],
                    replacement=get_replacement_fuyu,
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
- Fuyu (appends `boa_token`): [vllm/model_executor/models/fuyu.py](../../../vllm/model_executor/models/fuyu.py)
- Molmo (applies chat template which is not defined elsewhere): [vllm/model_executor/models/molmo.py](../../../vllm/model_executor/models/molmo.py)

### Custom HF processor

Some models don't define an HF processor class on HF Hub. In that case, you can define a custom HF processor that has the same call signature as HF processors and pass it to [_call_hf_processor][vllm.multimodal.processing.BaseMultiModalProcessor._call_hf_processor].

Examples:

- DeepSeek-VL2: [vllm/model_executor/models/deepseek_vl2.py](../../../vllm/model_executor/models/deepseek_vl2.py)
- InternVL: [vllm/model_executor/models/internvl.py](../../../vllm/model_executor/models/internvl.py)
- Qwen-VL: [vllm/model_executor/models/qwen_vl.py](../../../vllm/model_executor/models/qwen_vl.py)
