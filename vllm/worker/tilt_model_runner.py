import dataclasses
import itertools
import weakref
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import torch
import torch.distributed

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.backends.utils import PAD_SLOT_ID, CommonMetadataBuilder
from vllm.attention.backends.xformers import XFormersMetadata
from vllm.attention.selector import (get_env_variable_attn_backend,
                                     get_global_forced_attn_backend)
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.multimodal.inputs import (BatchedTensorInputs, MultiModalKwargs,
                                    NestedTensors)
from vllm.multimodal.ocr_document import OCR_DOCUMENT_PLUGIN_KEY, OcrDocument
from vllm.platforms import _Backend
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.sequence import (IntermediateTensors, SequenceData,
                           SequenceGroupMetadata)
from vllm.utils import (STR_NOT_IMPL_ENC_DEC_BACKEND, async_tensor_h2d,
                        make_tensor_with_pad)
from vllm.worker.enc_dec_model_runner import (EncoderDecoderModelInput,
                                              EncoderDecoderModelRunner)
from vllm.worker.model_runner import ModelInputForGPUBuilder

logger = init_logger(__name__)


@dataclasses.dataclass
class TiltXFormersMetadata(XFormersMetadata):
    # Encoder metadata is used for two different purposes:
    #   encoding and cross-attention.
    # Metadata for both purposes are the same in the original support of
    # encoder-decoder models, which assumes that entire encoder and decoder
    # prefill is scheduled at the same time. However, in TILT models, the
    # encoder and decoder sequences are scheduled independently.
    # Attributes below have been introduced to store cross-attention metadata.
    # Metadata for encoding are stored in the parent class, XFormersMetadata.
    cross_encoder_seq_lens: Optional[List[int]] = None
    cross_encoder_seq_lens_tensor: Optional[torch.Tensor] = None
    cross_max_encoder_seq_len: Optional[int] = None


class TiltXFormersMetadataBuilder(CommonMetadataBuilder[TiltXFormersMetadata]):
    _metadata_cls = TiltXFormersMetadata


@dataclasses.dataclass(frozen=True)
class TiltModelInput(EncoderDecoderModelInput):
    """
    Used by the TiltModelRunner.
    """

    encoder_chunk_ids: Optional[torch.Tensor] = None
    fusion_in_decoder_mask: Optional[torch.Tensor] = None

    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        tensor_dict = super().as_broadcastable_tensor_dict()
        tensor_dict["encoder_chunk_ids"] = self.encoder_chunk_ids
        tensor_dict["fusion_in_decoder_mask"] = self.fusion_in_decoder_mask
        tensor_dict["multi_modal_kwargs"] = self.multi_modal_kwargs
        return tensor_dict


class TiltModelInputBuilder(ModelInputForGPUBuilder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.attn_backend is not None:
            self.attn_metadata_builder = TiltXFormersMetadataBuilder(
                weakref.proxy(self))

    def _compute_multi_modal_input(
        self,
        inter_data: ModelInputForGPUBuilder.InterDataForSeqGroup,
        seq_group_metadata: SequenceGroupMetadata,
    ):
        """TILT model runner handles multi-modal data elsewhere."""
        return

    def build(self) -> TiltModelInput:
        """Copy of ModelInputForGPUBuilder.build() with changes.

        Main changes is: when there are no decoder tokens, it is not assumed
        that all tokens are cached and the computation can be skipped. There
        can still be some computation to be performed in the encoder.

        """
        # Combine and flatten intermediate data.
        input_tokens = []
        token_types = []
        for inter_data in self.inter_data_list:
            for cur_input_tokens in inter_data.input_tokens:
                input_tokens.extend(cur_input_tokens)
            for cur_token_types in inter_data.token_types:
                token_types.extend(cur_token_types)

        input_positions = []
        for inter_data in self.inter_data_list:
            for cur_input_positions in inter_data.input_positions:
                input_positions.extend(cur_input_positions)

        seq_lens = []
        query_lens = []
        max_decode_seq_len = 0
        max_encoder_seq_len = 0
        for inter_data in self.inter_data_list:
            seq_lens.extend(inter_data.seq_lens)
            query_lens.extend(inter_data.query_lens)
            if not inter_data.is_prompt:
                max_decode_seq_len = max(max_decode_seq_len,
                                         max(inter_data.seq_lens))
                if self.runner.model_config.is_encoder_decoder:
                    max_encoder_seq_len = max(max_encoder_seq_len,
                                              inter_data.encoder_seq_len)

        # Mapping from request IDs to sequence IDs. Used for Jamba models
        # that manages the cache by itself.
        request_ids_to_seq_ids = {
            data.request_id: data.seq_ids
            for data in self.inter_data_list
        }

        cuda_graph_pad_size = self._get_cuda_graph_pad_size(
            num_seqs=len(seq_lens),
            max_decode_seq_len=max_decode_seq_len,
            max_encoder_seq_len=max_encoder_seq_len,
        )

        batch_size = len(input_tokens)
        if cuda_graph_pad_size != -1:
            # If cuda graph can be used, pad tensors accordingly.
            # See `capture_model` API for more details.
            # vLLM uses cuda graph only for decoding requests.
            batch_size += cuda_graph_pad_size

        # Tokens and positions.
        if cuda_graph_pad_size:
            input_tokens.extend(itertools.repeat(0, cuda_graph_pad_size))
        assert self.runner.device is not None
        input_tokens_tensor = async_tensor_h2d(input_tokens, torch.long,
                                               self.runner.device,
                                               self.runner.pin_memory)

        token_types_tensor = (async_tensor_h2d(
            token_types, torch.long, self.runner.device,
            self.runner.pin_memory) if token_types else None)

        input_positions.extend(itertools.repeat(0, cuda_graph_pad_size))
        input_positions_tensor = async_tensor_h2d(input_positions, torch.long,
                                                  self.runner.device,
                                                  self.runner.pin_memory)
        # Sequence and query lengths.
        if cuda_graph_pad_size:
            seq_lens.extend(itertools.repeat(1, cuda_graph_pad_size))

        # Attention metadata.
        attn_metadata = self.attn_metadata_builder.build(
            seq_lens, query_lens, cuda_graph_pad_size, batch_size)

        # LoRA data.
        lora_requests = set()
        lora_mapping = None
        if self.enable_lora:
            raise NotImplementedError("TILT does not support LoRA")

        # Prompt adapter data.
        prompt_adapter_requests: Set[PromptAdapterRequest] = set()
        prompt_adapter_mapping = None
        if self.enable_prompt_adapter:
            raise NotImplementedError("TILT does not support prompt adapters")

        return self.model_input_cls(
            input_tokens=input_tokens_tensor,
            input_positions=input_positions_tensor,
            token_types=token_types_tensor,
            attn_metadata=attn_metadata,
            seq_lens=seq_lens,
            query_lens=query_lens,
            lora_mapping=lora_mapping,
            lora_requests=lora_requests,
            multi_modal_kwargs={},
            request_ids_to_seq_ids=request_ids_to_seq_ids,
            finished_requests_ids=self.finished_requests_ids,
            prompt_adapter_mapping=prompt_adapter_mapping,
            prompt_adapter_requests=prompt_adapter_requests,
        )


class TiltModelRunner(EncoderDecoderModelRunner):
    """Model runner with support for TILT encoder-decoder model.

    Changes:
      - Support for multimodal data with custom batching.
      - Support for TILT Long inference.

    """

    _model_input_cls: Type[TiltModelInput] = TiltModelInput
    _builder_cls: Type[TiltModelInputBuilder] = TiltModelInputBuilder

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_config = self.model_config.hf_config
        self.chunk_size = model_config.chunk_length
        self.max_images_in_chunk = model_config.max_images_in_chunk
        self.max_image_size = (model_config.max_image_height,
                               model_config.image_width)

    def _maybe_force_supported_attention_backend(self):
        """
        Force vLLM to use the XFormers attention backend,
        which is currently the only supported option.
        """

        def raise_backend_err():
            # The user has specified an attention backend override
            # which is invalid for encoder/decoder models
            raise NotImplementedError(STR_NOT_IMPL_ENC_DEC_BACKEND)

        maybe_env_var_forced_backend = get_env_variable_attn_backend()
        maybe_global_forced_backend = get_global_forced_attn_backend()
        is_forced_by_global = maybe_global_forced_backend is not None
        is_forced_by_env_var = maybe_env_var_forced_backend is not None
        if is_forced_by_global:  # noqa: SIM102
            # Backend override enforced by global variable takes
            # precedence over vLLM backend environment variable.
            if maybe_global_forced_backend not in [_Backend.XFORMERS]:
                raise_backend_err()
        elif is_forced_by_env_var:  # noqa: SIM102
            # Backend override enforced by vLLM backend
            # environment variable
            if maybe_env_var_forced_backend not in [_Backend.XFORMERS]:
                raise_backend_err()

    @torch.inference_mode()
    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None,
    ) -> TiltModelInput:
        model_input = self._prepare_model_input_tensors(
            seq_group_metadata_list, finished_requests_ids)

        (
            attn_metadata,
            encoder_input_tokens_tensor,
            encoder_input_positions_tensor,
            encoder_multi_modal_data,
            encoder_chunk_ids_tensor,
            fusion_in_decoder_mask_tensor,
        ) = self._prepare_encoder_model_input_tensors(seq_group_metadata_list,
                                                      model_input)

        is_prompt = (seq_group_metadata_list[0].is_prompt
                     if seq_group_metadata_list else None)

        # Inject attn_metadata encoder/cross-attention fields &
        # encoder input tokens/positions into model_input.
        # Frozen dataclass fields cannot be modified, so use
        # dataclasses.replace to construct a new model input
        # instance.
        model_input = dataclasses.replace(
            model_input,
            attn_metadata=attn_metadata,
            encoder_input_tokens=encoder_input_tokens_tensor,
            encoder_input_positions=encoder_input_positions_tensor,
            encoder_chunk_ids=encoder_chunk_ids_tensor,
            fusion_in_decoder_mask=fusion_in_decoder_mask_tensor,
            multi_modal_kwargs=encoder_multi_modal_data,
        )

        generators = self.get_generators(finished_requests_ids)
        sampling_metadata = SamplingMetadata.prepare(
            seq_group_metadata_list,
            model_input.seq_lens,
            model_input.query_lens,
            self.device,
            self.pin_memory,
            generators=generators,
        )
        return dataclasses.replace(
            model_input,
            sampling_metadata=sampling_metadata,
            is_prompt=is_prompt,
            virtual_engine=virtual_engine,
        )

    def _prepare_encoder_model_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        model_input: EncoderDecoderModelInput,
    ) -> Tuple[
            AttentionMetadata,
            Optional[torch.Tensor],
            Optional[torch.Tensor],
            Optional[torch.Tensor],
            Optional[torch.Tensor],
            Optional[torch.Tensor],
    ]:
        if len(seq_group_metadata_list) == 0:
            return (model_input.attn_metadata, None, None, None, None, None)

        # Since we are not supporting chunked prefill either the entire
        # batch is prefill or it is decode
        is_prompt = seq_group_metadata_list[0].is_prompt

        # Build encoder inputs
        encoder_seq_lens: List[int] = []
        encoder_input_tokens = []
        encoder_input_positions = []
        encoder_multi_modal_data = []
        current_chunk_id = 0
        encoder_chunk_ids = []
        fusion_in_decoder_mask = []
        cross_encoder_seq_lens: List[int] = []
        cross_slot_mapping = []
        cross_block_tables = []
        for seq_group_metadata in seq_group_metadata_list:
            # Profiling run
            is_profile_run = seq_group_metadata.block_tables is None
            if is_prompt and is_profile_run:
                if seq_group_metadata.encoder_seq_data is None:
                    continue

                seq_len = seq_group_metadata.encoder_seq_data.get_len()
                # During memory profiling, the block tables are not
                # initialized yet. In this case, we just use a dummy
                # slot mapping.
                cross_slot_mapping.extend([PAD_SLOT_ID] * seq_len)

                token_ids = seq_group_metadata.encoder_seq_data.get_token_ids()
                encoder_seq_lens.append(seq_len)
                encoder_chunk_ids.extend([-1] * seq_len)
                fusion_in_decoder_mask.extend([True] * seq_len)

                # Build encoder input tokens
                encoder_input_tokens.extend(token_ids)
                encoder_input_positions.extend(list(range(0, seq_len)))

                # Multi-modal data
                mm_data = seq_group_metadata.multi_modal_data
                if mm_data is not None:
                    mm_data = self.multi_modal_input_mapper(mm_data)
                encoder_multi_modal_data.append(mm_data)
                continue

            prefix_len = seq_group_metadata.encoder_prefix_seq_data.get_len()
            seq_len = seq_group_metadata.encoder_seq_data.get_len()

            # If decoder tokens are scheduled:
            if seq_group_metadata.token_chunk_size > 0:
                for _ in range(len(seq_group_metadata.seq_data)):
                    cross_block_table = seq_group_metadata.cross_block_table
                    if cross_block_table is None:
                        cross_block_table = []
                    for _ in range(seq_group_metadata.token_chunk_size):
                        # Duplicate cross attention block table for every token.
                        #
                        # Usually, TILT schedules only one decoder token (both
                        # in prefill and decode phases). Preemption can create
                        # a decoder prefill request with many tokens.
                        # Duplication should not be a problem most of the time.
                        #
                        # TODO: specialized PagedAttn kernel to avoid creating
                        # a block table too large
                        cross_block_tables.append(cross_block_table)
                        cross_encoder_seq_lens.append(prefix_len + seq_len)

            # If encoder tokens are scheduled:
            if seq_group_metadata.encoder_token_chunk_size > 0:
                # Get prefix data
                prefix_computed_len = (
                    seq_group_metadata.encoder_prefix_seq_data.
                    get_num_computed_tokens())
                assert (
                    prefix_computed_len == 0
                    or prefix_computed_len == prefix_len
                ), f"prefix_computed_len={prefix_computed_len}, prefix_len={prefix_len}"
                prefix_token_ids = (
                    seq_group_metadata.encoder_prefix_seq_data.get_token_ids())
                prefix_mm_data = seq_group_metadata.encoder_prefix_multi_modal_data
                if prefix_computed_len is not None:
                    prefix_mm_data = self.multi_modal_input_mapper(
                        prefix_mm_data)
                is_first_chunk = prefix_computed_len == 0

                # Compute chunk count and usable chunk size
                usable_chunk_size = self.chunk_size - prefix_len
                num_computed_encoder_tokens = (
                    seq_group_metadata.encoder_seq_data.
                    get_num_computed_tokens())
                if num_computed_encoder_tokens == 0:
                    assert (prefix_computed_len == 0
                            ), f"prefix_computed_len={prefix_computed_len}"
                num_new_encoder_tokens = seq_group_metadata.encoder_token_chunk_size
                last_new_encoder_tokens = (num_computed_encoder_tokens +
                                           num_new_encoder_tokens)
                chunk_count = (num_new_encoder_tokens + usable_chunk_size -
                               1) // usable_chunk_size

                # Get sequence data
                token_ids = seq_group_metadata.encoder_seq_data.get_token_ids()
                mm_data = seq_group_metadata.multi_modal_data
                mm_data = self.multi_modal_input_mapper(mm_data)

                # Build encoder input tokens
                current_encoder_token_idx = num_computed_encoder_tokens
                while current_encoder_token_idx < last_new_encoder_tokens:
                    # Add a chunk to encoder inputs
                    encoder_input_tokens.extend(prefix_token_ids)
                    encoder_multi_modal_data.append(prefix_mm_data)
                    chunk_token_ids = token_ids[
                        current_encoder_token_idx:current_encoder_token_idx +
                        usable_chunk_size]
                    encoder_input_tokens.extend(chunk_token_ids)
                    encoder_input_positions.extend(
                        list(range(0, prefix_len + len(chunk_token_ids))))
                    encoder_multi_modal_data.append({
                        "ocr_document":
                        self._subseq_document_mm_data(
                            mm_data["ocr_document"],
                            current_encoder_token_idx,
                            current_encoder_token_idx + usable_chunk_size,
                        )
                    })
                    encoder_chunk_ids.extend(
                        [current_chunk_id] *
                        (prefix_len + len(chunk_token_ids)))
                    fusion_in_decoder_mask.extend([is_first_chunk] *
                                                  prefix_len)
                    fusion_in_decoder_mask.extend([True] *
                                                  len(chunk_token_ids))

                    current_chunk_id += 1
                    current_encoder_token_idx += usable_chunk_size
                    is_first_chunk = False

                # Find slots for new encoder tokens
                for i in range(
                        prefix_computed_len + num_computed_encoder_tokens,
                        prefix_len + last_new_encoder_tokens,
                ):
                    block_number = seq_group_metadata.cross_block_table[
                        i // self.block_size]
                    block_offset = i % self.block_size
                    slot = block_number * self.block_size + block_offset
                    cross_slot_mapping.append(slot)

                encoder_seq_lens.append(num_new_encoder_tokens +
                                        chunk_count * prefix_len)
        if (model_input.attn_metadata is not None
                and model_input.attn_metadata.use_cuda_graph):
            raise NotImplementedError(
                "CUDA graphs are not supported in TILT yet"
            )  # TODO: CUDA graphs
        else:
            max_len_of_block_table = max(
                (len(block_table) for block_table in cross_block_tables),
                default=0)

        # Convert tokens/positions & cross-attention
        # slot-mapping to encoder input tensors
        encoder_input_tokens_tensor = self._list_to_long_tensor(
            encoder_input_tokens)
        encoder_input_positions_tensor = self._list_to_long_tensor(
            encoder_input_positions)
        encoder_multi_modal_data_batched = self.batch(encoder_multi_modal_data)
        encoder_chunk_ids_tensor = self._list_to_long_tensor(encoder_chunk_ids)
        fusion_in_decoder_mask_tensor = torch.tensor(fusion_in_decoder_mask,
                                                     dtype=torch.bool,
                                                     device=self.device)

        cross_slot_mapping_tensor = self._list_to_long_tensor(
            cross_slot_mapping)
        cross_block_tables = make_tensor_with_pad(
            cross_block_tables,
            max_len=max_len_of_block_table,
            pad=0,
            dtype=torch.int32,
            device=self.device,
        )
        cross_max_encoder_seq_len = max(cross_encoder_seq_lens, default=0)
        cross_encoder_seq_lens_tensor = self._list_to_int32_tensor(
            cross_encoder_seq_lens)

        # Compute encoder sequence lengths & encoder
        # sequence starting offset tensors
        max_encoder_seq_len = max(encoder_seq_lens, default=0)
        encoder_seq_lens_tensor = self._list_to_int32_tensor(encoder_seq_lens)
        encoder_seq_start_loc = torch.zeros(encoder_seq_lens_tensor.shape[0] +
                                            1,
                                            dtype=torch.int32,
                                            device=self.device)
        torch.cumsum(
            encoder_seq_lens_tensor,
            dim=0,
            dtype=encoder_seq_start_loc.dtype,
            out=encoder_seq_start_loc[1:],
        )

        # Update attention metadata with encoder-oriented attributes
        attn_metadata = model_input.attn_metadata
        assert attn_metadata is not None
        (
            attn_metadata.num_encoder_tokens,
            attn_metadata.encoder_seq_lens,
            attn_metadata.encoder_seq_lens_tensor,
            attn_metadata.max_encoder_seq_len,
            attn_metadata.encoder_seq_start_loc,
            attn_metadata.cross_slot_mapping,
            attn_metadata.cross_block_tables,
            attn_metadata.cross_max_encoder_seq_len,
            attn_metadata.cross_encoder_seq_lens,
            attn_metadata.cross_encoder_seq_lens_tensor,
        ) = (
            sum(encoder_seq_lens),
            encoder_seq_lens,
            encoder_seq_lens_tensor,
            max_encoder_seq_len,
            encoder_seq_start_loc,
            cross_slot_mapping_tensor,
            cross_block_tables,
            cross_max_encoder_seq_len,
            cross_encoder_seq_lens,
            cross_encoder_seq_lens_tensor,
        )

        return (
            attn_metadata,
            encoder_input_tokens_tensor,
            encoder_input_positions_tensor,
            encoder_multi_modal_data_batched,
            encoder_chunk_ids_tensor,
            fusion_in_decoder_mask_tensor,
        )

    def batch(self,
              inputs_list: List[MultiModalKwargs]) -> BatchedTensorInputs:
        """Batch TILT multimodal data."""
        if len(inputs_list) == 0:
            return {}

        # We need to consider the case where each item in the batch
        # contains different modalities (i.e. different keys).
        item_lists = defaultdict[str, list[NestedTensors]](list)

        for inputs in inputs_list:
            for k, v in inputs.items():
                item_lists[k].append(v)

        res = {}
        for k, item_list in item_lists.items():
            if k != OCR_DOCUMENT_PLUGIN_KEY:
                raise RuntimeError("TiltModelRunner supports only multi-modal"
                                   f' data type "{OCR_DOCUMENT_PLUGIN_KEY}",'
                                   f' but the request includes "{k}" type.')
            res[k] = self._batch_document_mm_data(item_list)
        return res

    def _batch_document_mm_data(self,
                                data_list: List[OcrDocument]) -> OcrDocument:
        all_token_bboxes = []
        all_images = []
        all_bboxes = []
        all_token_indices = []
        all_pages = []
        offset = 0
        page_offset = 0
        # TODO: identify duplicated images

        for data in data_list:
            all_token_bboxes.append(data.token_bboxes)

            all_images.extend(data.images)
            bboxes = data.roi_bboxes.clone()
            bboxes[:, 0] += page_offset
            all_bboxes.append(bboxes)
            if data.roi_token_indices is not None:
                all_token_indices.append(data.roi_token_indices + offset)
            elif (roi_token_count := data.roi_bboxes.shape[0]) > 0:
                all_token_indices.append(
                    torch.arange(offset, offset + roi_token_count))
            all_pages.append(data.pages)

            offset += data.token_count
            page_offset += data.page_count

        token_bboxes = torch.cat(all_token_bboxes, dim=0)
        images = self._batch_images(all_images)
        roi_bboxes = torch.cat(all_bboxes, dim=0)
        roi_token_indices = torch.cat(all_token_indices, dim=0)
        pages = torch.cat(all_pages, dim=0)
        return OcrDocument(
            token_bboxes=token_bboxes,
            images=images,
            roi_bboxes=roi_bboxes,
            roi_token_indices=roi_token_indices,
            pages=pages,
            token_count=offset,
            page_count=page_offset,
        )

    def _batch_images(
            self,
            images: list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        # Find max image size
        H = 0
        W = 0
        for image in images:
            _, h, w = image.shape
            H = max(h, H)
            W = max(w, W)
        # Pad to multiple of 64
        H += -H % 64
        W += -W % 64

        padded_images = []
        for image in images:
            _, h, w = image.shape
            padded_images.append(
                torch.nn.functional.pad(image, (0, W - w, 0, H - h),
                                        value=1.0))

        batch = torch.stack(padded_images, dim=0)
        return batch

    def _subseq_document_mm_data(self, data: OcrDocument, start_token: int,
                                 end_token: int) -> OcrDocument:
        """Select a chunk from multimodal data."""
        assert data.roi_token_indices is None

        token_bboxes = data.token_bboxes[start_token:end_token]
        roi_bboxes = data.roi_bboxes[start_token:end_token].clone()
        image_idxs, new_roi_image_idx = torch.unique(roi_bboxes[:, 0],
                                                     return_inverse=True)
        images = []
        for image_idx in image_idxs.to(torch.long).tolist():
            images.append(data.images[image_idx])
        roi_bboxes[:, 0] = new_roi_image_idx
        pages = data.pages[start_token:end_token]
        token_count = token_bboxes.shape[0]

        return OcrDocument(
            token_bboxes=token_bboxes,
            images=images,
            roi_bboxes=roi_bboxes,
            roi_token_indices=None,
            pages=pages,
            token_count=token_count,
            page_count=len(images),
        )

    @torch.inference_mode()
    def profile_run(self) -> None:
        logger.info("Starting profile run for a TILT model.")
        # Enable top-k sampling to reflect the accurate memory usage.
        sampling_params = SamplingParams(top_p=0.99, top_k=self.vocab_size - 1)

        # TODO: move to TiltDummyInputsBuilder
        max_num_batched_decoder_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_batched_encoder_tokens = (
            self.scheduler_config.max_num_batched_encoder_tokens)
        encoder_chunk_size = self.scheduler_config.encoder_chunk_size

        max_encoder_chunks = (max_num_batched_encoder_tokens -
                              1) // encoder_chunk_size + 1
        max_num_seqs = self.scheduler_config.max_num_seqs

        max_images_in_chunk = self.max_images_in_chunk
        max_image_size = self.max_image_size

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_{decoder,encoder}_tokens.
        seqs: List[SequenceGroupMetadata] = []

        page_count = max_images_in_chunk * max_encoder_chunks
        multi_modal_data = {
            "ocr_document":
            OcrDocument(
                token_bboxes=torch.zeros((max_num_batched_encoder_tokens, 4),
                                         dtype=torch.float32),
                images=[
                    torch.zeros((1, *max_image_size), dtype=torch.float32)
                    for _ in range(page_count)
                ],
                roi_bboxes=torch.zeros((max_num_batched_encoder_tokens, 5),
                                       dtype=torch.float32),
                roi_token_indices=None,
                pages=torch.randint(0,
                                    page_count,
                                    size=(max_num_batched_encoder_tokens, )),
                token_count=max_num_batched_encoder_tokens,
                page_count=page_count,
            )
        }
        batch_size = 0
        for group_id in range(max_num_seqs):
            seq_len = max_num_batched_decoder_tokens // max_num_seqs + (
                group_id < max_num_batched_decoder_tokens % max_num_seqs)
            batch_size += seq_len
            encoder_seq_len = max_num_batched_encoder_tokens // max_num_seqs + (
                group_id < max_num_batched_encoder_tokens % max_num_seqs)

            decoder_dummy_data = SequenceData.from_prompt_token_counts(
                (0, seq_len))
            encoder_dummy_data = SequenceData.from_prompt_token_counts(
                (0, encoder_seq_len))

            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: decoder_dummy_data},
                sampling_params=sampling_params,
                block_tables=None,
                encoder_seq_data=encoder_dummy_data,
                cross_block_table=None,
                # Small hack: first request contains multimodal data for all sequences.
                # Subsequent requests will contain empty data.
                multi_modal_data=multi_modal_data,
                multi_modal_placeholders={},
            )
            seqs.append(seq)
            multi_modal_data = {}

        finished_requests_ids = [seq.request_id for seq in seqs]
        model_input = self.prepare_model_input(
            seqs, finished_requests_ids=finished_requests_ids)
        intermediate_tensors = None
        self.execute_model(model_input, None, intermediate_tensors)
        torch.cuda.synchronize()
        return

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: EncoderDecoderModelInput,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:
        if num_steps > 1:
            raise ValueError("num_steps > 1 is not supported in "
                             "EncoderDecoderModelRunner")

        if (model_input.attn_metadata is not None
                and model_input.attn_metadata.prefill_metadata is None
                and model_input.attn_metadata.decode_metadata.use_cuda_graph):
            assert model_input.input_tokens is not None
            graph_batch_size = model_input.input_tokens.shape[0]
            model_executable = self.graph_runners[
                model_input.virtual_engine][graph_batch_size]
        else:
            model_executable = self.model

        seqlen_agnostic_kwargs = (
            {
                "finished_requests_ids": model_input.finished_requests_ids,
                "request_ids_to_seq_ids": model_input.request_ids_to_seq_ids,
            } if self.has_inner_state else {})

        multi_modal_kwargs = model_input.multi_modal_kwargs or {}
        with set_forward_context(model_input.attn_metadata, self.vllm_config,
                                 model_input.virtual_engine):
            hidden_or_intermediate_states = model_executable(
                input_ids=model_input.input_tokens,
                positions=model_input.input_positions,
                encoder_input_ids=model_input.encoder_input_tokens,
                encoder_positions=model_input.encoder_input_positions,
                encoder_chunk_ids=model_input.encoder_chunk_ids,
                fusion_in_decoder_mask=model_input.fusion_in_decoder_mask,
                kv_caches=kv_caches,
                attn_metadata=model_input.attn_metadata,
                intermediate_tensors=intermediate_tensors,
                **MultiModalKwargs.as_kwargs(multi_modal_kwargs,
                                             device=self.device),
                **seqlen_agnostic_kwargs,
            )

        if hidden_or_intermediate_states is None:
            return [SamplerOutput()]

        logits = self.model.compute_logits(hidden_or_intermediate_states,
                                           model_input.sampling_metadata)

        if not self.is_driver_worker:
            return []

        if model_input.async_callback is not None:
            model_input.async_callback()

        # Sample the next token.
        output: SamplerOutput = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )

        return [output]
