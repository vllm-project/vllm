# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from array import array
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from vllm.config.ec_manager_config import EncoderCacheManagerMetadata
from vllm.multimodal.utils import strip_covered_mm_data

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
    from vllm.lora.request import LoRARequest
    from vllm.multimodal.inputs import MultiModalFeatureSpec
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
    from vllm.v1.request import Request
else:
    ECConnectorMetadata = object
    KVConnectorMetadata = object
    KVCacheBlockCopy = object
    LoRARequest = object
    MultiModalFeatureSpec = object
    PoolingParams = object
    SamplingParams = object
    Request = object


@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec]
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    lora_request: LoRARequest | None
    prompt_embeds: "torch.Tensor | None" = None
    prompt_is_token_ids: list[bool] | None = None

    # Only used for v2 model runner.
    prefill_token_ids: list[int] | None = None

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: tuple[list[int], ...],
        prefill_token_ids: list[int] | None = None,
        uses_mrope: bool = False,
        uses_xdrope: bool = False,
    ) -> "NewRequestData":
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            mm_features=strip_covered_mm_data(
                request.mm_features,
                request.num_computed_tokens,
                uses_mrope=uses_mrope,
                uses_xdrope=uses_xdrope,
            ),
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
            prompt_embeds=request.prompt_embeds,
            prompt_is_token_ids=request.prompt_is_token_ids,
            prefill_token_ids=prefill_token_ids,
        )

    @property
    def prompt_len(self) -> int:
        if self.prompt_token_ids is not None:
            return len(self.prompt_token_ids)
        if self.prompt_embeds is not None:
            return self.prompt_embeds.shape[0]
        return 0

    def __repr__(self) -> str:
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None
        )
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids={self.prompt_token_ids},"
            f"prefill_token_ids={self.prefill_token_ids},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            ")"
        )

    # Version of __repr__ with the prompt data obfuscated
    def anon_repr(self) -> str:
        prompt_token_ids_len = (
            len(self.prompt_token_ids) if self.prompt_token_ids is not None else None
        )
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None
        )
        prefill_token_ids_len = (
            len(self.prefill_token_ids) if self.prefill_token_ids is not None else None
        )
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids_len={prompt_token_ids_len},"
            f"prefill_token_ids_len={prefill_token_ids_len},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            ")"
        )


@dataclass
class CachedRequestData:
    req_ids: list[str]
    # For request ids not in resumed_req_ids, new_block_ids will be appended to
    # the request's block IDs. For those in the set, new_block_ids will be used as the
    # request's block IDs instead of appending to the existing block IDs.
    resumed_req_ids: set[str]
    # NOTE(woosuk): new_token_ids is only used for pipeline parallelism.
    # When PP is not used, new_token_ids will be empty.
    new_token_ids: list[list[int]]
    # MRV1-only: For requests not scheduled in the last step, propagate the token ids
    # to the connector. Won't contain requests scheduled in the prior step.
    all_token_ids: dict[str, list[int]]
    new_block_ids: list[tuple[list[int], ...] | None]
    num_computed_tokens: list[int]
    num_output_tokens: list[int]

    # Version of dataclass repr with token IDs obfuscated.
    def anon_repr(self) -> str:
        new_token_ids_lens = [len(toks) for toks in self.new_token_ids]
        all_token_ids_lens = {
            req_id: len(toks) for req_id, toks in self.all_token_ids.items()
        }
        return (
            f"CachedRequestData("
            f"req_ids={self.req_ids},"
            f"resumed_req_ids={self.resumed_req_ids},"
            f"new_token_ids_lens={new_token_ids_lens},"
            f"all_token_ids_lens={all_token_ids_lens},"
            f"new_block_ids={self.new_block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"num_output_tokens={self.num_output_tokens}"
            f")"
        )

    def __repr__(self) -> str:
        return self.anon_repr()

    @property
    def num_reqs(self) -> int:
        return len(self.req_ids)

    @cached_property
    def _req_id_to_num_output_tokens(self) -> dict[str, int]:
        """Cache mapping of req_id to num_output_tokens for O(1) lookup.

        This cached property is safe because CachedRequestData instances
        are created fresh each scheduling iteration and not mutated during
        computation of iteration details.
        """
        return dict(zip(self.req_ids, self.num_output_tokens))

    def is_context_phase(self, req_id: str) -> bool:
        num_output_tokens = self._req_id_to_num_output_tokens.get(req_id)
        return num_output_tokens is not None and num_output_tokens == 0

    @classmethod
    def make_empty(cls) -> "CachedRequestData":
        return cls(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        )


@dataclass
class ScheduledEncoderInputStats:
    """Stats for encoder inputs scheduled in one iteration."""

    num_inputs: int = 0
    output_tokens: int = 0


@dataclass
class KVConnectorBlockState:
    """Scheduler-local block state offered to a producer-side KV connector."""

    # Authoritative current block-table snapshots.
    block_ids: dict[str, tuple[list[int], ...]]
    # Exact Mamba "align" boundary-state hand-offs.
    boundary_state_offloads: dict[str, list[tuple[int, int, int]]]


@dataclass
class SchedulerOutput:
    # list of the requests that are scheduled for the first time.
    # We cache the request's data in each worker process, so that we don't
    # need to re-send it every scheduling step.
    scheduled_new_reqs: list[NewRequestData]
    # list of the requests that have been scheduled before.
    # Since the request's data is already cached in the worker processes,
    # we only send the diff to minimize the communication cost.
    scheduled_cached_reqs: CachedRequestData

    # req_id -> num_scheduled_tokens
    # Number of tokens scheduled for each request.
    num_scheduled_tokens: dict[str, int]
    # Total number of tokens scheduled for all requests.
    # Equal to sum(num_scheduled_tokens.values())
    total_num_scheduled_tokens: int
    # req_id -> spec_token_ids
    # If a request does not have any spec decode tokens, it will not be
    # included in the dictionary.
    scheduled_spec_decode_tokens: dict[str, list[int]]
    # req_id -> encoder input indices that need processing.
    # E.g., if a request has [0, 1], it could mean the vision encoder needs
    # to process that the request's 0-th and 1-th images in the current step.
    scheduled_encoder_inputs: dict[str, list[int]]
    # Number of common prefix blocks for all requests in each KV cache group.
    # This can be used for cascade attention.
    num_common_prefix_blocks: list[int]

    # Request IDs that are finished in between the previous and the current
    # steps. This is used to notify the workers about the finished requests
    # so that they can free the cached states for those requests.
    finished_req_ids: set[str]
    # list of mm_hash strings associated with the encoder outputs to be
    # freed from the encoder cache.
    free_encoder_mm_hashes: list[str]

    scheduled_encoder_input_stats: ScheduledEncoderInputStats | None = None

    # Request IDs that are preempted in this step.
    # Only used for v2 model runner.
    preempted_req_ids: set[str] | None = None

    # Whether any of the scheduled requests use structured output.
    # Set only in async scheduling case.
    has_structured_output_requests: bool = False

    # Whether the scheduled requests have all the output tokens they
    # need to perform grammar bitmask computation.
    pending_structured_output_tokens: bool = False

    # Used for adjusting acceptance rate calculation.
    num_invalid_spec_tokens: dict[str, int] | None = None

    # KV Cache Connector metadata.
    kv_connector_metadata: KVConnectorMetadata | None = None

    # Whether any scheduled request consumes KV that the connector loads
    # synchronously during this step (load_async=False).
    has_sync_kv_loads: bool = False

    # EC Cache Connector metadata
    ec_connector_metadata: ECConnectorMetadata | None = None
    # EC Cache Manager metadata
    ec_manager_metadata: EncoderCacheManagerMetadata | None = None
    # Block IDs freshly allocated from the pool during this scheduling step.
    # The worker zeros the corresponding GPU memory before the blocks are used,
    # preventing stale NaN/data from corrupting attention or SSM computation.
    new_block_ids_to_zero: list[int] | None = None

    # CoW copies to apply after zeroing new blocks and before forward.
    kv_cache_block_copies: list[KVCacheBlockCopy] | None = None

    # Scheduler-local; always None by the time this reaches a worker.
    kv_connector_block_state: KVConnectorBlockState | None = None

    # Dynamic speculative decoding: optimal K chosen by scheduler.
    # Number of spec tokens to schedule for the next step.
    num_spec_tokens_to_schedule: int = 0

    @classmethod
    def make_empty(cls) -> "SchedulerOutput":
        return cls(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )


EXECUTE_MODEL_FAST_PATH_TAG = "__vllm_execute_model_fast_path_v1__"


def pack_scheduler_output_for_execute_model_fast_path(
    scheduler_output: SchedulerOutput,
) -> tuple:
    """Pack SchedulerOutput as positional tuples for execute_model RPC.

    The regular multiprocessing RPC pickles a dataclass object graph. This fixed
    schema avoids dataclass __dict__ and field-name metadata while using
    array('i') for int lists to speed up pickle/unpickle.
    """

    def pack_int_list(value: list[int] | None):
        return None if value is None else array("i", value)

    def pack_int_list_dict(value: dict[str, list[int]]):
        return {key: pack_int_list(val) for key, val in value.items()}

    def pack_block_ids(value: tuple[list[int], ...]):
        return tuple(pack_int_list(block_ids) for block_ids in value)

    def pack_optional_block_ids(value: tuple[list[int], ...] | None):
        return None if value is None else pack_block_ids(value)

    def pack_kv_connector_block_state(
        value: KVConnectorBlockState | None,
    ):
        if value is None:
            return None
        return (
            {
                req_id: pack_block_ids(block_ids)
                for req_id, block_ids in value.block_ids.items()
            },
            value.boundary_state_offloads,
        )

    cached = scheduler_output.scheduled_cached_reqs
    scheduled_new_reqs = tuple(
        (
            req.req_id,
            pack_int_list(req.prompt_token_ids),
            req.mm_features,
            req.sampling_params,
            req.pooling_params,
            pack_block_ids(req.block_ids),
            req.num_computed_tokens,
            req.lora_request,
            req.prompt_embeds,
            req.prompt_is_token_ids,
            pack_int_list(req.prefill_token_ids),
        )
        for req in scheduler_output.scheduled_new_reqs
    )
    scheduled_cached_reqs = (
        cached.req_ids,
        cached.resumed_req_ids,
        [pack_int_list(token_ids) for token_ids in cached.new_token_ids],
        pack_int_list_dict(cached.all_token_ids),
        [pack_optional_block_ids(block_ids) for block_ids in cached.new_block_ids],
        cached.num_computed_tokens,
        cached.num_output_tokens,
    )
    return (
        scheduled_new_reqs,
        scheduled_cached_reqs,
        scheduler_output.num_scheduled_tokens,
        scheduler_output.total_num_scheduled_tokens,
        pack_int_list_dict(scheduler_output.scheduled_spec_decode_tokens),
        pack_int_list_dict(scheduler_output.scheduled_encoder_inputs),
        pack_int_list(scheduler_output.num_common_prefix_blocks),
        scheduler_output.finished_req_ids,
        scheduler_output.free_encoder_mm_hashes,
        scheduler_output.scheduled_encoder_input_stats,
        scheduler_output.preempted_req_ids,
        scheduler_output.has_structured_output_requests,
        scheduler_output.pending_structured_output_tokens,
        scheduler_output.num_invalid_spec_tokens,
        scheduler_output.kv_connector_metadata,
        scheduler_output.has_sync_kv_loads,
        scheduler_output.ec_connector_metadata,
        scheduler_output.ec_manager_metadata,
        pack_int_list(scheduler_output.new_block_ids_to_zero),
        scheduler_output.kv_cache_block_copies,
        pack_kv_connector_block_state(scheduler_output.kv_connector_block_state),
        scheduler_output.num_spec_tokens_to_schedule,
    )


def unpack_scheduler_output_from_execute_model_fast_path(
    payload: tuple,
) -> SchedulerOutput:
    """Rebuild SchedulerOutput from the execute_model fast-path payload."""

    def unpack_int_list(value):
        return None if value is None else list(value)

    def unpack_int_list_dict(value):
        return {key: unpack_int_list(val) for key, val in value.items()}

    def unpack_block_ids(value):
        return tuple(unpack_int_list(block_ids) for block_ids in value)

    def unpack_optional_block_ids(value):
        return None if value is None else unpack_block_ids(value)

    def unpack_kv_connector_block_state(value):
        if value is None:
            return None
        block_ids_payload, boundary_state_offloads = value
        return KVConnectorBlockState(
            block_ids={
                req_id: unpack_block_ids(block_ids)
                for req_id, block_ids in block_ids_payload.items()
            },
            boundary_state_offloads=boundary_state_offloads,
        )

    (
        scheduled_new_reqs_payload,
        scheduled_cached_reqs_payload,
        num_scheduled_tokens,
        total_num_scheduled_tokens,
        scheduled_spec_decode_tokens,
        scheduled_encoder_inputs,
        num_common_prefix_blocks,
        finished_req_ids,
        free_encoder_mm_hashes,
        scheduled_encoder_input_stats,
        preempted_req_ids,
        has_structured_output_requests,
        pending_structured_output_tokens,
        num_invalid_spec_tokens,
        kv_connector_metadata,
        has_sync_kv_loads,
        ec_connector_metadata,
        ec_manager_metadata,
        new_block_ids_to_zero,
        kv_cache_block_copies,
        kv_connector_block_state,
        num_spec_tokens_to_schedule,
    ) = payload
    scheduled_new_reqs = [
        NewRequestData(
            req_id=req_id,
            prompt_token_ids=unpack_int_list(prompt_token_ids),
            mm_features=mm_features,
            sampling_params=sampling_params,
            pooling_params=pooling_params,
            block_ids=unpack_block_ids(block_ids),
            num_computed_tokens=num_computed_tokens,
            lora_request=lora_request,
            prompt_embeds=prompt_embeds,
            prompt_is_token_ids=prompt_is_token_ids,
            prefill_token_ids=unpack_int_list(prefill_token_ids),
        )
        for (
            req_id,
            prompt_token_ids,
            mm_features,
            sampling_params,
            pooling_params,
            block_ids,
            num_computed_tokens,
            lora_request,
            prompt_embeds,
            prompt_is_token_ids,
            prefill_token_ids,
        ) in scheduled_new_reqs_payload
    ]
    (
        req_ids,
        resumed_req_ids,
        new_token_ids,
        all_token_ids,
        new_block_ids,
        cached_num_computed_tokens,
        num_output_tokens,
    ) = scheduled_cached_reqs_payload
    scheduled_cached_reqs = CachedRequestData(
        req_ids=req_ids,
        resumed_req_ids=resumed_req_ids,
        new_token_ids=[unpack_int_list(token_ids) for token_ids in new_token_ids],
        all_token_ids=unpack_int_list_dict(all_token_ids),
        new_block_ids=[
            unpack_optional_block_ids(block_ids) for block_ids in new_block_ids
        ],
        num_computed_tokens=cached_num_computed_tokens,
        num_output_tokens=num_output_tokens,
    )
    return SchedulerOutput(
        scheduled_new_reqs=scheduled_new_reqs,
        scheduled_cached_reqs=scheduled_cached_reqs,
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        scheduled_spec_decode_tokens=unpack_int_list_dict(scheduled_spec_decode_tokens),
        scheduled_encoder_inputs=unpack_int_list_dict(scheduled_encoder_inputs),
        num_common_prefix_blocks=unpack_int_list(num_common_prefix_blocks),
        finished_req_ids=finished_req_ids,
        free_encoder_mm_hashes=free_encoder_mm_hashes,
        scheduled_encoder_input_stats=scheduled_encoder_input_stats,
        preempted_req_ids=preempted_req_ids,
        has_structured_output_requests=has_structured_output_requests,
        pending_structured_output_tokens=pending_structured_output_tokens,
        num_invalid_spec_tokens=num_invalid_spec_tokens,
        kv_connector_metadata=kv_connector_metadata,
        has_sync_kv_loads=has_sync_kv_loads,
        ec_connector_metadata=ec_connector_metadata,
        ec_manager_metadata=ec_manager_metadata,
        new_block_ids_to_zero=unpack_int_list(new_block_ids_to_zero),
        kv_cache_block_copies=kv_cache_block_copies,
        kv_connector_block_state=unpack_kv_connector_block_state(
            kv_connector_block_state
        ),
        num_spec_tokens_to_schedule=num_spec_tokens_to_schedule,
    )


@dataclass
class GrammarOutput:
    # ids of structured output requests.
    structured_output_request_ids: list[str]
    # Bitmask ordered as structured_output_request_ids.
    grammar_bitmask: "npt.NDArray[np.int32]"
