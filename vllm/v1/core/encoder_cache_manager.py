from typing import TYPE_CHECKING, Dict, List, Set, Tuple

from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.utils import cdiv
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.config import ModelConfig, SchedulerConfig

logger = init_logger(__name__)


class EncoderCacheManager:

    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.num_free_slots = cache_size
        # req_id -> cached input ids
        self.cached: Dict[str, Set[int]] = {}
        # List of [req_id, input_id]
        self.freed: List[Tuple[str, int]] = []

    def has_cache(self, request: Request, input_id: int) -> bool:
        req_id = request.request_id
        return req_id in self.cached and input_id in self.cached[req_id]

    def can_allocate(self, request: Request, input_id: int) -> bool:
        num_tokens = request.get_num_encoder_tokens(input_id)
        return num_tokens <= self.num_free_slots

    def allocate(self, request: Request, input_id: int) -> None:
        req_id = request.request_id
        if req_id not in self.cached:
            self.cached[req_id] = set()
        self.cached[req_id].add(input_id)
        self.num_free_slots -= request.get_num_encoder_tokens(input_id)

    def get_cached_input_ids(self, request: Request) -> Set[int]:
        return self.cached.get(request.request_id, set())

    def free(self, request: Request, input_id: int) -> None:
        req_id = request.request_id
        if req_id not in self.cached:
            return

        self.cached[req_id].discard(input_id)
        if len(self.cached[req_id]) == 0:
            del self.cached[req_id]
        self.num_free_slots += request.get_num_encoder_tokens(input_id)
        self.freed.append((req_id, input_id))

    def get_freed_ids(self) -> List[Tuple[str, int]]:
        freed = self.freed
        self.freed = []
        return freed


def compute_encoder_cache_budget(
    model_config: "ModelConfig",
    scheduler_config: "SchedulerConfig",
) -> int:
    """Compute the encoder cache budget based on the model and scheduler 
    configurations.
    """

    encoder_cache_budget = 0
    if not model_config.is_multimodal_model:
        return encoder_cache_budget

    max_tokens_by_modality_dict = MULTIMODAL_REGISTRY.get_max_tokens_per_item_by_modality(  # noqa: E501
        model_config)

    modality, max_tokens_per_mm_item = max(max_tokens_by_modality_dict.items(),
                                           key=lambda item: item[1])

    max_num_batched_tokens = scheduler_config.max_num_batched_tokens
    max_num_reqs = scheduler_config.max_num_seqs

    # In case that the biggest possible multimodal item takes space more
    # than the batch size, then it needs to be cached and chunk prefilled.
    if max_tokens_per_mm_item > max_num_batched_tokens:
        num_items = 1

    # In case that the biggest possible multimodal item takes space less
    # the batch size, then all items will be full prefilled except one.
    else:
        num_items = cdiv(max_num_batched_tokens, max_tokens_per_mm_item)

    # NOTE: We need the encoder cache to be able to compute & hold ONE
    # ADDITIONAL multimodal item, and is required only when:
    # - Two requests in the current batch share the same prefix with such item
    #   as part of the prefix.
    # - AND the prefix length is divisible by the block size, triggering the
    #   recomputation of the last block.
    # - AND the part of the embeddings of the item is in this last block.

    # This can be improved when we have a global encoder cache that does
    # not associate items to request id only.
    num_items += 1

    # Number of items needed cannot be bigger than max number of running
    # requests.
    num_items = min(num_items, max_num_reqs)

    encoder_cache_budget = num_items * max_tokens_per_mm_item
    logger.info(
        "Encoder cache will be initialized with a budget of %s tokens, and "
        "profiled with %s %s items of the maximum feature size.",
        encoder_cache_budget, num_items, modality)

    return encoder_cache_budget
