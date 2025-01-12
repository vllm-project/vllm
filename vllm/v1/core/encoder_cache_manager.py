from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

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

    Args:
        model_config: Model configuration.
        scheduler_config: Scheduler configuration.

    Returns:
        The encoder cache budget, in unit of number of tokens 
        in the input sequence.
    """

    encoder_cache_budget = 0

    if not model_config.is_multimodal_model:
        return encoder_cache_budget

    # TODO: handle encoder-decoder models once we support them.
    encoder_cache_budget, _, _ = compute_encoder_cache_budget_multimodal(
        model_config, scheduler_config)

    return encoder_cache_budget


def compute_encoder_cache_budget_multimodal(
    model_config: "ModelConfig",
    scheduler_config: "SchedulerConfig",
) -> tuple[int, Optional[str], int]:
    """Compute the encoder cache budget based on the model and scheduler 
    configurations for a multimodal model.

    Args:
        model_config: Model configuration.
        scheduler_config: Scheduler configuration.

    Returns:
        - The encoder cache budget, in unit of number of tokens in the 
            input sequence.
        - The modality of the multimodal item that requires the most tokens.
        - The number of multimodal items used to compute the encoder cache 
            budget.
    """

    encoder_cache_budget = 0
    max_tokens_by_modality_dict = MULTIMODAL_REGISTRY.get_max_tokens_per_item_by_nonzero_modality(  # noqa: E501
        model_config)

    if not max_tokens_by_modality_dict:
        logger.warning(
            "All non-text modalities supported by the model have been "
            "explicitly disabled via limit_mm_per_prompt. Encoder cache will "
            "not be initialized.")
        return encoder_cache_budget, None, 0

    modality, max_tokens_per_mm_item = max(max_tokens_by_modality_dict.items(),
                                           key=lambda item: item[1])

    max_num_batched_tokens = scheduler_config.max_num_batched_tokens
    max_num_reqs = scheduler_config.max_num_seqs

    # The biggest possible multimodal item cannot be fully prefilled in a
    # batch, so every batch can partially prefill at most one of such item.
    if max_tokens_per_mm_item > max_num_batched_tokens:
        num_items = 1

    # A batch can fully cover multiple biggest possible multimodal items, and
    # one that will be partially prefilled.
    else:
        num_items = cdiv(max_num_batched_tokens, max_tokens_per_mm_item)

    # NOTE: We need the encoder cache to be able to compute & hold ONE
    # ADDITIONAL multimodal item, and is required only when:
    # - Two requests in the current batch share the same prefix with such item
    #   as part of the prefix.
    # - AND the prefix length is divisible by the block size, triggering the
    #   recomputation of the last block.
    # - AND the part of the embeddings of the item is in this last block.

    # This issue can be fundamentally resolved by supporting num_new_tokens=0
    # on the model runner.
    num_items += 1

    # Number of items needed cannot be bigger than max number of running
    # requests * max number of multimodal items per request.
    max_mm_items_per_req = max(
        MULTIMODAL_REGISTRY.get_mm_limits_per_prompt(model_config).values())

    num_items = min(num_items, max_num_reqs * max_mm_items_per_req)
    encoder_cache_budget = num_items * max_tokens_per_mm_item

    logger.info(
        "Encoder cache will be initialized with a budget of %s tokens,"
        " and profiled with %s %s items of the maximum feature size.",
        encoder_cache_budget, num_items, modality)

    return encoder_cache_budget, modality, num_items
