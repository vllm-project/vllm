from typing import TYPE_CHECKING, Dict, List, Set, Tuple

from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
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


def compute_encoder_budget(
    model_config: "ModelConfig",
    scheduler_config: "SchedulerConfig",
) -> Tuple[int, int]:
    """Compute the encoder cache budget based on the model and scheduler 
    configurations.

    Args:
        model_config: Model configuration.
        scheduler_config: Scheduler configuration.

    Returns:
        - Compute budget for encoder execution, in unit of number of tokens 
            in the input sequence.
        - Space budget for encoder cache size, in unit of number of tokens 
            in the input sequence.
    """

    if not model_config.is_multimodal_model:
        return 0, 0

    # TODO: handle encoder-decoder models once we support them.
    (
        encoder_compute_budget,
        encoder_cache_size,
    ) = _compute_encoder_budget_multimodal(model_config, scheduler_config)

    return encoder_compute_budget, encoder_cache_size


def _compute_encoder_budget_multimodal(
    model_config: "ModelConfig",
    scheduler_config: "SchedulerConfig",
) -> Tuple[int, int]:
    """Compute the encoder cache budget based on the model and scheduler 
    configurations for a multimodal model.

    Args:
        model_config: Model configuration.
        scheduler_config: Scheduler configuration.

    Returns:
        - Compute budget for encoder execution, in unit of number of tokens 
            in the input sequence.
        - Space budget for encoder cache size, in unit of number of tokens 
            in the input sequence.
    """

    max_tokens_by_modality_dict = MULTIMODAL_REGISTRY.get_max_tokens_per_item_by_nonzero_modality(  # noqa: E501
        model_config)

    if not max_tokens_by_modality_dict:
        logger.warning(
            "All non-text modalities supported by the model have been "
            "explicitly disabled via limit_mm_per_prompt. Encoder cache will "
            "not be initialized.")
        return 0, 0

    _, max_tokens_per_mm_item = max(max_tokens_by_modality_dict.items(),
                                    key=lambda item: item[1])

    encoder_compute_budget = max(scheduler_config.max_num_encoder_input_tokens,
                                 max_tokens_per_mm_item)
    encoder_cache_size = max(scheduler_config.encoder_cache_size,
                             max_tokens_per_mm_item)

    return encoder_compute_budget, encoder_cache_size
