from typing import Any, Dict, List, Tuple

from vllm.pooling_params import PoolingParams


class PoolingMetadata:
    """Metadata for pooling operations in the Pooler layer.

    This class holds the necessary information for pooling operations,
    providing context for how to perform pooling and other related operations.

    Attributes:
        seq_groups: List of (seq_ids, pooling_params).
        seq_data: A mapping of sequence ID to additional sequence data.
        prompt_lens: List of the lengths of each prompt.
    """

    def __init__(
        self,
        seq_groups: List[Tuple[List[int], PoolingParams]],
        seq_data: Dict[int, Any],  # Specific data related to sequences
        prompt_lens: List[int],
    ) -> None:
        self.seq_groups = seq_groups
        self.seq_data = seq_data
        self.prompt_lens = prompt_lens

    def __repr__(self) -> str:
        return ("PoolingMetadata("
                f"seq_groups={self.seq_groups}, "
                f"seq_data={self.seq_data}, "
                f"prompt_lens={self.prompt_lens}, ")
