# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import nn

from vllm.model_executor.utils import is_weights_pre_processed, set_weight_attrs


class PerTokenTable(nn.Module):
    """This TP rank's hash columns of the checkpoint's per-token table."""

    dp_size = 1
    cpu_offload = False

    def __init__(
        self,
        comp_vocab: int,
        n_hash_cols: int,
        head_dim: int,
        tp_size: int,
        tp_rank: int,
    ):
        super().__init__()
        self.n_hash_cols = n_hash_cols
        self.tp_size = tp_size
        self.part_n_hash_cols = -(-n_hash_cols // tp_size)
        self.head_start = tp_rank * self.part_n_hash_cols
        self.weight = nn.Parameter(
            torch.full(
                (comp_vocab, self.part_n_hash_cols, head_dim),
                float("nan"),
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

    def process_weights_after_loading(self) -> None:
        """Reject a missing or NaN checkpoint table after loading."""
        if is_weights_pre_processed() or self.weight.device.type == "meta":
            return
        if torch.isnan(self.weight).any():
            raise ValueError(
                "engram mode 'per_token' needs the checkpoint tensor "
                "'...engram.per_token_table.weight' (expected shape "
                "(compressed_vocab, n_hash_cols, head_dim), bf16), but this "
                "table was never loaded (still NaN) or contains NaN. Export a "
                "per-token checkpoint or use engram mode 'ngram'."
            )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        expected = (param.shape[0], self.n_hash_cols, param.shape[-1])
        if tuple(loaded_weight.shape) != expected:
            raise ValueError(
                f"engram per_token_table.weight has shape "
                f"{tuple(loaded_weight.shape)}, expected {expected} "
                "(compressed_vocab, n_hash_cols, head_dim)"
            )
        if loaded_weight.dtype != torch.bfloat16:
            raise ValueError(
                "engram per_token_table.weight must be bfloat16, "
                f"got {loaded_weight.dtype}"
            )
        used = max(min(self.n_hash_cols - self.head_start, self.part_n_hash_cols), 0)
        param.data[:, :used].copy_(
            loaded_weight[:, self.head_start : self.head_start + used]
        )
        param.data[:, used:].zero_()


def skip_engram_weight(name: str, per_token: bool) -> bool:
    if per_token:
        return ".engram.embed_tokens." in name
    return ".engram.per_token_table." in name


def per_token_comp_ids(
    token_map: torch.Tensor, input_ids: torch.Tensor, keep_mask: torch.Tensor
) -> torch.Tensor:
    """Map masked positions to a valid row before the gate drops them."""
    ids = torch.where(keep_mask, input_ids, 0).long()
    return token_map[ids.clamp(0, token_map.shape[0] - 1)].long()
