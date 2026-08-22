# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
LoRA kernels metadata preparation utilities.
"""

import bisect
from dataclasses import dataclass, field

import torch


@dataclass
class LoRAKernelMeta:
    token_lora_mapping: torch.Tensor
    token_indices_sorted_by_lora_ids: torch.Tensor
    active_lora_ids: torch.Tensor
    num_tokens_per_lora: torch.Tensor
    lora_token_start_loc: torch.Tensor

    # The V1 architecture uses the traced torch.compile graphs to execute
    # a forward pass. Things to note about this process,
    # 1. The tracing infers all python scalar datatype objects into a constant
    # value.
    # 2. The tracing cannot handle dynamic control flow. (dynamic control flow
    # is an experimental feature in pytorch)
    # 3. The internals of torch.ops functions are not traced.
    # We disguise the "no_lora" flag as a cpu tensor and leverage point number 3
    # to early exit from inside the lora_expand / lora_shrink torch operation.
    no_lora_flag_cpu: torch.Tensor

    # Number of active LoRAs (unique non-(-1) values in token_lora_mapping).
    # Stored as a CPU tensor (not a Python int) so that torch.compile treats
    # it as a dynamic value rather than baking it as a constant at trace time.
    # This follows the same pattern as no_lora_flag_cpu above.
    num_active_loras_cpu: torch.Tensor

    # Default num_active_loras value (max_loras + 1) as a CPU tensor,
    # used when specialize_active_lora is False to avoid allocating a
    # new tensor on every meta_args() call.
    default_num_active_loras_cpu: torch.Tensor

    # Captured LoRA counts for cudagraph specialization (sorted list).
    # When specialize_active_lora is enabled, num_active_loras is rounded up
    # to the nearest value in this list to match cudagraph capture keys.
    # Empty list means no specialization (use actual count).
    captured_lora_counts: list[int] = field(default_factory=list)

    # ---- Per-request LoRA scaling (optional) ----------------------------
    # Only allocated when the engine is started with
    # `--enable-per-request-lora-scale`. Two requests in the same batch may
    # share a LoRA adapter (same lora id, same A/B weights) while asking for
    # different strengths. Following the persistent-batch pattern used by
    # SamplingStates, the scale is kept as [max_num_reqs] per-request state
    # and combined with a [max_num_tokens] token -> request index map, rather
    # than materializing a per-token copy of the scale.
    #
    #   effective_scale(token) = base_scale * request_scales[token_to_req[token]]
    #
    # token_lora_mapping decides *which* adapter a token uses;
    # request_scales decides *how strongly* it is applied. The two are
    # orthogonal, so a single adapter slot serves any number of scales.
    request_scales: torch.Tensor | None = None
    token_to_req: torch.Tensor | None = None

    @staticmethod
    def make(
        max_loras: int,
        max_num_tokens: int,
        device: torch.device | str,
        captured_lora_counts: list[int] | None = None,
        max_num_reqs: int | None = None,
    ) -> "LoRAKernelMeta":
        token_lora_mapping = torch.empty(
            max_num_tokens, dtype=torch.int32, device=device
        )

        token_indices_sorted_by_lora_ids = torch.empty(
            max_num_tokens, dtype=torch.int32, device=device
        )

        # +1 because "no-lora" is also a possibility
        # example: let max_loras be 3, active_lora_ids of [-1, 0, 2, 1]
        # is a possibility.
        active_lora_ids = torch.empty(max_loras + 1, dtype=torch.int32, device=device)

        # using running example, [3, 10, 5, 2] is a possibility.
        num_tokens_per_lora = torch.zeros(
            max_loras + 1, dtype=torch.int32, device=device
        )

        # +2 for this because, the first index is always 0.
        # using running example, lora_token_start_loc
        # is [0, 3, 13, 18, 20].
        lora_token_start_loc = torch.zeros(
            max_loras + 2, dtype=torch.int32, device=device
        )

        no_lora_flag_cpu = torch.tensor([False], dtype=torch.bool, device="cpu")

        num_active_loras_cpu = torch.tensor([0], dtype=torch.int32, device="cpu")
        default_num_active_loras_cpu = torch.tensor(
            [max_loras + 1], dtype=torch.int32, device="cpu"
        )

        request_scales = None
        token_to_req = None
        if max_num_reqs is not None:
            # Persistent per-request scale buffer. Defaults to 1.0 so any slot
            # that is never written behaves exactly like stock vLLM.
            request_scales = torch.ones(
                max_num_reqs, dtype=torch.float32, device=device
            )
            token_to_req = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)

        return LoRAKernelMeta(
            token_lora_mapping=token_lora_mapping,
            token_indices_sorted_by_lora_ids=token_indices_sorted_by_lora_ids,
            active_lora_ids=active_lora_ids,
            num_tokens_per_lora=num_tokens_per_lora,
            lora_token_start_loc=lora_token_start_loc,
            no_lora_flag_cpu=no_lora_flag_cpu,
            num_active_loras_cpu=num_active_loras_cpu,
            default_num_active_loras_cpu=default_num_active_loras_cpu,
            captured_lora_counts=sorted(captured_lora_counts)
            if captured_lora_counts
            else [],
            request_scales=request_scales,
            token_to_req=token_to_req,
        )

    @property
    def has_request_scales(self) -> bool:
        return self.request_scales is not None

    def prepare_request_scales(
        self,
        request_scales: torch.Tensor,
        token_to_req: torch.Tensor,
    ) -> None:
        """
        Stage the per-request LoRA scales for the current forward pass.

        Args:
            request_scales: float32 CPU tensor of shape [num_reqs] holding
                each request's scale multiplier.
            token_to_req: int32 CPU tensor of shape [num_tokens] mapping every
                token to the index of the request it belongs to.
        """
        assert self.request_scales is not None
        assert self.token_to_req is not None

        num_reqs = request_scales.size(0)
        num_tokens = token_to_req.size(0)
        assert num_reqs <= self.request_scales.size(0)
        assert num_tokens <= self.token_to_req.size(0)

        self.request_scales[:num_reqs].copy_(request_scales, non_blocking=True)
        # Slots beyond num_reqs keep whatever they had; token_to_req never
        # points at them, so they are unreachable this step.
        self.token_to_req[:num_tokens].copy_(token_to_req, non_blocking=True)

    def _reset(self):
        self.active_lora_ids.fill_(-1)
        self.num_tokens_per_lora.fill_(0)
        self.lora_token_start_loc.fill_(0)
        self.no_lora_flag_cpu.fill_(False)
        self.num_active_loras_cpu.fill_(0)

    def prepare_tensors(self, token_lora_mapping: torch.Tensor) -> None:
        """
        Prepare kernel metadata tensors for the current forward pass.

        Args:
            token_lora_mapping (torch.Tensor): Tensor containing lora indices
                for each input token.
        """

        self._reset()

        # Check and record no-lora case.
        no_lora = torch.all(token_lora_mapping == -1)
        self.no_lora_flag_cpu[0] = no_lora

        if no_lora:
            # Early exit. LoRA kernels will not be run.
            return

        num_tokens = token_lora_mapping.size(0)

        # copy token lora mapping
        self.token_lora_mapping[:num_tokens].copy_(
            token_lora_mapping, non_blocking=True
        )

        # token_indices_sorted_by_lora_ids
        _, token_indices_sorted_by_lora_ids = torch.sort(
            token_lora_mapping, stable=True
        )
        # start gpu transfer
        self.token_indices_sorted_by_lora_ids[:num_tokens].copy_(
            token_indices_sorted_by_lora_ids, non_blocking=True
        )

        # active_lora_ids, num_tokens_per_lora
        lora_ids, num_tokens_per_lora = torch.unique(
            token_lora_mapping, sorted=True, return_counts=True
        )
        self.active_lora_ids[: lora_ids.size(0)].copy_(lora_ids, non_blocking=True)
        self.num_tokens_per_lora[: num_tokens_per_lora.size(0)].copy_(
            num_tokens_per_lora, non_blocking=True
        )

        num_active_loras = lora_ids.size(0)

        # Round up num_active_loras to match cudagraph capture keys.
        # This ensures the kernel grid dimension matches the captured graph.
        if self.captured_lora_counts and num_active_loras > 0:
            idx = bisect.bisect_left(self.captured_lora_counts, num_active_loras)
            if idx < len(self.captured_lora_counts):
                num_active_loras = self.captured_lora_counts[idx]

        self.num_active_loras_cpu[0] = num_active_loras

        # lora_token_start_loc
        lora_token_start_loc = torch.cumsum(num_tokens_per_lora, dim=0)
        self.lora_token_start_loc[1 : 1 + lora_token_start_loc.size(0)].copy_(
            lora_token_start_loc, non_blocking=True
        )

    def meta_args(
        self,
        token_nums: int,
        specialize_active_lora: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        This function returns the kernel metadata required for the current
        forward pass execution of the kernel. The function returns all the
        metadata required by the kernel, in order, as a tuple, so it can be
        unpacked directly during the lora_shrink/lora_expand function call.

        Args:
            token_nums (int): Number of input tokens in the current forward
                pass of the kernel.
        """
        if specialize_active_lora:
            num_active_loras = self.num_active_loras_cpu
        else:
            num_active_loras = self.default_num_active_loras_cpu
        return (
            self.token_lora_mapping[:token_nums],
            self.token_indices_sorted_by_lora_ids[:token_nums],
            self.num_tokens_per_lora,
            self.lora_token_start_loc,
            self.active_lora_ids,
            self.no_lora_flag_cpu,
            num_active_loras,
        )
