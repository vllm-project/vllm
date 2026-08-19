"""Normal-DeepEP route alignment for the vLLM low-latency MoE layout.

Route ordering follows THUDM/slime a74ae3a0; execution uses vLLM primitives.
"""

from __future__ import annotations

import torch

_BACKWARD_CHUNK_ROWS = 1024

def _ordered_route_backward(
    *,
    route_values: torch.Tensor,
    topk_weights: torch.Tensor,
    output_index: torch.Tensor,
    grad_output: torch.Tensor,
    grad_routes: torch.Tensor | None,
    grad_weights: torch.Tensor | None,
    static_mapping_valid: bool | None = None,
) -> None:
    """Differentiate the ordered top-k gather with an optional static fast path."""
    routes_alias_values = (
        grad_routes is not None
        and grad_routes.untyped_storage().data_ptr() == route_values.untyped_storage().data_ptr()
    )
    use_static_mapping = static_mapping_valid is not None
    if use_static_mapping and static_mapping_valid is None:
        # The hot DeepEP caller passes this bit from its forward scatter,
        # where torch.nonzero has already exposed the route count to Python.
        # Other internal callers retain a safe fallback instead of assuming
        # that padded token slots own an expert-output row.
        static_mapping_valid = bool(torch.all(output_index >= 0).item())
    if use_static_mapping and static_mapping_valid:
        # Dropless fixed-top-k routing produces exactly one valid route row for
        # every flattened [token, top-k] slot.  Express that static mapping
        # directly instead of materializing torch.nonzero's data-dependent
        # output and synchronizing once per MoE layer.
        num_routes = output_index.numel()
        topk = output_index.shape[1]
        flat_output_index = output_index.reshape(-1)
        flat_weights = topk_weights.reshape(-1)
        flat_grad_weights = grad_weights.reshape(-1) if grad_weights is not None else None
        chunk_rows = _BACKWARD_CHUNK_ROWS
        # When the forward combine input is dead after this operation, its
        # route-sized storage can hold the route gradient.  Complete every
        # probability gradient first because it still reads the original
        # route values; only then overwrite that storage with route gradients.
        if routes_alias_values and flat_grad_weights is not None:
            for start in range(0, num_routes, chunk_rows):
                end = min(start + chunk_rows, num_routes)
                flat_positions = torch.arange(
                    start,
                    end,
                    device=output_index.device,
                    dtype=torch.long,
                )
                token_rows = torch.div(flat_positions, topk, rounding_mode="floor")
                route_rows = flat_output_index[start:end].to(dtype=torch.long)
                token_grads = grad_output.index_select(0, token_rows)
                selected_route_values = route_values.index_select(0, route_rows)
                weight_grads = (token_grads.float() * selected_route_values.float()).sum(dim=-1)
                flat_grad_weights[start:end].copy_(weight_grads)

        fused_route_grad = False
        if grad_routes is not None and grad_output.is_cuda:
            from vllm.models.deepseek_v4.training_route_kernels import ordered_route_grad

            ordered_route_grad(
                grad_output.contiguous(),
                topk_weights.contiguous(),
                output_index.contiguous(),
                grad_routes,
            )
            fused_route_grad = True

        if (grad_routes is not None and not fused_route_grad) or (
            flat_grad_weights is not None and not routes_alias_values
        ):
            for start in range(0, num_routes, chunk_rows):
                end = min(start + chunk_rows, num_routes)
                flat_positions = torch.arange(
                    start,
                    end,
                    device=output_index.device,
                    dtype=torch.long,
                )
                token_rows = torch.div(flat_positions, topk, rounding_mode="floor")
                route_rows = flat_output_index[start:end].to(dtype=torch.long)
                token_grads = grad_output.index_select(0, token_rows)
                weights = flat_weights[start:end]

                if grad_routes is not None and not fused_route_grad:
                    route_grads = (token_grads.float() * weights.float().unsqueeze(1)).to(dtype=grad_routes.dtype)
                    # Every static top-k slot maps to one distinct route row.
                    grad_routes.index_copy_(0, route_rows, route_grads)
                if flat_grad_weights is not None and not routes_alias_values:
                    selected_route_values = route_values.index_select(0, route_rows)
                    weight_grads = (token_grads.float() * selected_route_values.float()).sum(dim=-1)
                    flat_grad_weights[start:end].copy_(weight_grads)
        return

    if use_static_mapping:
        # Padding-aware fixed-shape path.  Mapping every masked slot to row 0
        # keeps all intermediates bounded by chunk_rows and avoids allocating
        # torch.nonzero's data-dependent route table.  Masked slots write an
        # exact zero; row 0 is restored in a final ordered kernel, so duplicate
        # sentinel writes cannot affect the visible gradient.
        num_slots = output_index.numel()
        num_route_rows = route_values.shape[0]
        flat_output_index = output_index.reshape(-1)
        flat_weights = topk_weights.reshape(-1)
        flat_grad_weights = grad_weights.reshape(-1) if grad_weights is not None else None
        chunk_rows = _BACKWARD_CHUNK_ROWS

        if num_route_rows == 0:
            if grad_routes is not None:
                grad_routes.zero_()
            if flat_grad_weights is not None:
                flat_grad_weights.zero_()
            return

        def probability_grad_chunk(start: int, end: int) -> None:
            flat_positions = torch.arange(
                start,
                end,
                device=output_index.device,
                dtype=torch.long,
            )
            token_rows = torch.div(
                flat_positions,
                output_index.shape[1],
                rounding_mode="floor",
            )
            route_rows = flat_output_index[start:end].to(dtype=torch.long)
            valid_rows = route_rows >= 0
            safe_route_rows = route_rows.clamp(min=0, max=num_route_rows - 1)
            token_grads = grad_output.index_select(0, token_rows)
            selected_route_values = route_values.index_select(0, safe_route_rows)
            weight_grads = (token_grads.float() * selected_route_values.float()).sum(dim=-1)
            weight_grads.masked_fill_(~valid_rows, 0)
            flat_grad_weights[start:end].copy_(weight_grads)

        if routes_alias_values and flat_grad_weights is not None:
            for start in range(0, num_slots, chunk_rows):
                probability_grad_chunk(start, min(start + chunk_rows, num_slots))

        if grad_routes is not None and routes_alias_values:
            grad_routes.zero_()

        for start in range(0, num_slots, chunk_rows):
            end = min(start + chunk_rows, num_slots)
            flat_positions = torch.arange(
                start,
                end,
                device=output_index.device,
                dtype=torch.long,
            )
            token_rows = torch.div(
                flat_positions,
                output_index.shape[1],
                rounding_mode="floor",
            )
            route_rows = flat_output_index[start:end].to(dtype=torch.long)
            valid_rows = route_rows >= 0
            safe_route_rows = route_rows.clamp(min=0, max=num_route_rows - 1)
            token_grads = grad_output.index_select(0, token_rows)

            if grad_routes is not None:
                weights = flat_weights[start:end].float().masked_fill(~valid_rows, 0)
                route_grads = (token_grads.float() * weights.unsqueeze(1)).to(dtype=grad_routes.dtype)
                grad_routes.index_copy_(0, safe_route_rows, route_grads)
            if flat_grad_weights is not None and not routes_alias_values:
                selected_route_values = route_values.index_select(0, safe_route_rows)
                weight_grads = (token_grads.float() * selected_route_values.float()).sum(dim=-1)
                weight_grads.masked_fill_(~valid_rows, 0)
                flat_grad_weights[start:end].copy_(weight_grads)

        if grad_routes is not None:
            route_zero_matches = flat_output_index == 0
            torch._assert_async(
                torch.any(route_zero_matches),
                "non-empty expert output has no route mapped to row zero",
            )
            route_zero_position = torch.argmax(route_zero_matches.to(dtype=torch.int32)).reshape(1)
            route_zero_token = torch.div(
                route_zero_position,
                output_index.shape[1],
                rounding_mode="floor",
            )
            route_zero_weight = flat_weights.index_select(0, route_zero_position).float()
            route_zero_grad = (
                grad_output.index_select(0, route_zero_token).float() * route_zero_weight.unsqueeze(1)
            ).to(dtype=grad_routes.dtype)
            grad_routes.narrow(0, 0, 1).copy_(route_zero_grad)
        return

    valid_positions = torch.nonzero(output_index >= 0, as_tuple=False)
    # If the returned input gradient aliases route_values, finish every
    # probability gradient before clearing or overwriting that storage.
    # Padded DeepEP batches contain -1 output indices, and aligned expert rows
    # not referenced by a real token must receive a zero gradient.
    if routes_alias_values and grad_weights is not None:
        for start in range(0, valid_positions.shape[0], _BACKWARD_CHUNK_ROWS):
            end = min(start + _BACKWARD_CHUNK_ROWS, valid_positions.shape[0])
            positions = valid_positions[start:end]
            token_rows = positions[:, 0]
            topk_columns = positions[:, 1]
            route_rows = output_index[token_rows, topk_columns].to(dtype=torch.long)
            token_grads = grad_output.index_select(0, token_rows)
            selected_route_values = route_values.index_select(0, route_rows)
            weight_grads = (token_grads.float() * selected_route_values.float()).sum(dim=-1)
            grad_weights[token_rows, topk_columns] = weight_grads

    if grad_routes is not None and routes_alias_values:
        grad_routes.zero_()

    for start in range(0, valid_positions.shape[0], _BACKWARD_CHUNK_ROWS):
        end = min(start + _BACKWARD_CHUNK_ROWS, valid_positions.shape[0])
        positions = valid_positions[start:end]
class _DeepEPScatterWithDeterministicBackward(torch.autograd.Function):
    """Scatter routes with a deterministic backward."""

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        output_index: torch.Tensor,
        total_rows: int,
    ) -> torch.Tensor:
        output = hidden_states.new_zeros((int(total_rows), hidden_states.shape[1]))
        if hidden_states.is_cuda:
            from vllm.models.deepseek_v4.training_route_kernels import scatter_routes_forward

            scatter_routes_forward(
                hidden_states.contiguous(),
                output_index.contiguous(),
                output,
            )
        else:
            valid_positions = torch.nonzero(output_index >= 0, as_tuple=False)
            for start in range(0, valid_positions.shape[0], _BACKWARD_CHUNK_ROWS):
                positions = valid_positions[start : start + _BACKWARD_CHUNK_ROWS]
                token_rows = positions[:, 0]
                topk_columns = positions[:, 1]
                destination_rows = output_index[token_rows, topk_columns].to(dtype=torch.long)
                output.index_copy_(
                    0,
                    destination_rows,
                    hidden_states.index_select(0, token_rows),
                )

        ctx.input_shape = tuple(hidden_states.shape)
        ctx.input_dtype = hidden_states.dtype
        ctx.input_device = hidden_states.device
        ctx.save_for_backward(output_index)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if not ctx.needs_input_grad[0]:
            return None, None, None

        (output_index,) = ctx.saved_tensors
        grad_input = torch.zeros(
            ctx.input_shape,
            dtype=ctx.input_dtype,
            device=ctx.input_device,
        )
        if grad_output.is_cuda:
            from vllm.models.deepseek_v4.training_route_kernels import scatter_routes_backward

            scatter_routes_backward(
                grad_output.contiguous(),
                output_index.contiguous(),
                grad_input,
            )
        else:
            # Accumulate top-k slots in their original order.  Each token row
            # is owned by one thread here, avoiding the nondeterministic
            # atomics that an index_add over expert-major occurrences would
            # require.
            for start in range(0, output_index.shape[0], _BACKWARD_CHUNK_ROWS):
                end = min(start + _BACKWARD_CHUNK_ROWS, output_index.shape[0])
                grad_chunk = grad_input[start:end]
                for column in range(output_index.shape[1]):
                    route_rows = output_index[start:end, column].to(dtype=torch.long)
                    valid_rows = route_rows >= 0
                    safe_route_rows = route_rows.clamp(min=0, max=max(grad_output.shape[0] - 1, 0))
                    if grad_output.shape[0] == 0:
                        continue
                    selected = grad_output.index_select(0, safe_route_rows)
                    selected.masked_fill_(~valid_rows.unsqueeze(1), 0)
                    grad_chunk.add_(selected)
        return grad_input, None, None


def _scatter_deepep_routes_with_padding(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    *,
    return_route_positions: bool = False,
    expected_route_count: int | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    bool,
    torch.Tensor,
]:
    """Build vLLM LL's expert-major DeepEP layout deterministically."""
    if hidden_states.ndim != 2 or topk_indices.ndim != 2:
        raise ValueError("DeepEP scatter expects [tokens, hidden] and [tokens, topk]")
    if topk_indices.shape != topk_weights.shape:
        raise ValueError("DeepEP top-k indices and weights must have identical shapes")
    if hidden_states.shape[0] != topk_indices.shape[0]:
        raise ValueError("DeepEP hidden-state and top-k token dimensions differ")
    if topk_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"DeepEP top-k indices must be integer, got {topk_indices.dtype}")
    if topk_weights.dtype != torch.float32:
        raise TypeError(f"DeepEP top-k weights must be FP32, got {topk_weights.dtype}")

    if tokens_per_expert.device.type == "cpu":
        # Normal DeepEP returns this metadata as a CPU tensor constructed from
        # its receive-count list.  The output row count is consequently
        # already known to Python; do not upload the counts only to block on a
        # device sum immediately afterwards.
        count_values = tuple(int(value) for value in tokens_per_expert.reshape(-1).tolist())
        total_rows = sum(count_values)
    else:
        count_values = None

    num_experts = tokens_per_expert.numel()
    valid = (topk_indices >= 0) & (topk_indices < num_experts)
    sanitized_indices = topk_indices.masked_fill(~valid, -1)
    real_counts = torch.bincount(
        sanitized_indices.masked_select(valid).to(dtype=torch.long),
        minlength=num_experts,
    )

    # The route-preserving metadata handle gives Python the exact number of
    # received routes.  Normal DeepEP's CPU count list is unpadded in this
    # case, so ``real_counts`` is the same per-expert metadata already resident
    # on CUDA.  Reusing it avoids a blocking pageable-CPU-to-CUDA upload once
    # per MoE layer (and again during checkpoint recomputation).  Retain the
    # original upload for aligned/padded layouts and callers without the exact
    # route count.
    counts_are_exact_unpadded = (
        count_values is not None and expected_route_count is not None and total_rows == expected_route_count
    )
    if counts_are_exact_unpadded:
        counts = real_counts
        torch._assert_async(
            real_counts.sum() == expected_route_count,
            "DeepEP received route count differs from its route-preserving metadata",
        )
    else:
        counts = tokens_per_expert.to(
            device=topk_indices.device,
            dtype=torch.long,
        ).reshape(-1)
    torch._assert_async(
        torch.all(real_counts <= counts),
        "DeepEP real route count exceeds its aligned expert count",
    )

    if count_values is None:
        total_rows = int(counts.sum().item())
    permuted_probs = topk_weights.new_zeros((total_rows,))
    output_index = torch.full_like(topk_indices, -1)
    routing_map = torch.zeros(
        (topk_indices.shape[0], num_experts),
        device=topk_indices.device,
        dtype=torch.bool,
    )
    expert_offsets = torch.cumsum(counts, dim=0) - counts

    if expected_route_count is not None and valid.is_cuda:
        from vllm.models.deepseek_v4.training_route_kernels import compact_route_positions

        occurrences = compact_route_positions(valid.contiguous(), expected_route_count)
    else:
        occurrences = torch.nonzero(valid, as_tuple=False)
    # The metadata handle exposes the compact route count as tensor shape.  If
    # it is available, use that static value instead of forcing nonzero to
    # report its data-dependent output size to Python.
    route_count = occurrences.shape[0] if expected_route_count is None else expected_route_count
    all_routes_valid = route_count == topk_indices.numel()
    if occurrences.numel():
        token_rows = occurrences[:, 0]
        topk_columns = occurrences[:, 1]
        route_experts = sanitized_indices[token_rows, topk_columns].to(dtype=torch.long)
        expert_order = torch.argsort(route_experts, stable=True)
        token_rows = token_rows.index_select(0, expert_order)
        topk_columns = topk_columns.index_select(0, expert_order)
        route_experts = route_experts.index_select(0, expert_order)

        real_offsets = torch.cumsum(real_counts, dim=0) - real_counts
        within_expert = torch.arange(
            route_experts.numel(),
            device=hidden_states.device,
            dtype=torch.long,
        ) - real_offsets.index_select(0, route_experts)
        destination_rows = expert_offsets.index_select(0, route_experts) + within_expert
        permuted_probs.index_copy_(
            0,
            destination_rows,
            topk_weights[token_rows, topk_columns],
        )
        output_index[token_rows, topk_columns] = destination_rows.to(dtype=output_index.dtype)
        routing_map[token_rows, route_experts] = True

    torch._assert_async(
        torch.all(~valid | (output_index >= 0)),
        "A valid DeepEP route has no expert-major output row",
    )
    permuted_hidden = _DeepEPScatterWithDeterministicBackward.apply(
        hidden_states,
        output_index,
        total_rows,
    )
    result = (
        permuted_hidden,
        permuted_probs,
        output_index,
        sanitized_indices,
        routing_map,
        all_routes_valid,
        real_counts,
    )
    if return_route_positions:
        return (*result, occurrences)
    return result


class _VLLMEPGatherWithBF16Backward(torch.autograd.Function):
    """vLLM rollout's ordered FP32 gather with a deterministic BF16 backward."""

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        output_index: torch.Tensor,
        reuse_input_for_grad: bool,
        static_mapping_valid: bool | None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 2 or hidden_states.dtype != torch.bfloat16:
            raise TypeError(
                "vLLM ep_gather requires BF16 [expert_rows, hidden], got "
                f"{hidden_states.dtype} {tuple(hidden_states.shape)}"
            )
        if topk_indices.shape != topk_weights.shape or topk_indices.shape != output_index.shape:
            raise ValueError("DeepEP gather IDs, weights, and output indices must align")
        from vllm.model_executor.layers.fused_moe.deep_gemm_utils import ep_gather

        output_shape = (topk_indices.shape[0], hidden_states.shape[1])
        output = torch.empty(
            output_shape,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        ep_gather(
            hidden_states,
            topk_indices,
            topk_weights,
            output_index,
            None,
            output,
        )
        ctx.reuse_input_for_grad = bool(reuse_input_for_grad)
        ctx.static_mapping_valid = static_mapping_valid
        ctx.save_for_backward(hidden_states, topk_weights, output_index)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden_states, topk_weights, output_index = ctx.saved_tensors
        needs_hidden = ctx.needs_input_grad[0]
        needs_weights = ctx.needs_input_grad[2]
        if needs_hidden and ctx.reuse_input_for_grad:
            # The caller guarantees this combine input has no forward consumer
            # after ep_gather.  Returning its detached storage as the input
            # gradient avoids a second route-sized BF16 allocation.
            grad_hidden = hidden_states.detach()
        else:
            grad_hidden = torch.zeros_like(hidden_states) if needs_hidden else None
        grad_weights = torch.zeros_like(topk_weights) if needs_weights else None

        _ordered_route_backward(
            route_values=hidden_states,
            topk_weights=topk_weights,
            output_index=output_index,
            grad_output=grad_output,
            grad_routes=grad_hidden,
            grad_weights=grad_weights,
            static_mapping_valid=ctx.static_mapping_valid,
        )

        return grad_hidden, None, grad_weights, None, None, None


def _compact_route_preserving_metadata_inputs(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    assume_all_routes_valid: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Build the small route-level dispatch used by the normal-mode prototype.

    The real hidden state remains rank-deduplicated in the primary DeepEP
    dispatch.  This second dispatch only carries a sixteen-BF16 fingerprint and
    creates a normal DeepEP handle whose logical tokens are ``(token, slot)``.
    The handle can consequently transport the real route outputs during
    combine without multiplying dispatch hidden traffic by top-k.

    Sixteen BF16 values are the smallest payload accepted by DeepEP's normal
    intranode dispatch: its TMA path requires the hidden payload to be a
    multiple of 32 bytes.  Keeping that constraint explicit here prevents a
    device-side assertion for otherwise-valid route metadata.
    """
    if hidden_states.ndim != 2 or hidden_states.dtype != torch.bfloat16:
        raise TypeError(
            "Route-preserving DeepEP metadata requires BF16 [tokens, hidden], "
            f"got {hidden_states.dtype} {tuple(hidden_states.shape)}"
        )
    if hidden_states.shape[1] < 16:
        raise ValueError("Route-preserving DeepEP fingerprints require hidden >= 16")
    if topk_indices.ndim != 2 or topk_weights.shape != topk_indices.shape:
        raise ValueError("Route-preserving DeepEP top-k IDs and weights must align")
    if topk_indices.shape[0] != hidden_states.shape[0]:
        raise ValueError("Route-preserving DeepEP token counts do not align")
    if topk_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"Route-preserving DeepEP IDs must be integer, got {topk_indices.dtype}")
    if topk_weights.dtype != torch.float32:
        raise TypeError(f"Route-preserving DeepEP weights must be FP32, got {topk_weights.dtype}")

    flat_indices = topk_indices.reshape(-1)
    if assume_all_routes_valid:
        # With no capacity factor and no router token mask, the router emits a
        # fixed valid top-k for every source token.  Keep this fast path tied
        # to that structural guarantee instead of synchronizing on nonzero's
        # data-dependent output once per MoE layer.
        torch._assert_async(
            torch.all(flat_indices >= 0),
            "Route-preserving DeepEP expected a valid fixed top-k",
        )
        compact_indices = flat_indices.reshape(-1, 1).contiguous()
        compact_weights = topk_weights.detach().reshape(-1, 1).contiguous()
        fingerprints = (
            hidden_states.detach()
            .narrow(1, 0, 16)
            .unsqueeze(1)
            .expand(-1, topk_indices.shape[1], -1)
            .reshape(-1, 16)
            .contiguous()
        )
        output_index = torch.arange(
            flat_indices.numel(),
            device=flat_indices.device,
            dtype=torch.long,
        ).reshape_as(topk_indices)
        return compact_indices, compact_weights, fingerprints, output_index, True

    valid_positions = torch.nonzero(flat_indices >= 0, as_tuple=False).reshape(-1)
    if valid_positions.numel() == 0:
        raise RuntimeError("Route-preserving DeepEP received no valid expert routes")
    compact_indices = flat_indices.index_select(0, valid_positions).reshape(-1, 1).contiguous()
    compact_weights = topk_weights.detach().reshape(-1).index_select(0, valid_positions).reshape(-1, 1).contiguous()
    token_rows = torch.div(valid_positions, topk_indices.shape[1], rounding_mode="floor")
    fingerprints = hidden_states.detach().narrow(1, 0, 16).index_select(0, token_rows).contiguous()
    output_index = torch.full_like(topk_indices, -1, dtype=torch.long)
    output_index.reshape(-1).index_copy_(
        0,
        valid_positions,
        torch.arange(valid_positions.numel(), device=valid_positions.device, dtype=torch.long),
    )
    all_routes_valid = valid_positions.numel() == topk_indices.numel()
    return compact_indices, compact_weights, fingerprints, output_index, all_routes_valid


def _deepep_route_handle_received_rows(handle: tuple) -> int:
    """Return the received route count encoded by a normal DeepEP handle."""
    if not isinstance(handle, tuple):
        raise TypeError(f"DeepEP route handle must be a tuple, got {type(handle).__name__}")
    if len(handle) == 6:
        # Intranode: (..., recv_src_idx, ...).
        received_metadata = handle[3]
    elif len(handle) == 10:
        # Internode: (..., recv_src_meta, ...).
        received_metadata = handle[7]
    else:
        raise ValueError(f"Unsupported normal DeepEP route handle length: {len(handle)}")
    if not isinstance(received_metadata, torch.Tensor) or received_metadata.ndim < 1:
        raise TypeError("DeepEP route handle has invalid received-source metadata")
    return received_metadata.shape[0]


def _validate_and_order_route_preserving_outputs(
    expert_outputs: torch.Tensor,
    received_tokens: torch.Tensor,
    received_topk_indices: torch.Tensor,
    received_topk_weights: torch.Tensor,
    output_index: torch.Tensor,
    route_fingerprints: torch.Tensor,
    route_indices: torch.Tensor,
    route_weights: torch.Tensor,
    *,
    order_outputs: bool = True,
    route_positions: torch.Tensor | None = None,
    return_route_rows: bool = False,
) -> torch.Tensor:
    """Return expert outputs in the route handle's receive order.

    Normal DeepEP's primary dispatch is rank-deduplicated: its received route
    probabilities are not a reliable per-(token, slot) identity when one token
    reaches multiple experts on the same rank.  Pair the primary hidden rows
    with the route-preserving metadata by expert ID and a sixteen-BF16 source
    fingerprint.  The metadata dispatch and the caller-owned source weights
    remain authoritative for exact route probabilities.
    """
    if expert_outputs.ndim != 2 or received_tokens.ndim != 2:
        raise ValueError("Route-preserving DeepEP expects 2D hidden tensors")
    if received_topk_indices.shape != received_topk_weights.shape:
        raise ValueError("Received DeepEP IDs and weights do not align")
    if output_index.shape != received_topk_indices.shape:
        raise ValueError("Received DeepEP route mapping does not align")

    positions = torch.nonzero(output_index >= 0, as_tuple=False) if route_positions is None else route_positions
    if positions.shape[0] != route_indices.numel():
        raise RuntimeError(
            "Route-preserving DeepEP route count mismatch: "
            f"primary={positions.shape[0]} metadata={route_indices.numel()}"
        )
    token_rows = positions[:, 0]
    topk_slots = positions[:, 1]
    expected_indices = received_topk_indices[token_rows, topk_slots].reshape(-1)
    expected_fingerprints = received_tokens.narrow(1, 0, 16).index_select(0, token_rows)
    if route_fingerprints.shape != expected_fingerprints.shape:
        raise RuntimeError(
            "Route-preserving DeepEP fingerprint shape mismatch: "
            f"{tuple(route_fingerprints.shape)} != {tuple(expected_fingerprints.shape)}"
        )

    def stable_key_order(
        fingerprints: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Lexicographically order exact route identity without hashing."""
        order = torch.arange(
            indices.numel(), device=indices.device, dtype=torch.long
        )
        fingerprint_bits = fingerprints.contiguous().view(torch.int16)
        columns = [
            *(fingerprint_bits[:, column] for column in range(fingerprint_bits.shape[1])),
            indices.to(dtype=torch.long),
        ]
        # Stable least-significant-to-most-significant sorts implement a
        # deterministic lexicographic order and retain duplicate multiplicity.
        for column in reversed(columns):
            permutation = torch.argsort(
                column.index_select(0, order), stable=True
            )
            order = order.index_select(0, permutation)
        return order

    primary_order = stable_key_order(expected_fingerprints, expected_indices)
    metadata_order = stable_key_order(
        route_fingerprints,
        route_indices.to(dtype=expected_indices.dtype),
    )
    torch._assert_async(
        torch.all(
            expected_indices.index_select(0, primary_order)
            == route_indices.to(dtype=expected_indices.dtype).index_select(
                0, metadata_order
            )
        ),
        "Route-preserving DeepEP metadata changed expert identities",
    )
    torch._assert_async(
        torch.all(torch.isfinite(route_weights)),
        "Route-preserving DeepEP metadata contains nonfinite route probabilities",
    )
    torch._assert_async(
        torch.all(
            expected_fingerprints.contiguous()
            .view(torch.int16)
            .index_select(0, primary_order)
            == route_fingerprints.contiguous()
            .view(torch.int16)
            .index_select(0, metadata_order)
        ),
        "Route-preserving DeepEP metadata changed source fingerprints",
    )

    # metadata_to_primary[m] is the corresponding route occurrence in the
    # primary hidden dispatch.  Stable ordering gives duplicate identities a
    # deterministic one-to-one pairing; identical duplicates are semantically
    # interchangeable.
    metadata_to_primary = torch.empty_like(metadata_order)
    metadata_to_primary.scatter_(0, metadata_order, primary_order)
    primary_route_rows = output_index[token_rows, topk_slots].to(dtype=torch.long)
    route_rows = primary_route_rows.index_select(0, metadata_to_primary)
    if return_route_rows:
        return route_rows
    if not order_outputs:
        return expert_outputs
    return expert_outputs.index_select(0, route_rows)
