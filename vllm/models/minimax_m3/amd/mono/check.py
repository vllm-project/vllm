# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Debug tooling for the mono path.

``forward_checked`` (``VLLM_ROCM_MINIMAX_M3_MONO_CHECK=1``, meant for
``--enforce-eager``) runs every sparse layer twice on the same (ar, res): first
the mono layer kernel, then the original decoder layer. Both insert the same cache
slots (the kernel writes the same bytes up to the re-quantized attention weights),
so running both is side-effect free. Rank 0 logs, per layer and token, the
difference of the residual stream and of the reduced MoE output.
"""

import torch

from vllm.logger import init_logger
from vllm.models.minimax_m3.amd.mono.config import HIDDEN

logger = init_logger(__name__)


def _diff(a: torch.Tensor, b: torch.Tensor) -> str:
    a, b = a.float().flatten(), b.float().flatten()
    d = (a - b).abs()
    rel = (d.norm() / b.norm().clamp(min=1e-30)).item()
    return f"max {d.max().item():.3e} rel {rel:.3e}"


def forward_checked(
    runner,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    inputs_embeds: torch.Tensor | None,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """``runner.forward`` with the original decoder layer run beside every mono
    layer; the mono chain's output is returned."""
    aux_hidden_states: list[torch.Tensor] | None = (
        [] if runner.model.aux_hidden_state_layers else None
    )
    res = runner.start(input_ids, positions, inputs_embeds, aux_hidden_states)
    n = res.shape[0]
    tables = runner.step_tables(n)
    ars = runner.ars
    for i, lw in enumerate(runner.sparse):
        layer = runner.model.layers[lw.layer_id]
        h_in = ars[i % 2][:n].clone()
        res_in = res.clone()
        res = runner.run_sparse_layer(i, lw, tables, positions, res)
        ref_h, ref_res = layer(positions, h_in, res_in)
        if aux_hidden_states is not None:
            runner.model._maybe_add_hidden_state(
                aux_hidden_states,
                lw.layer_id + 1,
                ars[(i + 1) % 2][:n],
                res,
            )
        if runner.rank == 0:
            for k in range(n):
                logger.info(
                    "mono check layer %d token %d/%d pos %d seq_len %d: residual %s | "
                    "reduced out %s",
                    lw.layer_id, k, n, int(positions[k]), int(tables.seq_lens[k]),
                    _diff(res[k], ref_res.view(n, HIDDEN)[k]),
                    _diff(ars[(i + 1) % 2][k], ref_h.view(n, HIDDEN)[k]),
                )  # fmt: skip
    hidden = runner.run_final_norm(res)
    if aux_hidden_states is not None:
        return hidden, aux_hidden_states
    return hidden
