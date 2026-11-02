# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.config import SpeculativeConfig
from vllm.logger import init_logger
from vllm.v1.worker.gpu.spec_decode.dflash.config import DFlashSpeculatorConfig

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.spec_decode.draft_token_head import DraftTokenHead

logger = init_logger(__name__)


class XPressSpeculatorConfig(DFlashSpeculatorConfig):
    """The DFlash speculator with the XPress draft token head."""

    supported_options = frozenset({"xpress_topc"})
    # The DSpark loader body is architecture-agnostic (embed/lm_head sharing,
    # non-causal attention config, KV setup), so it is reused as-is.
    draft_model_loader = "vllm.v1.worker.gpu.spec_decode.dspark.utils.load_dspark_model"

    @classmethod
    def apply_options(cls, spec: SpeculativeConfig) -> None:
        hf_config = spec.draft_model_config.hf_config
        # XPress is trained in the fill-in layout -- slot 0 carries the
        # verified anchor and only slots 1..N are predicted -- and the
        # refiner's kernels bake that layout into their write masks and
        # readout row count. DFlash rejects the nested dflash_config form; this
        # covers the top-level key XPress checkpoints carry.
        if getattr(hf_config, "sample_from_anchor", False):
            raise ValueError(
                "sample_from_anchor=True is not supported for XPress: the refiner "
                "is trained and served in the fill-in layout (anchor at slot 0, "
                "slots 1..N predicted)."
            )
        xpress_topc = spec.xpress_topc
        if xpress_topc is None:
            xpress_topc = getattr(hf_config, "xpress_topc", None)
        if xpress_topc is None:
            return
        draft_vocab_size = (
            getattr(hf_config, "draft_vocab_size", None) or hf_config.vocab_size
        )
        if not 0 <= xpress_topc <= draft_vocab_size:
            raise ValueError(
                "xpress_topc must be between 0 and the draft "
                f"vocabulary size ({draft_vocab_size})"
            )
        hf_config.xpress_topc = xpress_topc

    def create_draft_token_head(self) -> "DraftTokenHead":
        # Imported lazily: speculator configs are used in every process.
        from vllm.v1.worker.gpu.spec_decode.xpress.draft_token_head import (
            XPressDraftTokenHead,
        )

        hf_config = self.spec_config.draft_model_config.hf_config
        num_jacobi_passes = int(getattr(hf_config, "xpress_num_passes", 6))
        logger.info(
            "XPress: %d Jacobi passes, top-C=%s",
            num_jacobi_passes,
            getattr(hf_config, "xpress_topc", 512) or "off",
        )
        return XPressDraftTokenHead(num_jacobi_passes)
