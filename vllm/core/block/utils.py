# SPDX-License-Identifier: Apache-2.0
"""Block manager utils."""
from vllm.sequence import SequenceGroup
from vllm.utils import (STR_NOT_IMPL_ENC_DEC_PREFIX_CACHE,
                        STR_NOT_IMPL_ENC_DEC_SWA)


def check_no_caching_or_swa_for_blockmgr_encdec(
        block_mgr, seq_group: SequenceGroup) -> None:
    '''
    Enforce that prefix caching & sliding-window attention (SWA)
    are currently unsupported *specifically* for encoder/decoder models.

    Raises NotImplementedError if unsupported scenario is detected.

    Arguments:

    * block_mgr: BlockSpaceManager instance
    * seq_group: SequenceGroup passed to block_mgr
    '''

    if seq_group.is_encoder_decoder():
        if block_mgr.max_block_sliding_window is not None:
            raise NotImplementedError(STR_NOT_IMPL_ENC_DEC_SWA)

        if block_mgr.enable_caching:
            raise NotImplementedError(STR_NOT_IMPL_ENC_DEC_PREFIX_CACHE)
