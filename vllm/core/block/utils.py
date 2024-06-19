"""Block manager utils."""
from vllm.sequence import SequenceGroup
from vllm.utils import (STR_NOT_IMPL_ENC_DEC_PREFIX_CACHE,
                        STR_NOT_IMPL_ENC_DEC_SWA)


def _get_block_mgr_sliding_window_attr(block_mgr):
    '''
    BlockManagerV1 and BlockManagerV2 have slightly different
    members related to sliding window attention (SWA). This
    function extracts the appropriate member to use for determining
    whether SWA is enabled.

    Arguments:

    * block_mgr: BlockManagerV1 or BlockManagerV2 instance
    '''

    if hasattr(block_mgr, 'block_sliding_window'):
        return block_mgr.block_sliding_window
    if hasattr(block_mgr, 'max_block_sliding_window'):
        return block_mgr.max_block_sliding_window

    raise AttributeError("Block manager instance has neither " + \
                         "block_sliding_window nor " + \
                         "max_block_sliding_window attributes.")


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
        if _get_block_mgr_sliding_window_attr(block_mgr) is not None:
            raise NotImplementedError(STR_NOT_IMPL_ENC_DEC_SWA)

        if block_mgr.enable_caching:
            raise NotImplementedError(STR_NOT_IMPL_ENC_DEC_PREFIX_CACHE)
