"""Block manager utils."""
from vllm.sequence import SequenceGroup
'''
Exception strings for non-implemented block manager encoder/decoder scenarios
'''

STR_NOT_IMPL_ENC_DEC_SWA = \
    "Sliding window attention for encoder/decoder models " + \
                    "is not currently supported."

STR_NOT_IMPL_ENC_DEC_PREFIX_CACHE = \
    "Prefix caching for encoder/decoder models " + \
                    "is not currently supported."


def check_no_caching_or_swa_for_blckmgr_encdec(
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
        if block_mgr.block_sliding_window is not None:
            raise NotImplementedError(STR_NOT_IMPL_ENC_DEC_SWA)

        if block_mgr.enable_caching:
            raise NotImplementedError(STR_NOT_IMPL_ENC_DEC_PREFIX_CACHE)
