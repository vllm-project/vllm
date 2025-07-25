from vllm.core.block.block_table import BlockTable
from vllm.core.block.utils import check_no_caching_or_swa_for_blockmgr_encdec
from vllm.core.block_manager import SelfAttnBlockSpaceManager
from vllm.core.interfaces import AllocStatus
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device


class TiltBlockSpaceManager(SelfAttnBlockSpaceManager):

    def can_allocate(self,
                     seq_group: SequenceGroup,
                     num_lookahead_slots: int = 0) -> AllocStatus:
        # Same as SelfAttnBlockSpaceManager.can_allocate, but also takes into
        # account encoder prefix sequence.
        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        num_required_blocks = BlockTable.get_num_required_blocks(
            seq.get_token_ids(),
            block_size=self.block_size,
            num_lookahead_slots=num_lookahead_slots,
        )

        if seq_group.is_encoder_decoder():
            encoder_seq = seq_group.get_encoder_seq()
            assert encoder_seq is not None
            encoder_token_ids = encoder_seq.get_token_ids()
            if (encoder_prefix_seq :=
                    seq_group.encoder_prefix_seq) is not None:
                encoder_token_ids = (encoder_prefix_seq.get_token_ids() +
                                     encoder_token_ids)

            num_required_blocks += BlockTable.get_num_required_blocks(
                encoder_token_ids,
                block_size=self.block_size,
            )

        if self.max_block_sliding_window is not None:
            raise RuntimeError(
                "Sliding window attention is not supported for TILT")

        num_free_gpu_blocks = self.block_allocator.get_num_free_blocks(
            device=Device.GPU)

        # Use watermark to avoid frequent cache eviction.
        if self.num_total_gpu_blocks - num_required_blocks < self.watermark_blocks:
            return AllocStatus.NEVER
        if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def _allocate_tilt_cross_sequence(self, encoder_prefix_seq: Sequence,
                                      encoder_seq: Sequence) -> BlockTable:
        block_table = BlockTable(
            block_size=self.block_size,
            block_allocator=self.block_allocator,
            max_block_sliding_window=self.max_block_sliding_window,
        )
        assert encoder_prefix_seq.get_token_ids()
        assert encoder_seq.get_token_ids()
        assert encoder_prefix_seq.extra_hash() == encoder_seq.extra_hash()
        block_table.allocate(
            token_ids=encoder_prefix_seq.get_token_ids() +
            encoder_seq.get_token_ids(),
            extra_hash=encoder_seq.extra_hash(),
        )

        return block_table

    def allocate(self, seq_group: SequenceGroup) -> None:
        # Same as SelfAttnBlockSpaceManager.allocate, but with support for
        # allocating TILT cross-attention KV cache.

        # Allocate self-attention block tables for decoder sequences
        waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
        assert not (set(seq.seq_id for seq in waiting_seqs)
                    & self.block_tables.keys()), "block table already exists"

        # NOTE: Here we assume that all sequences in the group have the same
        # prompt.
        seq = waiting_seqs[0]
        block_table: BlockTable = self._allocate_sequence(seq)
        self.block_tables[seq.seq_id] = block_table

        # Track seq
        self._last_access_blocks_tracker.add_seq(seq.seq_id)

        # Assign the block table for each sequence.
        for seq in waiting_seqs[1:]:
            self.block_tables[seq.seq_id] = block_table.fork()

            # Track seq
            self._last_access_blocks_tracker.add_seq(seq.seq_id)

        # Allocate cross-attention block table for encoder sequence
        #
        # NOTE: Here we assume that all sequences in the group have the same
        # encoder prompt.
        request_id = seq_group.request_id

        assert request_id not in self.cross_block_tables, "block table already exists"

        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        if seq_group.is_encoder_decoder():
            encoder_seq = seq_group.get_encoder_seq()
            assert encoder_seq is not None
            if seq_group.encoder_prefix_seq is None:
                block_table = self._allocate_sequence(encoder_seq)
            else:
                block_table = self._allocate_tilt_cross_sequence(
                    seq_group.encoder_prefix_seq, encoder_seq)
            self.cross_block_tables[request_id] = block_table
