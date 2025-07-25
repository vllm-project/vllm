from vllm.config import CacheConfig
from vllm.inputs import TokenInputs
from vllm.sequence import Logprob, Sequence, SequenceGroup
from vllm.tilt.scheduler import Scheduler, TiltSchedulerConfig


def test_scheduler():
    scheduler_config = TiltSchedulerConfig(
        runner_type="tilt",
        max_num_batched_tokens=100,
        max_num_seqs=64,
        max_model_len=16,
        encoder_chunk_size=32,
        max_num_batched_encoder_tokens=32,
    )
    block_size = 8
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 4
    cache_config.num_gpu_blocks = 32

    scheduler = Scheduler(scheduler_config, cache_config, None)

    decoder_seq = Sequence(
        seq_id=0,
        inputs=TokenInputs(type="token", prompt_token_ids=[1]),
        block_size=block_size,
        eos_token_id=0,
    )
    encoder_seq = Sequence(
        seq_id=0,
        inputs=TokenInputs(type="token", prompt_token_ids=[2] * 64),
        block_size=block_size,
    )
    encoder_prefix_seq = Sequence(
        seq_id=0,
        inputs=TokenInputs(type="token", prompt_token_ids=[3] * 8),
        block_size=block_size,
    )

    seq_group = SequenceGroup(
        request_id="0",
        seqs=[decoder_seq],
        encoder_seq=encoder_seq,
        encoder_prefix_seq=encoder_prefix_seq,
        arrival_time=0,
    )

    scheduler.add_seq_group(seq_group)
    meta, out, _ = scheduler.schedule()
    print(meta, out)

    assert len(out.scheduled_seq_groups) == 1
    assert out.scheduled_seq_groups[0].token_chunk_size == 0
    assert out.scheduled_seq_groups[0].encoder_token_chunk_size == 24
    assert out.scheduled_seq_groups[0].budgeted_encoder_token_chunk_size == 32
    out.scheduled_seq_groups[
        0].seq_group.encoder_prefix_seq.data.update_num_computed_tokens(8)
    out.scheduled_seq_groups[
        0].seq_group.encoder_seq.data.update_num_computed_tokens(24)

    meta, out, _ = scheduler.schedule()
    print(meta, out)

    assert len(out.scheduled_seq_groups) == 1
    assert out.scheduled_seq_groups[0].token_chunk_size == 0
    assert out.scheduled_seq_groups[0].encoder_token_chunk_size == 24
    assert out.scheduled_seq_groups[0].budgeted_encoder_token_chunk_size == 32
    out.scheduled_seq_groups[
        0].seq_group.encoder_seq.data.update_num_computed_tokens(24)

    meta, out, _ = scheduler.schedule()
    print(meta, out)

    assert len(out.scheduled_seq_groups) == 1
    assert out.scheduled_seq_groups[0].token_chunk_size == 1
    assert out.scheduled_seq_groups[0].encoder_token_chunk_size == 16
    assert out.scheduled_seq_groups[0].budgeted_encoder_token_chunk_size == 24
    out.scheduled_seq_groups[0].seq_group.update_num_computed_tokens(1)
    out.scheduled_seq_groups[
        0].seq_group.encoder_seq.data.update_num_computed_tokens(16)
    seq_group.seqs[0].append_token_id(0, {0: Logprob(0.0)})

    meta, out, _ = scheduler.schedule()
    print(meta, out)

    assert len(out.scheduled_seq_groups) == 1
    assert out.scheduled_seq_groups[0].token_chunk_size == 1
    assert out.scheduled_seq_groups[0].encoder_token_chunk_size == 0
    out.scheduled_seq_groups[0].seq_group.update_num_computed_tokens(1)
    seq_group.seqs[0].append_token_id(0, {0: Logprob(0.0)})
