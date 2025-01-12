from typing import List

import torch

from vllm.v1.utils import bind_kv_cache


def test_bind_kv_cache():
    from vllm.attention import Attention

    ctx = {
        'layers.0.self_attn': Attention(32, 128, 0.1),
        'layers.1.self_attn': Attention(32, 128, 0.1),
        'layers.2.self_attn': Attention(32, 128, 0.1),
        'layers.3.self_attn': Attention(32, 128, 0.1),
    }
    kv_cache = {
        'layers.0.self_attn': torch.zeros((1, )),
        'layers.1.self_attn': torch.zeros((1, )),
        'layers.2.self_attn': torch.zeros((1, )),
        'layers.3.self_attn': torch.zeros((1, )),
    }
    runner_kv_caches: List[torch.Tensor] = []
    bind_kv_cache(ctx, runner_kv_caches, kv_cache)
    assert ctx['layers.0.self_attn'].kv_cache[0] is kv_cache[
        'layers.0.self_attn']
    assert ctx['layers.1.self_attn'].kv_cache[0] is kv_cache[
        'layers.1.self_attn']
    assert ctx['layers.2.self_attn'].kv_cache[0] is kv_cache[
        'layers.2.self_attn']
    assert ctx['layers.3.self_attn'].kv_cache[0] is kv_cache[
        'layers.3.self_attn']

    assert runner_kv_caches[0] is kv_cache['layers.0.self_attn']
    assert runner_kv_caches[1] is kv_cache['layers.1.self_attn']
    assert runner_kv_caches[2] is kv_cache['layers.2.self_attn']
    assert runner_kv_caches[3] is kv_cache['layers.3.self_attn']


def test_bind_kv_cache_non_attention():
    from vllm.attention import Attention

    # example from Jamba PP=2
    ctx = {
        'model.layers.20.attn': Attention(32, 128, 0.1),
        'model.layers.28.attn': Attention(32, 128, 0.1),
    }
    kv_cache = {
        'model.layers.20.attn': torch.zeros((1, )),
        'model.layers.28.attn': torch.zeros((1, )),
    }

    runner_kv_caches: List[torch.Tensor] = []
    bind_kv_cache(ctx, runner_kv_caches, kv_cache)

    assert ctx['model.layers.20.attn'].kv_cache[0] is kv_cache[
        'model.layers.20.attn']
    assert ctx['model.layers.28.attn'].kv_cache[0] is kv_cache[
        'model.layers.28.attn']

    assert runner_kv_caches[0] is kv_cache['model.layers.20.attn']
    assert runner_kv_caches[1] is kv_cache['model.layers.28.attn']


def test_bind_kv_cache_encoder_decoder():
    from vllm.attention import Attention, AttentionType

    # example from bart
    ctx = {
        'encoder.layers.0.self_attn.attn':
        Attention(32, 128, 0.1, attn_type=AttentionType.ENCODER),
        'decoder.layers.0.encoder_attn.attn':
        Attention(32, 128, 0.1, attn_type=AttentionType.ENCODER_DECODER),
        'decoder.layers.0.self_attn.attn':
        Attention(32, 128, 0.1, attn_type=AttentionType.DECODER),
    }

    kv_cache_tensor = torch.zeros((1, ))
    kv_cache = {
        'decoder.layers.0.encoder_attn.attn': kv_cache_tensor,
        'decoder.layers.0.self_attn.attn': kv_cache_tensor,
    }
    encoder_kv_cache = ctx['encoder.layers.0.self_attn.attn'].kv_cache

    runner_kv_caches: List[torch.Tensor] = []
    bind_kv_cache(ctx, runner_kv_caches, kv_cache)
    assert ctx['encoder.layers.0.self_attn.attn'].kv_cache is encoder_kv_cache
    assert ctx['decoder.layers.0.encoder_attn.attn'].kv_cache[0] is kv_cache[
        'decoder.layers.0.encoder_attn.attn']
    assert ctx['decoder.layers.0.self_attn.attn'].kv_cache[0] is kv_cache[
        'decoder.layers.0.self_attn.attn']

    assert runner_kv_caches[0] is kv_cache_tensor
