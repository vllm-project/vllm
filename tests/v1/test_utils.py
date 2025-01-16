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
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)
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
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)

    assert ctx['model.layers.20.attn'].kv_cache[0] is kv_cache[
        'model.layers.20.attn']
    assert ctx['model.layers.28.attn'].kv_cache[0] is kv_cache[
        'model.layers.28.attn']

    assert runner_kv_caches[0] is kv_cache['model.layers.20.attn']
    assert runner_kv_caches[1] is kv_cache['model.layers.28.attn']
