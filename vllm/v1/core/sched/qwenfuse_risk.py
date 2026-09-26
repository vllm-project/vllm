"""Experimental local full-attention risk-triggered admission; diagnostic only."""
from vllm.v1.core.sched.scheduler import Scheduler
from .qwenfuse_risk_predict import predict_fresh
from vllm.v1.kv_cache_interface import FullAttentionSpec

class RiskPerfScheduler(Scheduler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_risk_policy()

    def enable_risk_policy(self):
        groups = self.kv_cache_config.kv_cache_groups
        if len(groups) != 1 or not isinstance(groups[0].kv_cache_spec, FullAttentionSpec) or self.connector is not None or (self.lora_config is not None) or self.is_encoder_decoder or self.num_spec_tokens or self.prefix_replay_tokens or self.num_lookahead_tokens or self.scheduler_config.async_scheduling:
            raise ValueError('Risk experiment requires synchronous single-group full attention, no KV connector, LoRA, encoder, speculation or prefix replay')
        self._qwenfuse_risk_enabled = bool(self.cache_aware_window)

    def _qwenfuse_risk_reorder(self, request, num_new_tokens, num_new_local_computed_tokens, new_computed_blocks):
        result = predict_fresh(request, dict(num_new_tokens=num_new_tokens, num_new_computed_tokens=num_new_local_computed_tokens, new_computed_blocks=new_computed_blocks), self.kv_cache_manager.block_pool.free_block_queue, self.block_size)
        if result is None:
            return False
        if not result['supported']:
            raise ValueError(f'Unsupported risk prediction: {result}')
        if result['full_risk']['reason'] != 'cached_block_exposed':
            return False
        before = self.waiting.peek_request()
        threshold = self.cache_aware_threshold
        try:
            self.cache_aware_threshold = 0.0
            self._reorder_waiting_by_cached_prefix()
        finally:
            self.cache_aware_threshold = threshold
        changed_head = self.waiting.peek_request() is not before
        return changed_head
