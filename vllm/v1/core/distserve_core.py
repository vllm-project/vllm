from vllm.v1.engine.core import EngineCore
from vllm.v1.core.kv_cache_transfer_manager import KvCacheTransferManager

class DistServeCore:
    def __init__(self):
        self.kvcache_transfer_manager = KvCacheTransferManager()
        self.prefill_engine = EngineCore(stage='prefill', 
        on_execute_model_finish_callback = self.kvcache_transfer_manager.on_execute_model_finish_callback)
        self.decode_engine = EngineCore(stage='decode', 
        on_scheduling_finished_callback = self.kvcache_transfer_manager.on_scheduling_finished_callback)

        self.request_prefill_decode_id_map = {}
    
    def add_request(self, new_request):
        # initialize request for both prefill and decode engine with same request id
        # both engine will try to allocate blocks, but decode engine will not run execute_model
        # see core.py::step()
        prefill_request_id = self.prefill_engine.add_request(new_request)
        decode_request_id = self.decode_engine.add_request(new_request)
        self.request_prefill_decode_id_map[new_request.id] = [prefill_request_id, decode_request_id]
        self.kvcache_transfer_manager.add_prefill_decode_id_map(new_request.id, [prefill_request_id, decode_request_id])

