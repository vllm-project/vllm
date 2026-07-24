import sys
from unittest.mock import MagicMock

# 1. Initialize the root package mock and explicitly declare it as a package
# by assigning the __path__ attribute to prevent ModuleNotFoundError.
mock_lmcache = MagicMock()
mock_lmcache.__path__ = []

# 2. Mock the utils module and bypass the specific NVTX decorator.
mock_utils = MagicMock()
mock_utils._lmcache_nvtx_annotate = lambda f: f

# 3. Register the package and all its required submodules in sys.modules.
sys.modules['lmcache'] = mock_lmcache
sys.modules['lmcache.utils'] = mock_utils
sys.modules['lmcache.config'] = MagicMock()
sys.modules['lmcache.v1'] = MagicMock()
sys.modules['lmcache.v1.multiprocess'] = MagicMock()
sys.modules['lmcache.v1.multiprocess.custom_types'] = MagicMock()
sys.modules['lmcache.v1.multiprocess.mq'] = MagicMock()
sys.modules['lmcache.v1.multiprocess.protocol'] = MagicMock()

# Now the target adapter can be safely imported without external dependencies.
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.multi_process_adapter import LMCacheMPWorkerAdapter

def test_get_finished_filters_aborted_requests():
    adapter = LMCacheMPWorkerAdapter.__new__(LMCacheMPWorkerAdapter)
    adapter.store_futures = {}
    adapter.retrieve_futures = {}
    adapter.finished_stores = set()
    adapter.previously_finished = set()
    
    dead_req_id = "req_dead_1"
    mock_future = MagicMock()
    mock_future.query.return_value = True
    mock_future.result.return_value = [True]
    
    adapter.retrieve_futures[dead_req_id] = (mock_future, [])
    finished_req_ids_from_engine = {dead_req_id}
    
    ret_stores, finished_retrieves = adapter.get_finished(finished_req_ids_from_engine)
    
    assert dead_req_id not in finished_retrieves, "Failed: Aborted request leaked to finished_retrieves"
    assert dead_req_id not in adapter.retrieve_futures, "Failed: Aborted request remains in retrieve_futures"
