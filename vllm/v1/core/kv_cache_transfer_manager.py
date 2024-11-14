from enum import Enum

class MigrationState(Enum):
    Queue = 0
    BlockAvailable = 1
    MigrationStart = 2
    MigrationFinish = 3

class KvCacheTransferManager:
    def __init__(
        self,
        prefill_cache_config,
        ...,
    ) -> None:
        self.prefill_migration_queue = []
        self.decode_migration_queue = []
        self.ongoing_migration_queue = []
        self.prefill_migration_state = {}

        self.prefill_cache_config = prefill_cache_config

        self.torch_nccl_migrator = TorchNcclMigrator()
        self.request_prefill_decode_id_map = {}
    
    # helper function to map prefill and decode request id
    def add_prefill_decode_id_map(self, distserve_id, prefill_decode_id):
        self.request_prefill_decode_id_map[distserve_id.id] = prefill_decode_id

    def _prefill_to_decode_id(self, prefill_id):
        for prefill, decode in self.request_prefill_decode_id_map.values():
            if prefill == prefill_id:
                return decode
    
    def _decode_to_prefill_id(self, decode_id):
        for prefill, decode in self.request_prefill_decode_id_map.values():
            if decode == decode_id:
                return prefill


    # prefill callback
    # replace running_requests to all requests that already being migrated, 
    # so we can continue freeing kvcache and output encoder
    def on_execute_model_finish_callback(self, running_requests):
        migrate_requests_data = []
        for req in running_requests:
            if req.num_computed_tokens == req.num_total_tokens:
                migrate_requests_data.append(req)
                # remove running queue and add to free_cache_wait queue
                # save to move on by just removing request from running
                running_requests.remove(req.id)
                self.prefill_migration_state[req.request_id] = req

		# non blocking
		# requests data include request id and block ids
		# add the request to the queue
        self.on_migrate_request_initialized(migrate_requests_data)

        # update running request with finished cache migration
        # collect all the finished migration request
        running_requests.extend(self.on_finished_migrations())
        return running_requests

    # prefill helper
    # function to add migration request from finished prefill
    def on_migrate_request_initialized(self, prefill_requests):
        self.prefill_migration_queue.extend(prefill_requests)
        for req in prefill_requests:
            self.prefill_migration_state[req.request_id] = MigrationState.Queue
        
    # function to inform prefill engine which request already migrated
    def on_finished_migrations(self):
        finished_request_migration = []	
        for req_id in self.prefill_migration_state:
            if self.prefill_migration_state[req_id] == MigrationState.MigrationFinished:
                finished_request_migration.append(req_id)
                self.prefill_migration_state.remove(req_id)
        return finished_request_migration

    # decode callback
    # remove all new decode requests which are not migrated yet, replace it with request that ready to be migrated
    # note: torch.send wont do migration until torch.recv called (in model_runner) which is a blocking function
    def on_scheduling_finished_callback(self, scheduler_output, max_num_scheduled_tokens):
        new_decode_reqs = []
        new_total_num_scheduled_tokens = 0
        # add all new decode request to decode migration queue
        self.decode_migration_queue.extend(scheduler_output.scheduled_new_reqs)

        # check if prefill already initiate migration queue
        for request in self.decode_migration_queue:
            # prefill kvcache is available
            if self._decode_to_prefill_id(request.request_id) in self.prefill_migration_state and \
            new_total_num_scheduled_tokens + scheduler_output.num_scheduled_tokens[request.request_id] < max_num_scheduled_tokens:
                new_decode_reqs.append(request)
                total_num_scheduled_tokens += scheduler_output.num_scheduled_tokens[request.request_id]
                self.prefill_migration_state[req.request_id] = MigrationState.BlockAvailable
            else:
                # add to decode migration wait
                self.decode_migration_queue.append(request)
        
        # overwrite scheduler_output requests
        scheduler_output.scheduled_new_reqs = new_decode_reqs
        scheduler_output.total_num_scheduled_tokens = new_total_num_scheduled_tokens
        return scheduler_output
        

    # callback to be called from the model_runner decode engine to override kvcache with the migrated 
    # blocked until migration finish as we use torch.recv
    def on_retrieve_migrated_kvcache(self, request_id):
        migrated_kvcache = self.torch_nccl_migrator.recv_pipe(request_id, tensor, rank)
        if not migrated_kv_cache.success:
            # fault tolerance
            return
        # update state
        self.prefill_migration_state[request_id] = MigrationState.Finished	
        self.ongoing_migration_queue.pop(request_id)
        return migrated_kvcache

    # periodical kvcache transfer manager check and perform migration
    # periodically start migration process on request that have block allocated
    def _migrate_step(self):
        for migration_req in self.prefill_migration_queue:
            if self.prefill_migration_state[migration_req] == MigrationState.BlockAvailable:
                # adjust implementation after learning how to transfer cache from block id
                self.torch_nccl_migrator.send_pipe(migration_req.request_id, tensor, rank)
                self.ongoing_migration_queue.push(migration_req.request_id)
                self.prefill_migration_queue.pop(migration_req)

    def start_event_loop(self):
        while True:
            #1: perform migration onto specified block in decode engine
            self._migrate_step()
