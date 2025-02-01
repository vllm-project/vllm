# Copyright (c) Microsoft Corporation.


class PrefixSharingGroupManager:
    """
    To manage the prefix sharing group.
    """

    def __init__(self, scheduler, block_manager):
        self.scheduler = scheduler
        self.block_manager = block_manager
        # given request_id of the shared-prefix request,
        # it holds the value of the corresponding non-shared
        # context request ids.
        self.group_dict = {}
        self.active_group_dict = {}
        # given request_id of the shared-prefix request,
        # it holds the value of the corresponding sequence id
        # and sequence length.
        self.sharedprefix2seqid_dict = {}
        # given request_id of the non_shared context request,
        # it holds the value of the corresponding shared-prefix request id.
        self.nonshared2shared_dict = {}
        # Current shared-prefix request id in the loop of `add_request()`
        self.prefix_sharing_group_request_id_reg = None

    def add_shared_prefix_request(self, seq_group):
        request_id = seq_group.request_id
        self.prefix_sharing_group_request_id_reg = request_id
        self.group_dict[request_id] = []

    def add_non_shared_request(self, seq_group):
        request_id = seq_group.request_id
        if self.prefix_sharing_group_request_id_reg is not None:
            self.group_dict[self.prefix_sharing_group_request_id_reg].extend(
                [request_id])
            self.nonshared2shared_dict[
                request_id] = self.prefix_sharing_group_request_id_reg
        else:
            # Handle the case where prefix_sharing_group_request_id_reg is None
            raise ValueError("Shared-prefix request ID is not registered.")

    def remove_finished_request(self, seq_group):
        request_id = seq_group.request_id
        # shared-prefix request
        if self.is_psgroup_shared(seq_group):
            self.sharedprefix2seqid_dict[request_id] = {
                'seq_id': seq_group.get_seqs()[0].seq_id,
                'seq_len': seq_group.get_seqs()[0].data.get_prompt_len()
            }
            self.active_group_dict[request_id] = self.group_dict.pop(
                request_id)
            # pop sequence groups from waiting to the dist waiting queue.
            dist_num = len(self.active_group_dict[request_id])
            for i in range(dist_num):
                self.scheduler.non_shared_ready.appendleft(
                    self.scheduler.non_shared_waiting.popleft())
        # non-shared request
        elif self.is_psgroup_non_shared(seq_group):
            shared_prefix_request_id = self.nonshared2shared_dict[request_id]
            self.active_group_dict[shared_prefix_request_id].remove(request_id)
            if self.active_group_dict[shared_prefix_request_id] == []:
                shared_prefix_seq_id = self.sharedprefix2seqid_dict[
                    shared_prefix_request_id]['seq_id']
                self.block_manager.free(None, shared_prefix_seq_id)
                del self.active_group_dict[shared_prefix_request_id]
                del self.sharedprefix2seqid_dict[shared_prefix_request_id]
                del self.nonshared2shared_dict[request_id]
        else:
            return

    def is_psgroup_shared(self, seq_group):
        return seq_group.request_id in self.group_dict \
            or seq_group.request_id in self.active_group_dict

    def is_psgroup_non_shared(self, seq_group):
        return seq_group.request_id in self.nonshared2shared_dict

    def is_psgroup(self, seq_group):
        return self.is_psgroup_shared(seq_group) or self.is_psgroup_non_shared(
            seq_group)

    def get_shared_prefix_request_id(self, seq_group):
        request_id = seq_group.request_id
        if request_id in self.nonshared2shared_dict:
            return self.nonshared2shared_dict[request_id]
        else:
            return request_id

    def get_common_request(self, seq_group):
        request_id = seq_group.request_id
        shared_prefix_request_id = self.get_shared_prefix_request_id(seq_group)
        shared_block_tables, shared_seq_len = [], 0
        if request_id in self.nonshared2shared_dict:
            # non_shared request
            shared_prefix_seq_id = self.sharedprefix2seqid_dict[
                shared_prefix_request_id]['seq_id']
            shared_block_tables = self.block_manager.get_block_table(
                None, shared_prefix_seq_id)
            shared_seq_len = self.sharedprefix2seqid_dict[
                shared_prefix_request_id]['seq_len']
        return shared_prefix_request_id, shared_block_tables, shared_seq_len
