# vllm_mnnvl_compat.py
from vllm.distributed import (get_tensor_model_parallel_group,
                             get_pipeline_model_parallel_group,
                             get_dp_group)
import torch

# class vLLMToMPIShim:
#     """A shim that makes vLLM's communication look like MPI"""
#     def __init__(self, group_type="tp"):
#         """
#         Args:
#             group_type: Can be "tp" (tensor parallel), "pp" (pipeline parallel),
#                         or "cp" (context parallel - if applicable)
#         """
#         from vllm.distributed import (get_tensor_model_parallel_group,
#                                     get_pipeline_model_parallel_group)
        
#         self.group_type = group_type
#         self.tp_group = get_tensor_model_parallel_group()
#         self.pp_group = get_pipeline_model_parallel_group()
#         self.dp_group = get_dp_group()
        
#         # For context parallel, if vLLM doesn't have it, we might need to implement
#         # something custom or use the TP group as a fallback
#         self.cp_group = self.tp_group  # Default fallback
    
#     def Get_size(self):
#         print("tttttt calling vLLMToMPIShim")
#         if self.group_type == "tp":
#             return self.tp_group.size()
#         elif self.group_type == "pp":
#             return self.pp_group.size()
#         elif self.group_type == "cp":
#             return self.cp_group.size()
#         return self.tp_group.size()  # Default
    
#     def Get_rank(self):
#         if self.group_type == "tp":
#             return self.tp_group.rank()
#         elif self.group_type == "pp":
#             return self.pp_group.rank()
#         elif self.group_type == "cp":
#             return self.cp_group.rank()
#         return self.tp_group.rank()  # Default
    
#     def allgather(self, data):
#         import torch
#         from vllm.distributed import (tensor_model_parallel_all_gather,
#                                      pipeline_model_parallel_all_gather)
        
#         if isinstance(data, int):
#             tensor = torch.tensor([data], dtype=torch.int64, device='cuda')
#             if self.group_type == "tp":
#                 gathered = tensor_model_parallel_all_gather(tensor)
#             elif self.group_type == "pp":
#                 gathered = pipeline_model_parallel_all_gather(tensor)
#             else:
#                 gathered = tensor_model_parallel_all_gather(tensor)  # Default
#             return gathered.view(-1).cpu().numpy().tolist()
#         else:
#             raise NotImplementedError("Only integer allgather implemented")
    
#     def Split(self, color, key):
#         """
#         Mimics MPI's Split operation using vLLM's groups.
        
#         The original code uses:
#         mapping.pp_rank * mapping.cp_size + mapping.cp_rank as color
#         mapping.tp_rank as key
        
#         This effectively creates groups where:
#         - Same color = same pp_rank and cp_rank (same pipeline and context parallel group)
#         - Within each color, ranks are ordered by tp_rank
#         """
#         # In vLLM, we can approximate this by:
#         # - color determines the PP+CP group
#         # - key determines the TP rank within that group
        
#         # For vLLM, we'll just return a new shim focused on the TP group
#         # since vLLM already handles the PP/CP grouping separately
#         return vLLMToMPIShim(group_type="tp")

from flashinfer.comm.mnnvl import CommBackend as CommBackend
class vLLMCommBackend(CommBackend):
    def __init__(self):
        # self._group = get_tensor_model_parallel_group()
        self._group = get_dp_group()
    
    def Get_rank(self) -> int:
        return self._group.rank_in_group
    
    def Get_size(self) -> int:
        print('compat.py'*100)
        return self._group.world_size
    
    def allgather(self, data: int) -> list[int]:
        tensor = torch.tensor([data], device="cuda")
        # gathered = tensor_model_parallel_all_gather(tensor)
        print(f"hereallgather:{self._group}")
        gathered = self._group.all_gather(tensor)
        return gathered.cpu().tolist()#.cpu().tolist()
    
    def allgather_bytes(self, data: bytes):
        # Step 1: Convert bytes to tensor
        local_tensor = torch.ByteTensor(list(data)).unsqueeze(0).to("cuda")  # or "cpu"

        # # Step 2: Gather sizes of each data chunk from all ranks
        # local_size = torch.IntTensor([len(local_tensor)]).to("cuda")
        # world_size = self.Get_size()
        # sizes = [torch.IntTensor([0]).to("cuda") for _ in range(world_size)]
        # self._group.all_gather(sizes, local_size)

        # # Step 3: Pad to max size (required for all_gather)
        # max_size = max(s.item() for s in sizes)
        # padded = torch.zeros(max_size, dtype=torch.uint8, device="cuda")
        # padded[:len(local_tensor)] = local_tensor

        # Step 4: All-gather the padded tensors
        # gathered = [torch.empty(max_size, dtype=torch.uint8, device="cuda") for _ in range(world_size)]
        # self._group.all_gather(gathered, padded)


        # # Step 5: Trim and convert back to bytes
        # results = [bytes(t[:s.item()].tolist()) for t, s in zip(gathered, sizes)]
        # return results
        print(f"before gather: {local_tensor.shape}")
        gathered = self._group.all_gather(local_tensor, dim=0)
        print(f"what is gathered:{gathered}")
        result = [bytes(gathered[i].cpu().tolist()) for i in range(self.Get_size())]
        # result = gathered.tolist()
        # print(f"after to list:{result}")
        # result = [bytes(i[0]) for i in result]
        print(f"result: {result}")
        return result#
    
    def Split(self, color: int, key: int) -> 'vLLMCommBackend':
        # vLLM handles this automatically via its groups
        return self
