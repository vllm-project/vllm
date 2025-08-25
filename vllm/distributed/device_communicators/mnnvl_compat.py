from vllm.distributed import get_dp_group
import torch

from flashinfer.comm.mnnvl import CommBackend as CommBackend

class vLLMCommBackend(CommBackend):
    def __init__(self):
        self._group = get_dp_group()
    
    def Get_rank(self) -> int:
        return self._group.rank_in_group
    
    def Get_size(self) -> int:
        return self._group.world_size
    
    def allgather(self, data: int) -> list[int]:
        tensor = torch.tensor([data], device="cuda")
        gathered = self._group.all_gather(tensor)
        return gathered.cpu().tolist()
    
    def allgather_bytes(self, data: bytes):
        device_id = torch.cuda.current_device()
        device_str = f"cuda:{device_id}"
        local_tensor = torch.ByteTensor(list(data)).unsqueeze(0).to(device_str)
        gathered = self._group.all_gather(local_tensor, dim=0)
        result = [bytes(gathered[i].cpu().tolist()) for i in range(self.Get_size())]
        return result

    def allgather(self, data: int | bytes):
        device = f"cuda:{torch.cuda.current_device()}"

        if isinstance(data, int):
            # Handle integer case
            local_tensor = torch.tensor([data], device=device, dtype=torch.int32)
            gathered = self._group.all_gather(local_tensor)
            return [int(x.item()) for x in gathered]

        elif isinstance(data, bytes):
            # Handle bytes case
            local_tensor = torch.ByteTensor(list(data)).unsqueeze(0).to(device)
            gathered = self._group.all_gather(local_tensor, dim=0)
            return [bytes(gathered[i].cpu().tolist()) for i in range(self.Get_size())]

        else:
            raise TypeError(f"Unsupported type for allgather: {type(data)}")

    def Split(self, color: int, key: int) -> 'vLLMCommBackend':
        # vLLM handles this automatically via its groups
        return self
