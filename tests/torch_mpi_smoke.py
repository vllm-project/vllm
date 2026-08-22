import os

import torch
import torch.distributed as dist

os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
os.environ["LOCAL_RANK"] = "0"
os.environ.setdefault("MASTER_ADDR", "172.31.12.228")
os.environ.setdefault("MASTER_PORT", "29599")
os.environ.setdefault("GLOO_SOCKET_IFNAME", "enp39s0")

dist.init_process_group("gloo", init_method="env://")
value = torch.tensor([dist.get_rank() + 1], dtype=torch.int64)
dist.all_reduce(value)
if dist.get_rank() == 0:
    expected = dist.get_world_size() * (dist.get_world_size() + 1) // 2
    print(
        f"MPI_GLOO_SMOKE={'PASS' if value.item() == expected else 'FAIL'} "
        f"world_size={dist.get_world_size()} sum={value.item()} expected={expected}"
    )
dist.destroy_process_group()
