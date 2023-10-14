import torch
import vllm_sort
from loguru import logger


class CudaTimer:

    def __init__(self,name ):
        self.name = name
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start_event.record()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.end_event.record()
        self.end_event.synchronize()
        self.elapsed_time_ms = self.start_event.elapsed_time(self.end_event)  # 毫秒为单位
        elapsed_time_seconds = self.elapsed_time_ms / 1000  # 转换为秒
        formatted_float = "{:.8f}".format(elapsed_time_seconds)
        logger.info(f"{self.name} cost is : {formatted_float}s")
        print(torch.cuda.memory_summary())
        return True
def main(size):
    logger.error(f'size is {size}')
    out= torch.rand((1,size), device='cuda:0')
    tensor = torch.rand((1,size), device='cuda:0')
    with CudaTimer('pytorch sort'):
        torch.sort(tensor)
    # with CudaTimer('cub sort'):
    #     vllm_sort.vllm_sort(tensor,out)
    
if __name__=="__main__":
    main(10**8)
        




