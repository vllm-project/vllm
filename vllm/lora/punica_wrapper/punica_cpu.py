from typing import final

from .punica_gpu import PunicaWrapperGPU


@final
class PunicaWrapperCPU(PunicaWrapperGPU):
    """
    PunicaWrapperCPU is designed to manage and provide metadata for the punica 
    kernel. The main function is to maintain the state information for 
    Multi-LoRA, and to provide the interface for the pytorch punica ops.
    It uses the punica ops in the same manner as the GPU implementation.
    """
    pass
