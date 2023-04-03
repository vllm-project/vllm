import torch

# Global profile option

SYNC_FOR_PROFILING = False

def maybe_sync_for_profiling():
    if SYNC_FOR_PROFILING:
        torch.cuda.synchronize()

def get_sync_for_profiling():
    return SYNC_FOR_PROFILING

def set_sync_for_profiling(new_value: bool = True):
    global SYNC_FOR_PROFILING
    SYNC_FOR_PROFILING = new_value

# Communication latency

COMMUNICATION_LATENCY = 0.0

def reset_communication_latency():
    global COMMUNICATION_LATENCY
    COMMUNICATION_LATENCY = 0.0

def add_to_communication_latency(latency):
    global COMMUNICATION_LATENCY
    COMMUNICATION_LATENCY += latency

def get_communication_latency():
    return COMMUNICATION_LATENCY
