import threading
from vllm.core.logger import logger
from math import ceil

running_sequence_count = 0
in_count = 0
out_count = 0
lock = threading.Lock()

def increment_sequence_count(num_blocks):
    global running_sequence_count, in_count, out
    with lock:
        running_sequence_count += 1
        in_count += num_blocks
        logger.debug(f'in: {in_count}, out: {out_count}, running: {running_sequence_count}')

def get_sequence_count():
    global running_sequence_count
    with lock:
        return running_sequence_count
    
def leave_free_blocks():
    seq_cnt = get_sequence_count()
    return ceil(seq_cnt / 16) * 2
    # return 0

def decrement_sequence_count(num_blocks):
    global running_sequence_count, in_count, out_count
    with lock:
        running_sequence_count -= 1
        out_count += num_blocks
        logger.debug(f'in: {in_count}, out: {out_count}, running: {running_sequence_count}')