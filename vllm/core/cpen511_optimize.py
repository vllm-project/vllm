import threading
from vllm.core.logger import logger
from math import ceil

running_sequence_count = 0
in_count = 0
out_count = 0
swap_in_count = 0
swap_out_count = 0
factor = 0
lock = threading.Lock()
last_speed_report = (0, 0)

size_counter = []

def add_size(size):
    global size_counter
    size_counter.append(size)
    
def get_avg_size():
    global size_counter
    return sum(size_counter) / len(size_counter)

def increment_sequence_count(num_blocks):
    global running_sequence_count, in_count, out_count, swap_in_count, swap_out_count
    with lock:
        running_sequence_count += 1
        in_count += num_blocks
        if num_blocks > 0:
            swap_in_count += 1
        # logger.debug(f'in: {in_count}, out: {out_count}, running: {running_sequence_count}')

def get_sequence_count():
    global running_sequence_count
    with lock:
        return running_sequence_count
    
def leave_free_blocks():
    global factor
    seq_cnt = get_sequence_count()
    return ceil(seq_cnt / 16 * factor)

def decrement_sequence_count(num_blocks):
    global running_sequence_count, in_count, out_count, swap_out_count
    with lock:
        running_sequence_count -= 1
        out_count += num_blocks
        if num_blocks > 0:
            swap_out_count += 1
        # logger.debug(f'in: {in_count}, out: {out_count}, running: {running_sequence_count}')
        
def set_factor(f):
    global factor
    factor = f
    
def get_stats():
    global factor, in_count, out_count, last_speed_report, swap_in_count, swap_out_count
    with lock:
        in_spd = in_count - last_speed_report[0]
        out_spd = out_count - last_speed_report[1]
        last_speed_report = (in_count, out_count)
        return factor, in_count, out_count, in_spd, out_spd, swap_in_count, swap_out_count
    
def reset_counts():
    global in_count, out_count, swap_in_count, swap_out_count
    with lock:
        in_count = 0
        out_count = 0
        swap_in_count = 0
        swap_out_count = 0