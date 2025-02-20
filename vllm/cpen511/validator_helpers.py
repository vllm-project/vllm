'''
2025-02-20 14:43:00,643 - vllm_logger - DEBUG - Allocate sequence: 82, in blocks [122]
2025-02-20 14:43:00,643 - vllm_logger - DEBUG - Allocate sequence: 83, in blocks [123, 124]
2025-02-20 14:43:00,643 - vllm_logger - DEBUG - Allocate sequence: 84, in blocks [125, 126]
2025-02-20 14:43:00,695 - vllm_logger - DEBUG - Appended slots for sequence 3 at [127]
2025-02-20 14:43:00,696 - vllm_logger - DEBUG - Appended slots for sequence 16 at [128]
2025-02-20 14:43:00,698 - vllm_logger - DEBUG - Swapped out blocks for sequences [84] from GPU [125, 126] to CPU [0, 1]
2025-02-20 14:43:00,698 - vllm_logger - DEBUG - Appended slots for sequence 20 at [126]

....

2025-02-20 14:43:00,839 - vllm_logger - DEBUG - Freed block for sequence 62 at [95, 110]

....

2025-02-20 14:43:01,042 - vllm_logger - DEBUG - Swapped in blocks for sequences [83] from CPU [2, 3] to GPU [80, 41]

'''

operations = [
    'Allocate sequence',
    'Appended slots',
    'Swapped out blocks',
    'Freed block',
    'Swapped in blocks'
]

# write a parser to extract the sequence number and the block numbers from the log. One parser for each operation.
def parse_allocate_sequence(log):
    sequence = int(log.split(': ')[1].split(',')[0])
    blocks = list(map(int, log.split('[')[1].split(']')[0].split(', ')))
    return sequence, blocks

def parse_appended_slots(log):
    sequence = int(log.split('sequence ')[1].split(' ')[0])
    blocks = list(map(int, log.split('[')[1].split(']')[0].split(', ')))
    return sequence, blocks

def parse_swapped_out_blocks(log):
    if 'sequences ' not in log or '[' not in log or ']' not in log:
        raise ValueError(f"Log format is incorrect for {log}")
    sequence = int(log.split('sequences ')[1].split(']')[0][1:])
    gpu_blocks = list(map(int, log.split('[')[2].split(']')[0].split(', ')))
    cpu_blocks = list(map(int, log.split('[')[3].split(']')[0].split(', ')))
    return sequence, gpu_blocks, cpu_blocks

def parse_freed_block(log):
    sequence = int(log.split('sequence ')[1].split(' ')[0])
    blocks = list(map(int, log.split('[')[1].split(']')[0].split(', ')))
    return sequence, blocks

def parse_swapped_in_blocks(log):
    sequence = int(log.split('sequences ')[1].split(']')[0][1:])
    cpu_blocks = list(map(int, log.split('[')[2].split(']')[0].split(', ')))
    gpu_blocks = list(map(int, log.split('[')[3].split(']')[0].split(', ')))
    return sequence, cpu_blocks, gpu_blocks

class Sequence:
    def __init__(self, sequence_id):
        self.sequence_id = sequence_id
        self.block_ids = []
        self.gpu = False
        
    def __init__(self, sequence_id, block_ids):
        self.sequence_id = sequence_id
        self.block_ids = block_ids
        
    def __init__(self, sequence_id, block_id):
        self.sequence_id = sequence_id
        self.block_ids = [block_id]

    def add_sequence(self, sequence):
        self.sequences.append(sequence)

    def get_sequences(self):
        return self.sequences

    def __str__(self):
        return str(self.sequences)