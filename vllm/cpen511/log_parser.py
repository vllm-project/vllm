from validator_helpers import *
import sys

# get the first argument as the input file path
# input_file_path = sys.argv[1]
input_file_path = 'debug.log'

# dictionary of {block_id: sequence_id}
gpu = {}
cpu = {}    

# read the input file
with open(input_file_path, 'r') as f, open('output.csv', 'w') as out:
    out.write('operation,sequence_id\n')
    lines = f.readlines()
    for line in lines:
        for operation in operations:
            if operation in line:
                if operation == 'Allocate sequence':
                    sequence, blocks = parse_allocate_sequence(line)
                    for block in blocks:
                        if block in gpu:
                            print(f"Block {block} is already in the gpu dictionary at line {line}")
                            assert False
                        gpu[block] = sequence
                elif operation == 'Appended slots':
                    sequence, blocks = parse_appended_slots(line)
                    for block in blocks:
                        if block in gpu:
                            print(f"Block {block} is already in the gpu dictionary at line {line}")
                            assert False
                        gpu[block] = sequence
                elif operation == 'Swapped out blocks':
                    sequence, gpu_blocks, cpu_blocks = parse_swapped_out_blocks(line)
                    for block in gpu_blocks:
                        if block not in gpu:
                            print(f"Block {block} is not in the gpu dictionary at line {line}")
                            assert False
                        if gpu[block] != sequence:
                            print(f"Block {block} does not belong to sequence {sequence} at line {line}")
                            assert False
                        # remove the block from the gpu dictionary
                        gpu.pop(block)
                    for block in cpu_blocks:
                        if block in cpu:
                            print(f"Block {block} is already in the cpu dictionary at line {line}")
                            assert False
                        cpu[block] = sequence
                elif operation == 'Freed block':
                    sequence, blocks = parse_freed_block(line)
                    # not sure if it is in cpu or gpu, so check both
                    # for each block in blocks, check if it is in the cpu or gpu dictionary
                    # Ensure all blocks are in the same device
                    in_gpu = all(((block in gpu) and (gpu[block] == sequence)) for block in blocks)
                    in_cpu = all(((block in cpu) and (cpu[block] == sequence)) for block in blocks)
                    if not (in_gpu or in_cpu):
                        print(f"Not all blocks are in the same device or sequence not matching at line {line}")
                        assert False
                    if in_gpu:
                        for block in blocks:
                            gpu.pop(block)
                    if in_cpu:
                        for block in blocks:
                            cpu.pop(block)
                elif operation == 'Swapped in blocks':
                    sequence, cpu_blocks, gpu_blocks = parse_swapped_in_blocks(line)
                    for block in cpu_blocks:
                        if block not in cpu:
                            print(f"Block {block} is not in the cpu dictionary at line {line}")
                            assert False
                        if cpu[block] != sequence:
                            print(f"Block {block} does not belong to sequence {sequence} at line {line}")
                            assert False
                        # remove the block from the cpu dictionary
                        cpu.pop(block)
                    for block in gpu_blocks:
                        if block in gpu:
                            print(f"Block {block} is already in the gpu dictionary at line {line}")
                            assert False
                        gpu[block] = sequence
                        
                out.write(f'{operation},{sequence}\n')