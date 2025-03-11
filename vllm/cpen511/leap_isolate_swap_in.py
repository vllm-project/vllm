from validator_helpers import *
import sys
import pandas as pd


input_file_path = sys.argv[1]

inputs = pd.read_csv(input_file_path)

start = 0

# read each rwo, check it 'operation' col is 'Swapped in blocks', if so, write the sequence_id to leap_output.csv
with open('leap_output.csv', 'w') as out, open('pure_sequence.csv', 'w') as pure_out:
    out.write('sequence_id\n')
    pure_out.write('sequence_id\n')
    for index, row in inputs.iterrows():
        if row['operation'] == 'Swapped in blocks':
            current_sequence = int(row['sequence_id'])
            pure_out.write(f"{current_sequence}\n")
            out.write(f"{current_sequence - start}\n")
            start = current_sequence
            