from validator_helpers import *
import sys
import pandas as pd


input_file_path = sys.argv[1]
if len(sys.argv) > 2:
    length = int(sys.argv[2])
else:
    length = 8

inputs = pd.read_csv(input_file_path)
    
prev_history = None

# enum the operations with label encoding
operations = ['Allocate sequence', 'Appended slots', 'Swapped out blocks', 'Freed block', 'Swapped in blocks']
for i, operation in enumerate(operations):
    inputs['operation'] = inputs['operation'].replace(operation, i).infer_objects(copy=False)

# print out first 5 rows
print(inputs.head())

# transform all cols into int   
inputs = inputs.infer_objects(copy=False)

# create a queue of length length, which contains the first n length of the rows
queue = []
for i in range(length):
    queue.append((inputs.iloc[i]['operation'], inputs.iloc[i]['sequence_id']))

with open(f'predict_{length}_history.csv', 'w') as out:
    out.write('history,target\n')
    for _, row in inputs.iterrows():
        if row['operation'] == operations.index('Swapped in blocks'):
            if prev_history is not None:
                out.write(f"{prev_history},{row['sequence_id']}\n")
            else:
                out.write(f"{queue},{row['sequence_id']}\n")
            prev_history = queue
        elif operations.index('Swapped in blocks') not in queue:
            prev_history = None
        queue.pop(0)
        queue.append((int(row['operation']), int(row['sequence_id'])))