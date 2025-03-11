# This script runs the leap prefetcher on 
# a trace. 

from pandas import *
# Update the miss_trace_file here !!!
import sys, os
import math

# Get the arguments from the command line
arguments = sys.argv[1:]
filepath = arguments[0]

if len(arguments) > 1:
    portion = arguments[1]
else:
    portion = None

Good = True
if filepath == None:
    print("Please provide the path to the trace file")
    Good = False
  
if not Good:
    sys.exit(1)
    
# make them absolute paths
if not os.path.isabs(filepath):
    filepath = os.path.abspath(filepath)
print("Absolute path to trace file: ", filepath)

if portion == None:
    portion = 1
else:
    # if not portion.isnumeric():
    #     print("Please provide a valid portion number, not numeric")
    #     sys.exit(1)
    # else:
    portion = float(portion)
    if portion <= 0 or portion > 1:
        print("Please provide a valid portion number, not in the range (0, 1]")
        sys.exit(1)

# Declare the global variable here:
global max_access_hist, N_split, PW, PW_P, C_hits, disabled
max_access_hist = 32
N_split = 8 # maybe start with 8?
PW = 0
PW_P = 0
C_hits = 0
disabled = True
last_candidate = 0

    

trace_file = read_csv(filepath)
# trace_file = read_csv(filepath, skiprows=lambda x: x < len(read_csv(filepath)) * (1 - portion))

# Only take the last portion of the trace, with the given portion
trace_file = trace_file.tail(int(len(trace_file) * portion))

# Function to find majority element
def findMajority(arr, n):
    global max_access_hist
    candidate = -1
    votes = 0

    # Finding majority candidate
    for i in range (n):
        if (votes == 0):
            candidate = arr[i]
            votes = 1
        else:
            if (arr[i] == candidate):
                votes += 1
            else:
                votes -= 1
    count = 0

    # Checking if majority candidate occurs more than n/2
    # times
    for i in range (n):
        if (arr[i] == candidate):
            count += 1
    
    if (count > n // 2):
        return candidate
    else:
        return -1

# Find the trend of past accesses
def findTrend(arr):
    global max_access_hist, N_split
    w = int(max_access_hist / N_split)
    trend = 0
    while w <= max_access_hist:
        trend = findMajority(arr[-w:], w)
        if trend != -1:
            break
        w *= 2
    return trend

def getPrefetchWindowSize():
    global PW, PW_P, C_hits, disabled
    if C_hits == 0:
        if disabled:
            PW = 1
            disabled = False
        else:
            PW = 0
            disabled = True
    else:
        PW = 2**math.ceil(math.log2(C_hits + 1))
        PW = min(PW, max_access_hist)
    
    if PW < PW_P / 2:
        PW = math.floor(PW_P / 2)
    
    C_hits = 0
    PW_P = PW
    return PW

count = 0
report_time = 0
predicted = 0
not_predicted = 0
correct = 0
incorrect = 0
len_trace = len(trace_file)
while count + max_access_hist * 2 < len_trace:
    PW = getPrefetchWindowSize()
    if PW == 0:
        not_predicted += 1
        count += 1
    else:        
        deltas = trace_file['sequence_id'][count: count + max_access_hist].tolist()
        candidate = findTrend(deltas)
        if candidate == -1:
            candidate = last_candidate
        else:
            last_candidate = candidate
        delta = trace_file['sequence_id'][count + max_access_hist : count + max_access_hist + PW].tolist()
        predicted += PW
        # check the number of correct predictions in the prefetch window
        for i in range(PW):
            if delta[i] == candidate:
                correct += 1
                C_hits += 1
            else:
                incorrect += 1
        
        count += PW
        
        # Every 4096 accesses, report the stats
    
    if count > report_time:
        print("Predicted by Leap: ", predicted)
        print("Not predicted by Leap: ", not_predicted)
        print("Correct: ", correct)
        print("Incorrect: ", incorrect)
        # print("Count: ", count)
        print("Progress: ", count / len_trace)
        print("Accuracy: ", correct / (correct + incorrect))
        print("C_hits: ", C_hits)
        print("PW: ", PW)
        print("PW_P: ", PW_P)
        print("Disabled: ", disabled)
        # print("Max Access Hist: ", max_access_hist)
        # print("N_split: ", N_split)
        print("====================================")
        report_time += 0x10000

f = open("leap_stats.txt", "a")
# change stdout to a file object
sys.stdout = f

print(f"for trace file: {filepath}")
print("Length of trace: ", len_trace)
print("Portion of trace: ", portion)
accuracy = correct / (correct + incorrect)
coverage = correct / len_trace
print("Predicted by Leap: ", predicted)
print("Not predicted by Leap: ", not_predicted)
print("Correct: ", correct)
print("Incorrect: ", incorrect)
print("Accuracy: ", accuracy)
print("Coverage: ", coverage)
print()


f.close()


