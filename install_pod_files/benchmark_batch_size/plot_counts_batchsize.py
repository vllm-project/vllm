# Assisted by watsonx Code Assistant

import matplotlib.pyplot as plt
from collections import defaultdict

# Given data for 10 prompts
data = [[(2, 528, 3.76), (1, 119, 3.4), (4, 87, 3.69), (8, 31, 3.71), (16, 2, 5.14)]]

# Given data for 100 prompts
# data = [[(2, 622, 3.76), (1, 267, 3.4), (4, 183, 3.82), (16, 141, 4.1), (24, 131, 4.27), (8, 123, 3.96), (48, 71, 4.48), (40, 71, 4.5), (32, 37, 4.3), (56, 29, 4.52), (2048, 10, 24.87), (64, 9, 4.66)]]

# Given data for 1000 prompts
# data = [[(2, 651, 3.76), (1, 393, 3.4), (4, 239, 3.83), (16, 227, 4.16), (24, 155, 4.27), (8, 123, 3.96), (40, 114, 4.52), (48, 111, 4.51), (256, 85, 7.84), (56, 62, 5.61), (32, 59, 4.32), (2048, 55, 28.61), (64, 33, 5.78), (80, 26, 6.17), (112, 25, 6.72), (168, 24, 7.69), (104, 21, 6.64), (280, 16, 11.41), (272, 15, 12.03), (72, 13, 6.03), (136, 12, 7.26), (200, 12, 7.95), (144, 11, 7.47), (120, 11, 6.79), (88, 11, 6.27), (160, 10, 7.59), (232, 10, 8.39), (192, 10, 7.8), (240, 10, 8.48), (128, 10, 6.98), (264, 9, 11.19), (96, 9, 6.43), (288, 7, 11.4), (208, 7, 8.07), (176, 7, 7.74), (224, 6, 8.28), (152, 5, 7.62), (304, 5, 11.96), (312, 5, 13.24), (512, 5, 14.25), (216, 5, 8.11), (416, 4, 13.92), (296, 4, 10.71), (344, 4, 11.9), (248, 4, 8.59), (184, 4, 7.77), (320, 3, 12.01), (352, 3, 11.87), (472, 3, 13.66), (1031, 3, 23.8), (336, 3, 13.0), (828, 2, 20.08), (653, 2, 17.26), (551, 2, 14.69), (480, 2, 13.48), (806, 2, 19.56), (958, 2, 21.04), (432, 2, 14.27), (1091, 2, 23.66), (628, 2, 17.18), (424, 2, 12.97), (783, 2, 20.54), (1024, 2, 23.36), (582, 2, 18.8), (514, 2, 17.95), (1016, 2, 23.78), (1035, 2, 26.47), (360, 2, 14.79)]]


# Step 1: Parse the data and aggregate counts
batchsize_counts = defaultdict(int)

for process in data:
    for batchsize, count, _ in process:
        batchsize_counts[batchsize] += count

# Step 2: Prepare data for plotting
batchsizes = list(batchsize_counts.keys())
total_counts = list(batchsize_counts.values())

# Step 3: Plot the bar diagram
# plt.bar(batchsizes, total_counts)
# plt.xlabel('Batchsize')
# plt.ylabel('Total Counts')
# plt.title('Total Counts by Batchsize for prompts')
# plt.show()

sorted_by_key = dict(sorted(batchsize_counts.items()))
for k,v in sorted_by_key.items():
    print(k, v)
