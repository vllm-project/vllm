# a = tuple(range(1000))
# b = list(range(1000))

# print(a.__sizeof__()) # 8024
# print(b.__sizeof__()) # 9088

# # 由于tuples的操作拥有更小的size，也就意味着tuples在操作时相比list更快，当数据足够大的时候tuples的数据操作性能更优

from tqdm import tqdm
import time
requests = [1,2,3,4,5]
pbar = tqdm(total=len(requests), desc='Finished requests')
# 模拟一个耗时的任务
for i in tqdm(range(10)):
    time.sleep(1)  # 假装这里有一些需要耗时的操作