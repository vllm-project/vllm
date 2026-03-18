import random
from queue import Queue
from typing import AnyStr, Tuple, List, Dict

import math
import numpy


class Task:
    """模拟任务(仅包含长度信息)"""

    def __init__(self, task_id: AnyStr, task_length: int):
        self.id = task_id
        self.length = task_length
        self.bucket_idx = -1

    def __repr__(self):
        return f"Task(id={self.id}, length={self.length})"


class Bucket:
    """ 桶 """

    def __init__(self, bucket_ranges: Tuple[int, int]):
        self.min_length = bucket_ranges[0]
        self.max_length = bucket_ranges[1]
        # 统计信息
        self.task_count = 0
        self.total_load = 0.0


class DynamicBucketLoadBalancer:
    """
    基于任务长度静态分桶，并根据桶的负载和长度亲和性动态调整新任务分配以实现负载均衡
    """

    def __init__(self, num_buckets: int, buckets: List[Tuple[int, int]], sensitivity=100.0,
                 affinity_strength=1.0, log_func=print, all_neighbor=False):
        """
        初始化负载均衡器
        :param num_buckets: 桶的数量
        :param buckets: 每个桶的长度范围
        :param sensitivity: 对负载差距的敏感度系数，值越大，对差距越敏感（与长度相关，常见LLM序列范围优选值为100）
        :param affinity_strength: 长度亲和因子的强度系数，值越大，长度匹配度对概率的影响越大
        :param log_func: 日志打印函数
        :param all_neighbor: 是否将所有桶作为邻居（负载均衡的范围），False时仅将左右桶作为邻居
        """
        self.num_buckets = num_buckets
        self.sensitivity = sensitivity
        self.affinity_strength = affinity_strength
        self.log_func = log_func
        self.all_neighbor = all_neighbor

        # 初始化桶
        self.buckets = {
            idx: Bucket(bucket_ranges) for idx, bucket_ranges in enumerate(buckets)
        }

        # 负载均衡的阈值概率，与桶的数量相关，数量越多，阈值越低
        self.base_probability_threshold = 1 / (self.num_buckets * 2.0)
        self._log_info(f"Load Balance base_probability_threshold: {self.base_probability_threshold:.2f} ")

        # 保存task
        self.tasks: Dict[AnyStr, Queue] = {}

        # 统计信息
        self.redirected_tasks = 0
        self.total_tasks = 0

    def _log_info(self, msg, *args, **kwargs):
        if self.log_func:
            self.log_func(msg, *args, **kwargs)

    def _get_standard_bucket_index(self, task_length):
        """根据任务长度确定其标准所属的桶索引"""
        for bucket_idx, bucket in self.buckets.items():
            if bucket.min_length <= task_length < bucket.max_length:
                return bucket_idx
        return self.num_buckets - 1

    def _get_neighbor_indices(self, bucket_idx):
        """获取指定桶的左右邻居索引"""
        neighbors = []
        if self.all_neighbor:
            for idx in range(self.num_buckets):
                neighbors.append(idx)
            return neighbors

        if bucket_idx > 0:
            neighbors.append(bucket_idx - 1)
        if bucket_idx < self.num_buckets - 1:
            neighbors.append(bucket_idx + 1)
        return neighbors

    def _calculate_length_affinity(self, task_length, target_bucket_idx):
        """
        计算任务长度与目标桶的亲和因子 (0.0 到 1.0)
        1.0 表示完美匹配（长度在桶中心），0.0 表示在桶的边缘。
        """
        target_bucket = self.buckets[target_bucket_idx]
        bucket_min = target_bucket.min_length
        bucket_max = target_bucket.max_length

        bucket_center = (bucket_min + bucket_max) / 2.0
        bucket_half_width = (bucket_max - bucket_min) / 2.0

        # 计算任务长度到桶中心的距离
        distance_to_center = abs(task_length - bucket_center)

        # 计算亲和因子：距离中心越近，因子越接近1
        # 使用钟形曲线 (e.g., Gaussian-like) 来平滑衰减
        if bucket_half_width > 0:
            # 归一化距离
            normalized_distance = distance_to_center / bucket_half_width
            # 使用指数衰减计算亲和因子
            affinity = math.exp(-self.affinity_strength * normalized_distance)
        else:
            affinity = 1.0  # 理论上不会发生，但作为保护

        # 确保在 [0, 1] 范围内
        return max(0.0, min(affinity, 1.0))

    def _calculate_redirect_probability(self, task_length, standard_bucket_idx, neighbor_bucket_idx):
        """
        根据负载差距和长度亲和性计算重定向到邻居桶的概率
        """
        standard_load = self.buckets[standard_bucket_idx].total_load
        neighbor_load = self.buckets[neighbor_bucket_idx].total_load

        # --- 1. 基于负载差距计算基础概率 ---
        if standard_load <= 0:
            load_probability = 0.0  # 标准桶无负载，无需重定向
        else:
            # 计算负载比率
            load_ratio = standard_load / max(neighbor_load, 1e-9)  # 防止除以零

            # 使用对数函数使概率增长更平滑，对差距更敏感
            raw_probability = math.log(load_ratio) if load_ratio > 1 else 0

            # 应用敏感度系数并限制在 [0, 1] 范围内
            load_probability = 1 - math.exp(-self.sensitivity * raw_probability)
            load_probability = max(0.0, min(load_probability, 1))

        # --- 2. 计算长度亲和因子 ---
        affinity_factor = self._calculate_length_affinity(task_length, neighbor_bucket_idx)

        # --- 3. 结合两者计算最终概率 ---
        # 最终概率 = 基础负载概率 * 长度亲和因子
        # 这意味着：即使负载差距很大，如果长度不匹配，重定向概率也会被抑制。
        # 反之，如果长度非常匹配，即使负载差距一般，也可能获得较高的重定向概率。
        final_probability = load_probability * affinity_factor

        return final_probability

    def dispatch_single_task(self, task_id: AnyStr, task_length: int):
        return self.dispatch_task(Task(task_id, task_length))

    def dispatch_task_without_id(self, task_length: int):
        return self.dispatch_task(Task("Unknown", task_length))

    def dispatch_task(self, cur_task):
        """
        为新任务分配桶，考虑动态负载均衡和长度亲和性
        """
        self.total_tasks += 1
        standard_bucket_idx = self._get_standard_bucket_index(cur_task.length)   # 根据task长度决策进入哪个桶，返回桶的id

        # 获取邻居桶, 作用是：如果当前桶的负载较高，则将本task放入负载较低的邻居桶内，邻居的序列长度要尽量与本task接近，如果all_neighbor为true，则所有桶都为邻居桶
        neighbor_indices = self._get_neighbor_indices(standard_bucket_idx)

        best_neighbor_idx = None
        best_redirect_prob = 0.0

        # 检查所有邻居，找出综合考虑负载和长度亲和性后，重定向概率最高的那个
        for neighbor_idx in neighbor_indices:
            # 只考虑负载更低的邻居
            if self.buckets[neighbor_idx].total_load < self.buckets[standard_bucket_idx].total_load:
                prob = self._calculate_redirect_probability(cur_task.length, standard_bucket_idx, neighbor_idx)   # 根据长度和负载决策选择新桶的概率，期望是：与新桶的序列长度差距小；比新桶负载更重
                if prob > best_redirect_prob:
                    best_redirect_prob = prob
                    best_neighbor_idx = neighbor_idx

        # 决定最终分配的桶
        final_bucket_idx = standard_bucket_idx
        if best_neighbor_idx is not None and best_redirect_prob > 0:
            # 根据计算出的最佳概率决定是否重定向
            if self.base_probability_threshold < best_redirect_prob:
                final_bucket_idx = best_neighbor_idx
                self.redirected_tasks += 1
                self._log_info(f"{cur_task} redirected from bucket {standard_bucket_idx} to {final_bucket_idx}"
                               f"(prob={best_redirect_prob:.4f})")

        # 将任务分配给最终选定的桶（更新统计信息）
        self.buckets[final_bucket_idx].task_count += 1
        self.buckets[final_bucket_idx].total_load += cur_task.length
        cur_task.bucket_idx = final_bucket_idx
        if cur_task.id != "Unknown":
            if cur_task.id not in self.tasks:
                self.tasks[cur_task.id] = Queue()    # 【问题】：此处原理是什么？为什么是以task为单位建立队列，【回答】：支持为了做id唯一性检查，不对主流程产生影响
            self.tasks[cur_task.id].put(cur_task)

        return final_bucket_idx

    def release_task(self, task_id: AnyStr):
        if task_id in self.tasks and not self.tasks[task_id].empty():
            found_task = self.tasks[task_id].get()
            if self.tasks[task_id].empty():
                self.tasks.pop(task_id)
            if 0 <= found_task.bucket_idx < self.num_buckets:
                self.buckets[found_task.bucket_idx].task_count -= 1
                self.buckets[found_task.bucket_idx].total_load -= found_task.length
                return True
        self._log_info(f"Task {task_id} not found")
        return False

    def release_task_by_bucket_idx(self, bucket_idx: int, task_length: int):
        if 0 <= bucket_idx < self.num_buckets:
            self.buckets[bucket_idx].task_count -= 1
            self.buckets[bucket_idx].total_load -= task_length
            return True
        self._log_info(f"Bucket {bucket_idx} not found")
        return False

    def release_all_tasks(self):
        for bucket in self.buckets.values():
            bucket.task_count = 0
            bucket.total_load = 0
        self.tasks.clear()

    def print_status(self):
        """打印当前各桶的状态"""
        self._log_info("--- Bucket Status ---")
        load_list = []
        for idx in range(self.num_buckets):
            bucket = self.buckets[idx]
            load_list.append(bucket.total_load)
            avg_load = bucket.total_load / bucket.task_count if bucket.task_count > 0 else 0
            self._log_info(f"Bucket {idx}: {bucket.task_count} tasks, "
                           f"Total Load: {bucket.total_load:.2f}, Avg Load/Task: {avg_load:.2f}")
        if self.total_tasks > 0:
            self._log_info(f"Total Tasks: {self.total_tasks}, Redirected: {self.redirected_tasks}, "
                           f"Redirect Rate: {self.redirected_tasks / self.total_tasks * 100:.2f}%")
        load_std = numpy.std(load_list, ddof=1)
        self._log_info(f"Var Load: {load_std:.2f}")
        self._log_info("---------------------\n")
        return load_std


class NoStandardBucketLoadBalancer(DynamicBucketLoadBalancer):

    def __init__(self, num_buckets: int, max_length: int, log_func=print):
        bucket_range = math.ceil(max_length / num_buckets)
        start_length = 0
        buckets = []
        for _ in range(num_buckets):
            end_length = start_length + bucket_range
            if end_length > max_length:
                end_length = max_length
            buckets.append((start_length, end_length))
            start_length += bucket_range
        super().__init__(num_buckets=num_buckets, buckets=buckets, log_func=log_func,
                         sensitivity=100, affinity_strength=0, all_neighbor=True)


# --- 示例运行 ---
if __name__ == "__main__":
    # 1. 初始化均衡器，初始化2个桶，短序列（16K以内）放在第1个桶内，长序列（16K-64K）放在第2个桶内。
    balancer = DynamicBucketLoadBalancer(num_buckets=2,
                                         buckets=[(1, 16 * 1024), (16 * 1024, 64 * 1024)],
                                         sensitivity=100, affinity_strength=1, all_neighbor=True)
    # balancer = NoStandardBucketLoadBalancer(num_buckets=2, max_length=64*1024)

    balancer.print_status()

    # 2. 处理新任务流
    print("Processing new incoming tasks...")
    incoming_tasks = []
    total_tasks = 1920
    for i in range(total_tasks):
        # 生成任务，参考新浪数据集
        # 包含一些边界情况，测试亲和因子的作用
        r = random.random()
        if r < 0.11:  # 11% 1~4K
            length = random.randint(1, 4 * 1024)
        elif r < 0.25:  # 14% 4~8K
            length = random.randint(4 * 1024, 8 * 1024)
        elif r < 0.40:  # 15% 8~12K
            length = random.randint(8 * 1024, 12 * 1024)
        elif r < 0.55:  # 15% 12~16K
            length = random.randint(12 * 1024, 16 * 1024)
        elif r < 0.67:  # 12% 16~20K
            length = random.randint(16 * 1024, 20 * 1024)
        elif r < 0.76:  # 9% 20~24K
            length = random.randint(20 * 1024, 24 * 1024)
        elif r < 0.82:  # 6% 24~28K
            length = random.randint(24 * 1024, 28 * 1024)
        elif r < 0.87:  # 5% 28~32K
            length = random.randint(28 * 1024, 32 * 1024)
        else:  # 13% 32~64K
            length = random.randint(32 * 1024, 64 * 1024)

        incoming_tasks.append(Task(f"Incoming_{i}", length))

    print("--- Task Assignment Log ---")
    print(f'>>>>>>>>> incoming_tasks.len:: {len(incoming_tasks)}')
    for task in incoming_tasks:
        assigned_bucket = balancer.dispatch_task(task)    # 针对每个task进行分桶，分桶时考虑负载均衡，若重新分配会有打印信息

    balancer.print_status()

    print("--- Task Assignment Log ---")
    batch_num = 192
    batch = batch_num
    total_load_std = 0
    print(f'>>>>>>>>> incoming_tasks.len:: {len(incoming_tasks)}')   #
    for task in incoming_tasks:
        batch -= 1
        if batch < 0:
            total_load_std += balancer.print_status()    # 【问题】：这是组batch，为什么要先分组完再组batch？相同的task又分组了一把？【回答】：两者只用其一
            balancer.release_all_tasks()
            batch = batch_num - 1
        assigned_bucket = balancer.dispatch_task(task)   # 【问题】：192个task为1个batch，进行分组，然后送入模型求解，然后再组下一个batch吗？【回答】：总的流程是：上层来一个batch，调度层对其dispatch_task，然后进行release_all_tasks，在循环下一个。


    # 3. 打印统计结果
    total_load_std += balancer.print_status()
    print(f"--- Total Load Std --- {total_load_std / (total_tasks / batch_num) / 1024:.2f}")

    # 4. 释放请求
    balancer.release_task("Incoming_0")
    balancer.release_task("Incoming_1000")
    balancer.release_all_tasks()