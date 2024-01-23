from abc import ABC


# 使用了策略模式为具有不同优化目标的基于深度强化学习的作业调度模型提供不同的奖励计算方法
class RewardGiver(ABC):
    def __init__(self):
        self.simulation = None

    def attach(self, simulation):
        self.simulation = simulation

    def get_reward(self):
        if self.simulation is None:
            raise ValueError('Before calling method get_reward, the reward giver '
                             'must be attach to a simulation using method attach.')


# MakespanRewardGiver 给出用于优化完工时间（Makespan）的奖励
class MakespanRewardGiver(RewardGiver):
    name = 'Makespan'

    def __init__(self, reward_per_timestamp):
        super().__init__()
        self.reward_per_timestamp = reward_per_timestamp

    def get_reward(self):
        super().get_reward()
        return self.reward_per_timestamp


# AverageSlowDownRewardGiver 给出用于优化平均 SlowDown 的奖励
class AverageSlowDownRewardGiver(RewardGiver):
    name = 'AS'

    def get_reward(self):
        super().get_reward()
        cluster = self.simulation.cluster
        unfinished_tasks = cluster.unfinished_tasks
        reward = 0
        for task in unfinished_tasks:
            reward += (- 1 / task.task_config.duration)
        return reward


# AverageCompletionRewardGiver 给出用于优化平均完成时间的奖励
class AverageCompletionRewardGiver(RewardGiver):
    name = 'AC'

    def get_reward(self):
        super().get_reward()
        cluster = self.simulation.cluster
        unfinished_task_len = len(cluster.unfinished_tasks)
        return - unfinished_task_len


# 给出在边缘计算模式下完成时间的奖励
# 奖励r计算应当与以下数值有关系：
# 计算时延a，传输时延b，能耗c，
# r = xa+yb+zc
# 其中x+y+z=1
# 解：
# 获取当前完成的任务‘信息’和对应虚拟机‘信息’
class MakespanRewardGiver_edge(RewardGiver):
    name = 'Makespan_edge'

    def __init__(self, all_candidates, pair_index):
        super().__init__()
        self.candidates = all_candidates
        self.index = pair_index

    def get_reward(self):
        super().get_reward()
        reward = []
        counter = 0
        for machine, task in self.candidates:
            machine_cpu = machine.cpu_capacity
            machine_memory = machine.memory_capacity
            machine_disk = machine.disk_capacity
            machine_bandwidth = machine.bandwidth
            machine_energy_consumption_per_unit = machine.energy_consumption_per_unit
            task_cpu = task.task_config.cpu
            task_memory = task.task_config.memory
            task_disk = task.task_config.disk
            if machine_bandwidth is not None and machine_energy_consumption_per_unit is not None:
                # 计算时延和虚拟机的cpu(machine_cpu)、任务的cpu(task_cpu)有关
                # task_cpu固定时，machine_cpu越大计算时延越小
                # 1.较为直观的计算式：task_cpu/machine_cpu
                # 2.取关于machine_cpu的task_cpu的对数:math.log(task_cpu, machine_cpu)
                computation_latency = task_cpu / machine_cpu
                # 传输时延和虚拟机带宽(machine_bandwidth)、任务的memory和disk(task_memory,task_disk)
                # 计算式为：(task_memory + task_disk) / machine_bandwidth
                transmission_latency = (task_memory + task_disk) / machine_bandwidth
                # 能耗和计算时延有关
                energy_consumption = computation_latency * machine_energy_consumption_per_unit
                reward[counter] = computation_latency + transmission_latency + energy_consumption
            else:
                reward[counter] = task_cpu / machine_cpu
            counter += 1
        return reward[self.index]
