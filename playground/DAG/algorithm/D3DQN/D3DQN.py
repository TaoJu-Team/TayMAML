import math
import random
from math import e

import scipy.stats
import tensorflow as tf
import numpy as np

tf.enable_eager_execution()


class Node(object):
    def __init__(
            self,
            observation,
            action,
            reward,
            wighted_cost,
            computation_latency,
            transmission_latency,
            energy_consumption,
            clock):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.wighted_cost = wighted_cost
        self.computation_latency = computation_latency
        self.transmission_latency = transmission_latency
        self.energy_consumption = energy_consumption
        self.clock = clock


class machine_bad(object):
    def __init__(self):
        self.cpu = -1
        self.memory = -1
        self.disk = -1


class D3DQN(object):
    def __init__(self,
                 agent_target,
                 agent_eval,
                 action_probability,
                 learning_steps,
                 itr,
                 last_loss,
                 machines_number,
                 reward_giver,
                 features_normalize_func,
                 features_extract_func):
        self.agent_target = agent_target
        self.agent_eval = agent_eval
        self.action_probability = action_probability
        self.learning_steps = learning_steps
        self.itr = itr
        self.last_loss = last_loss
        self.reward = 0
        self.machines_number = machines_number
        self.reward_giver = reward_giver
        self.features_normalize_func = features_normalize_func
        self.features_extract_func = features_extract_func
        self.current_trajectory = []

    def extract_features(self, valid_pairs):
        features = []
        for machine, task in valid_pairs:
            features.append([machine.cpu, machine.memory, machine.bandwidth, machine.energy_consumption_per_unit] + self.features_extract_func(task))
        features = self.features_normalize_func(features)
        return features

    def communication_rate(self, bandwidth):
        noise = np.random.randn()
        m = 2
        nakagami_fading = scipy.stats.nakagami.stats(m)
        rate = bandwidth * math.log2(1 + 10 * math.log10((500 * abs(nakagami_fading[0]) ** 2) / abs(noise)))
        return rate

    def get_cost(self, all_candidates, pair_index):
        candidates = all_candidates
        machine = candidates[pair_index][0]
        task = candidates[pair_index][1]
        machine_cpu = machine.cpu
        machine_memory = machine.memory
        machine_disk = machine.disk
        machine_bandwidth = machine.bandwidth
        machine_energy_consumption_per_unit = machine.energy_consumption_per_unit
        task_cpu = task.task_config.cpu
        task_memory = task.task_config.memory
        task_disk = task.task_config.disk
        if machine_bandwidth != 0:
            communication_rate = self.communication_rate(machine_bandwidth)
            computation_latency = task_cpu / machine_cpu
            transmission_latency = task_disk / communication_rate
            energy_consumption = (computation_latency + transmission_latency) * machine_energy_consumption_per_unit
            cost = 0.2 * computation_latency + 0.7 * transmission_latency + 0.1 * energy_consumption
            return cost, computation_latency, transmission_latency, energy_consumption
        else:
            computation_latency = task_cpu / machine_cpu
            transmission_latency = 0.0
            energy_consumption = (computation_latency + transmission_latency) * machine_energy_consumption_per_unit
            cost = task_cpu / machine_cpu
            return cost, computation_latency, transmission_latency, energy_consumption

    def machine_load(self, machine):
        cpu_capacity = machine.cpu_capacity
        memory_capacity = machine.memory_capacity
        disk_capacity = machine.disk_capacity
        cpu = machine.cpu
        memory = machine.memory
        disk = machine.disk
        load_ratio = 0.4 * ((cpu_capacity - cpu) / cpu_capacity) + 0.3 * ((memory_capacity - memory) / memory_capacity) + 0.3 * ((disk_capacity - disk) / disk_capacity)

        if load_ratio <= 0.7:
            return True
        else:
            return False

    def in_list(self, element, array):
        if element in array:
            return True
        else:
            return False

    def __call__(self, cluster, clock):
        machines = cluster.machines
        tasks = cluster.ready_tasks_which_has_waiting_instance
        all_candidates = []
        exist_solution = False
        for task in tasks:
            for machine in machines:
                if machine.accommodate(task) and self.machine_load(machine):
                    exist_solution = True
                all_candidates.append((machine, task))
            break
        if not exist_solution:
            self.current_trajectory.append(Node(None,
                                                None,
                                                self.reward_giver.get_reward(),
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                clock))
            return None, None
        else:
            features = self.extract_features(all_candidates)
            features = tf.convert_to_tensor(features, dtype=np.float32)

            logits_target = self.agent_target.brain(features)
            logits_eval = self.agent_eval.brain(features)

            logits = []
            logits_temp = []
            array_target = logits_target.numpy()[0]
            array_eval = logits_eval.numpy()[0]
            for i in range(logits_target.shape[1]):
                logits_temp.append(array_target[i] * (1 - pow(e, -self.last_loss)**1)
                                   + array_eval[i] * pow(e, -self.last_loss)**1)
            logits.append(logits_temp)
            logits = tf.convert_to_tensor(logits)

            if random.random() <= (self.action_probability - pow(e, -self.last_loss)**2):
                pair_index = tf.squeeze(tf.multinomial(logits, num_samples=1), axis=1).numpy()[0]
            else:
                pair_index = np.argmax(logits.numpy())

            machine = all_candidates[pair_index][0]
            task = all_candidates[pair_index][1]
            if machine.accommodate(task) and self.machine_load(machine):
                node = Node(features,
                            pair_index,
                            0,
                            self.get_cost(all_candidates, pair_index)[0],
                            self.get_cost(all_candidates, pair_index)[1],
                            self.get_cost(all_candidates, pair_index)[2],
                            self.get_cost(all_candidates, pair_index)[3],
                            clock)
                self.current_trajectory.append(node)
                return all_candidates[pair_index]
            else:
                node = Node(None,
                            None,
                            self.reward_giver.get_reward(),
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            clock)
                self.current_trajectory.append(node)
                return None, None
