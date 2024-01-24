import os
import sys
import time

import numpy as np
import tensorflow as tf
import pandas as pd

from psutil import *
from playground.DAG.algorithm.D3DQN.D3DQN import D3DQN

from core.machine import MachineConfig
from playground.DAG.adapter.episode import Episode
from playground.DAG.algorithm.D3DQN.agent import Agent
from playground.DAG.algorithm.D3DQN.brain import BrainSmall
from playground.DAG.algorithm.D3DQN.reward_giver import MakespanRewardGiver
from playground.DAG.utils.csv_reader import CSVReader
from playground.DAG.utils.feature_functions import features_extract_func_ac, features_normalize_func_ac
from playground.auxiliary.tools import average_completion, average_slowdown

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
sys.path.append('..')

os.environ['CUDA_VISIBLE_DEVICES'] = ''

np.random.seed(41)
tf.random.set_random_seed(41)
# ************************ Parameters Setting Start ************************

machines_number_local = 1
machines_number_edge = 4
machines_number = machines_number_edge + machines_number_local
learning_steps = 4000
action_probability = 1
tasks_num = 10
jobs_start_index = 0
jobs_len = 5
jobs_csv = '../jobs_files/jobs.csv'

brain_target = BrainSmall(16, machines_number_local + machines_number_edge)
brain_eval = BrainSmall(16, machines_number_local + machines_number_edge)
reward_giver = MakespanRewardGiver(-1)
features_extract_func = features_extract_func_ac
features_normalize_func = features_normalize_func_ac

name = '%s-%s-m%d' % (reward_giver.name, brain_target.name, machines_number_local + machines_number_edge)
model_dir = './agents/%s' % name
# ************************ Parameters Setting End **************************
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

agent_target = Agent(name, brain_target, 1, reward_to_go=True, nn_baseline=True, normalize_advantages=True,
                     model_save_path='%s/model.ckpt' % model_dir)
agent_eval = Agent(name, brain_eval, 1, reward_to_go=True, nn_baseline=True, normalize_advantages=True,
                   model_save_path='%s/model.ckpt' % model_dir)

machine_configs = [MachineConfig(4, 4, 4, [False, 0, 1]) for i in range(machines_number_local)] \
                  + [MachineConfig(16, 16, 16, [True, 1, 1]) for i in range(machines_number_edge)]
csv_reader = CSVReader(jobs_csv)

prior_samples_buffer = []
last_loss = 100
last_loss_array = []
D3DQN_loss = []

D3DQN_makespan = []
D3DQN_job_makespan = []
D3DQN_job_cost = []
D3DQN_job_computation = []
D3DQN_job_transmission = []
D3DQN_job_energy_consumption = []
D3DQN_job_reward = []

for itr in range(learning_steps):

    # 优先级经验回放机制
    if itr % 50 == 0 and itr > 0:
        print('*****Learning previous experience with higher priority*****')
        length = len(prior_samples_buffer) / tasks_num
        length = int(length)
        sample_observations = []
        sample_actions = []
        sample_rewards = []
        for index in range(length):
            observations = []
            actions = []
            rewards = []
            counter = index * tasks_num
            for i in range(tasks_num):
                observations.append(prior_samples_buffer[counter + i][0])
                actions.append(prior_samples_buffer[counter + i][1])
                rewards.append(prior_samples_buffer[counter + i][2])
            sample_observations.append(observations)
            sample_actions.append(actions)
            sample_rewards.append(rewards)
        agent_eval.priority_sample(sample_observations, sample_actions, sample_rewards, None, 0, 0, False)
        prior_samples_buffer = []

    jobs_configs = csv_reader.generate(jobs_start_index, jobs_len)

    print("********** Iteration %i ************" % (itr + 1))
    all_observations = []
    all_actions = []
    all_rewards = []

    D3DQN_makespan_temp = []
    average_completions = []
    average_slowdowns = []
    trajectories = []

    job_makespan_temp = []
    job_cost_temp = []
    job_computation_temp = []
    job_transmission_temp = []
    job_energy_consumption_temp = []
    job_reward_temp = []
    # D3DQN
    for i in range(12):
        D3DQN_agent = D3DQN(agent_target,
                            agent_eval,
                            action_probability,
                            learning_steps,
                            itr + 1,
                            last_loss,
                            machines_number,
                            reward_giver,
                            features_extract_func=features_extract_func,
                            features_normalize_func=features_normalize_func)

        episode = Episode(machine_configs, jobs_configs, D3DQN_agent, None)
        D3DQN_agent.reward_giver.attach(episode.simulation)
        episode.run()
        trajectories.append(episode.simulation.scheduler.algorithm.current_trajectory)
        # ******完成时间makespans再加上传输时间、能耗等****** #
        sum_of_cost = 0.0
        sum_of_computation = 0.0
        sum_of_transmission = 0.0
        sum_of_energy_consumption = 0.0
        sum_of_reward = 0

        for node in episode.simulation.scheduler.algorithm.current_trajectory:
            sum_of_cost += node.wighted_cost
            sum_of_computation += node.computation_latency
            sum_of_transmission += node.transmission_latency
            sum_of_energy_consumption += node.energy_consumption
            sum_of_reward += node.reward
        D3DQN_makespan_temp.append(episode.simulation.env.now + sum_of_cost)
        job_cost_temp.append(sum_of_cost)
        job_computation_temp.append(sum_of_computation)
        job_transmission_temp.append(sum_of_transmission)
        job_energy_consumption_temp.append(sum_of_energy_consumption)
        job_reward_temp.append(sum_of_reward)
        job_makespan_temp.append(episode.simulation.env.now)
        # ********************结束********************** #
        average_completions.append(average_completion(episode))
        average_slowdowns.append(average_slowdown(episode))
        agent_eval.log('makespan', D3DQN_makespan_temp, agent_eval.global_step)
        agent_eval.log('average_completions', average_completions, agent_eval.global_step)
        agent_eval.log('average_slowdowns', average_slowdowns, agent_eval.global_step)
    D3DQN_makespan.append(np.mean(D3DQN_makespan_temp))
    D3DQN_job_cost.append(np.mean(job_cost_temp))
    D3DQN_job_computation.append(np.mean(job_computation_temp))
    D3DQN_job_transmission.append(np.mean(job_transmission_temp))
    D3DQN_job_energy_consumption.append(np.mean(job_energy_consumption_temp))
    D3DQN_job_makespan.append(np.mean(job_makespan_temp))
    D3DQN_job_reward.append(np.mean(job_reward_temp))

    for trajectory in trajectories:
        observations = []
        actions = []
        rewards = []
        for node in trajectory:
            observations.append(node.observation)
            actions.append(node.action)
            rewards.append(node.reward)
        all_observations.append(observations)
        all_actions.append(actions)
        all_rewards.append(rewards)
    losses_value = agent_eval.update_parameters(all_observations, all_actions, all_rewards)

    D3DQN_loss.append(np.mean(losses_value))

    # 优先级经验抽样机制
    prior_samples_buffer = prior_samples_buffer + agent_eval.priority_sample(all_observations,
                                                                             all_actions,
                                                                             all_rewards,
                                                                             losses_value,
                                                                             itr + 1,
                                                                             learning_steps,
                                                                             go_to_sample=True)

    if (itr + 1) % 200 == 0 and itr > 0:
        for t, e in zip(agent_target.brain.variables, agent_eval.brain.variables):
            tf.assign(t, e)
        print('*****Set target_net_parameters*****')

    last_loss_array.append(np.mean(losses_value))
    # last_loss = max(losses_value)
    if (itr + 1) % 100 == 0 and itr > 0:
        last_loss = np.mean(last_loss_array)
        last_loss_array = []

    print('Loss:', np.mean(losses_value))

agent_target.save()
agent_eval.save()

D3DQN_makespan = pd.DataFrame(data=D3DQN_makespan)
D3DQN_makespan.to_csv('../jobs_files/D3DQN.csv', index=False,
                      header=['D3DQN_makespan_sum'])
temp_reader = pd.read_csv('../jobs_files/D3DQN.csv')
temp_reader['D3DQN_makespan'] = D3DQN_job_makespan
temp_reader['D3DQN_cost'] = D3DQN_job_cost
temp_reader['D3DQN_computation'] = D3DQN_job_computation
temp_reader['D3DQN_transmission'] = D3DQN_job_transmission
temp_reader['D3DQN_energy_consumption'] = D3DQN_job_energy_consumption
temp_reader['D3DQN_reward'] = D3DQN_job_reward
temp_reader['D3DQN_loss'] = D3DQN_loss
temp_reader.to_csv('../jobs_files/D3DQN.csv', index=False)

