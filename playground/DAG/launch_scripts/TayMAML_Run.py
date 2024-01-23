import os
import sys
import time

import numpy as np
import tensorflow as tf
import pandas as pd
import random

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
learning_steps = 2000
learning_steps_application = 2000
action_probability = 1
tasks_num = 10
theta = 0.000314
jobs_start_index = 0
jobs_len = 4
jobs_start_index_application = 0
jobs_len_application = 8

jobs_csv_metaset1_train = '../jobs_files/TaskStructure[alpha=1.0]_train.csv'
jobs_csv_metaset2_train = '../jobs_files/TaskStructure[alpha=2.0]_train.csv'
jobs_csv_metaset3_train = '../jobs_files/CPU[0.5,1.0]Memory[0,1]Disk[0,1]_train.csv'
jobs_csv_metaset4_train = '../jobs_files/CPU[1.5,2.0]Memory[1,2]Disk[1,2]_train.csv'
jobs_csv_metaset1_test = '../jobs_files/TaskStructure[alpha=1.0]_test.csv'
jobs_csv_metaset2_test = '../jobs_files/TaskStructure[alpha=2.0]_test.csv'
jobs_csv_metaset3_test = '../jobs_files/CPU[0.5,1.0]Memory[0,1]Disk[0,1]_test.csv'
jobs_csv_metaset4_test = '../jobs_files/CPU[1.5,2.0]Memory[1,2]Disk[1,2]_test.csv'
jobs_csv_application1 = '../jobs_files/TayMAML_Application_Dataset.csv'

brain_target = BrainSmall(16, machines_number_local + machines_number_edge)
brain_eval = BrainSmall(16, machines_number_local + machines_number_edge)

brain_f_norm = BrainSmall(16, machines_number_local + machines_number_edge)

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

agent_f_norm = Agent(name, brain_f_norm, 1, reward_to_go=True, nn_baseline=True, normalize_advantages=True,
                     model_save_path='%s/model.ckpt' % model_dir)

machine_configs = [MachineConfig(4, 4, 4, [False, 0, 1]) for i in range(machines_number_local)] \
                  + [MachineConfig(16, 16, 16, [True, 1, 1]) for i in range(machines_number_edge)]

meta_csv_reader1_train = CSVReader(jobs_csv_metaset1_train)
meta_csv_reader2_train = CSVReader(jobs_csv_metaset2_train)
meta_csv_reader3_train = CSVReader(jobs_csv_metaset3_train)
meta_csv_reader4_train = CSVReader(jobs_csv_metaset4_train)
meta_csv_reader1_test = CSVReader(jobs_csv_metaset1_test)
meta_csv_reader2_test = CSVReader(jobs_csv_metaset2_test)
meta_csv_reader3_test = CSVReader(jobs_csv_metaset3_test)
meta_csv_reader4_test = CSVReader(jobs_csv_metaset4_test)
meta_csv_reader1_application = CSVReader(jobs_csv_application1)

prior_samples_buffer = []
prior_samples_buffer_application = []
last_loss = 100
last_loss_array = []
last_loss_array_application = []
D3DQN_loss = []
loss_application = []

D3DQN_makespan = []
D3DQN_job_makespan = []
D3DQN_job_cost = []
D3DQN_job_computation = []
D3DQN_job_transmission = []
D3DQN_job_energy_consumption = []
D3DQN_job_reward = []

makespan_application = []
job_makespan_application = []
job_cost_application = []
job_computation_application = []
job_transmission_application = []
job_energy_consumption_application = []
job_reward_application = []

meta_loss1 = []
meta_loss2 = []
meta_loss3 = []
meta_loss4 = []

meta_losses_array = []
meta_running_time = []
meta_cpu_usage_ratio = [cpu_percent(0.1)]
meta_memory_usage_ratio = [virtual_memory()[2]]

jobs_len_test1 = 0
jobs_len_test2 = 0
jobs_len_test3 = 0
jobs_len_test4 = 0
jobs_start_index_test1 = 0
jobs_start_index_test2 = 0
jobs_start_index_test3 = 0
jobs_start_index_test4 = 0
start_time = time.time()

variables_old = []

for itr in range(learning_steps):
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

    # meta
    jobs_configs1_train = meta_csv_reader1_train.generate(jobs_start_index, jobs_len)
    jobs_configs2_train = meta_csv_reader2_train.generate(jobs_start_index, jobs_len)
    jobs_configs3_train = meta_csv_reader3_train.generate(jobs_start_index, jobs_len)
    jobs_configs4_train = meta_csv_reader4_train.generate(jobs_start_index, jobs_len)
    jobs_start_index += jobs_len

    print("********** Iteration %i ************" % (itr + 1))
    # meta
    all_observations1 = []
    all_observations2 = []
    all_observations3 = []
    all_observations4 = []
    all_actions1 = []
    all_actions2 = []
    all_actions3 = []
    all_actions4 = []
    all_rewards1 = []
    all_rewards2 = []
    all_rewards3 = []
    all_rewards4 = []

    D3DQN_makespan_temp = []
    average_completions = []
    average_slowdowns = []

    # meta
    trajectories_1 = []
    trajectories_2 = []
    trajectories_3 = []
    trajectories_4 = []

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

        # episode = Episode(machine_configs, jobs_configs1, D3DQN_agent, None)

        # meta
        episode1 = Episode(machine_configs, jobs_configs1_train, D3DQN_agent, None)
        episode2 = Episode(machine_configs, jobs_configs2_train, D3DQN_agent, None)
        episode3 = Episode(machine_configs, jobs_configs3_train, D3DQN_agent, None)
        episode4 = Episode(machine_configs, jobs_configs4_train, D3DQN_agent, None)

        # D3DQN_agent.reward_giver.attach(episode.simulation)

        # meta
        D3DQN_agent.reward_giver.attach(episode1.simulation)
        D3DQN_agent.reward_giver.attach(episode2.simulation)
        D3DQN_agent.reward_giver.attach(episode3.simulation)
        D3DQN_agent.reward_giver.attach(episode4.simulation)

        # episode.run()

        # meta
        episode1.run()
        episode2.run()
        episode3.run()
        episode4.run()

        # trajectories.append(episode.simulation.scheduler.algorithm.current_trajectory)

        # meta
        trajectories_1.append(episode1.simulation.scheduler.algorithm.current_trajectory)
        trajectories_2.append(episode2.simulation.scheduler.algorithm.current_trajectory)
        trajectories_3.append(episode3.simulation.scheduler.algorithm.current_trajectory)
        trajectories_4.append(episode4.simulation.scheduler.algorithm.current_trajectory)

        # meta
        # episode1
        sum_of_cost = 0.0
        sum_of_computation = 0.0
        sum_of_transmission = 0.0
        sum_of_energy_consumption = 0.0
        sum_of_reward = 0
        for node in episode1.simulation.scheduler.algorithm.current_trajectory:
            sum_of_cost += node.wighted_cost
            sum_of_computation += node.computation_latency
            sum_of_transmission += node.transmission_latency
            sum_of_energy_consumption += node.energy_consumption
            sum_of_reward += node.reward
        D3DQN_makespan_temp.append(episode1.simulation.env.now + sum_of_cost)
        job_cost_temp.append(sum_of_cost)
        job_computation_temp.append(sum_of_computation)
        job_transmission_temp.append(sum_of_transmission)
        job_energy_consumption_temp.append(sum_of_energy_consumption)
        job_reward_temp.append(sum_of_reward)
        job_makespan_temp.append(episode1.simulation.env.now)
        # episode2
        sum_of_cost = 0.0
        sum_of_computation = 0.0
        sum_of_transmission = 0.0
        sum_of_energy_consumption = 0.0
        sum_of_reward = 0
        for node in episode2.simulation.scheduler.algorithm.current_trajectory:
            sum_of_cost += node.wighted_cost
            sum_of_computation += node.computation_latency
            sum_of_transmission += node.transmission_latency
            sum_of_energy_consumption += node.energy_consumption
            sum_of_reward += node.reward
        D3DQN_makespan_temp.append(episode2.simulation.env.now + sum_of_cost)
        job_cost_temp.append(sum_of_cost)
        job_computation_temp.append(sum_of_computation)
        job_transmission_temp.append(sum_of_transmission)
        job_energy_consumption_temp.append(sum_of_energy_consumption)
        job_reward_temp.append(sum_of_reward)
        job_makespan_temp.append(episode2.simulation.env.now)
        # episode3
        sum_of_cost = 0.0
        sum_of_computation = 0.0
        sum_of_transmission = 0.0
        sum_of_energy_consumption = 0.0
        sum_of_reward = 0
        for node in episode3.simulation.scheduler.algorithm.current_trajectory:
            sum_of_cost += node.wighted_cost
            sum_of_computation += node.computation_latency
            sum_of_transmission += node.transmission_latency
            sum_of_energy_consumption += node.energy_consumption
            sum_of_reward += node.reward
        D3DQN_makespan_temp.append(episode3.simulation.env.now + sum_of_cost)
        job_cost_temp.append(sum_of_cost)
        job_computation_temp.append(sum_of_computation)
        job_transmission_temp.append(sum_of_transmission)
        job_energy_consumption_temp.append(sum_of_energy_consumption)
        job_reward_temp.append(sum_of_reward)
        job_makespan_temp.append(episode3.simulation.env.now)
        # episode4
        sum_of_cost = 0.0
        sum_of_computation = 0.0
        sum_of_transmission = 0.0
        sum_of_energy_consumption = 0.0
        sum_of_reward = 0
        for node in episode4.simulation.scheduler.algorithm.current_trajectory:
            sum_of_cost += node.wighted_cost
            sum_of_computation += node.computation_latency
            sum_of_transmission += node.transmission_latency
            sum_of_energy_consumption += node.energy_consumption
            sum_of_reward += node.reward
        D3DQN_makespan_temp.append(episode4.simulation.env.now + sum_of_cost)
        job_cost_temp.append(sum_of_cost)
        job_computation_temp.append(sum_of_computation)
        job_transmission_temp.append(sum_of_transmission)
        job_energy_consumption_temp.append(sum_of_energy_consumption)
        job_reward_temp.append(sum_of_reward)
        job_makespan_temp.append(episode4.simulation.env.now)

        average_completions.append(average_completion(episode1) + average_completion(episode2) +
                                   average_completion(episode3) + average_completion(episode4))
        average_slowdowns.append(average_slowdown(episode1) + average_slowdown(episode2) +
                                 average_slowdown(episode3) + average_slowdown(episode4))

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

    if itr % 20 == 0:
        variables_old = agent_eval.brain.variables

    # meta
    # trajectories1
    for trajectory in trajectories_1:
        observations = []
        actions = []
        rewards = []
        for node in trajectory:
            observations.append(node.observation)
            actions.append(node.action)
            rewards.append(node.reward)
        all_observations1.append(observations)
        all_actions1.append(actions)
        all_rewards1.append(rewards)
    losses_value1 = agent_eval.update_parameters(all_observations1, all_actions1, all_rewards1)
    # trajectories2
    for trajectory in trajectories_2:
        observations = []
        actions = []
        rewards = []
        for node in trajectory:
            observations.append(node.observation)
            actions.append(node.action)
            rewards.append(node.reward)
        all_observations2.append(observations)
        all_actions2.append(actions)
        all_rewards2.append(rewards)
    losses_value2 = agent_eval.update_parameters(all_observations2, all_actions2, all_rewards2)
    # trajectories3
    for trajectory in trajectories_3:
        observations = []
        actions = []
        rewards = []
        for node in trajectory:
            observations.append(node.observation)
            actions.append(node.action)
            rewards.append(node.reward)
        all_observations3.append(observations)
        all_actions3.append(actions)
        all_rewards3.append(rewards)
    losses_value3 = agent_eval.update_parameters(all_observations3, all_actions3, all_rewards3)
    # trajectories4
    for trajectory in trajectories_4:
        observations = []
        actions = []
        rewards = []
        for node in trajectory:
            observations.append(node.observation)
            actions.append(node.action)
            rewards.append(node.reward)
        all_observations4.append(observations)
        all_actions4.append(actions)
        all_rewards4.append(rewards)
    losses_value4 = agent_eval.update_parameters(all_observations4, all_actions4, all_rewards4)

    meta_loss1 += losses_value1
    meta_loss2 += losses_value2
    meta_loss3 += losses_value3
    meta_loss4 += losses_value4

    D3DQN_loss.append((np.mean(losses_value1) + np.mean(losses_value2) +
                       np.mean(losses_value3) + np.mean(losses_value4)) / 4)

    prior_samples_buffer = prior_samples_buffer + agent_eval.priority_sample(all_observations1 + all_observations2 +
                                                                             all_observations3 + all_observations4,
                                                                             all_actions1 + all_actions2 +
                                                                             all_actions3 + all_actions4,
                                                                             all_rewards1 + all_rewards2 +
                                                                             all_rewards3 + all_rewards4,
                                                                             losses_value1 + losses_value2 +
                                                                             losses_value3 + losses_value4,
                                                                             itr + 1,
                                                                             learning_steps,
                                                                             go_to_sample=True)


    if (itr + 1) % 20 == 0 and itr > 0:
        print('*****Starting meta-update*****')
        NumOfTestExamples1 = int((np.mean(meta_loss1) / (np.mean(meta_loss1) + np.mean(meta_loss2) +
                                                         np.mean(meta_loss3) + np.mean(meta_loss4))) * 80)
        NumOfTestExamples2 = int((np.mean(meta_loss2) / (np.mean(meta_loss1) + np.mean(meta_loss2) +
                                                         np.mean(meta_loss3) + np.mean(meta_loss4))) * 80)
        NumOfTestExamples3 = int((np.mean(meta_loss3) / (np.mean(meta_loss1) + np.mean(meta_loss2) +
                                                         np.mean(meta_loss3) + np.mean(meta_loss4))) * 80)
        NumOfTestExamples4 = 80 - NumOfTestExamples1 - NumOfTestExamples2 - NumOfTestExamples3

        jobs_len_test1 = NumOfTestExamples1
        jobs_len_test2 = NumOfTestExamples2
        jobs_len_test3 = NumOfTestExamples3
        jobs_len_test4 = NumOfTestExamples4
        jobs_configs1_test = meta_csv_reader1_test.generate(jobs_start_index_test1, jobs_len_test1)
        jobs_configs2_test = meta_csv_reader2_test.generate(jobs_start_index_test2, jobs_len_test2)
        jobs_configs3_test = meta_csv_reader3_test.generate(jobs_start_index_test2, jobs_len_test3)
        jobs_configs4_test = meta_csv_reader4_test.generate(jobs_start_index_test2, jobs_len_test4)
        jobs_configs = jobs_configs1_test + jobs_configs2_test + jobs_configs3_test + jobs_configs4_test
        
        jobs_configs_f_norm = jobs_configs1_test[:random.randint(1, 4)] + jobs_configs2_test[:random.randint(1, 4)] + \
                              jobs_configs3_test[:random.randint(1, 4)] + jobs_configs4_test[:random.randint(1, 4)]
        jobs_start_index_test1 += jobs_len_test1
        jobs_start_index_test2 += jobs_len_test2
        jobs_start_index_test3 += jobs_len_test3
        jobs_start_index_test4 += jobs_len_test4

        all_observations = []
        all_actions = []
        all_rewards = []
        trajectories = []
        
        all_observations_f_norm = []
        all_actions_f_norm = []
        all_rewards_f_norm = []
        trajectories_f_norm = []

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

        for i in range(2):
            D3DQN_agent_f_norm = D3DQN(agent_target,
                                       agent_f_norm,
                                       action_probability,
                                       learning_steps,
                                       itr + 1,
                                       last_loss,
                                       machines_number,
                                       reward_giver,
                                       features_extract_func=features_extract_func,
                                       features_normalize_func=features_normalize_func)
            episode_f_norm = Episode(machine_configs, jobs_configs_f_norm, D3DQN_agent_f_norm, None)
            D3DQN_agent_f_norm.reward_giver.attach(episode_f_norm.simulation)
            episode_f_norm.run()
            trajectories_f_norm.append(episode_f_norm.simulation.scheduler.algorithm.current_trajectory)

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

        for trajectory in trajectories_f_norm:
            observations = []
            actions = []
            rewards = []
            for node in trajectory:
                observations.append(node.observation)
                actions.append(node.action)
                rewards.append(node.reward)
            all_observations_f_norm.append(observations)
            all_actions_f_norm.append(actions)
            all_rewards_f_norm.append(rewards)
        losses_value_f_norm = agent_f_norm.update_parameters(all_observations_f_norm, all_actions_f_norm,
                                                             all_rewards_f_norm)

        variables_train = agent_eval.brain.variables
        variables_test = agent_f_norm.brain.variables
        subtract = [p1 - p2 for p1, p2 in zip(variables_train, variables_test)]

        print("***********F-norm***********")

        def calculate_f_norm(parameters):
            f_norm = 0
            for parameter in parameters:
                param_array = parameter.numpy()
                f_norm += np.sum(np.square(param_array))
            return np.sqrt(f_norm)

        difference_f_norm = calculate_f_norm(subtract)
        print('F-norm:', difference_f_norm)

        # TayMAML
        variables_new = agent_eval.brain.variables
        meta_losses, cpu_usage_ratio, memory_usage_ratio = \
            agent_eval.meta_update_parameters(all_observations, all_actions, all_rewards,
                                              variables_old, variables_new, theta, difference_f_norm)
 
        end_time = time.time()
        meta_running_time.append(end_time - start_time)
        print('Meta_Loss:', np.mean(meta_losses))
        print('Meta_Running_time:', end_time - start_time)
        print('Mean_Cpu_Usage_Ration:', np.mean(cpu_usage_ratio))
        print('Mean_Memory_Usage_Ratio:', np.mean(memory_usage_ratio))
        meta_losses_array.append(np.mean(meta_losses))
        meta_cpu_usage_ratio.append(np.mean(cpu_usage_ratio))
        meta_memory_usage_ratio.append(np.mean(memory_usage_ratio))

        start_time = time.time()

    if (itr + 1) % 200 == 0 and itr > 0:
        for t, e in zip(agent_target.brain.variables, agent_eval.brain.variables):
            tf.assign(t, e)
        print('*****Set target_net_parameters*****')

    last_loss_array.append((np.mean(losses_value1) + np.mean(losses_value2) +
                            np.mean(losses_value3) + np.mean(losses_value4)) / 4)

    if (itr + 1) % 100 == 0 and itr > 0:
        last_loss = np.mean(last_loss_array)
        last_loss_array = []

    print('Loss:', (np.mean(losses_value1) + np.mean(losses_value2) +
                    np.mean(losses_value3) + np.mean(losses_value4)) / 4)

    if (itr + 1) == learning_steps:
        for itr_application in range(learning_steps_application):
            if itr_application % 50 == 0 and itr_application > 0:
                print('*****Learning previous experience with higher priority*****')
                length = len(prior_samples_buffer_application) / tasks_num
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
                        observations.append(prior_samples_buffer_application[counter + i][0])
                        actions.append(prior_samples_buffer_application[counter + i][1])
                        rewards.append(prior_samples_buffer_application[counter + i][2])
                    sample_observations.append(observations)
                    sample_actions.append(actions)
                    sample_rewards.append(rewards)
                agent_eval.priority_sample(sample_observations, sample_actions, sample_rewards, None, 0, 0, False)
                prior_samples_buffer_application = []

            jobs_configs1_application = meta_csv_reader1_application.generate(jobs_start_index_application,
                                                                              jobs_len_application)
            jobs_start_index_application += jobs_len_application
            print("********** Application_Iteration %i ************" % (itr_application + 1))
            all_observations = []
            all_actions = []
            all_rewards = []
            makespan_temp = []
            average_completions = []
            average_slowdowns = []
            trajectories = []
            job_makespan_temp = []
            job_cost_temp = []
            job_computation_temp = []
            job_transmission_temp = []
            job_energy_consumption_temp = []
            job_reward_temp = []
            for i in range(12):
                D3DQN_agent = D3DQN(agent_target,
                                    agent_eval,
                                    action_probability,
                                    learning_steps_application,
                                    itr_application + 1,
                                    last_loss,
                                    machines_number,
                                    reward_giver,
                                    features_extract_func=features_extract_func,
                                    features_normalize_func=features_normalize_func)

                episode = Episode(machine_configs, jobs_configs1_application, D3DQN_agent, None)
                D3DQN_agent.reward_giver.attach(episode.simulation)
                episode.run()
                trajectories.append(episode.simulation.scheduler.algorithm.current_trajectory)

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
                makespan_temp.append(episode.simulation.env.now + sum_of_cost)
                job_cost_temp.append(sum_of_cost)
                job_computation_temp.append(sum_of_computation)
                job_transmission_temp.append(sum_of_transmission)
                job_energy_consumption_temp.append(sum_of_energy_consumption)
                job_reward_temp.append(sum_of_reward)
                job_makespan_temp.append(episode.simulation.env.now)
                
                average_completions.append(average_completion(episode))
                average_slowdowns.append(average_slowdown(episode))
                agent_eval.log('makespan', makespan_temp, agent_eval.global_step)
                agent_eval.log('average_completions', average_completions, agent_eval.global_step)
                agent_eval.log('average_slowdowns', average_slowdowns, agent_eval.global_step)

            makespan_application.append(np.mean(makespan_temp))
            job_cost_application.append(np.mean(job_cost_temp))
            job_computation_application.append(np.mean(job_computation_temp))
            job_transmission_application.append(np.mean(job_transmission_temp))
            job_energy_consumption_application.append(np.mean(job_energy_consumption_temp))
            job_makespan_application.append(np.mean(job_makespan_temp))
            job_reward_application.append(np.mean(job_reward_temp))

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
            losses_value = agent_eval.update_parameters_application(all_observations, all_actions, all_rewards)

            loss_application.append(np.mean(losses_value))

            prior_samples_buffer_application = \
                prior_samples_buffer_application + agent_eval.priority_sample(
                    all_observations, all_actions, all_rewards,
                    losses_value, itr_application + 1, learning_steps_application, go_to_sample=True)

            if (itr_application + 1) % 200 == 0 and itr_application > 0:
                for t, e in zip(agent_target.brain.variables, agent_eval.brain.variables):
                    tf.assign(t, e)
                print('*****Set target_net_parameters*****')

            last_loss_array_application.append(np.mean(losses_value))

            # last_loss = max(losses_value)
            if (itr_application + 1) % 100 == 0 and itr_application > 0:
                last_loss = np.mean(last_loss_array_application)
                last_loss_array_application = []

            print('Loss:', np.mean(losses_value))

agent_target.save()
agent_eval.save()

D3DQN_makespan = pd.DataFrame(data=D3DQN_makespan)
D3DQN_makespan.to_csv('../jobs_files/TayMAML_in.csv', index=False,
                      header=['D3DQN_makespan_sum'])
temp_reader = pd.read_csv('../jobs_files/TayMAML_in.csv')
temp_reader['D3DQN_makespan'] = D3DQN_job_makespan
temp_reader['D3DQN_cost'] = D3DQN_job_cost
temp_reader['D3DQN_computation'] = D3DQN_job_computation
temp_reader['D3DQN_transmission'] = D3DQN_job_transmission
temp_reader['D3DQN_energy_consumption'] = D3DQN_job_energy_consumption
temp_reader['D3DQN_reward'] = D3DQN_job_reward
temp_reader['D3DQN_loss'] = D3DQN_loss
temp_reader.to_csv('../jobs_files/TayMAML_in.csv', index=False)

meta_losses_array = pd.DataFrame(data=meta_losses_array)
meta_losses_array.to_csv('../jobs_files/TayMAML_out.csv', index=False,
                         header=['Meta_loss'])
temp_reader = pd.read_csv('../jobs_files/TayMAML_out.csv')
temp_reader['Meta_running_time'] = meta_running_time
temp_reader.to_csv('../jobs_files/TayMAML_out.csv', index=False)

meta_cpu_usage_ratio = pd.DataFrame(data=meta_cpu_usage_ratio)
meta_cpu_usage_ratio.to_csv('../jobs_files/TayMAML_CPU_And_Memory_Usage.csv', index=False,
                            header=['CPU'])
temp_reader = pd.read_csv('../jobs_files/TayMAML_CPU_And_Memory_Usage.csv')
temp_reader['Memory'] = meta_memory_usage_ratio
temp_reader.to_csv('../jobs_files/TayMAML_CPU_And_Memory_Usage.csv', index=False)

makespan_application = pd.DataFrame(data=makespan_application)
makespan_application.to_csv('../jobs_files/TayMAMLApplication.csv', index=False,
                            header=['makespan_sum'])
temp_reader = pd.read_csv('../jobs_files/TayMAMLApplication.csv')
temp_reader['makespan'] = job_makespan_application
temp_reader['cost'] = job_cost_application
temp_reader['computation'] = job_computation_application
temp_reader['transmission'] = job_transmission_application
temp_reader['energy_consumption'] = job_energy_consumption_application
temp_reader['reward'] = job_reward_application
temp_reader['loss'] = loss_application
temp_reader.to_csv('../jobs_files/TayMAMLApplication.csv', index=False)

