import random
import time
import copy
import numpy as np
import tensorflow as tf
from psutil import *


class Agent(object):
    def __init__(self, name, brain, gamma, reward_to_go, nn_baseline, normalize_advantages, model_save_path=None,
                 summary_path=None):
        super().__init__()

        self.gamma = gamma
        self.reward_to_go = reward_to_go
        self.baseline = nn_baseline
        self.normalize_advantages = normalize_advantages

        self.alpha = 0.001
        self.beta = 0.0001

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha)
        self.global_step = tf.train.get_or_create_global_step()
        self.summary_path = summary_path if summary_path is not None else './tensorboard/%s--%s' % (
            name, time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))
        self.summary_writer = tf.contrib.summary.create_file_writer(self.summary_path)
        self.brain = brain
        self.checkpoint = tf.train.Checkpoint(brain=self.brain)
        self.model_save_path = model_save_path

    def restore(self, model_path):
        self.checkpoint.restore(model_path)

    def save(self):
        self.checkpoint.save(self.model_save_path)

    def log(self, name, loss_value, step):
        with self.summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar(name, loss_value, step=step)

    def _sum_of_rewards(self, rewards_n):
        q_s = []
        for re in rewards_n:
            q = []
            cur_q = 0
            for reward in reversed(re):
                cur_q = cur_q * self.gamma + reward
                q.append(cur_q)
            q = list(reversed(q))
            q_s.append(q)

        if self.reward_to_go:
            return q_s
        else:
            q_n = []
            for q in q_s:
                q_n.append([q[0]] * len(q))
            return q_n

    def _compute_advantage(self, q_n):
        if self.baseline:
            adv_n = copy.deepcopy(q_n)
            max_length = max([len(adv) for adv in adv_n])
            for adv in adv_n:
                while len(adv) < max_length:
                    adv.append(0.0)
            adv_n = np.array(adv_n)
            adv_n = adv_n - adv_n.mean(axis=0)
            adv_n__ = []
            for i in range(adv_n.shape[0]):
                original_length = len(q_n[i])
                adv_n__.append(list(adv_n[i][:original_length]))
            return adv_n__
        else:
            adv_n = q_n.copy()
            return adv_n

    def estimate_return(self, rewards_n):
        q_n = self._sum_of_rewards(rewards_n)
        adv_n = self._compute_advantage(q_n)

        # Advantage Normalization
        if self.normalize_advantages:
            # On the next line, implement a trick which is known empirically to reduce variance
            # in policy gradient methods: normalize adv_n to have mean zero and std=1.
            adv_s = []
            for advantages in adv_n:
                for advantage in advantages:
                    adv_s.append(advantage)
            adv_s = np.array(adv_s)
            mean = adv_s.mean()
            std = adv_s.std()
            adv_n__ = []
            for advantages in adv_n:
                advantages__ = []
                for advantage in advantages:
                    advantages__.append((advantage - mean) / (std + np.finfo(np.float32).eps))
                adv_n__.append(advantages__)
            adv_n = adv_n__
        return q_n, adv_n

    def _loss(self, X, y, adv):
        logits = self.brain(X)
        label = y[0]
        one_hot = []
        for i in range(5):
            if i == label:
                one_hot.append(1)
            else:
                one_hot.append(0)
        one_hot = tf.convert_to_tensor([one_hot])
        logprob = tf.losses.softmax_cross_entropy(onehot_labels=one_hot, logits=logits)
        return abs(logprob)

    def priority_sample(self, all_observations, all_actions, all_rewards, losses_value, steps_counter, learning_step,
                        go_to_sample):
        if go_to_sample:
            experience_pool = []
            for observations, actions, rewards in zip(all_observations, all_actions, all_rewards):
                for observation, action, reward in zip(observations, actions, rewards):
                    if observation is None or action is None:
                        continue
                    else:
                        experience_pool.append([observation, action, reward])
            counter = 0
            for example in experience_pool:
                example.append(losses_value[counter])
                counter += 1

            experience_pool.sort(key=lambda x: (x[3]))
            length = len(experience_pool)
            if steps_counter / learning_step <= 0.1:
                index_start = length * 0.45
                index_end = length * 0.55
            elif steps_counter / learning_step <= 0.3:
                index_start = length * 0.35
                index_end = length * 0.65
            elif steps_counter / learning_step <= 0.5:
                index_start = length * 0.25
                index_end = length * 0.75
            elif steps_counter / learning_step <= 0.7:
                index_start = length * 0.15
                index_end = length * 0.85
            else:
                index_start = length * 0.05
                index_end = length * 0.95

            start = int(index_start)
            end = int(index_end)
            while (end - start) % 10 != 0:
                start += 1
                end -= 1
            return experience_pool[round(start):round(end)]
        else:
            rewards_n, adv_n = self.estimate_return(all_rewards)
            for observations, actions, advantages in zip(all_observations, all_actions, adv_n):
                grads_by_trajectory = []
                cnt = 1
                for observation, action, advantage in zip(observations, actions, advantages):
                    if observation is None or action is None:
                        continue
                    with tf.GradientTape() as t:
                        loss_value = self._loss(observation, [action], advantage)
                    grads = t.gradient(loss_value, self.brain.variables)
                    grads_by_trajectory.append(grads)
                    if cnt % 1000 == 0:
                        self.optimize(grads_by_trajectory)
                        grads_by_trajectory = []
                    cnt += 1
                if len(grads_by_trajectory) > 0:
                    self.optimize(grads_by_trajectory)

    def optimize(self, grads_by_trajectory):
        average_grads = []
        for grads_by_layer in zip(*grads_by_trajectory):
            average_grads.append(np.array([grad.numpy() for grad in grads_by_layer]).mean(axis=0))

        assert len(average_grads) == len(self.brain.variables)
        for average_grad, variable in zip(average_grads, self.brain.variables):
            assert average_grad.shape == variable.shape

        self.optimizer.apply_gradients(zip(average_grads, self.brain.variables), self.global_step)

    def update_parameters(self, all_observations, all_actions, all_rewards):
        rewards_n, adv_n = self.estimate_return(all_rewards)
        losses_value = []
        advantages__ = []
        grads_by_trajectory = []
        for observations, actions, advantages in zip(all_observations, all_actions, adv_n):
            for observation, action, advantage in zip(observations, actions, advantages):
                if observation is None or action is None:
                    continue
                with tf.GradientTape() as t:
                    loss_value = self._loss(observation, [action], advantage)
                    grads = t.gradient(loss_value, self.brain.variables)
                    grads_by_trajectory.append(grads)
                losses_value.append(loss_value)
                advantages__.append(advantage)

        self.optimize(grads_by_trajectory)

        self.log('loss', np.mean(losses_value), self.global_step)
        self.log('advantage', np.mean(advantages__), self.global_step)

        return losses_value

    def meta_update_parameters(self, all_observations, all_actions, all_rewards,
                               variables_old, variables_new, theta, difference_f_norm):
        meta_losses_array = []
        cpu_usage_ratio = []
        memory_usage_ratio = []

        rewards_n, adv_n = self.estimate_return(all_rewards)
        grads_array = []
        for observations, actions, advantages in zip(all_observations, all_actions, adv_n):
            for observation, action, advantage in zip(observations, actions, advantages):
                if observation is None or action is None:
                    continue

                for t, e in zip(self.brain.variables, variables_old):
                    tf.assign(t, e + theta)
                output_add = self.brain(observation)
                loss_add = tf.losses.sparse_softmax_cross_entropy(labels=[action], logits=output_add)

                for t, e in zip(self.brain.variables, variables_old):
                    tf.assign(t, e - 2 * theta)
                output_minus = self.brain(observation)
                loss_minus = tf.losses.sparse_softmax_cross_entropy(labels=[action], logits=output_minus)

                for t, e in zip(self.brain.variables, variables_old):
                    tf.assign(t, e + theta)
                output = self.brain(observation)
                loss = tf.losses.sparse_softmax_cross_entropy(labels=[action], logits=output)

                coefficient = (loss_add + loss_minus - 2 * loss) / theta ** 2
                coefficient_grad = 1 - self.alpha * coefficient

                cpu_usage_ratio.append(cpu_percent())
                memory_usage_ratio.append(virtual_memory()[2])

                with tf.GradientTape() as t1:
                    for t, e in zip(self.brain.variables, variables_new):
                        tf.assign(t, e)
                    loss_value_true = self._loss(observation, [action], advantage)
                    loss_value = self._loss(observation, [action], advantage) + (0.0001 * difference_f_norm)
                    for t, e in zip(self.brain.variables, variables_new):
                        tf.assign(t, e)
                    grads = t1.gradient(loss_value, self.brain.variables)
                    grads_temp = []
                    for items in grads:
                        temp = []
                        for item in items:
                            temp.append(item.numpy() * coefficient_grad)
                        grads_temp.append(tf.convert_to_tensor(temp))
                    grads = grads_temp
                    grads_array.append(grads)
                meta_losses_array.append(loss_value_true)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.beta)
        for t, e in zip(self.brain.variables, variables_old):
            tf.assign(t, e)
        self.optimize(grads_array)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha)

        return meta_losses_array, cpu_usage_ratio, memory_usage_ratio

    def update_parameters_application(self, all_observations, all_actions, all_rewards):
        self.alpha = 0.0001
        rewards_n, adv_n = self.estimate_return(all_rewards)
        losses_value = []
        advantages__ = []
        grads_by_trajectory = []
        for observations, actions, advantages in zip(all_observations, all_actions, adv_n):
            for observation, action, advantage in zip(observations, actions, advantages):
                if observation is None or action is None:
                    continue
                with tf.GradientTape() as t:
                    loss_value = self._loss(observation, [action], advantage)
                    grads = t.gradient(loss_value, self.brain.variables)
                    grads_by_trajectory.append(grads)
                losses_value.append(loss_value)
                advantages__.append(advantage)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha)
        self.optimize(grads_by_trajectory)

        self.log('loss', np.mean(losses_value), self.global_step)
        self.log('advantage', np.mean(advantages__), self.global_step)

        return losses_value
