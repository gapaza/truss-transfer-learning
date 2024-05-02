import numpy as np
import time
import tensorflow as tf
from copy import deepcopy
import matplotlib.gridspec as gridspec
import random
import json
from tqdm import tqdm
import config
import matplotlib.pyplot as plt
import os
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from task.AbstractTask import AbstractTask
from problem.TrussDesign import TrussDesign as Design
import scipy.signal
from task.GA_Task import GA_Task
from model import get_multi_task_decoder as get_model
from collections import OrderedDict
import tensorflow_addons as tfa


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]



num_weight_samples = 4  # 4
num_task_samples = 1  # 1
repeat_size = 3  # 3
global_mini_batch_size = num_weight_samples * num_task_samples * repeat_size

conditioning_vars = config.num_conditioning_vars

freeze_actor = False
freeze_critic = False

plot_freq = 50

uniform_sampling = True
bimodal_sampling = False
bimodal_alternating_sampling = False

run_dir = 20
run_num = 1
task_epochs = 200
val_task = 0

random.seed(0)
tf.random.set_seed(0)

use_actor_warmup = True


class PPO_MultiTaskValidation(AbstractTask):

    def __init__(
            self,
            run_num=0,
            barrier=None,
            problem=None,
            epochs=50,
            actor_load_path=None,
            critic_load_path=None,
            debug=False,
            c_type='uniform',
            run_val=False,
            val_itr=0,
            val_task=0,
    ):
        super(PPO_MultiTaskValidation, self).__init__(run_num, barrier, problem, epochs, actor_load_path, critic_load_path)
        self.debug = debug
        self.c_type = c_type
        self.run_val = run_val
        self.val_itr = val_itr
        self.val_task = val_task

        # HV
        self.ref_point = np.array(config.hv_ref_point)  # value, weight
        self.hv_client = HV(self.ref_point)
        self.nds = NonDominatedSorting()
        self.unique_designs = set()
        self.unique_designs_vals = []
        self.unique_designs_in_window = []

        # Algorithm parameters
        self.pop_size = 30  # 32 FU_NSGA2, 10 U_NSGA2
        self.offspring_size = global_mini_batch_size  # 32 FU_NSGA2, 30 U_NSGA2
        self.mini_batch_size = global_mini_batch_size
        self.nfe = 0
        self.epochs = epochs
        self.steps_per_design = config.num_vars  # 30 | 60

        # PPO alg parameters
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_ratio = 0.2  # was 0.2
        self.target_kl = 0.001  # was 0.01
        self.entropy_coef = 0.02  # was 0.02 originally
        self.counter = 0
        self.decision_start_token_id = 1
        self.num_actions = 2
        self.curr_epoch = 0

        # Population
        self.population = []
        self.hv = []
        self.nfes = []

        # Results
        self.plot_freq = plot_freq

        # Objective Weights
        num_keys = 9
        # self.objective_weights = list(np.linspace(0.1, 0.9, num_keys))
        self.objective_weights = list(np.linspace(0.05, 0.95, num_keys))
        # self.objective_weights = [0.5]
        # self.objective_weights = list(np.linspace(0.0, 1.0, num_keys))

        # Tasks
        self.tasks = [i for i in range(len(self.problem.train_problems))]
        self.run_tasks = [val_task]  # Validation tasks

        # GA Comparison Data
        self.uniform_ga = self.init_comparison_data()
        # self.uniform_ga = []

    def init_comparison_data(self):
        print('--> RUNNING GA')
        task_runner = GA_Task(
            run_num=self.run_num,
            problem=problem,
            limit=100000,
            c_type='uniform',
            max_nfe=10000,
            problem_num=self.val_task,
            run_val=True,
        )
        performance = task_runner.run()
        return performance


    def build(self):

        # Optimizer parameters
        self.actor_learning_rate = 0.0001  # 0.0001
        self.critic_learning_rate = 0.0005  # 0.0001
        self.train_actor_iterations = 250  # was 250
        self.train_critic_iterations = 40  # was 40
        self.beta_1 = 0.9
        if self.run_val is False:
            self.beta_1 = 0.0

        if use_actor_warmup is True:
            self.actor_learning_rate = tf.keras.optimizers.schedules.CosineDecay(
                0.0,  # initial learning rate
                100000,  # decay_steps
                alpha=0.1,
                warmup_target=self.actor_learning_rate,
                warmup_steps=500
            )

        # Optimizers
        if self.actor_optimizer is None:
            self.actor_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.actor_learning_rate)
            # self.actor_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.actor_learning_rate, beta_1=self.beta_1)
            # self.actor_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=self.actor_learning_rate)
        if self.critic_optimizer is None:
            self.critic_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.critic_learning_rate)
            # self.critic_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.critic_learning_rate, beta_1=self.beta_1)
            # self.critic_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=self.critic_learning_rate)

        self.c_actor, self.c_critic = get_model(self.actor_load_path, self.critic_load_path)

        self.c_actor.summary()

    def run(self):
        self.build()

        for x in range(self.epochs):
            self.curr_epoch = x
            epoch_info = self.fast_mini_batch()
            self.prune_population()
            self.record(epoch_info)

        # Save final models
        t_actor_save_path = os.path.join(self.run_dir, 'actor_weights_val')
        t_critic_save_path = os.path.join(self.run_dir, 'critic_weights_val')
        self.c_actor.save_weights(t_actor_save_path)
        self.c_critic.save_weights(t_critic_save_path)

    # -------------------------------------
    # Population Functions
    # -------------------------------------

    def calc_pop_hv(self):
        objectives = self.eval_population()
        F = np.array(objectives)
        hv = self.hv_client.do(F)
        return hv

    def eval_population(self):
        evals = []
        for design in self.population:
            evals.append(design.evaluate())
        return evals

    def prune_population(self):

        # 1. Get Objectives
        objectives = self.eval_population()
        stiffness_ratios = [design.stiffness_ratio for design in self.population]
        in_stiffness_window = [design.is_feasible for design in self.population]

        # 2. Determine survivors
        fronts = self.custom_dominance_sorting(objectives, stiffness_ratios, in_stiffness_window)
        survivors = []
        exit_loop = False
        for k, front in enumerate(fronts, start=1):
            for idx in front:
                survivors.append(idx)
                if len(survivors) >= self.pop_size and k > 1:
                    exit_loop = True
                    break
            if exit_loop is True:
                break
        perished = [idx for idx in range(len(self.population)) if idx not in survivors]

        # 3. Update population (sometimes the first front is larger than pop_size)
        new_population = []
        for idx in survivors:
            new_population.append(self.population[idx])
        self.population = new_population

    def custom_dominance_sorting(self, objectives, deviations, in_stiffness_window):
        """
        Sort designs into Pareto fronts with custom dominance criteria.

        Parameters:
        - objectives: List of lists, where each sublist holds a design's vertical stiffness (negative value) and volume fraction (positive value).
        - deviations: List of design deviation values (positive values).

        Returns:
        - List of lists, where each list holds the indices of designs in that Pareto front.
        """
        num_designs = len(objectives)

        # Initialize dominance and front structures
        dominates = {i: set() for i in range(num_designs)}
        dominated_by = {i: set() for i in range(num_designs)}
        num_dominated = [0 for _ in range(num_designs)]
        fronts = [[] for _ in range(num_designs)]  # Worst case: each individual in its own front

        # Step 1: Determine dominance relationships
        for i in range(num_designs):
            for j in range(i + 1, num_designs):
                if self.is_dominant(i, j, objectives, deviations, in_stiffness_window):
                    dominates[i].add(j)
                    dominated_by[j].add(i)
                    num_dominated[j] += 1
                elif self.is_dominant(j, i, objectives, deviations, in_stiffness_window):
                    dominates[j].add(i)
                    dominated_by[i].add(j)
                    num_dominated[i] += 1

        # Step 2: Identify the first front
        current_front = []
        for i in range(num_designs):
            if num_dominated[i] == 0:
                current_front.append(i)
                fronts[0].append(i)

        # Step 3: Build subsequent fronts
        front_index = 0
        while current_front:
            next_front = []
            for i in current_front:
                for j in dominates[i]:
                    num_dominated[j] -= 1
                    if num_dominated[j] == 0:
                        next_front.append(j)
                        fronts[front_index + 1].append(j)
            front_index += 1
            current_front = next_front

        # Remove empty fronts
        fronts = [front for front in fronts if front]

        return fronts

    def is_dominant(self, i, j, objectives, deviations, in_stiffness_window):
        """
        Determine if design i dominates design j.

        Parameters:
        - i, j: Indices of the designs being compared.
        - objectives: List of lists of objectives.
        - deviations: List of deviation values.

        Returns:
        - True if i dominates j, False otherwise.
        """

        # Design A and Design B
        # - case: one design in stiffness window one not
        # - If design A is in stiffness window and design B is not, design A dominates design B
        if in_stiffness_window[i] is True and in_stiffness_window[j] is False:
            return True
        if in_stiffness_window[i] is False and in_stiffness_window[j] is True:
            return False

        # - case: both designs are not in stiffness window
        # If design A is and design B are not in stiffness window, determine dominance by stiffness ratio delta
        if in_stiffness_window[i] is False and in_stiffness_window[j] is False:
            if deviations[i] < deviations[j]:
                return True
            elif deviations[i] > deviations[j]:
                return False
            # Break ties with objective value dominance
            for obj_i, obj_j in zip(objectives[i], objectives[j]):
                if obj_i > obj_j:  # Since lower values are better, i does not dominate j if any of i's objectives are worse
                    return False
            return True

        # - case: both designs are in stiffness window
        # If design A and design B are in stiffness ratio window, determine dominance by objective values
        if in_stiffness_window[i] is True and in_stiffness_window[j] is True:
            for obj_i, obj_j in zip(objectives[i], objectives[j]):
                if obj_i > obj_j:  # Since lower values are better, i dominates j if both objectives are smaller
                    return False
            return True

        raise ValueError('-- NO DOMINANCE CONDITIONS WERE MET, CHECK DOMINANCE FUNC')

    # -------------------------------------
    # PPO Functions
    # -------------------------------------

    def get_cross_obs(self):

        # --- Token weight selection --- #
        # Want weight tensor of shape: (mini_batch_size, 2)

        # Uniform weight sampling
        if uniform_sampling is True:
            weight_samples = random.sample(self.objective_weights, num_weight_samples)
        elif bimodal_sampling is True:
            # Bimodal weight sampling
            half_point = int(len(self.objective_weights) / 2)
            objective_weights_bottom_half = self.objective_weights[:half_point]
            objective_weights_top_half = self.objective_weights[half_point:]
            half_samples = int(num_weight_samples / 2)
            weight_samples = []
            weight_samples.extend(random.sample(objective_weights_bottom_half, half_samples))
            weight_samples.extend(random.sample(objective_weights_top_half, half_samples))
        elif bimodal_alternating_sampling is True:
            half_point = int(len(self.objective_weights) / 2)
            objective_weights_bottom_half = self.objective_weights[:half_point]
            objective_weights_top_half = self.objective_weights[half_point:]
            if self.curr_epoch % 2 == 0:
                weight_samples = random.sample(objective_weights_bottom_half, num_weight_samples)
            else:
                weight_samples = random.sample(objective_weights_top_half, num_weight_samples)





        task_samples = random.sample(self.run_tasks, num_task_samples)
        cross_obs_vars = []
        weight_samples_all = []
        task_samples_all = []
        for task in task_samples:
            task_cond_vars = self.problem.val_problems_norm[task]
            for weight in weight_samples:
                weight_vec = [weight, task_cond_vars[1], task_cond_vars[2], task_cond_vars[3]]
                if len(weight_vec) > conditioning_vars:
                    weight_vec = weight_vec[:conditioning_vars]
                cross_obs_vars.append(weight_vec)
                weight_samples_all.append(weight)
                task_samples_all.append(task)

        weight_samples_all = [element for element in weight_samples_all for _ in range(repeat_size)]
        task_samples_all = [element for element in task_samples_all for _ in range(repeat_size)]
        cross_obs_vars = [element for element in cross_obs_vars for _ in range(repeat_size)]

        cross_obs_tensor = tf.convert_to_tensor(cross_obs_vars, dtype=tf.float32)

        return cross_obs_tensor, weight_samples_all, task_samples_all

    def fast_mini_batch(self):
        children = []

        all_total_rewards = []
        all_actions = [[] for _ in range(self.mini_batch_size)]
        all_rewards = [[] for _ in range(self.mini_batch_size)]
        all_logprobs = [[] for _ in range(self.mini_batch_size)]
        designs = [[] for x in range(self.mini_batch_size)]
        epoch_designs = []
        observation = [[self.decision_start_token_id] for x in range(self.mini_batch_size)]
        critic_observation_buffer = [[] for x in range(self.mini_batch_size)]

        # Get cross attention observation input
        cross_obs_tensor, weight_samples_all, task_samples_all = self.get_cross_obs()

        # -------------------------------------
        # Sample Actor
        # -------------------------------------
        curr_time = time.time()

        for t in range(self.steps_per_design):
            action_log_prob, action, all_action_probs = self.sample_actor(observation, cross_obs_tensor)  # returns shape: (batch,) and (batch,)
            action_log_prob = action_log_prob.numpy().tolist()

            observation_new = deepcopy(observation)
            for idx, act in enumerate(action.numpy()):
                all_actions[idx].append(deepcopy(act))
                all_logprobs[idx].append(action_log_prob[idx])
                m_action = int(deepcopy(act))
                designs[idx].append(m_action)
                observation_new[idx].append(m_action + 2)

            # Determine reward for each batch element
            if len(designs[0]) == self.steps_per_design:
                done = True

                eval_start_time = time.time()
                for idx, design in enumerate(designs):
                    # Record design
                    design_bitstr = ''.join([str(bit) for bit in design])
                    epoch_designs.append(design_bitstr)

                    # Evaluate design
                    reward, design_obj = self.calc_reward(
                        design_bitstr,
                        weight_samples_all[idx],
                        task_samples_all[idx],
                        run_val=True
                    )
                    all_rewards[idx].append(reward)
                    children.append(design_obj)
                    all_total_rewards.append(reward)
                total_eval_time = time.time() - eval_start_time
            else:
                done = False
                reward = 0.0
                for idx, _ in enumerate(designs):
                    all_rewards[idx].append(reward)

            # Update the observation
            if done is True:
                critic_observation_buffer = deepcopy(observation_new)
            else:
                observation = observation_new

        # print('Sample Actor Time:', time.time() - curr_time, 'Total Eval Time:', total_eval_time)
        # -------------------------------------
        # Sample Critic
        # -------------------------------------
        curr_time = time.time()

        # --- SINGLE CRITIC PREDICTION --- #
        value_t = self.sample_critic(critic_observation_buffer, cross_obs_tensor)
        value_t = value_t.numpy().tolist()  # (30, 31)
        for idx, value in zip(range(self.mini_batch_size), value_t):
            last_reward = value[-1]
            all_rewards[idx].append(last_reward)

        # print('Sample Critic Time:', time.time() - curr_time)
        # -------------------------------------
        # Calculate Advantage and Return
        # -------------------------------------

        proc_time = time.time()
        all_advantages = [[] for _ in range(self.mini_batch_size)]
        all_returns = [[] for _ in range(self.mini_batch_size)]
        for idx in range(len(all_rewards)):
            rewards = np.array(all_rewards[idx])
            values = np.array(value_t[idx])
            deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
            adv_tensor = discounted_cumulative_sums(
                deltas, self.gamma * self.lam
            )
            all_advantages[idx] = adv_tensor

            ret_tensor = discounted_cumulative_sums(
                rewards, self.gamma
            )  # [:-1]
            ret_tensor = np.array(ret_tensor, dtype=np.float32)
            all_returns[idx] = ret_tensor

        advantage_mean, advantage_std = (
            np.mean(all_advantages),
            np.std(all_advantages),
        )
        all_advantages = (all_advantages - advantage_mean) / advantage_std

        observation_tensor = tf.convert_to_tensor(observation, dtype=tf.float32)
        action_tensor = tf.convert_to_tensor(all_actions, dtype=tf.int32)
        logprob_tensor = tf.convert_to_tensor(all_logprobs, dtype=tf.float32)
        advantage_tensor = tf.convert_to_tensor(all_advantages, dtype=tf.float32)
        critic_observation_tensor = tf.convert_to_tensor(critic_observation_buffer, dtype=tf.float32)
        return_tensor = tf.convert_to_tensor(all_returns, dtype=tf.float32)
        return_tensor = tf.expand_dims(return_tensor, axis=-1)

        # -------------------------------------
        # Train Actor
        # -------------------------------------
        curr_time = time.time()

        policy_update_itr = 0
        kl, entr, policy_loss, actor_loss = 0, 0, 0, 0
        if freeze_actor is False:
            for i in range(self.train_actor_iterations):
                policy_update_itr += 1
                kl, entr, policy_loss, actor_loss = self.train_actor(
                    observation_tensor,
                    action_tensor,
                    logprob_tensor,
                    advantage_tensor,
                    cross_obs_tensor
                )
                if kl > 1.5 * self.target_kl:
                    # Early Stopping
                    break
            kl = kl.numpy()
            entr = entr.numpy()
            policy_loss = policy_loss.numpy()
            actor_loss = actor_loss.numpy()

        # print('Train Actor Time:', time.time() - curr_time)
        # -------------------------------------
        # Train Critic
        # -------------------------------------
        curr_time = time.time()

        value_loss = 0
        if freeze_critic is False:
            for i in range(self.train_critic_iterations):
                value_loss = self.train_critic(
                    critic_observation_tensor,
                    return_tensor,
                    cross_obs_tensor,
                )
            value_loss = value_loss.numpy()

        # print('Train Critic Time:', time.time() - curr_time)

        # Update results tracker
        epoch_info = {
            'mb_return': np.mean(all_total_rewards),
            'c_loss': value_loss,
            'p_loss': policy_loss,
            'p_iter': policy_update_itr,
            'entropy': entr,
            'kl': kl,
            'tasks': list(set(task_samples_all)),
        }

        # Update population
        self.population.extend(children)

        return epoch_info

    def calc_reward(self, bitstr, weight, task, run_val=True):

        h_stiffness, v_stiffness, stiff_ratio, vol_frac, constraints = self.problem.evaluate(bitstr, problem_num=task, run_val=True)
        h_stiffness = float(h_stiffness)
        v_stiffness = float(v_stiffness)
        stiff_ratio = float(stiff_ratio)
        vol_frac = float(vol_frac)

        # print('Generated Design Objectives:', v_stiffness, vol_frac, h_stiffness)

        # -------------------------------------
        # Calculate reward
        # -------------------------------------

        w1 = weight
        w2 = 1.0 - weight
        v_stiff_term = w1 * v_stiffness
        vol_frac_term = w2 * (1.0 - vol_frac)
        reward = v_stiff_term + vol_frac_term

        # -------------------------------------
        # Create design
        # -------------------------------------
        design = Design(design_vector=[int(i) for i in bitstr], evaluator=self.problem, num_bits=self.steps_per_design,
                        c_type=self.c_type, p_num=task, val=run_val)
        design.is_feasible = True
        design.stiffness_ratio = stiff_ratio
        design.set_objectives(v_stiffness * -1.0, vol_frac)
        design.h_stiffness = h_stiffness
        if bitstr not in self.unique_designs:
            self.unique_designs.add(bitstr)
            self.unique_designs_vals.append([v_stiffness * -1.0, vol_frac])
            self.unique_designs_in_window.append(design.is_feasible)
            self.nfe += 1

        return reward, design

    # -------------------------------------
    # Actor-Critic Functions
    # -------------------------------------

    def sample_actor(self, observation, cross_obs):
        inf_idx = len(observation[0]) - 1  # all batch elements have the same length
        observation_input = deepcopy(observation)
        observation_input = tf.convert_to_tensor(observation_input, dtype=tf.float32)
        inf_idx = tf.convert_to_tensor(inf_idx, dtype=tf.int32)
        return self._sample_actor(observation_input, cross_obs, inf_idx)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),  # shape=(global_mini_batch_size, None)
        tf.TensorSpec(shape=(None, conditioning_vars), dtype=tf.float32),  # shape=(global_mini_batch_size, 1)
        tf.TensorSpec(shape=(), dtype=tf.int32)
    ])
    def _sample_actor(self, observation_input, cross_input, inf_idx):
        # print('sampling actor', inf_idx)
        pred_probs = self.c_actor([observation_input, cross_input])

        # Batch sampling
        all_token_probs = pred_probs[:, inf_idx, :]  # shape (batch, 2)
        all_token_log_probs = tf.math.log(all_token_probs + 1e-10)
        samples = tf.random.categorical(all_token_log_probs, 1)  # shape (batch, 1)
        next_bit_ids = tf.squeeze(samples, axis=-1)  # shape (batch,)
        batch_indices = tf.range(0, tf.shape(all_token_log_probs)[0], dtype=tf.int64)  # shape (batch,)
        next_bit_probs = tf.gather_nd(all_token_log_probs, tf.stack([batch_indices, next_bit_ids], axis=-1))

        actions = next_bit_ids  # (batch,)
        actions_log_prob = next_bit_probs  # (batch,)
        return actions_log_prob, actions, all_token_probs



    def sample_critic(self, observation, parent_obs):
        inf_idx = len(observation[0]) - 1
        observation_input = tf.convert_to_tensor(observation, dtype=tf.float32)
        inf_idx = tf.convert_to_tensor(inf_idx, dtype=tf.int32)
        return self._sample_critic(observation_input, parent_obs, inf_idx)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(global_mini_batch_size, None), dtype=tf.float32),
        tf.TensorSpec(shape=(global_mini_batch_size, conditioning_vars), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    ])
    def _sample_critic(self, observation_input, parent_input, inf_idx):
        t_value = self.c_critic([observation_input, parent_input])  # (batch, seq_len, 2)
        t_value = t_value[:, :, 0]
        return t_value
        # t_value_stiff = t_value[:, :, 0]  # (batch, 1)
        # t_value_vol = t_value[:, :, 1]  # (batch, 1)
        # return t_value_stiff, t_value_vol

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(global_mini_batch_size, config.num_vars), dtype=tf.float32),
        tf.TensorSpec(shape=(global_mini_batch_size, config.num_vars), dtype=tf.int32),
        tf.TensorSpec(shape=(global_mini_batch_size, config.num_vars), dtype=tf.float32),
        tf.TensorSpec(shape=(global_mini_batch_size, config.num_vars), dtype=tf.float32),
        tf.TensorSpec(shape=(global_mini_batch_size, conditioning_vars), dtype=tf.float32),
    ])
    def train_actor(
            self,
            observation_buffer,
            action_buffer,
            logprobability_buffer,
            advantage_buffer,
            parent_buffer
    ):

        with tf.GradientTape() as tape:
            pred_probs = self.c_actor([observation_buffer, parent_buffer])  # shape: (batch, seq_len, 2)
            pred_log_probs = tf.math.log(pred_probs)  # shape: (batch, seq_len, 2)
            logprobability = tf.reduce_sum(
                tf.one_hot(action_buffer, self.num_actions) * pred_log_probs, axis=-1
            )  # shape (batch, seq_len)

            # Total loss
            loss = 0

            # PPO Surrogate Loss
            ratio = tf.exp(
                logprobability - logprobability_buffer
            )
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + self.clip_ratio) * advantage_buffer,
                (1 - self.clip_ratio) * advantage_buffer,
            )
            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantage_buffer, min_advantage)
            )
            loss += policy_loss

            # Entropy Term
            entr = -tf.reduce_sum(pred_probs * pred_log_probs, axis=-1)  # shape (batch, seq_len)
            entr = tf.reduce_mean(entr)  # Higher positive value means more exploration - shape (batch,)
            loss = loss - (self.entropy_coef * entr)

        policy_grads = tape.gradient(loss, self.c_actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(policy_grads, self.c_actor.trainable_variables))

        #  KL Divergence
        pred_probs = self.c_actor([observation_buffer, parent_buffer])
        pred_log_probs = tf.math.log(pred_probs)
        logprobability = tf.reduce_sum(
            tf.one_hot(action_buffer, self.num_actions) * pred_log_probs, axis=-1
        )  # shape (batch, seq_len)
        kl = tf.reduce_mean(
            logprobability_buffer - logprobability
        )
        kl = tf.reduce_sum(kl)

        return kl, entr, policy_loss, loss

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(global_mini_batch_size, config.num_vars+1), dtype=tf.float32),
        tf.TensorSpec(shape=(global_mini_batch_size, config.num_vars+1, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(global_mini_batch_size, conditioning_vars), dtype=tf.float32),
    ])
    def train_critic(
            self,
            observation_buffer,
            return_buffer,
            parent_buffer,
    ):

        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            pred_values = self.c_critic(
                [observation_buffer, parent_buffer])  # (batch, seq_len, 2)

            # Value Loss (mse)
            value_loss = tf.reduce_mean((return_buffer - pred_values) ** 2)

        critic_grads = tape.gradient(value_loss, self.c_critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.c_critic.trainable_variables))

        return value_loss

    def record(self, epoch_info):
        if epoch_info is None:
            return

        # Record new epoch / print
        if self.debug is True:
            print(f"Proc GA_Task {self.run_num} - {self.curr_epoch} ", end=' ')
            for key, value in epoch_info.items():
                print(f"{key}: {value}", end=' | ')
            print('')

        # Update metrics
        self.returns.append(epoch_info['mb_return'])
        self.c_loss.append(epoch_info['c_loss'])
        self.p_loss.append(epoch_info['p_loss'])
        self.p_iter.append(epoch_info['p_iter'])
        self.entropy.append(epoch_info['entropy'])
        self.kl.append(epoch_info['kl'])
        self.hv.append(self.calc_pop_hv())
        self.nfes.append(self.nfe)

        if len(self.entropy) % self.plot_freq == 0:
            print('--> PLOTTING')
            self.plot_ppo()
        else:
            return

    def plot_ppo(self):

        # --- Plotting ---
        epochs = [x for x in range(len(self.returns))]
        gs = gridspec.GridSpec(3, 2)
        fig = plt.figure(figsize=(16, 8))  # default [6.4, 4.8], W x H  9x6, 12x8
        fig.suptitle('Results', fontsize=16)

        # Returns plot
        plt.subplot(gs[0, 0])
        plt.plot(epochs, self.returns)
        plt.xlabel('Epoch')
        plt.ylabel('Mini-batch Return')
        plt.title('PPO Return Plot')

        # Critic loss plot
        plt.subplot(gs[0, 1])
        if len(self.c_loss) < 100:
            c_loss = self.c_loss
            c_epochs = epochs
        else:
            c_loss = self.c_loss[50:]
            c_epochs = epochs[50:]
        plt.plot(c_epochs, c_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Critic loss')
        plt.title('Critic Loss Plot')

        # Policy entropy plot
        plt.subplot(gs[1, 0])
        plt.plot(epochs, self.entropy)
        plt.xlabel('Epoch')
        plt.ylabel('Entropy')
        plt.title('Policy Entropy Plot')

        # KL divergence plot
        plt.subplot(gs[1, 1])
        plt.plot(epochs, self.kl)
        plt.xlabel('Epoch')
        plt.ylabel('KL')
        plt.title('KL Divergence Plot')

        # HV Plot
        plt.subplot(gs[2, 0])
        plt.plot(self.nfes, self.hv, label='PPO HV (MultiTask)')
        plt.plot([r[0] for r in self.uniform_ga], [r[1] for r in self.uniform_ga], label='Uniform GA HV')
        plt.xlabel('NFE')
        plt.ylabel('HV')
        plt.title('Hypervolume Plot')
        plt.legend()

        # Design Plot
        plt.subplot(gs[2, 1])
        for idx, obj_vals in enumerate(self.unique_designs_vals):
            plt.scatter(obj_vals[0] * -1.0, obj_vals[1], color='blue')
        plt.xlim(0, 1.1)
        plt.ylim(0, 1.1)
        plt.xlabel('Vertical Stiffness')
        plt.ylabel('Volume Fraction')
        plt.title('All Designs')


        # Save and close
        plt.tight_layout()
        save_path = os.path.join(self.run_dir, 'plots.png')
        if self.run_val is True:
            save_path = os.path.join(self.run_dir, 'plots_val_' + str(self.val_itr) + '.png')
        plt.savefig(save_path)
        plt.close('all')

        # HV file
        hv_prog_file_path = os.path.join(self.run_dir, 'hv.json')
        if self.run_val is True:
            hv_prog_file_path = os.path.join(self.run_dir, 'hv_val_' + str(self.val_itr) + '.json')
        hv_progress = [(n, h) for n, h in zip(self.nfes, self.hv)]
        with open(hv_prog_file_path, 'w', encoding='utf-8') as file:
            json.dump(hv_progress, file, ensure_ascii=False, indent=4)

        # Save current population to file
        pop_file_path = os.path.join(self.run_dir, 'pop.json')
        if self.run_val is True:
            pop_file_path = os.path.join(self.run_dir, 'pop_val_' + str(self.val_itr) + '.json')
        pop_data = []
        for design in self.population:
            pop_data.append({
                'design': design.get_vector_str(),
                'objectives': design.objectives,
            })
        with open(pop_file_path, 'w', encoding='utf-8') as file:
            json.dump(pop_data, file, ensure_ascii=False, indent=4)

from problem.TrussProblem import TrussProblem

if __name__ == '__main__':
    problem = TrussProblem(sidenum=config.sidenum)

    # actor_save_path = os.path.join(config.results_save_dir, 'run_' + str(run_dir), 'actor_weights_init')
    # critic_save_path = os.path.join(config.results_save_dir, 'run_' + str(run_dir), 'critic_weights_init')

    # actor_save_path = os.path.join(config.results_save_dir, 'run_' + str(run_dir), 'actor_weights_400')
    # critic_save_path = os.path.join(config.results_save_dir, 'run_' + str(run_dir), 'critic_weights_400')

    # actor_save_path = os.path.join(config.results_save_dir, 'run_' + str(run_dir), 'actor_weights_val')
    # critic_save_path = os.path.join(config.results_save_dir, 'run_' + str(run_dir), 'critic_weights_val')

    # actor_save_path = os.path.join(config.results_save_dir, 'run_' + str(run_dir), 'pretrained', 'actor_weights_500')
    # critic_save_path = os.path.join(config.results_save_dir, 'run_' + str(run_dir), 'pretrained', 'critic_weights_500')

    actor_save_path = os.path.join(config.results_save_dir, 'run_' + str(3), 'pretrained', 'actor_weights_4850')  # 4850
    critic_save_path = os.path.join(config.results_save_dir, 'run_' + str(3), 'pretrained', 'critic_weights_4850')  # 4850

    # actor_save_path = None
    # critic_save_path = None


    # alg = PPO_MultiTaskValidation(
    #     run_num=run_dir,
    #     problem=problem,
    #     epochs=task_epochs,
    #     actor_load_path=actor_save_path,
    #     critic_load_path=critic_save_path,
    #     debug=True,
    #     c_type='uniform',
    #     run_val=True,
    #     val_itr=run_num,
    #     val_task=val_task
    # )
    # alg.run()


    runs = 30
    for x in range(1, 30):
        val_itr = 110 + x
        alg = PPO_MultiTaskValidation(
            run_num=run_dir,
            problem=problem,
            epochs=task_epochs,
            actor_load_path=actor_save_path,
            critic_load_path=critic_save_path,
            debug=True,
            c_type='uniform',
            run_val=True,
            val_itr=val_itr,
            val_task=val_task
        )
        alg.run()

