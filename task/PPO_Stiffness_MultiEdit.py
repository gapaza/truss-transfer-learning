import numpy as np
import time
import tensorflow as tf
from copy import deepcopy
import matplotlib.gridspec as gridspec
import random
import json
import config
import matplotlib.pyplot as plt
import os
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from task.AbstractTask import AbstractTask
from problem.TrussDesign import TrussDesign as Design
import scipy.signal
from task.GA_Constrained_Task import GA_Constrained_Task
from model import get_stiffness_decoder as get_model
from collections import OrderedDict
import tensorflow_addons as tfa


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]



# Sampling parameters
global_mini_batch_size = 10

population_redundancy = 1
population_size = 100
use_population = True


uniform_sampling = True
bimodal_sampling = False
bimodal_alternating_sampling = False
trimodal_sampling = False

num_conditioning_vars = config.num_conditioning_vars


# Fine-tuning parameters
update_actor = True
update_critic = True
freeze_actor_base = False
freeze_critic_base = False


# Run parameters
save_init_weights = False
load_init_weights = False
run_dir = 0
run_num = 8
plot_freq = 50

# Problem parameters
problem_num = 0
target_stiffness_ratio = 0.7
feasible_stiffness_delta = 0.00001

# Training Parameters
task_epochs = 10000
clip_ratio = 0.2
target_kl = 0.005
entropy_coef = 0.02

use_actor_train_call = False
use_critic_train_call = False

# Set random seed and tensorflow seed
random.seed(0)
tf.random.set_seed(0)




class PPO_Stiffness_Decoder(AbstractTask):

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
        super(PPO_Stiffness_Decoder, self).__init__(run_num, barrier, problem, epochs, actor_load_path, critic_load_path)
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
        self.unique_designs_feasible = []
        self.constraint_returns = []
        self.constraint_term_history = []
        self.min_stiffness_delta = 1000.0

        # Algorithm parameters
        self.pop_size = 30  # 32 FU_NSGA2, 10 U_NSGA2
        self.offspring_size = global_mini_batch_size  # 32 FU_NSGA2, 30 U_NSGA2
        self.mini_batch_size = global_mini_batch_size
        self.nfe = 0
        self.epochs = epochs
        self.steps_per_design = 30  # 30 | 60

        # PPO alg parameters
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_ratio = clip_ratio  # was 0.2
        self.target_kl = target_kl  # was 0.005
        self.entropy_coef = entropy_coef  # was 0.02 originally
        self.counter = 0
        self.decision_start_token_id = 1
        self.num_actions = 2
        self.curr_epoch = 0

        # Population
        self.population = []
        self.sample_population = PPO_Stiffness_Decoder.get_random_designs(sample_size=population_size, design_len=30)
        self.sample_population = self.sample_population * population_redundancy
        self.sample_population_vals = [100] * len(self.sample_population)
        self.hv = []
        self.nfes = []

        # Results
        self.plot_freq = plot_freq

        # Init weights dir
        self.init_weights_dir = os.path.join(self.run_dir, 'init_weights')
        if not os.path.exists(self.init_weights_dir):
            os.makedirs(self.init_weights_dir)




    def build(self):

        # Optimizer parameters
        self.actor_learning_rate = 0.0001  # 0.0001
        self.critic_learning_rate = 0.0001  # 0.0001
        self.train_actor_iterations = 250  # was 250
        self.train_critic_iterations = 10  # was 40
        self.beta_1 = 0.9
        if self.run_val is False:
            self.beta_1 = 0.0

        # Optimizers
        if self.actor_optimizer is None:
            self.actor_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.actor_learning_rate)
            # self.actor_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.actor_learning_rate, beta_1=self.beta_1)
            # self.actor_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=self.actor_learning_rate)
        if self.critic_optimizer is None:
            self.critic_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.critic_learning_rate)
            # self.critic_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.critic_learning_rate, beta_1=self.beta_1)
            # self.critic_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=self.critic_learning_rate)

        # Load models
        if load_init_weights is True and len(os.listdir(self.init_weights_dir)) != 0:
            self.actor_load_path = os.path.join(self.init_weights_dir, 'actor_weights_init')
            self.critic_load_path = os.path.join(self.init_weights_dir, 'critic_weights_init')

        self.c_actor, self.c_critic = get_model(self.actor_load_path, self.critic_load_path)

        # Save initial weights
        if save_init_weights is True:
            if len(os.listdir(self.init_weights_dir)) == 0:
                t_actor_save_path = os.path.join(self.init_weights_dir, 'actor_weights_init')
                t_critic_save_path = os.path.join(self.init_weights_dir, 'critic_weights_init')
                self.c_actor.save_weights(t_actor_save_path)
                self.c_critic.save_weights(t_critic_save_path)

        # Freeze weights
        if freeze_actor_base is True:
            self.c_actor.decoder_1.trainable = False
            self.c_actor.decoder_2.trainable = False
        if freeze_critic_base is True:
            self.c_critic.decoder_1.trainable = False

        self.c_actor.summary()


    def run(self):
        self.build()

        for x in range(self.epochs):
            self.curr_epoch = x
            epoch_info = self.fast_mini_batch()
            # self.prune_population()
            self.record(epoch_info)

    # -------------------------------------
    # Population Functions
    # -------------------------------------


    def calc_pop_hv(self):
        objectives = self.eval_population()

        feasible_objectives = []
        for design, d_objectives in zip(self.population, objectives):
            if design.is_feasible is True:
                feasible_objectives.append(d_objectives)
        if len(feasible_objectives) <= 1:
            return 0.0

        F = np.array(feasible_objectives)
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
        feasibility_scores = [design.feasibility_score for design in self.population]
        is_feasible = [design.is_feasible for design in self.population]

        # 2. Determine survivors
        fronts = self.custom_dominance_sorting(objectives, feasibility_scores, is_feasible)
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

    @staticmethod
    def bit_list_to_str(bit_list):
        return ''.join([str(bit) for bit in bit_list])

    @staticmethod
    def get_random_designs(sample_size=global_mini_batch_size, design_len=30):
        num_ones = np.random.randint(1, design_len, sample_size)
        bitstrings = []
        for i in range(sample_size):
            design = np.zeros(design_len)
            design[:num_ones[i]] = 1
            np.random.shuffle(design)
            design_int = [int(x) for x in design]
            bitstrings.append(list(design_int))
        return bitstrings

    def fast_mini_batch(self):
        children = []

        all_constraint_terms = []
        all_constraint_returns = []
        all_total_rewards = []
        all_actions = [[] for _ in range(self.mini_batch_size)]
        all_rewards = [[] for _ in range(self.mini_batch_size)]
        all_logprobs = [[] for _ in range(self.mini_batch_size)]
        bit_flips = [[] for x in range(self.mini_batch_size)]
        epoch_designs = []
        observation = [[self.decision_start_token_id] for x in range(self.mini_batch_size)]
        critic_observation_buffer = [[] for x in range(self.mini_batch_size)]
        all_stiffness_ratio_deltas = []


        if use_population is True:
            start_designs = random.sample(self.sample_population, self.mini_batch_size)
        else:
            start_designs = self.get_random_designs(sample_size=self.mini_batch_size, design_len=30)


        start_designs_str = ["".join([str(bit) for bit in design]) for design in start_designs]
        cross_obs_tensor = tf.convert_to_tensor(start_designs, dtype=tf.float32)

        # -------------------------------------
        # Sample Actor
        # -------------------------------------
        sample_population_strings = [PPO_Stiffness_Decoder.bit_list_to_str(design) for design in self.sample_population]


        for t in range(self.steps_per_design):
            action_log_prob, action, all_action_probs = self.sample_actor(observation, cross_obs_tensor)  # returns shape: (batch,) and (batch,)
            action_log_prob = action_log_prob.numpy().tolist()

            observation_new = deepcopy(observation)
            for idx, act in enumerate(action.numpy()):
                all_actions[idx].append(deepcopy(act))
                all_logprobs[idx].append(action_log_prob[idx])
                m_action = int(deepcopy(act))
                bit_flips[idx].append(m_action)
                observation_new[idx].append(m_action + 2)

            # Determine reward for each batch element
            if len(bit_flips[0]) == self.steps_per_design:
                done = True
                for idx, design_bit_flips in enumerate(bit_flips):

                    # Get edited design
                    edited_design = deepcopy(start_designs[idx])
                    for idx2, flip in enumerate(design_bit_flips):
                        if flip == 1:
                            edited_design[idx2] = int(1 - edited_design[idx2])


                    design_bitstr = ''.join([str(bit) for bit in edited_design])
                    epoch_designs.append(design_bitstr)

                    # Evaluate design
                    reward, stiff_ratio_delta = self.calc_reward(
                        design_bitstr,
                        start_designs_str[idx],
                        sample_population_strings,
                        run_val=True
                    )
                    all_rewards[idx].append(reward)
                    all_total_rewards.append(reward)
                    all_stiffness_ratio_deltas.append(stiff_ratio_delta)
            else:
                done = False
                reward = 0.0
                for idx, _ in enumerate(bit_flips):
                    all_rewards[idx].append(reward)

            # Update the observation
            if done is True:
                critic_observation_buffer = deepcopy(observation_new)
            else:
                observation = observation_new


        # first_bit_flip = bit_flips[0]
        # first_bit_flip_str = "".join([str(bit) for bit in first_bit_flip])
        # print('- Original Design:', start_designs_str[0])
        # print('--------- Actions:', first_bit_flip_str)
        # print('--- Edited Design:', epoch_designs[0])
        # exit(0)


        for stiffness_ratio_delta, design_bitstr in zip(all_stiffness_ratio_deltas, epoch_designs):
            max_pop_val = max(self.sample_population_vals)
            max_pop_val_index = self.sample_population_vals.index(max_pop_val)
            if stiffness_ratio_delta < max_pop_val and design_bitstr not in sample_population_strings:
                epoch_design_int = [int(x) for x in design_bitstr]
                self.sample_population[max_pop_val_index] = epoch_design_int
                self.sample_population_vals[max_pop_val_index] = stiffness_ratio_delta



        # -------------------------------------
        # Sample Critic
        # -------------------------------------

        # --- SINGLE CRITIC PREDICTION --- #
        value_t = self.sample_critic(critic_observation_buffer, cross_obs_tensor)
        value_t = value_t.numpy().tolist()  # (30, 31)
        for idx, value in zip(range(self.mini_batch_size), value_t):
            last_reward = value[-1]
            all_rewards[idx].append(last_reward)

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
        if update_actor is True:
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

        # -------------------------------------
        # Train Critic
        # -------------------------------------

        curr_time = time.time()
        value_loss = 0
        if update_critic is True:
            for i in range(self.train_critic_iterations):
                value_loss = self.train_critic(
                    critic_observation_tensor,
                    return_tensor,
                    cross_obs_tensor,
                )
            value_loss = value_loss.numpy()

        # Update results tracker
        epoch_info = {
            'mb_return': np.mean(all_total_rewards),
            'c_loss': value_loss,
            'p_loss': policy_loss,
            'p_iter': policy_update_itr,
            'entropy': entr,
            'kl': kl,
        }

        return epoch_info


    def calc_reward(self, bitstr, original_design, samp_pop_str, run_val=True):

        # if bitstr in samp_pop_str:
        #     return -1, 1

        if bitstr == original_design:
            return -1, 1

        # Evaluate Edited
        h_stiffness, v_stiffness, stiff_ratio, vol_frac, constraints = self.problem.evaluate(bitstr, problem_num=problem_num, run_val=run_val)
        h_stiffness = float(h_stiffness)
        v_stiffness = float(v_stiffness)
        stiff_ratio = float(stiff_ratio)
        vol_frac = float(vol_frac)

        stiffness_ratio_delta = abs(self.problem.target_stiffness_ratio - stiff_ratio)
        stiffness_ratio_delta_edited = deepcopy(stiffness_ratio_delta)
        if stiffness_ratio_delta < self.min_stiffness_delta:
            self.min_stiffness_delta = stiffness_ratio_delta
        if stiffness_ratio_delta > 1.0:
            stiffness_ratio_delta = 1.0
        exp = 9
        # edited_reward = (1 - stiffness_ratio_delta) ** exp
        edited_reward = (1.0 / stiffness_ratio_delta) / 100.0
        return edited_reward, stiffness_ratio_delta_edited


        # # Evaluate Original
        # h_stiffness_orig, v_stiffness_orig, stiff_ratio_orig, vol_frac_orig, constraints_orig = self.problem.evaluate(original_design, problem_num=problem_num, run_val=run_val)
        # h_stiffness_orig = float(h_stiffness_orig)
        # v_stiffness_orig = float(v_stiffness_orig)
        # stiff_ratio_orig = float(stiff_ratio_orig)
        # vol_frac_orig = float(vol_frac_orig)
        #
        # stiffness_ratio_delta_orig = abs(self.problem.target_stiffness_ratio - stiff_ratio_orig)
        # if stiffness_ratio_delta_orig > 1.0:
        #     stiffness_ratio_delta_orig = 1.0
        # exp = 3
        # original_reward = (1 - stiffness_ratio_delta_orig) ** exp
        #
        # reward = edited_reward - original_reward
        #
        #
        # return reward, stiffness_ratio_delta_edited

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
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),  # shape=(global_mini_batch_size, 1)
        tf.TensorSpec(shape=(), dtype=tf.int32)
    ])
    def _sample_actor(self, observation_input, cross_input, inf_idx):
        # print('sampling actor', inf_idx)
        pred_probs = self.c_actor([observation_input, cross_input], training=use_actor_train_call)  # shape (batch, seq_len, 2

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
        tf.TensorSpec(shape=(global_mini_batch_size, None), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    ])
    def _sample_critic(self, observation_input, parent_input, inf_idx):
        t_value = self.c_critic([observation_input, parent_input], training=use_critic_train_call)  # (batch, seq_len, 2)
        t_value = t_value[:, :, 0]
        return t_value
        # t_value_stiff = t_value[:, :, 0]  # (batch, 1)
        # t_value_vol = t_value[:, :, 1]  # (batch, 1)
        # return t_value_stiff, t_value_vol

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(global_mini_batch_size, 30), dtype=tf.float32),
        tf.TensorSpec(shape=(global_mini_batch_size, 30), dtype=tf.int32),
        tf.TensorSpec(shape=(global_mini_batch_size, 30), dtype=tf.float32),
        tf.TensorSpec(shape=(global_mini_batch_size, 30), dtype=tf.float32),
        tf.TensorSpec(shape=(global_mini_batch_size, 30), dtype=tf.float32),
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
            pred_probs = self.c_actor([observation_buffer, parent_buffer],
                                      training=use_actor_train_call)  # shape: (batch, seq_len, 2)
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
        pred_probs = self.c_actor([observation_buffer, parent_buffer], training=use_actor_train_call)
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
        tf.TensorSpec(shape=(global_mini_batch_size, 31), dtype=tf.float32),
        tf.TensorSpec(shape=(global_mini_batch_size, 31, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(global_mini_batch_size, 30), dtype=tf.float32),
    ])
    def train_critic(
            self,
            observation_buffer,
            return_buffer,
            parent_buffer,
    ):

        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            pred_values = self.c_critic(
                [observation_buffer, parent_buffer],
                training=use_critic_train_call
            )  # (batch, seq_len, 2)

            # Value Loss (mse)
            value_loss = tf.reduce_mean((return_buffer - pred_values) ** 2)

        critic_grads = tape.gradient(value_loss, self.c_critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.c_critic.trainable_variables))

        return value_loss


    def record(self, epoch_info):
        if epoch_info is None:
            return

        # determine number of repeats in sample population
        sample_population_strings = [PPO_Stiffness_Decoder.bit_list_to_str(design) for design in self.sample_population]
        unique_designs = set(sample_population_strings)
        num_repeats = len(self.sample_population) - len(unique_designs)

        # Record new epoch / print
        if self.debug is True:
            print(f"Proc PPO_Task {self.run_num} - {self.curr_epoch} ", end=' ')
            for key, value in epoch_info.items():
                if isinstance(value, list):
                    print(f"{key}: {value}", end=' | ')
                else:
                    print("%s: %.5f" % (key, value), end=' | ')
            print(self.min_stiffness_delta, '|', np.mean(self.sample_population_vals), '|', num_repeats)

        # Update metrics
        self.returns.append(epoch_info['mb_return'])
        self.c_loss.append(epoch_info['c_loss'])
        self.p_loss.append(epoch_info['p_loss'])
        self.p_iter.append(epoch_info['p_iter'])
        self.entropy.append(epoch_info['entropy'])
        self.kl.append(epoch_info['kl'])
        self.nfes.append(self.nfe)

        if len(self.entropy) % self.plot_freq == 0:
            print('--> PLOTTING')
            self.plot_ppo()
        else:
            return

    def plot_ppo(self):

        # --- Plotting ---
        epochs = [x for x in range(len(self.returns))]
        gs = gridspec.GridSpec(2, 2)
        fig = plt.figure(figsize=(12, 8))  # default [6.4, 4.8], W x H  9x6, 12x8
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





from problem.TrussProblem import TrussProblem

if __name__ == '__main__':
    problem = TrussProblem(
        sidenum=3,
        calc_constraints=True,
        target_stiffness_ratio=target_stiffness_ratio,
        feasible_stiffness_delta=feasible_stiffness_delta,
    )

    actor_save_path = None
    critic_save_path = None

    alg = PPO_Stiffness_Decoder(
        run_num=run_dir,
        problem=problem,
        epochs=task_epochs,
        actor_load_path=actor_save_path,
        critic_load_path=critic_save_path,
        debug=True,
        c_type='uniform',
        run_val=True,
        val_itr=run_num,
        val_task=problem_num
    )
    alg.run()






