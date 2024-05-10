import numpy as np
import time
import tensorflow as tf
from copy import deepcopy
import matplotlib.gridspec as gridspec
import random
import config
import matplotlib.pyplot as plt
import os
from pymoo.indicators.hv import HV
from task.AbstractTask import AbstractTask
from problem.TrussDesign import TrussDesign as Design
import scipy.signal
from modelC import get_multi_task_decoder_constraint as get_model
from problem import get_str_targets

# One variable for the objective weighting, one variable for the target stiffness ratio
if config.num_conditioning_vars != 2:
    raise ValueError('Number of conditioning variables must be 2 for the multiple constraint tasks')


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


# Sampling parameters
num_weight_samples = 4
num_task_samples = 1
num_constraint_samples = 6
repeat_size = 1
global_mini_batch_size = num_weight_samples * num_task_samples * repeat_size * num_constraint_samples

# Run parameters
run_dir = 20000
run_num = 3
plot_freq = 50
save_freq = 50

# Problem parameters
num_tasks = 1  # was 36, for now make first validation task
val_tasks = True

min_str, max_str, str_delta = 0.2, 1.8, 0.01
target_ratios, target_ratios_norm = get_str_targets(min_str, max_str, str_delta)
val_target_tasks_mod = 10
val_target_tasks = []
target_tasks = []
cnt = 1
for x, y in zip(target_ratios, target_ratios_norm):
    if cnt % val_target_tasks_mod == 0:
        val_target_tasks.append((x, y))
    else:
        target_tasks.append((x, y))
    cnt += 1
str_delta = 0.00001


# Training Parameters
task_epochs = 10000
clip_ratio = 0.2
target_kl = 0.005
entropy_coef = 0.1  # was 0.1

# Reward weight terms
reward_coef = 1.0
perf_term_weight = 1.0
str_multiplier = 10.0
fea_multiplier = 1.0
constraint_term_weights_epochs = [250]
constraint_term_weights = [0.4, 0.4]  # was 0.1

use_actor_train_call = False
use_critic_train_call = False

# Learning Rates
actor_learning_rate = 0.0001
critic_learning_rate = 0.0001
train_actor_iterations = 250
train_critic_iterations = 40

# Set seeds to 0
random.seed(0)
tf.random.set_seed(0)

def get_constraint_weight(epoch):
    for idx, threshold in enumerate(constraint_term_weights_epochs):
        if epoch <= threshold:
            return constraint_term_weights[idx]
    return constraint_term_weights[-1]

class PPO_Constrained_MultiTask(AbstractTask):

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
            num_tasks=9,
    ):
        super(PPO_Constrained_MultiTask, self).__init__(run_num, barrier, problem, epochs, actor_load_path, critic_load_path)
        self.debug = debug
        self.c_type = c_type
        self.run_val = run_val
        self.val_itr = val_itr
        self.num_tasks = num_tasks

        # HV
        self.ref_point = np.array(config.hv_ref_point)  # value, weight
        self.hv_client = HV(self.ref_point)
        self.unique_designs_all = set()
        self.unique_designs = [set() for _ in range(self.num_tasks)]
        self.unique_designs_vals = [[] for _ in range(self.num_tasks)]
        self.unique_designs_feasible = [[] for _ in range(self.num_tasks)]
        self.unique_designs_epochs = [[] for _ in range(self.num_tasks)]

        # Algorithm parameters
        self.pop_size = 30  # 32 FU_NSGA2, 10 U_NSGA2
        self.offspring_size = global_mini_batch_size  # 32 FU_NSGA2, 30 U_NSGA2
        self.mini_batch_size = global_mini_batch_size
        self.nfe = 0
        self.nfes = [0 for _ in range(self.num_tasks)]
        self.task_nfes = [[] for _ in range(self.num_tasks)]
        self.epochs = epochs
        self.steps_per_design = config.num_vars  # 30 | 60

        # PPO alg parameters
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl  # was 0.01
        self.entropy_coef = entropy_coef  # was 0.02 originally
        self.counter = 0
        self.decision_start_token_id = 1
        self.num_actions = 2
        self.curr_epoch = 0

        # Population
        self.population = [[] for _ in range(self.num_tasks)]
        self.hv = [[] for _ in range(self.num_tasks)]

        # Results
        self.plot_freq = plot_freq
        self.performance_returns = []
        self.constraint_returns = []

        # Objective Weights
        num_keys = 9
        self.objective_weights = list(np.linspace(0.05, 0.95, num_keys))

        # Tasks
        self.run_tasks = [i for i in range(self.num_tasks)]
        self.constraint_tasks = target_tasks
        self.val_constraint_tasks = val_target_tasks

        # GA Comparison Data
        self.uniform_ga = []

        # Pretrain save dir
        self.pretrain_save_dir = os.path.join(self.run_dir, 'pretrained')
        if not os.path.exists(self.pretrain_save_dir):
            os.makedirs(self.pretrain_save_dir)
        self.actor_pretrain_save_path = os.path.join(self.pretrain_save_dir, 'actor_weights')
        self.critic_pretrain_save_path = os.path.join(self.pretrain_save_dir, 'critic_weights')


    def build(self):

        # Optimizer parameters
        self.actor_learning_rate = actor_learning_rate  # 0.0001
        self.critic_learning_rate = critic_learning_rate  # 0.0001
        self.train_actor_iterations = train_actor_iterations  # was 250
        self.train_critic_iterations = train_critic_iterations  # was 40
        self.beta_1 = 0.9
        if self.run_val is False:
            self.beta_1 = 0.0

        # Optimizers
        if self.actor_optimizer is None:
            self.actor_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.actor_learning_rate)
        if self.critic_optimizer is None:
            self.critic_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.critic_learning_rate)

        self.c_actor, self.c_critic = get_model(self.actor_load_path, self.critic_load_path)

        self.c_actor.summary()

    def run(self):
        self.build()

        for x in range(self.epochs):
            self.curr_epoch = x
            epoch_info = self.fast_mini_batch()

            self.record(epoch_info)

            if self.curr_epoch % save_freq == 0:
                t_actor_save_path = os.path.join(self.pretrain_save_dir, 'actor_weights_' + str(self.curr_epoch))
                t_critic_save_path = os.path.join(self.pretrain_save_dir, 'critic_weights_' + str(self.curr_epoch))
                self.c_actor.save_weights(t_actor_save_path)
                self.c_critic.save_weights(t_critic_save_path)

    # -------------------------------------
    # Population Functions
    # -------------------------------------

    def calc_pop_hv(self, task_num):
        objectives = self.eval_population(task_num)

        feasible_objectives = []
        for design, d_objectives in zip(self.population[task_num], objectives):
            if design.is_feasible is True:
                feasible_objectives.append(d_objectives)

        F = np.array(feasible_objectives)
        hv = self.hv_client.do(F)
        return hv

    def eval_population(self, task_num):
        evals = []
        for design in self.population[task_num]:
            evals.append(design.evaluate())
        return evals

    def prune_population(self, task_num):

        # 0. Get Objectives
        objectives = self.eval_population(task_num)
        feasibility_scores = [design.feasibility_score for design in self.population[task_num]]
        is_feasible = [design.is_feasible for design in self.population[task_num]]

        # 1. Determine survivors
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

        # 2. Update population (sometimes the first front is larger than pop_size)
        new_population = []
        for idx in survivors:
            new_population.append(self.population[task_num][idx])
        self.population[task_num] = new_population

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

        # Uniform weight sampling
        weight_samples = random.sample(self.objective_weights, num_weight_samples)

        # Uniform task sampling
        task_samples = random.sample(self.run_tasks, num_task_samples)

        # Uniform constraint sampling
        constraint_samples = random.sample(self.constraint_tasks, num_constraint_samples)

        cross_obs_vars = []
        weight_samples_all = []
        task_samples_all = []
        constraint_samples_all = []
        for task in task_samples:
            task_cond_vars = self.problem.train_problems_norm[task]
            for weight in weight_samples:
                for constraint, constraint_norm in constraint_samples:
                    # cross_obs_vars.append([weight, task_cond_vars[1], task_cond_vars[2], task_cond_vars[3], constraint_norm])
                    cross_obs_vars.append([weight, constraint_norm])
                    weight_samples_all.append(weight)
                    task_samples_all.append(task)
                    constraint_samples_all.append(constraint)

        weight_samples_all = [element for element in weight_samples_all for _ in range(repeat_size)]
        task_samples_all = [element for element in task_samples_all for _ in range(repeat_size)]
        constraint_samples_all = [element for element in constraint_samples_all for _ in range(repeat_size)]
        cross_obs_vars = [element for element in cross_obs_vars for _ in range(repeat_size)]

        cross_obs_tensor = tf.convert_to_tensor(cross_obs_vars, dtype=tf.float32)

        return cross_obs_tensor, weight_samples_all, task_samples_all, constraint_samples_all

    def fast_mini_batch(self):
        children = []

        all_perf_returns = []
        all_constraint_returns = []
        all_total_rewards = []
        all_actions = [[] for _ in range(self.mini_batch_size)]
        all_rewards = [[] for _ in range(self.mini_batch_size)]
        all_logprobs = [[] for _ in range(self.mini_batch_size)]
        all_logprobs_full = [[] for _ in range(self.mini_batch_size)]  # Logprobs for all actions
        all_probs_full = [[] for _ in range(self.mini_batch_size)]
        designs = [[] for x in range(self.mini_batch_size)]
        epoch_designs = []
        observation = [[self.decision_start_token_id] for x in range(self.mini_batch_size)]
        critic_observation_buffer = [[] for x in range(self.mini_batch_size)]

        # Get cross attention observation input
        cross_obs_tensor, weight_samples_all, task_samples_all, constraint_samples_all = self.get_cross_obs()

        # -------------------------------------
        # Sample Actor
        # -------------------------------------
        curr_time = time.time()
        eval_time = 0


        for t in range(self.steps_per_design):
            action_log_prob, action, all_action_log_probs, all_action_probs = self.sample_actor(observation, cross_obs_tensor)  # returns shape: (batch,) and (batch,)
            action_log_prob = action_log_prob.numpy().tolist()
            all_action_log_probs = all_action_log_probs.numpy().tolist()
            all_action_probs = all_action_probs.numpy().tolist()

            observation_new = deepcopy(observation)
            for idx, act in enumerate(action.numpy()):
                all_actions[idx].append(deepcopy(act))
                all_logprobs[idx].append(action_log_prob[idx])
                all_logprobs_full[idx].append(all_action_log_probs[idx])
                all_probs_full[idx].append(all_action_probs[idx])
                m_action = int(deepcopy(act))
                designs[idx].append(m_action)
                observation_new[idx].append(m_action + 2)

            # Determine reward for each batch element
            if len(designs[0]) == self.steps_per_design:
                done = True
                for idx, design in enumerate(designs):
                    # Record design
                    design_bitstr = ''.join([str(bit) for bit in design])
                    epoch_designs.append(design_bitstr)

                    # Evaluate design
                    curr_eval_time = time.time()
                    reward, design_obj, perf_return, constraint_return = self.calc_reward(
                        design_bitstr,
                        weight_samples_all[idx],
                        task_samples_all[idx],
                        constraint_samples_all[idx],
                    )
                    eval_time += time.time() - curr_eval_time
                    all_rewards[idx].append(reward)
                    children.append(design_obj)
                    all_total_rewards.append(reward)
                    all_perf_returns.append(perf_return)
                    all_constraint_returns.append(constraint_return)
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


        # print('Sample Actor Time:', (time.time() - curr_time) - eval_time)
        # print('Eval Time:', eval_time)
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
        all_returns_mo = [[] for _ in range(self.mini_batch_size)]
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

        all_logprobs_tensor = tf.convert_to_tensor(all_logprobs_full, dtype=tf.float32)
        all_probs_tensor = tf.convert_to_tensor(all_probs_full, dtype=tf.float32)

        # -------------------------------------
        # Train Actor
        # -------------------------------------

        curr_time = time.time()
        policy_update_itr = 0
        for i in range(self.train_actor_iterations):
            policy_update_itr += 1
            kl, entr, policy_loss, actor_loss = self.train_actor(
                observation_tensor,
                action_tensor,
                logprob_tensor,
                advantage_tensor,
                cross_obs_tensor,
                all_logprobs_tensor,
                all_probs_tensor
            )
            if kl > 1.5 * self.target_kl:
                # Early Stopping
                break


        # print('Train actor time:', time.time() - curr_time)
        # -------------------------------------
        # Train Critic
        # -------------------------------------

        curr_time = time.time()
        for i in range(self.train_critic_iterations):
            value_loss = self.train_critic(
                critic_observation_tensor,
                return_tensor,
                cross_obs_tensor,
            )

        # Update results tracker
        epoch_info = {
            'mb_return': np.mean(all_total_rewards),
            'perf_return': np.mean(all_perf_returns),
            'constraint_return': np.mean(all_constraint_returns),
            'c_loss': value_loss.numpy(),
            'p_loss': policy_loss.numpy(),
            'p_iter': policy_update_itr,
            'entropy': entr.numpy(),
            'kl': kl.numpy(),
            'tasks': list(set(task_samples_all)),
            # 'constraints': list(set(constraint_samples_all)),
        }

        # for task in task_samples_all:
        #     child = children[task]
        #     self.population[task].append(child)

        return epoch_info

    def calc_reward(self, bitstr, weight, task, constraint):

        h_stiffness, v_stiffness, stiff_ratio, vol_frac, constraints = self.problem.evaluate(bitstr, problem_num=task, run_val=val_tasks)
        h_stiffness = float(h_stiffness)
        v_stiffness = float(v_stiffness)
        stiff_ratio = float(stiff_ratio)
        vol_frac = float(vol_frac)

        # If invalid, penalize stiffness ratio
        # if vol_frac >= 1.0 and v_stiffness <= 0.0:
        #     stiff_ratio = 0.0

        # -------------------------------------
        # Calculate performance reward
        # -------------------------------------

        w1 = weight
        w2 = 1.0 - weight
        v_stiff_term = w1 * v_stiffness
        vol_frac_term = w2 * (1.0 - vol_frac)
        performance_term = v_stiff_term + vol_frac_term
        performance_term = performance_term * perf_term_weight

        # -------------------------------------
        # Constraints
        # -------------------------------------
        # 1. Feasibility
        # 2. Connectivity
        # 3. Stiffness Ratio

        # 1. Feasibility Term (minimize)
        feasibility_constraint = float(constraints[0])
        feasibility_term = (1.0 - feasibility_constraint)
        feasibility_term = 0.0

        # 2. Connectivity Term (minimize)
        connectivity_constraint = float(constraints[1])
        connectivity_term = (1.0 - connectivity_constraint)

        # 3. Stiffness Ratio Term (minimize)
        in_stiffness_window = False
        stiffness_ratio_delta = abs(constraint - stiff_ratio)
        if stiffness_ratio_delta > 3.0:  # Clip term
            stiffness_ratio_delta = 3.0
        stiffness_ratio_term = stiffness_ratio_delta
        if stiffness_ratio_delta < (1.0 * str_delta):
            in_stiffness_window = True
            stiffness_ratio_term = 0.0

        # -----------------------
        # --- Constraint Term ---
        # -----------------------
        # - /15.0 works best so far for untrained model
        # - testing for pretrained model

        # minimize constraint term
        constraint_sum = (feasibility_term + connectivity_term + stiffness_ratio_term)

        # stiffness_term_scaled = 2 - (1 / (stiffness_ratio_term + 0.5))
        constraint_term = ((feasibility_term * fea_multiplier) + connectivity_term + (stiffness_ratio_term * str_multiplier))
        constraint_term *= -1.0
        constraint_term = constraint_term * get_constraint_weight(self.curr_epoch)

        # maximize constraint term
        # - /= 10.0 overall and *3.0 STR term works best so far for pretrained model (run 215)
        # constraint_term = (feasibility_constraint + connectivity_constraint + ((1.0 - stiffness_ratio_term)*str_multiplier))
        # constraint_term = constraint_term * constraint_term_weight

        # -------------------------------------
        # Actual Reward
        # -------------------------------------

        is_feasible = False
        # if feasibility_constraint == 1.0 and connectivity_constraint == 1.0 and in_stiffness_window is True:
        if connectivity_constraint == 1.0 and in_stiffness_window is True:
            is_feasible = True
            # reward = performance_term * perf_term_weight
        else:
            is_feasible = False
            # reward = 0.0

        reward = performance_term + constraint_term
        reward = reward * reward_coef

        # -------------------------------------
        # Create design
        # -------------------------------------
        design = Design(design_vector=[int(i) for i in bitstr], evaluator=self.problem, num_bits=self.steps_per_design,
                        c_type=self.c_type, p_num=task, val=val_tasks, constraints=True)
        design.is_feasible = is_feasible
        design.feasibility_score = (feasibility_constraint + connectivity_constraint + (1.0 - stiffness_ratio_delta)) * -1.0
        design.stiffness_ratio = stiff_ratio
        design.set_objectives(v_stiffness * -1.0, vol_frac)
        design.h_stiffness = h_stiffness
        design.evaluated = True
        if bitstr not in self.unique_designs[task]:
            self.unique_designs[task].add(bitstr)
            self.unique_designs_vals[task].append([v_stiffness, vol_frac])
            self.unique_designs_epochs[task].append(self.curr_epoch)
            self.unique_designs_feasible[task].append(design.is_feasible)
            self.nfes[task] += 1
        if bitstr not in self.unique_designs_all:
            self.unique_designs_all.add(bitstr)
            self.nfe += 1

        return reward, design, performance_term, constraint_sum

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
        pred_probs = self.c_actor([observation_input, cross_input], training=use_actor_train_call)

        # Batch sampling
        all_token_probs = pred_probs[:, inf_idx, :]  # shape (batch, 2)
        all_token_log_probs = tf.math.log(all_token_probs + 1e-10)
        samples = tf.random.categorical(all_token_log_probs, 1)  # shape (batch, 1)
        next_bit_ids = tf.squeeze(samples, axis=-1)  # shape (batch,)
        batch_indices = tf.range(0, tf.shape(all_token_log_probs)[0], dtype=tf.int64)  # shape (batch,)
        next_bit_probs = tf.gather_nd(all_token_log_probs, tf.stack([batch_indices, next_bit_ids], axis=-1))

        actions = next_bit_ids  # (batch,)
        actions_log_prob = next_bit_probs  # (batch,)
        return actions_log_prob, actions, all_token_log_probs, all_token_probs

    def sample_critic(self, observation, parent_obs):
        inf_idx = len(observation[0]) - 1
        observation_input = tf.convert_to_tensor(observation, dtype=tf.float32)
        inf_idx = tf.convert_to_tensor(inf_idx, dtype=tf.int32)
        return self._sample_critic(observation_input, parent_obs, inf_idx)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(global_mini_batch_size, None), dtype=tf.float32),
        tf.TensorSpec(shape=(global_mini_batch_size, config.num_conditioning_vars), dtype=tf.float32),
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
        tf.TensorSpec(shape=(global_mini_batch_size, config.num_vars), dtype=tf.float32),
        tf.TensorSpec(shape=(global_mini_batch_size, config.num_vars), dtype=tf.int32),
        tf.TensorSpec(shape=(global_mini_batch_size, config.num_vars), dtype=tf.float32),
        tf.TensorSpec(shape=(global_mini_batch_size, config.num_vars), dtype=tf.float32),
        tf.TensorSpec(shape=(global_mini_batch_size, None), dtype=tf.float32),
        tf.TensorSpec(shape=(global_mini_batch_size, config.num_vars, 2), dtype=tf.float32),
        tf.TensorSpec(shape=(global_mini_batch_size, config.num_vars, 2), dtype=tf.float32),
    ])
    def train_actor(
            self,
            observation_buffer,
            action_buffer,
            logprobability_buffer,
            advantage_buffer,
            parent_buffer,
            all_logprobability_buffer,
            all_probs_buffer
    ):
        # print('-- TRAIN ACTOR --')
        # print('observation buffer:', observation_buffer.shape)
        # print('action buffer:', action_buffer.shape)
        # print('logprob buffer:', logprobability_buffer.shape)
        # print('advantage buffer:', advantage_buffer.shape)
        # print('parent buffer:', parent_buffer.shape)
        # print('pop vector:', pop_vector.shape)

        with tf.GradientTape() as tape:
            pred_probs = self.c_actor([observation_buffer, parent_buffer], training=use_actor_train_call)  # shape: (batch, seq_len, 2)
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
        pred_log_probs = tf.math.log(pred_probs + 1e-10)
        logprobability = tf.reduce_sum(
            tf.one_hot(action_buffer, self.num_actions) * pred_log_probs, axis=-1
        )  # shape (batch, seq_len)
        kl = tf.reduce_mean(
            logprobability_buffer - logprobability
        )
        kl = tf.reduce_sum(kl)

        #  KL Divergence
        # pred_probs = self.c_actor([observation_buffer, parent_buffer], training=False)
        # pred_log_probs = tf.math.log(pred_probs + 1e-10)  # shape (9, 280, 2)
        # true_kl = tf.reduce_sum(
        #     all_probs_buffer * (all_logprobability_buffer - pred_log_probs),
        #     axis=-1
        # )  # shape (9, 280)
        # kl = tf.reduce_mean(true_kl)  # shape (1,)

        return kl, entr, policy_loss, loss

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(global_mini_batch_size, config.num_vars+1), dtype=tf.float32),
        tf.TensorSpec(shape=(global_mini_batch_size, config.num_vars+1, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(global_mini_batch_size, None), dtype=tf.float32),
    ])
    def train_critic(
            self,
            observation_buffer,
            return_buffer,
            parent_buffer,
    ):

        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            pred_values = self.c_critic(
                [observation_buffer, parent_buffer], training=use_critic_train_call)  # (batch, seq_len, 2)

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
                if isinstance(value, list):
                    print(f"{key}: {value}", end=' | ')
                else:
                    print("%s: %.5f" % (key, value), end=' | ')
            print('nfe', self.nfe)


        # Update metrics
        self.returns.append(epoch_info['mb_return'])
        self.performance_returns.append(epoch_info['perf_return'])
        self.c_loss.append(epoch_info['c_loss'])
        self.p_loss.append(epoch_info['p_loss'])
        self.p_iter.append(epoch_info['p_iter'])
        self.entropy.append(epoch_info['entropy'])
        self.kl.append(epoch_info['kl'])
        self.constraint_returns.append(epoch_info['constraint_return'])

        if len(self.entropy) % self.plot_freq == 0:
            print('--> PLOTTING')
            self.plot_ppo()
            self.plot_designs()
        else:
            return

    def plot_designs(self):

        # Plot self.unique_designs_vals
        all_vals = []
        all_epochs = []
        for idx, ud in enumerate(self.unique_designs_vals):
            all_vals.extend(ud)
            all_epochs.extend(self.unique_designs_epochs[idx])
        all_x_vals = [x[0] for x in all_vals]
        all_y_vals = [x[1] for x in all_vals]

        # --- Plotting ---
        plt.figure(figsize=(8, 6))
        plt.scatter(all_x_vals, all_y_vals, c=all_epochs, s=10)
        plt.xlabel('Vertical Stiffness')
        plt.ylabel('Volume Fraction')
        plt.xlim(0, 1.1)
        plt.ylim(0, 1.1)
        plt.colorbar()
        plt.title('Design Space')
        file_name = os.path.join(self.run_dir, 'design_space_' + str(self.val_itr) + '.png')
        plt.savefig(file_name)
        plt.close('all')


    def plot_ppo(self):

        # --- Plotting ---
        epochs = [x for x in range(len(self.returns))]
        gs = gridspec.GridSpec(3, 2)
        fig = plt.figure(figsize=(16, 12))  # default [6.4, 4.8], W x H  9x6, 12x8
        fig.suptitle('Results', fontsize=16)

        # Returns plot
        plt.subplot(gs[0, 0])
        plt.plot(epochs, self.returns)
        plt.xlabel('Epoch')
        plt.ylabel('Mini-batch Return')
        plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5)
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
        plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5)
        plt.title('Critic Loss Plot')

        # Policy entropy plot
        plt.subplot(gs[1, 0])
        plt.plot(epochs, self.entropy)
        plt.xlabel('Epoch')
        plt.ylabel('Entropy')
        plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5)
        plt.title('Policy Entropy Plot')

        # KL divergence plot
        plt.subplot(gs[1, 1])
        plt.plot(epochs, self.kl)
        plt.xlabel('Epoch')
        plt.ylabel('KL')
        plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5)
        plt.title('KL Divergence Plot')

        # Performance Returns plot
        plt.subplot(gs[2, 0])
        plt.plot(epochs, self.performance_returns)
        plt.xlabel('Epoch')
        plt.ylabel('Performance Return')
        plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5)
        plt.title('Performance Return Plot')

        # Constraint Returns plot
        plt.subplot(gs[2, 1])
        plt.plot(epochs, self.constraint_returns)
        plt.xlabel('Epoch')
        plt.ylabel('Constraint Return')
        plt.title('Constraint Return Plot')
        plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5)
        plt.ylim(0.0, 0.6)

        # Save and close
        plt.tight_layout()
        save_path = os.path.join(self.run_dir, 'plots.png')
        if self.run_val is True:
            save_path = os.path.join(self.run_dir, 'plots_val_' + str(self.val_itr) + '.png')
        plt.savefig(save_path)
        plt.close('all')


from problem.TrussProblem import TrussProblem

if __name__ == '__main__':
    problem = TrussProblem(
        sidenum=config.sidenum,
        calc_constraints=True,
        target_stiffness_ratio=1.0,
        feasible_stiffness_delta=0.1,
    )

    actor_save_path = None
    critic_save_path = None

    # actor_save_path = os.path.join(config.results_save_dir, 'run_' + str(run_dir), 'pretrained', 'actor_weights_2600')
    # critic_save_path = os.path.join(config.results_save_dir, 'run_' + str(run_dir), 'pretrained', 'critic_weights_2600')

    alg = PPO_Constrained_MultiTask(
        run_num=run_dir,
        problem=problem,
        epochs=task_epochs,
        actor_load_path=actor_save_path,
        critic_load_path=critic_save_path,
        debug=True,
        c_type='uniform',
        run_val=True,
        val_itr=run_num,
        num_tasks=num_tasks,  # Tested with 9 tasks, try upping substantially
    )
    alg.run()