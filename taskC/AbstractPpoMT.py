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
from modelC import get_multi_task_decoder as get_model
from collections import OrderedDict
import tensorflow_addons as tfa


# Sampling parameters
# num_weight_samples = 4
# num_task_samples = 6
# repeat_size = 2
# global_mini_batch_size = num_weight_samples * num_task_samples * repeat_size

# Set seeds to 0
random.seed(0)
tf.random.set_seed(0)


class AbstractPpoMT(AbstractTask):

    @staticmethod
    def discounted_cumulative_sums(x, discount):
        # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def __init__(
            self,
            run_num=0,
            problem=None,
            epochs=1000,
            actor_load_path=None,
            critic_load_path=None,
            c_type='uniform',
            run_val=False,
            val_itr=0,
            num_tasks=9
    ):
        super(AbstractPpoMT, self).__init__(run_num, None, problem, epochs, actor_load_path, critic_load_path)
        self.c_type = c_type
        self.run_val = run_val
        self.val_itr = val_itr
        self.num_tasks = num_tasks

        # Pretrain save dir
        self.pretrain_save_dir = os.path.join(self.run_dir, 'pretrained')
        if not os.path.exists(self.pretrain_save_dir):
            os.makedirs(self.pretrain_save_dir)
        self.actor_pretrain_save_path = os.path.join(self.pretrain_save_dir, 'actor_weights')
        self.critic_pretrain_save_path = os.path.join(self.pretrain_save_dir, 'critic_weights')

        # Optimizer parameters
        self.actor_learning_rate = 0.0001  # 0.0001
        self.critic_learning_rate = 0.0001  # 0.0001
        self.train_actor_iterations = 250  # was 250
        self.train_critic_iterations = 40  # was 40
        self.use_actor_train_call = False
        self.use_critic_train_call = False

        # HV
        self.ref_point = np.array(config.hv_ref_point)  # value, weight
        self.hv_client = HV(self.ref_point)
        self.unique_designs = [set() for _ in range(self.num_tasks)]
        self.unique_designs_vals = [[] for _ in range(self.num_tasks)]
        self.unique_designs_feasible = [[] for _ in range(self.num_tasks)]

        # Algorithm parameters
        self.nfe = 0
        self.nfes = [0 for _ in range(self.num_tasks)]
        self.task_nfes = [[] for _ in range(self.num_tasks)]
        self.epochs = epochs
        self.steps_per_design = config.num_vars  # 30 | 60

        # Sampling parameters
        self.weight_samples = 4
        self.task_samples = 6
        self.repeat_size = 2
        self.mini_batch_size = self.weight_samples * self.task_samples * self.repeat_size

        # PPO Parameters
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_ratio = 0.2
        self.target_kl = 0.005    # was 0.01
        self.entropy_coef = 0.02  # was 0.02 originally
        self.counter = 0
        self.decision_start_token_id = 1
        self.num_actions = 2
        self.curr_epoch = 0

        # Plotting / Saving
        self.plot_freq = 50
        self.save_freq = 50

        # Define objectives
        self.objective_weights = self.define_objectives(num_weights=9, lb=0.05, ub=0.95)
        self.objective_weights_idx = [x for x in range(len(self.objective_weights))]

        # Define tasks
        self.tasks = [i for i in range(len(self.problem.train_problems))]
        self.run_tasks = [i for i in range(self.num_tasks)]

        # Metrics / History
        self.metrics = {}
        self.design_history = []
        self.history_epochs = 100


    def define_objectives(self, num_weights=9, lb=0.05, ub=0.95):
        return list(np.linspace(lb, ub, num_weights))

    def update_history(self):
        epoch_list = [x[2] for x in self.design_history]
        max_epoch = max(epoch_list)
        min_epoch = max_epoch - self.history_epochs
        if min_epoch < 0:
            min_epoch = 0
        self.design_history = [x for x in self.design_history if x[2] >= min_epoch]


    def build(self, warmup=None, summary=True):
        if warmup is not None:
            warmup_steps, decay_coeff, decay_steps = warmup
            self.actor_learning_rate = tf.keras.optimizers.schedules.CosineDecay(
                0.0,  # initial learning rate
                decay_steps,  # decay_steps
                alpha=decay_coeff,
                warmup_target=self.actor_learning_rate,
                warmup_steps=warmup_steps
            )

        # Optimizers
        if self.actor_optimizer is None:
            self.actor_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.actor_learning_rate)
            # self.actor_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=self.actor_learning_rate)
        if self.critic_optimizer is None:
            self.critic_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.critic_learning_rate)
            # self.critic_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=self.critic_learning_rate)

        self.c_actor, self.c_critic = self.load_models()

        if summary is True:
            self.c_actor.summary()
            # self.c_critic.summary()

    def load_models(self):
        self.c_actor, self.c_critic = get_model(self.actor_load_path, self.critic_load_path)
        return self.c_actor, self.c_critic

    def save_pretrained(self):
        t_actor_save_path = os.path.join(self.pretrain_save_dir, 'actor_weights_' + str(self.curr_epoch))
        t_critic_save_path = os.path.join(self.pretrain_save_dir, 'critic_weights_' + str(self.curr_epoch))
        self.c_actor.save_weights(t_actor_save_path)
        self.c_critic.save_weights(t_critic_save_path)

    def run(self):
        self.build()

        for x in range(self.epochs):

            # Run mini-batch
            self.curr_epoch = x
            epoch_info = self.exec_mini_batch()

            # Record results
            self.record(epoch_info)

            # Save pre-trained weights
            if self.curr_epoch % self.save_freq == 0:
                self.save_pretrained()

        # Plot results
        self.save_pretrained()

    # -------------------------------------
    # PPO Functions
    # -------------------------------------

    def get_cross_obs(self, num_weights, num_tasks, num_repeats):

        # Weight samples
        weight_samples = random.sample(self.objective_weights, num_weights)

        # Task samples
        task_samples = random.sample(self.run_tasks, num_tasks)

        # Build samples
        cross_obs_vecs = []
        weight_samples_all = []
        task_samples_all = []
        for weight in weight_samples:
            for task in task_samples:
                task_cond_vars = self.problem.train_problems_norm[task]
                cross_obs_vector = [weight, task_cond_vars[1], task_cond_vars[2], task_cond_vars[3]]
                cross_obs_vecs.append(cross_obs_vector)
                weight_samples_all.append(weight)
                task_samples_all.append(task)

        # Repeat samples
        weight_samples_all = [element for element in weight_samples_all for _ in range(num_repeats)]
        task_samples_all = [element for element in task_samples_all for _ in range(num_repeats)]
        cross_obs_vecs = [element for element in cross_obs_vecs for _ in range(num_repeats)]

        # Convert to tensor
        cross_obs_tensor = tf.convert_to_tensor(cross_obs_vecs, dtype=tf.float32)

        return cross_obs_tensor, weight_samples_all, task_samples_all


    def exec_mini_batch(self):
        return {}

    # -------------------------------------
    # Reward Functions
    # -------------------------------------

    def calc_reward(self, bitstr, params):
        weight, task = params

        # 1. Performance reward
        performance_term, results = self.calc_performance(bitstr, params)
        h_stiffness, v_stiffness, stiff_ratio, vol_frac, constraints, features = results

        # 2. Constraint reward
        constraint_term, c_terms, is_feasible = self.calc_constraints(results, params)

        # 3. Calculate total reward
        total_reward = performance_term + constraint_term

        # 4. Add to task specific unique designs
        if bitstr not in self.unique_designs[task]:
            self.unique_designs[task].add(bitstr)
            self.unique_designs_vals[task].append([v_stiffness, vol_frac])
            self.unique_designs_feasible[task].append(is_feasible)
            self.nfes[task] += 1

        # 4. Add to design history
        self.design_history.append([v_stiffness, vol_frac, self.curr_epoch])

        # 5. Aggregate terms
        terms = [performance_term, constraint_term]
        terms.extend(c_terms)
        terms.append(results[-1])
        return total_reward, terms


    def calc_performance(self, bitstr, params):
        weight, task = params[0], params[1]

        results = self.problem.evaluate(bitstr, problem_num=task, run_val=self.run_val)
        h_stiffness, v_stiffness, stiff_ratio, vol_frac, constraints, features = results
        h_stiffness = float(h_stiffness)
        v_stiffness = float(v_stiffness)
        stiff_ratio = float(stiff_ratio)
        vol_frac = float(vol_frac)

        # -------------------------------------
        # Calculate performance reward
        # -------------------------------------

        w1 = weight
        w2 = 1.0 - weight
        v_stiff_term = w1 * v_stiffness
        vol_frac_term = w2 * (1.0 - vol_frac)
        performance_term = v_stiff_term + vol_frac_term
        performance_term = performance_term * self.get_performance_weights()

        return performance_term, results

    def get_performance_weights(self):
        return 1.0

    def calc_constraints(self, results, params):
        constraints = results[4]

        # 1. Feasibility Term (minimize)
        feasibility_constraint = float(constraints[0])
        feasibility_term = (1.0 - feasibility_constraint)

        # 2. Connectivity Term (minimize)
        connectivity_constraint = float(constraints[1])
        connectivity_term = (1.0 - connectivity_constraint)

        # 3. Stiffness Ratio Term (minimize)
        stiff_ratio = results[2]
        stiffness_ratio_delta = abs(self.problem.target_stiffness_ratio - stiff_ratio)
        if stiffness_ratio_delta < self.problem.feasible_stiffness_delta:
            in_stiffness_window = True
            stiffness_ratio_term = 0.0
        else:
            in_stiffness_window = False
            if stiffness_ratio_delta > 10.0:
                stiffness_ratio_term = 10.0
            else:
                stiffness_ratio_term = stiffness_ratio_delta

        # 4. Calculate constraint term
        fea_multiplier, conn_multiplier, str_multiplier, constraint_multiplier = self.get_constraint_weights()
        constraint_term = ((feasibility_term * fea_multiplier) + (connectivity_term * conn_multiplier) + (stiffness_ratio_term * str_multiplier))
        constraint_term *= -1.0
        constraint_term = constraint_term * constraint_multiplier

        # 5. Check feasibility
        is_feasible = False
        if feasibility_constraint == 1.0 and connectivity_constraint == 1.0 and in_stiffness_window is True:
            is_feasible = True

        # 6. Gather terms for recording
        terms = [feasibility_term, connectivity_term, stiffness_ratio_term]
        return constraint_term, terms, is_feasible

    def get_constraint_weights(self):
        feasibility_coeff = 1.0
        connectivity_coeff = 1.0
        stiffness_ratio_coeff = 1.0
        constraint_coeff = 1.0
        return feasibility_coeff, connectivity_coeff, stiffness_ratio_coeff, constraint_coeff

    # -------------------------------------
    # Actor-Critic Sampling
    # -------------------------------------

    def sample_actor(self, observation, cross_obs):
        inf_idx = len(observation[0]) - 1  # all batch elements have the same length
        observation_input = deepcopy(observation)
        observation_input = tf.convert_to_tensor(observation_input, dtype=tf.float32)
        inf_idx = tf.convert_to_tensor(inf_idx, dtype=tf.int32)
        return self._sample_actor(observation_input, cross_obs, inf_idx)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),  # shape=(global_mini_batch_size, None)
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    ])
    def _sample_actor(self, observation_input, cross_input, inf_idx):
        # print('sampling actor', inf_idx)
        pred_probs = self.c_actor([observation_input, cross_input], training=self.use_actor_train_call)

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
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    ])
    def _sample_critic(self, observation_input, parent_input, inf_idx):
        t_value = self.c_critic([observation_input, parent_input], training=self.use_critic_train_call)  # (batch, seq_len, 2)
        t_value = t_value[:, :, 0]
        return t_value

    # -------------------------------------
    # Actor-Critic Training
    # -------------------------------------

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
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

        with tf.GradientTape() as tape:
            pred_probs = self.c_actor([observation_buffer, parent_buffer], training=self.use_actor_train_call)  # shape: (batch, seq_len, 2)
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

        # KL Divergence
        # kl = self.calc_kl_divergence(observation_buffer, parent_buffer, all_probs_buffer, all_logprobability_buffer)
        kl = self.approx_kl_divergence(observation_buffer, parent_buffer, action_buffer, logprobability_buffer)

        # Return terms
        return kl, entr, policy_loss, loss


    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
    ])
    def calc_kl_divergence(self, observation_buffer, parent_buffer, all_probs_buffer, all_logprobability_buffer):
        pred_probs = self.c_actor([observation_buffer, parent_buffer], training=False)
        pred_log_probs = tf.math.log(pred_probs + 1e-10)  # shape (9, 280, 2)
        true_kl = tf.reduce_sum(
            all_probs_buffer * (all_logprobability_buffer - pred_log_probs),
            axis=-1
        )  # shape (9, 280)
        kl = tf.reduce_mean(true_kl)  # shape (1,)
        return kl

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    ])
    def approx_kl_divergence(self, observation_buffer, parent_buffer, action_buffer, logprobability_buffer):
        pred_probs = self.c_actor([observation_buffer, parent_buffer], training=self.use_actor_train_call)
        pred_log_probs = tf.math.log(pred_probs + 1e-10)
        logprobability = tf.reduce_sum(
            tf.one_hot(action_buffer, self.num_actions) * pred_log_probs, axis=-1
        )  # shape (batch, seq_len)
        kl = tf.reduce_mean(
            logprobability_buffer - logprobability
        )
        kl = tf.reduce_sum(kl)
        return kl

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    ])
    def train_critic(
            self,
            observation_buffer,
            return_buffer,
            parent_buffer,
    ):
        with tf.GradientTape() as tape:
            pred_values = self.c_critic([observation_buffer, parent_buffer], training=self.use_critic_train_call)  # (batch, seq_len, 2)

            # Value Loss (mse)
            value_loss = tf.reduce_mean((return_buffer - pred_values) ** 2)

        critic_grads = tape.gradient(value_loss, self.c_critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.c_critic.trainable_variables))

        return value_loss

    # -------------------------------------
    # Record
    # -------------------------------------

    def record(self, epoch_info):
        if epoch_info is None:
            return

        # 1. Print results
        print(f"PPO-MT {self.run_num} - {self.curr_epoch} ", end=' ')
        for key, value in epoch_info.items():
            if isinstance(value, list):
                print(f"{key}: {value}", end=' | ')
            else:
                print("%s: %.5f" % (key, value), end=' | ')
        print(sum(self.nfes))

        # 2. Update metrics
        for key, value in epoch_info.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

        # 2.1 Trim design history to max len
        self.update_history()

        # 3. Plot results
        if self.curr_epoch % self.plot_freq == 0:
            self.plot_results()

    # -------------------------------------
    # Plot Results
    # -------------------------------------

    def plot_results(self):
        pass

