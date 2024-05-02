import numpy as np
import time
import tensorflow as tf
from copy import deepcopy
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os
import config
from math import cos, pi
import keras_nlp

# from modelC import get_multi_task_decoder as get_model
# from modelC import get_multi_task_decoder_moe as get_model
from modelC import get_hierarchical_mtd as get_model

from problem.TrussProblem import TrussProblem
from taskC.AbstractPpoMT import AbstractPpoMT

# Mini-batch Samples
weight_samples = 4
task_samples = 6
num_repeats = 2

# Run parameters
run_dir = 12
run_num = 3
task_epochs = 10000
num_tasks = 36

class HierarchicalMT(AbstractPpoMT):

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
        super(HierarchicalMT, self).__init__(run_num, problem, epochs, actor_load_path, critic_load_path, c_type, run_val, val_itr, num_tasks)


        # --- Hyperparameters ---
        self.clip_ratio = 0.2
        self.target_kl = 0.005  # was 0.01
        self.entropy_coef = 0.02  # was 0.02 originally
        self.actor_learning_rate = 0.0001  # 0.0001
        self.critic_learning_rate = 0.0001  # 0.0001

        # Plotting / Saving
        self.plot_freq = 25
        self.save_freq = 50
        self.history_epochs = 1000


    def get_constraint_weights(self):
        feasibility_coeff = 1.0
        connectivity_coeff = 1.0
        stiffness_ratio_coeff = 1.0
        # constraint_coeff = self.warup_learning_rate(500, 0.2, delay=150)
        constraint_coeff = 0.2
        return feasibility_coeff, connectivity_coeff, stiffness_ratio_coeff, constraint_coeff

    def warup_learning_rate(self, warmup_steps, final_learning_rate, delay=0):
        if delay > self.curr_epoch:
            return 0.0
        step = min(self.curr_epoch - delay, warmup_steps)
        warmup = step / warmup_steps
        curr_lr = warmup * final_learning_rate
        return curr_lr

    def load_models(self):
        self.c_actor, self.c_critic, self.c_critic_2 = get_model(self.actor_load_path, self.critic_load_path)
        return self.c_actor, self.c_critic, self.c_critic_2

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
        self.critic_optimizer_2 = tf.keras.optimizers.legacy.Adam(learning_rate=self.critic_learning_rate)

        self.c_actor, self.c_critic, self.c_critic_2 = self.load_models()

        if summary is True:
            self.c_actor.summary()
            # self.c_critic.summary()

    def exec_mini_batch(self):
        all_perf_returns = []
        all_constraint_returns = []
        all_total_rewards = []
        all_actions = [[] for _ in range(self.mini_batch_size)]
        all_rewards = [[] for _ in range(self.mini_batch_size)]
        all_logprobs = [[] for _ in range(self.mini_batch_size)]
        all_logprobs_full = [[] for _ in range(self.mini_batch_size)]  # Logprobs for all actions
        all_probs_full = [[] for _ in range(self.mini_batch_size)]
        designs = [[] for x in range(self.mini_batch_size)]
        observation = [[self.decision_start_token_id] for x in range(self.mini_batch_size)]
        critic_observation_buffer = [[] for x in range(self.mini_batch_size)]


        all_perf_returns_c = []
        all_total_rewards_c = []
        all_actions_c = [[] for _ in range(self.mini_batch_size)]
        all_rewards_c = [[] for _ in range(self.mini_batch_size)]
        all_logprobs_c = [[] for _ in range(self.mini_batch_size)]
        designs_c = [[] for x in range(self.mini_batch_size)]
        observation_c = [[self.decision_start_token_id] for x in range(self.mini_batch_size)]
        critic_observation_buffer_c = [[] for x in range(self.mini_batch_size)]

        # Get cross attention observation input
        cross_obs_tensor, weight_samples_all, task_samples_all = self.get_cross_obs(weight_samples, task_samples, num_repeats)

        # -------------------------------------
        # Sample Actor
        # -------------------------------------

        for t in range(self.steps_per_design):
            action_log_prob, action, all_action_log_probs, all_action_probs, actions_log_prob_c, actions_c, all_token_log_probs_c, all_token_probs_c = self.sample_actor(observation, cross_obs_tensor)  # returns shape: (batch,) and (batch,)
            action_log_prob = action_log_prob.numpy().tolist()
            all_action_log_probs = all_action_log_probs.numpy().tolist()
            all_action_probs = all_action_probs.numpy().tolist()

            actions_log_prob_c = actions_log_prob_c.numpy().tolist()

            observation_new = deepcopy(observation)
            observation_new_c = deepcopy(observation_c)

            for idx, act in enumerate(action.numpy()):
                all_actions[idx].append(deepcopy(act))
                all_logprobs[idx].append(action_log_prob[idx])
                all_logprobs_full[idx].append(all_action_log_probs[idx])
                all_probs_full[idx].append(all_action_probs[idx])
                m_action = int(deepcopy(act))
                designs[idx].append(m_action)
                observation_new[idx].append(m_action + 2)

                all_actions_c[idx].append(deepcopy(actions_c[idx]))
                all_logprobs_c[idx].append(actions_log_prob_c[idx])
                m_action_c = int(deepcopy(actions_c[idx]))
                designs_c[idx].append(m_action_c)
                observation_new_c[idx].append(m_action_c + 2)

            # Determine reward for each batch element
            if len(designs[0]) == self.steps_per_design:
                done = True
                for idx, design in enumerate(designs):
                    design_bitstr = ''.join([str(bit) for bit in design])
                    reward, terms = self.calc_reward(
                        design_bitstr,
                        [weight_samples_all[idx], task_samples_all[idx], False]
                    )
                    all_rewards[idx].append(reward)
                    all_total_rewards.append(reward)
                    all_perf_returns.append(terms[0])
                for idx, design_c in enumerate(designs_c):
                    design_bitstr_c = ''.join([str(bit) for bit in design_c])
                    reward_c, terms_c = self.calc_reward(
                        design_bitstr_c,
                        [weight_samples_all[idx], task_samples_all[idx], True]
                    )
                    all_rewards_c[idx].append(reward_c)
                    all_total_rewards_c.append(reward_c)
                    all_perf_returns_c.append(terms_c[0])
                    all_constraint_returns.append(terms_c[1])
            else:
                done = False
                reward = 0.0
                for idx, _ in enumerate(designs):
                    all_rewards[idx].append(reward)
                    all_rewards_c[idx].append(reward)

            # Update the observation
            if done is True:
                critic_observation_buffer = deepcopy(observation_new)
                critic_observation_buffer_c = deepcopy(observation_new_c)
            else:
                observation = observation_new
                observation_c = observation_new_c

        # -------------------------------------
        # Sample Critic
        # -------------------------------------

        # --- SINGLE CRITIC PREDICTION --- #
        value_t = self.sample_critic(critic_observation_buffer, cross_obs_tensor)
        value_t = value_t.numpy().tolist()  # (30, 31)
        for idx, value in zip(range(self.mini_batch_size), value_t):
            last_reward = value[-1]
            all_rewards[idx].append(last_reward)

        # --- CONSTRAINED CRITIC PREDICTION --- #
        value_t_c = self.sample_critic_c(critic_observation_buffer_c, cross_obs_tensor)
        value_t_c = value_t_c.numpy().tolist()  # (30, 31)
        for idx, value_c in zip(range(self.mini_batch_size), value_t_c):
            last_reward_c = value_c[-1]
            all_rewards_c[idx].append(last_reward_c)

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
            adv_tensor = AbstractPpoMT.discounted_cumulative_sums(
                deltas, self.gamma * self.lam
            )
            all_advantages[idx] = adv_tensor

            ret_tensor = AbstractPpoMT.discounted_cumulative_sums(
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
        # Calculate Advantage and Return Constrained
        # -------------------------------------

        all_advantages_c = [[] for _ in range(self.mini_batch_size)]
        all_returns_c = [[] for _ in range(self.mini_batch_size)]
        for idx in range(len(all_rewards_c)):
            rewards_c = np.array(all_rewards_c[idx])
            values_c = np.array(value_t_c[idx])
            deltas_c = rewards_c[:-1] + self.gamma * values_c[1:] - values_c[:-1]
            adv_tensor_c = AbstractPpoMT.discounted_cumulative_sums(
                deltas_c, self.gamma * self.lam
            )
            all_advantages_c[idx] = adv_tensor_c

            ret_tensor_c = AbstractPpoMT.discounted_cumulative_sums(
                rewards_c, self.gamma
            )  # [:-1]
            ret_tensor_c = np.array(ret_tensor_c, dtype=np.float32)
            all_returns_c[idx] = ret_tensor_c

        advantage_mean_c, advantage_std_c = (
            np.mean(all_advantages_c),
            np.std(all_advantages_c),
        )
        all_advantages_c = (all_advantages_c - advantage_mean_c) / advantage_std_c

        observation_tensor_c = tf.convert_to_tensor(observation_c, dtype=tf.float32)
        action_tensor_c = tf.convert_to_tensor(all_actions_c, dtype=tf.int32)
        logprob_tensor_c = tf.convert_to_tensor(all_logprobs_c, dtype=tf.float32)
        advantage_tensor_c = tf.convert_to_tensor(all_advantages_c, dtype=tf.float32)
        critic_observation_tensor_c = tf.convert_to_tensor(critic_observation_buffer_c, dtype=tf.float32)
        return_tensor_c = tf.convert_to_tensor(all_returns_c, dtype=tf.float32)
        return_tensor_c = tf.expand_dims(return_tensor_c, axis=-1)

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
                all_probs_tensor,

                observation_tensor_c,
                action_tensor_c,
                logprob_tensor_c,
                advantage_tensor_c,
            )
            if kl > 1.5 * self.target_kl:
                # Early Stopping
                break

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
            value_loss += self.train_critic_c(
                critic_observation_tensor_c,
                return_tensor_c,
                cross_obs_tensor,
            )

        # -------------------------------------
        # Update Results
        # -------------------------------------

        # Update results tracker
        epoch_info = {
            'return': np.mean(all_total_rewards),
            'return_c': np.mean(all_total_rewards_c),
            'perf_return': np.mean(all_perf_returns),
            'constraint_return': np.mean(all_constraint_returns),
            'a_loss': actor_loss.numpy(),  # 'c_loss': value_loss.numpy(),
            'c_loss': value_loss.numpy(),
            'p_loss': policy_loss.numpy(),
            'p_iter': policy_update_itr,
            'entropy': entr.numpy(),
            'kl': kl.numpy(),
            'tasks': list(set(task_samples_all)),
        }
        return epoch_info


    def calc_reward(self, bitstr, params):
        weight, task, use_constraints = params[0], params[1], params[2]

        # 1. Performance reward
        performance_term, results = self.calc_performance(bitstr, params)
        h_stiffness, v_stiffness, stiff_ratio, vol_frac, constraints, features = results

        # 2. Constraint reward
        constraint_term, c_terms, is_feasible = self.calc_constraints(results, params)

        # 3. Calculate total reward
        if use_constraints == False:
            total_reward = performance_term
        else:
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


    def sample_critic_c(self, observation, parent_obs):
        inf_idx = len(observation[0]) - 1
        observation_input = tf.convert_to_tensor(observation, dtype=tf.float32)
        inf_idx = tf.convert_to_tensor(inf_idx, dtype=tf.int32)
        return self._sample_critic_c(observation_input, parent_obs, inf_idx)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    ])
    def _sample_critic_c(self, observation_input, parent_input, inf_idx):
        t_value = self.c_critic_2([observation_input, parent_input], training=self.use_critic_train_call)  # (batch, seq_len, 2)
        t_value = t_value[:, :, 0]
        return t_value



    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),  # shape=(global_mini_batch_size, None)
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    ])
    def _sample_actor(self, observation_input, cross_input, inf_idx):
        # print('sampling actor', inf_idx)
        pred_probs, pred_probs_c = self.c_actor([observation_input, cross_input], training=self.use_actor_train_call)

        # Batch sampling
        all_token_probs = pred_probs[:, inf_idx, :]  # shape (batch, 2)
        all_token_log_probs = tf.math.log(all_token_probs + 1e-10)
        samples = tf.random.categorical(all_token_log_probs, 1)  # shape (batch, 1)
        next_bit_ids = tf.squeeze(samples, axis=-1)  # shape (batch,)
        batch_indices = tf.range(0, tf.shape(all_token_log_probs)[0], dtype=tf.int64)  # shape (batch,)
        next_bit_probs = tf.gather_nd(all_token_log_probs, tf.stack([batch_indices, next_bit_ids], axis=-1))

        actions = next_bit_ids  # (batch,)
        actions_log_prob = next_bit_probs  # (batch,)

        # Batch Sampling Constrained
        all_token_probs_c = pred_probs_c[:, inf_idx, :]  # shape (batch, 2)
        all_token_log_probs_c = tf.math.log(all_token_probs_c + 1e-10)
        samples_c = tf.random.categorical(all_token_log_probs_c, 1)  # shape (batch, 1)
        next_bit_ids_c = tf.squeeze(samples_c, axis=-1)  # shape (batch,)
        batch_indices_c = tf.range(0, tf.shape(all_token_log_probs_c)[0], dtype=tf.int64)  # shape (batch,)
        next_bit_probs_c = tf.gather_nd(all_token_log_probs_c, tf.stack([batch_indices_c, next_bit_ids_c], axis=-1))

        actions_c = next_bit_ids_c  # (batch,)
        actions_log_prob_c = next_bit_probs_c  # (batch,)

        return actions_log_prob, actions, all_token_log_probs, all_token_probs, actions_log_prob_c, actions_c, all_token_log_probs_c, all_token_probs_c

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),

        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    ])
    def train_actor(
            self,
            observation_buffer,
            action_buffer,
            logprobability_buffer,
            advantage_buffer,
            parent_buffer,
            all_logprobability_buffer,
            all_probs_buffer,

            observation_buffer_c,
            action_buffer_c,
            logprobability_buffer_c,
            advantage_buffer_c,
    ):

        with tf.GradientTape() as tape:
            pred_probs, pred_probs_c = self.c_actor([observation_buffer, parent_buffer], training=self.use_actor_train_call)  # shape: (batch, seq_len, 2)

            ### PPO Loss ###
            pred_log_probs = tf.math.log(pred_probs + 1e-10)  # shape: (batch, seq_len, 2)
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

            ### PPO Loss Constrained ###
            pred_log_probs_c = tf.math.log(pred_probs_c + 1e-10)  # shape: (batch, seq_len, 2)
            logprobability_c = tf.reduce_sum(
                tf.one_hot(action_buffer_c, self.num_actions) * pred_log_probs_c, axis=-1
            )  # shape (batch, seq_len)
            loss_c = 0
            ratio_c = tf.exp(
                logprobability_c - logprobability_buffer_c
            )
            min_advantage_c = tf.where(
                advantage_buffer_c > 0,
                (1 + self.clip_ratio) * advantage_buffer_c,
                (1 - self.clip_ratio) * advantage_buffer_c,
            )
            policy_loss_c = -tf.reduce_mean(
                tf.minimum(ratio_c * advantage_buffer_c, min_advantage_c)
            )
            loss_c += policy_loss_c

            # Entropy Term Constrained
            entr_c = -tf.reduce_sum(pred_probs_c * pred_log_probs_c, axis=-1)  # shape (batch, seq_len)
            entr_c = tf.reduce_mean(entr_c)  # Higher positive value means more exploration - shape (batch,)
            loss_c = loss_c - (self.entropy_coef * entr_c)

            # Combine Losses
            loss = loss + loss_c

        policy_grads = tape.gradient(loss, self.c_actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(policy_grads, self.c_actor.trainable_variables))

        # KL Divergence
        kl = self.approx_kl_divergence(observation_buffer, parent_buffer, action_buffer, logprobability_buffer)
        kl_c = self.approx_kl_divergence_c(observation_buffer_c, parent_buffer, action_buffer_c, logprobability_buffer_c)
        kl = (kl + kl_c) / 2

        # Return terms
        return kl, entr, policy_loss, loss

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    ])
    def approx_kl_divergence(self, observation_buffer, parent_buffer, action_buffer, logprobability_buffer):
        pred_probs, pred_probs_c = self.c_actor([observation_buffer, parent_buffer], training=self.use_actor_train_call)
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
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    ])
    def approx_kl_divergence_c(self, observation_buffer, parent_buffer, action_buffer, logprobability_buffer):
        pred_probs, pred_probs_c = self.c_actor([observation_buffer, parent_buffer], training=self.use_actor_train_call)
        pred_log_probs = tf.math.log(pred_probs_c + 1e-10)
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
    def train_critic_c(
            self,
            observation_buffer,
            return_buffer,
            parent_buffer,
    ):
        with tf.GradientTape() as tape:
            pred_values = self.c_critic_2([observation_buffer, parent_buffer], training=self.use_critic_train_call)  # (batch, seq_len, 2)

            # Value Loss (mse)
            value_loss = tf.reduce_mean((return_buffer - pred_values) ** 2)

        critic_grads = tape.gradient(value_loss, self.c_critic_2.trainable_variables)
        self.critic_optimizer_2.apply_gradients(zip(critic_grads, self.c_critic_2.trainable_variables))

        return value_loss

    def plot_results(self):
        self.plot_metrics()
        self.plot_designs()

    def plot_metrics(self):

        # --- Plotting ---
        epochs = [x for x in range(len(self.metrics['return']))]
        gs = gridspec.GridSpec(3, 2)
        fig = plt.figure(figsize=(16, 12))  # default [6.4, 4.8], W x H  9x6, 12x8
        fig.suptitle('Results', fontsize=16)

        # Returns plot
        plt.subplot(gs[0, 0])
        plt.plot(epochs, self.metrics['return'])
        plt.xlabel('Epoch')
        plt.ylabel('Mini-batch Return')
        plt.title('PPO Return Plot')

        # Critic loss plot
        plt.subplot(gs[0, 1])
        if len(self.metrics['c_loss']) < 100:
            c_loss = self.metrics['c_loss']
            c_epochs = epochs
        else:
            c_loss = self.metrics['c_loss'][50:]
            c_epochs = epochs[50:]
        plt.plot(c_epochs, c_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Critic loss')
        plt.title('Critic Loss Plot')

        # Policy entropy plot
        plt.subplot(gs[1, 0])
        plt.plot(epochs, self.metrics['entropy'])
        plt.xlabel('Epoch')
        plt.ylabel('Entropy')
        plt.title('Policy Entropy Plot')

        # KL divergence plot
        plt.subplot(gs[1, 1])
        plt.plot(epochs, self.metrics['kl'])
        plt.xlabel('Epoch')
        plt.ylabel('KL')
        plt.title('KL Divergence Plot')

        # Performance Returns plot
        plt.subplot(gs[2, 0])
        plt.plot(epochs, self.metrics['perf_return'])
        plt.xlabel('Epoch')
        plt.ylabel('Performance Return')
        plt.title('Performance Return Plot')

        # Constraint Returns plot
        plt.subplot(gs[2, 1])
        plt.plot(epochs, self.metrics['constraint_return'])
        plt.xlabel('Epoch')
        plt.ylabel('Constraint Return')
        plt.title('Constraint Return Plot')

        # Save and close
        plt.tight_layout()
        # save_path = os.path.join(self.run_dir, 'plots.png')
        save_path = os.path.join(self.run_dir, 'plots_val_' + str(self.val_itr) + '.png')
        plt.savefig(save_path)
        plt.close('all')


    def plot_designs(self):

        # Plot design history
        points_x = [val[0] for val in self.design_history]
        points_y = [val[1] for val in self.design_history]
        epoch_class = [val[2] for val in self.design_history]

        # plot with epoch class as color dimension
        scatter = plt.scatter(points_x, points_y, c=epoch_class, cmap='viridis', s=10)
        plt.colorbar(scatter, label='Epoch')
        plt.xlim(0, 1.1)
        plt.ylim(0, 1.1)
        plt.xlabel('Vertical Stiffness')
        plt.ylabel('Volume Fraction')
        save_path = os.path.join(self.run_dir, 'designs_val_' + str(self.val_itr) + '.png')
        plt.savefig(save_path)
        plt.close('all')


if __name__ == '__main__':
    target_stiffness_ratio = 1.0
    feasible_stiffness_delta = 0.01
    problem = TrussProblem(
        sidenum=config.sidenum,
        calc_constraints=True,
        target_stiffness_ratio=target_stiffness_ratio,
        feasible_stiffness_delta=feasible_stiffness_delta,
    )

    actor_save_path = None
    critic_save_path = None
    alg = HierarchicalMT(
        run_num=run_dir,
        problem=problem,
        epochs=task_epochs,
        actor_load_path=actor_save_path,
        critic_load_path=critic_save_path,
        c_type='uniform',
        run_val=False,
        val_itr=run_num,
        num_tasks=num_tasks,  # Tested with 9 tasks, try upping substantially
    )
    alg.run()








