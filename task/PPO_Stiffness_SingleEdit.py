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
from task.GA_Task import GA_Task
from model import get_stiffness_encoder as get_model
from collections import OrderedDict
import tensorflow_addons as tfa


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


design_len = 30


# Local Population
use_local_population = True
local_population_size = 30

# Mini-batch parameters
global_mini_batch_size = 10
trajectory_len = 10

# Task parameters
run_dir = 0
val_itr = 2
task_epochs = 1000



class PPO_Stiffness(AbstractTask):

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
        super(PPO_Stiffness, self).__init__(run_num, barrier, problem, epochs, actor_load_path, critic_load_path)
        self.debug = debug
        self.c_type = c_type
        self.run_val = run_val
        self.val_itr = val_itr
        self.val_task = val_task

        # HV
        self.ref_point = np.array(config.hv_ref_point)  # value, weight
        self.hv_client = HV(self.ref_point)
        self.nds = NonDominatedSorting()
        self.unique_designs = {}
        self.unique_designs_in_window = []

        # Algorithm parameters
        self.pop_size = 30  # 32 FU_NSGA2, 10 U_NSGA2
        self.offspring_size = global_mini_batch_size  # 32 FU_NSGA2, 30 U_NSGA2
        self.mini_batch_size = global_mini_batch_size
        self.nfe = 0
        self.epochs = epochs
        self.steps_per_trajectory = trajectory_len  # 30 | 60

        # PPO alg parameters
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_ratio = 0.2  # was 0.2
        self.target_kl = 0.001  # was 0.01
        self.entropy_coef = 0.000  # was 0.02 originally
        self.counter = 0
        self.decision_start_token_id = 1
        self.num_actions = 30
        self.curr_epoch = 0

        # Population
        self.population = self.get_random_designs(sample_size=local_population_size, design_len=30)
        self.hv = []
        self.nfes = []

        # Results
        self.plot_freq = 50

        # Problem Parameters
        self.min_stiffness_ratio_delta = 100.0
        self.stiffness_ratio_returns = []


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

        self.c_actor, self.c_critic = get_model(self.actor_load_path, self.critic_load_path)

    def run(self):
        self.build()

        for x in range(self.epochs):
            self.curr_epoch = x
            epoch_info = self.fast_mini_batch()
            self.record(epoch_info)




    # -------------------------------------
    # PPO Functions
    # -------------------------------------

    @staticmethod
    def get_random_designs(sample_size=global_mini_batch_size, design_len=30):
        num_ones = np.random.randint(1, design_len, sample_size)
        bitstrings = []
        for i in range(sample_size):
            design = np.zeros(design_len)
            design[:num_ones[i]] = 1
            np.random.shuffle(design)
            bitstrings.append(list(design))
        return bitstrings

    def flip_bit(self, bit_list, flip_idx):
        bit_list[flip_idx] = 1 - bit_list[flip_idx]
        return bit_list


    def fast_mini_batch(self):
        children = []
        designs = []


        all_total_rewards = []
        all_actions = [[] for _ in range(self.mini_batch_size)]
        all_rewards = [[] for _ in range(self.mini_batch_size)]
        all_logprobs = [[] for _ in range(self.mini_batch_size)]
        all_critic_rewards = [[] for _ in range(self.mini_batch_size)]
        all_stiffness_ratio_deltas = [[] for _ in range(self.mini_batch_size)]

        observation = []  # Fill with trajectories, each trajectory is a list of designs

        if use_local_population is True:
            start_designs = deepcopy(random.sample(self.population, self.mini_batch_size))
        else:
            start_designs = self.get_random_designs(sample_size=self.mini_batch_size, design_len=30)

        for idx, start_design in enumerate(start_designs):
            trajectory = []
            trajectory.append(start_design)
            observation.append(trajectory)

        critic_observation_buffer = None

        # -------------------------------------
        # Sample Actor / Critic
        # -------------------------------------

        for t in range(self.steps_per_trajectory):
            last_observations = deepcopy([traj[-1] for traj in observation])
            action_log_prob, action, all_action_probs = self.sample_actor(last_observations)
            action_log_prob = action_log_prob.numpy().tolist()

            # Get next state for each action
            next_observations = deepcopy(last_observations)
            for idx, act in enumerate(action.numpy()):
                all_actions[idx].append(act)
                all_logprobs[idx].append(action_log_prob[idx])
                m_action = int(deepcopy(act))
                self.flip_bit(next_observations[idx], m_action)

            # Get rewards for each action
            idx = 0
            for last_obs, next_obs in zip(last_observations, next_observations):
                reward, stiff_ratio_delta = self.calc_reward(last_obs, next_obs)
                all_rewards[idx].append(reward)
                all_stiffness_ratio_deltas[idx].append(stiff_ratio_delta)
                idx += 1

            # Get critic rewards for each action
            critic_rewards = self.sample_critic(last_observations)
            critic_rewards = critic_rewards.numpy().tolist()
            for idx, c_reward in enumerate(critic_rewards):
                all_critic_rewards[idx].append(c_reward)

            # Update observation
            if t == self.steps_per_trajectory - 1:
                critic_observation_buffer = deepcopy(observation)
                for idx, next_obs in enumerate(next_observations):
                    critic_observation_buffer[idx].append(next_obs)
            else:
                for idx, next_obs in enumerate(next_observations):
                    observation[idx].append(next_obs)

            # Finish trajectory if we are on last step
            if t == self.steps_per_trajectory - 1:
                critic_rewards = self.sample_critic(next_observations)
                critic_rewards = critic_rewards.numpy().tolist()
                for idx, c_reward in enumerate(critic_rewards):
                    all_critic_rewards[idx].append(c_reward)
                    all_rewards[idx].append(c_reward)


        # get all vals for final designs
        # print('Trajectory 1 Rewards:', all_rewards[0])
        all_total_rewards = [np.mean(traj_rewards) for traj_rewards in all_rewards]
        all_stiffness_ratio_delta_mins = [min(traj_stiffness_ratio_deltas) for traj_stiffness_ratio_deltas in all_stiffness_ratio_deltas]


        # -------------------------------------
        # Calculate Advantage and Return
        # -------------------------------------

        all_advantages = [[] for _ in range(self.mini_batch_size)]
        all_returns = [[] for _ in range(self.mini_batch_size)]
        for idx in range(len(all_rewards)):
            rewards = np.array(all_rewards[idx])
            values = np.array(all_critic_rewards[idx])
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

        observation_tensor = tf.convert_to_tensor(observation, dtype=tf.float32)  # (trajectories, steps, design_len)
        action_tensor = tf.convert_to_tensor(all_actions, dtype=tf.int32)  # (trajectories, steps)
        logprob_tensor = tf.convert_to_tensor(all_logprobs, dtype=tf.float32)  # (trajectories, steps)
        advantage_tensor = tf.convert_to_tensor(all_advantages, dtype=tf.float32)  # (trajectories, steps)

        critic_observation_tensor = tf.convert_to_tensor(critic_observation_buffer, dtype=tf.float32)  # (trajectories, steps+1, design_len)
        return_tensor = tf.convert_to_tensor(all_returns, dtype=tf.float32)  # (trajectories, steps+1)

        # -- Flatten trajectories and steps axis to batch axis for training --
        observation_tensor = tf.reshape(observation_tensor, (-1, design_len))
        action_tensor = tf.reshape(action_tensor, (-1,))
        logprob_tensor = tf.reshape(logprob_tensor, (-1,))
        advantage_tensor = tf.reshape(advantage_tensor, (-1,))

        critic_observation_tensor = tf.reshape(critic_observation_tensor, (-1, design_len))
        return_tensor = tf.reshape(return_tensor, (-1,))

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
        for i in range(self.train_critic_iterations):
            value_loss = self.train_critic(
                critic_observation_tensor,
                return_tensor,
            )
        value_loss = value_loss.numpy()

        # Update results tracker
        epoch_info = {
            'mb_return': np.mean(all_total_rewards),
            'sr_delta_min': np.mean(all_stiffness_ratio_delta_mins),
            'c_loss': value_loss,
            'p_loss': policy_loss,
            'p_iter': policy_update_itr,
            'entropy': entr,
            'kl': kl,
        }

        return epoch_info


    # -------------------------------------
    # Reward Calculation
    # -------------------------------------

    def calc_reward(self, last_obs, next_obs):

        # Did we move closer or further from the target?
        # Move delta will be positive if we moved closer
        # Move delta will be negative if we moved further
        last_obs_stiffness_ratio = self.get_or_eval_design(last_obs)
        next_obs_stiffness_ratio = self.get_or_eval_design(next_obs)

        last_obs_stiff_delta = abs(last_obs_stiffness_ratio - self.problem.target_stiffness_ratio)
        next_obs_stiff_delta = abs(next_obs_stiffness_ratio - self.problem.target_stiffness_ratio)

        if next_obs_stiff_delta > 1.0:
            next_stiff_delta_used = 1.0
        else:
            next_stiff_delta_used = next_obs_stiff_delta

        if last_obs_stiff_delta > 1.0:
            last_stiff_delta_used = 1.0
        else:
            last_stiff_delta_used = last_obs_stiff_delta

        if min(last_obs_stiff_delta, next_obs_stiff_delta) < self.min_stiffness_ratio_delta:
            self.min_stiffness_ratio_delta = min(last_obs_stiff_delta, next_obs_stiff_delta)

        exp = 9.0
        last_obs_reward = (1.0 - last_stiff_delta_used) ** exp
        next_obs_reward = (1.0 - next_stiff_delta_used) ** exp

        reward = next_obs_reward - last_obs_reward
        return reward, next_obs_stiff_delta




        # next_obs_stiffness_ratio = self.get_or_eval_design(next_obs)
        # next_obs_stiff_delta = abs(next_obs_stiffness_ratio - self.problem.target_stiffness_ratio)
        # if next_obs_stiff_delta < self.min_stiffness_ratio_delta:
        #     self.min_stiffness_ratio_delta = next_obs_stiff_delta
        #
        # # Inverse reward
        # # if next_obs_stiff_delta == 0.0:
        # #     next_obs_stiff_delta = 0.0000000000000001
        # # reward = 1 / next_obs_stiff_delta
        # # reward /= 100.0
        #
        # # Exponential reward
        # if next_obs_stiff_delta > 1.0:
        #     stiff_delta_used = 1.0
        # else:
        #     stiff_delta_used = next_obs_stiff_delta
        #
        # # exp = 9.0
        # # reward = (1.0 - stiff_delta_used) ** exp
        # reward = 1.0 - stiff_delta_used
        # reward *= 0.001
        #
        # # Reward is the move delta
        # return reward, next_obs_stiff_delta

    def get_or_eval_design(self, design):
        design_str = ''.join([str(int(x)) for x in design])
        if design_str in self.unique_designs:
            return self.unique_designs[design_str]
        else:
            h_stiffness, v_stiffness, stiff_ratio, vol_frac, constraints = self.problem.evaluate(
                design_str,
                problem_num=0,
                run_val=True
            )
            self.unique_designs[design_str] = stiff_ratio
            self.nfe += 1
            return stiff_ratio

    # -------------------------------------
    # Sampling Functions
    # -------------------------------------

    def sample_critic(self, last_observations):
        # (batch_size, design_len)
        observation_input = tf.convert_to_tensor(last_observations, dtype=tf.float32)
        return self._sample_critic(observation_input)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(global_mini_batch_size, design_len), dtype=tf.float32),
        # shape=(global_mini_batch_size, None)
    ])
    def _sample_critic(self, observation_input):
        t_value = self.c_critic(observation_input)
        t_value = t_value[:, 0]  # shape (batch,)
        return t_value


    def sample_actor(self, last_observations):
        # (batch_size, design_len)
        observation_input = tf.convert_to_tensor(last_observations, dtype=tf.float32)
        # print('--> SAMPLE ACTOR INPUT', observation_input.shape)
        return self._sample_actor(observation_input)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(global_mini_batch_size, design_len), dtype=tf.float32),  # shape=(global_mini_batch_size, None)
    ])
    def _sample_actor(self, observation_input):
        # print('sampling actor', inf_idx)
        pred_probs, pred_log_probs = self.c_actor(observation_input)  # shape: (batch, 30)

        all_token_probs = pred_probs  # (batch_size, 30)
        all_token_log_probs = pred_log_probs
        samples = tf.random.categorical(all_token_log_probs, 1)  # shape (batch, 1)
        samples_clamped = tf.minimum(samples, 29)  # Ensure values do not exceed 29
        flip_bit_ids = tf.squeeze(samples_clamped, axis=-1)  # shape (batch,)

        batch_indices = tf.range(0, tf.shape(all_token_log_probs)[0], dtype=tf.int64)  # shape (batch,)
        flip_bit_probs = tf.gather_nd(all_token_log_probs, tf.stack([batch_indices, flip_bit_ids], axis=-1))

        actions = flip_bit_ids
        action_log_probs = flip_bit_probs
        return action_log_probs, actions, all_token_probs

    # -------------------------------------
    # Training Functions
    # -------------------------------------

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(global_mini_batch_size * (trajectory_len + 1), 30), dtype=tf.float32),
        tf.TensorSpec(shape=(global_mini_batch_size * (trajectory_len + 1)), dtype=tf.float32),
    ])
    def train_critic(self, observation, return_tensor):
        with tf.GradientTape() as tape:
            pred_values = self.c_critic(observation)

            # Value loss (MSE)
            value_loss = tf.reduce_mean((return_tensor - pred_values) ** 2)

        critic_grads = tape.gradient(value_loss, self.c_critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.c_critic.trainable_variables))

        return value_loss

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(global_mini_batch_size * trajectory_len, design_len), dtype=tf.float32),
        tf.TensorSpec(shape=(global_mini_batch_size * trajectory_len), dtype=tf.int32),
        tf.TensorSpec(shape=(global_mini_batch_size * trajectory_len), dtype=tf.float32),
        tf.TensorSpec(shape=(global_mini_batch_size * trajectory_len), dtype=tf.float32),
    ])
    def train_actor(self, observation, action, logprob, advantage):

        with tf.GradientTape() as tape:
            pred_probs, pred_log_probs = self.c_actor(observation)  # shape: (batch, 30)
            one_hot_actions = tf.one_hot(action, self.num_actions)  # shape: (batch, 30)
            logprobability = tf.reduce_sum(
                one_hot_actions * pred_log_probs, axis=-1
            )  # shape (batch,)

            # Total Loss
            loss = 0

            # PPO Surrogate Loss
            ratio = tf.exp(
                logprobability - logprob
            )
            min_advantage = tf.where(
                advantage > 0,
                (1 + self.clip_ratio) * advantage,
                (1 - self.clip_ratio) * advantage,
            )
            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantage, min_advantage)
            )
            loss += policy_loss

            # Entropy Term
            entropy = -tf.reduce_sum(pred_probs * pred_log_probs, axis=-1)
            entropy = tf.reduce_mean(entropy)
            loss = loss - (self.entropy_coef * entropy)

        policy_grads = tape.gradient(loss, self.c_actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(policy_grads, self.c_actor.trainable_variables))

        # KL Divergence
        pred_probs, pred_log_probs = self.c_actor(observation)
        # pred_log_probs = tf.math.log(pred_probs)  # shape: (batch, 30)
        logprobability = tf.reduce_sum(
            tf.one_hot(action, self.num_actions) * pred_log_probs, axis=-1
        )  # shape (batch,)
        kl = tf.reduce_mean(
            logprob - logprobability
        )
        kl = tf.reduce_sum(kl)

        return kl, entropy, policy_loss, loss


    # -------------------------------------
    # Record Functions
    # -------------------------------------

    def record(self, epoch_info):
        if epoch_info is None:
            return

        # Record new epoch / print
        if self.debug is True:
            print(f"Proc PPO_Task {self.run_num} - {self.curr_epoch} ", end=' ')
            for key, value in epoch_info.items():
                if isinstance(value, list):
                    print(f"{key}: {value}", end=' | ')
                else:
                    print("%s: %.5f" % (key, value), end=' | ')
            print(self.min_stiffness_ratio_delta, '|', self.nfe)

        # Update metrics
        self.returns.append(epoch_info['mb_return'])
        self.stiffness_ratio_returns.append(epoch_info['sr_delta_min'])
        self.c_loss.append(epoch_info['c_loss'])
        self.p_loss.append(epoch_info['p_loss'])
        self.p_iter.append(epoch_info['p_iter'])
        self.entropy.append(epoch_info['entropy'])
        self.kl.append(epoch_info['kl'])

        # Plot
        if self.curr_epoch % self.plot_freq == 0:
            self.plot_ppo()

    def plot_ppo(self):

        # --- Plotting ---
        epochs = [x for x in range(len(self.returns))]
        gs = gridspec.GridSpec(3, 2)
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

        # Stiffness ratio delta returns
        plt.subplot(gs[2, 0])
        plt.plot(epochs, self.stiffness_ratio_returns)
        plt.xlabel('Epoch')
        plt.ylabel('Stiffness Ratio Delta')
        plt.title('Stiffness Ratio Delta Returns')





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
        target_stiffness_ratio=0.7,
        feasible_stiffness_delta=0.00001,
    )



    actor_save_path = None
    critic_save_path = None

    alg = PPO_Stiffness(
        run_num=run_dir,
        problem=problem,
        epochs=task_epochs,
        actor_load_path=actor_save_path,
        critic_load_path=critic_save_path,
        debug=True,
        c_type='uniform',
        run_val=True,
        val_itr=val_itr,
        val_task=0
    )
    alg.run()




