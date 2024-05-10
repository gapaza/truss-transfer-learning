import numpy as np
import time
import tensorflow as tf
from copy import deepcopy
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os
import config
from math import cos, pi


# from modelC import get_multi_task_decoder as get_model
from modelC import get_multi_task_decoder_moe as get_model

from problem.TrussProblem import TrussProblem
from taskC.AbstractPpoMT import AbstractPpoMT

# Mini-batch Samples
weight_samples = 4
task_samples = 6
num_repeats = 2

# Run parameters
run_dir = 1
run_num = 10
task_epochs = 10000
num_tasks = 36

class ConstrainedMT(AbstractPpoMT):

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
        super(ConstrainedMT, self).__init__(run_num, problem, epochs, actor_load_path, critic_load_path, c_type, run_val, val_itr, num_tasks)


        # --- Hyperparameters ---
        self.clip_ratio = 0.2
        self.target_kl = 0.0025  # was 0.01
        self.entropy_coef = 0.04  # was 0.02 originally
        self.actor_learning_rate = 0.00005  # 0.0001
        self.critic_learning_rate = 0.0001  # 0.0001

        # Plotting / Saving
        self.plot_freq = 25
        self.save_freq = 50
        self.history_epochs = 1000


    def get_constraint_weights(self):
        feasibility_coeff = 1.0
        connectivity_coeff = 1.0
        stiffness_ratio_coeff = 1.0
        constraint_coeff = self.warup_learning_rate(500, 0.2, delay=150)
        # if self.curr_epoch < 150:
        #     constraint_coeff = 0.0
        return feasibility_coeff, connectivity_coeff, stiffness_ratio_coeff, constraint_coeff

    def warup_learning_rate(self, warmup_steps, final_learning_rate, delay=0):
        if delay > self.curr_epoch:
            return 0.0
        step = min(self.curr_epoch - delay, warmup_steps)
        warmup = step / warmup_steps
        curr_lr = warmup * final_learning_rate
        return curr_lr

    def load_models(self):
        self.c_actor, self.c_critic = get_model(self.actor_load_path, self.critic_load_path)
        return self.c_actor, self.c_critic

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
        epoch_designs = []
        observation = [[self.decision_start_token_id] for x in range(self.mini_batch_size)]
        critic_observation_buffer = [[] for x in range(self.mini_batch_size)]

        # Get cross attention observation input
        cross_obs_tensor, weight_samples_all, task_samples_all = self.get_cross_obs(weight_samples, task_samples, num_repeats)

        # -------------------------------------
        # Sample Actor
        # -------------------------------------

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
                    reward, terms = self.calc_reward(
                        design_bitstr,
                        [weight_samples_all[idx], task_samples_all[idx]]
                    )
                    all_rewards[idx].append(reward)
                    all_total_rewards.append(reward)
                    all_perf_returns.append(terms[0])
                    all_constraint_returns.append(terms[1])
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

        # -------------------------------------
        # Update Results
        # -------------------------------------

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
        }
        return epoch_info







    def plot_results(self):
        self.plot_metrics()
        self.plot_designs()

    def plot_metrics(self):

        # --- Plotting ---
        epochs = [x for x in range(len(self.metrics['mb_return']))]
        gs = gridspec.GridSpec(3, 2)
        fig = plt.figure(figsize=(16, 12))  # default [6.4, 4.8], W x H  9x6, 12x8
        fig.suptitle('Results', fontsize=16)

        # Returns plot
        plt.subplot(gs[0, 0])
        plt.plot(epochs, self.metrics['mb_return'])
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
    alg = ConstrainedMT(
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








