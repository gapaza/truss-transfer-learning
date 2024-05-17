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
from problem.knapsack.KnapsackDesign import KnapsackDesign as Design
import scipy.signal
from task.GA_Knapsack_Task import GA_Knapsack_Task
from modelC import get_q_decoder as get_model

# Set random seed
seed_num = 0
random.seed(seed_num)
tf.random.set_seed(seed_num)

# Run parameters
save_init_weights = False
load_init_weights = False
run_dir = 11
run_num = 1  # ------------------------- RUN NUM
plot_freq = 50
task_epochs = 10000

# Trajectory Sampling
total_weight_samples = 9
num_weight_samples = 9  # 4
repeat_size = 1
global_mini_batch_size = num_weight_samples * repeat_size

# Target Network Update
value_learning_rate = 0.001
use_warmup = False
update_batch_size_max = 256
update_batch_size_min = 32
update_batch_iterations = 200
update_target_network_freq = 5

# Replay Buffer
replay_buffer_size = 10000

# Epsilon Greedy
epsilon = 0.99  # was 0.99
epsilon_end = 0.01
decay_steps = 50 * config.num_vars

# Reward
perf_term_weight = 1.0


class QLearning:

    def __init__(
            self,
            run_num=0,
            barrier=None,
            problem=None,
            epochs=50,
            q_network_load_path=None,
            debug=False,
            c_type='uniform',
            run_val=False,
            val_itr=0,
            val_task=0,
    ):
        self.debug = debug
        self.run_num = run_num
        self.barrier = barrier
        self.problem = problem
        self.epochs = epochs
        self.c_type = c_type
        self.run_val = run_val
        self.val_itr = val_itr
        self.val_task = val_task
        self.q_network_load_path = q_network_load_path

        # Writing
        self.run_root = config.results_save_dir
        self.run_dir = os.path.join(self.run_root, 'run_' + str(self.run_num))
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

        # Save file
        self.q_network_save_path = os.path.join(self.run_dir, 'q_network_weights')

        # HV
        self.pop_size = 100
        self.ref_point = np.array(config.hv_ref_point)  # value, weight
        self.hv_client = HV(self.ref_point)
        self.nds = NonDominatedSorting()
        self.unique_designs = set()
        self.unique_designs_vals = []
        self.unique_designs_epoch = []
        self.unique_designs_feasible = []
        self.constraint_returns = []

        # Algorithm parameters
        self.mini_batch_size = global_mini_batch_size
        self.nfe = 0
        self.epochs = epochs
        self.curr_epoch = 0
        self.num_actions = 2
        self.decision_start_token_id = 1
        self.steps_per_design = config.num_vars  # 30 | 60
        self.gamma = 0.99

        # Q Algorithm Parameters
        self.value_network = None
        self.target_value_network = None
        self.replay_buffer = []
        self.replay_buffer_size = replay_buffer_size
        self.step = 0
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.decay_steps = decay_steps

        # Population
        self.population = []
        self.hv = []
        self.nfes = []

        # Model steps
        self.q_network_steps = 0

        # Results
        self.plot_freq = plot_freq

        # Objective Weights
        self.objective_weights = list(np.linspace(0.00, 1.0, total_weight_samples))

        # Tasks
        self.run_tasks = [val_task]  # Validation tasks

        # GA Comparison Data
        self.uniform_ga = self.init_comparison_data()

        self.additional_info = {
            'loss': [],
            'epsilons': [],
            'avg_rewards': [],
        }


    def init_comparison_data(self):
        print('--> RUNNING GA')
        task_runner = GA_Knapsack_Task(
            run_num=self.run_num,
            problem=self.problem,
            limit=100000,
            c_type='uniform',
            max_nfe=10000,
            problem_num=self.val_task,
            run_val=True,
            pop_size=50,
            offspring_size=50,
        )
        performance = task_runner.run()
        return performance

    def build(self):
        self.value_learning_rate = value_learning_rate
        if use_warmup is True:
            self.value_learning_rate = tf.keras.optimizers.schedules.CosineDecay(
                0.0,  # initial learning rate
                1000,  # decay_steps
                alpha=0.1,
                warmup_target=self.value_learning_rate,
                warmup_steps=1000
            )
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=self.value_learning_rate)

        # Q Network
        self.value_network, self.target_value_network = get_model(self.q_network_load_path)
        self.value_network.summary()

    def update_target_network(self):
        self.target_value_network.load_target_weights(self.value_network)

    def run(self):
        self.build()

        for x in range(self.epochs):
            self.curr_epoch = x
            avg_reward = self.gen_trajectories()
            self.prune_population()
            epoch_info = None
            # epoch_info = self.train_q_network(use_pop=True)
            for x in range(update_batch_iterations):
                epoch_info = self.update_q_network(use_pop=False)
            if epoch_info:
                epoch_info['avg_reward'] = avg_reward
            self.record(epoch_info)

            if self.curr_epoch % update_target_network_freq == 0:
                self.update_target_network()

    def linear_decay(self):
        if self.step > self.decay_steps:
            return self.epsilon_end
        return self.epsilon - self.step * (self.epsilon - self.epsilon_end) / self.decay_steps

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
    # Trajectory Sampling
    # -------------------------------------

    def get_cross_obs(self):

        # Weight sampling
        weight_samples = []
        weight_samples = random.sample(self.objective_weights, num_weight_samples)

        # Construct conditioning tensor
        cross_obs_vars = []
        weight_samples_all = []
        task_samples_all = []
        for weight in weight_samples:
            sample_vars = [weight]
            cross_obs_vars.append(sample_vars)
            weight_samples_all.append(weight)
            task_samples_all.append(0)

        weight_samples_all = [element for element in weight_samples_all for _ in range(repeat_size)]
        task_samples_all = [element for element in task_samples_all for _ in range(repeat_size)]
        cross_obs_vars = [element for element in cross_obs_vars for _ in range(repeat_size)]

        cross_obs_tensor = tf.convert_to_tensor(cross_obs_vars, dtype=tf.float32)

        return cross_obs_tensor, weight_samples_all, task_samples_all

    def gen_trajectories(self):

        # -------------------------------------
        # Sample Trajectories
        # -------------------------------------
        observation = [[self.decision_start_token_id] for x in range(self.mini_batch_size)]

        all_actions = [[] for _ in range(self.mini_batch_size)]
        all_actions_values = [[] for _ in range(self.mini_batch_size)]
        all_actions_values_full = [[] for _ in range(self.mini_batch_size)]

        all_rewards = [[] for _ in range(self.mini_batch_size)]

        all_values = [[] for _ in range(self.mini_batch_size)]

        designs = [[] for x in range(self.mini_batch_size)]
        epoch_designs = []

        # Get cross attention observation input
        cross_obs_tensor, weight_samples_all, task_samples_all = self.get_cross_obs()

        for t in range(self.steps_per_design):
            self.step += 1
            actions, actions_values, actions_values_full = self.sample_value_network(observation, cross_obs_tensor)

            for idx, act in enumerate(actions):
                all_actions[idx].append(deepcopy(act))
                all_actions_values[idx].append(deepcopy(actions_values[idx]))
                all_actions_values_full[idx].append(deepcopy(actions_values_full[idx]))
                m_action = int(deepcopy(act))
                designs[idx].append(m_action)
                observation[idx].append(m_action + 2)

            # Assume all rewards are 0 for now
            for idx, act in enumerate(actions):
                all_rewards[idx].append(0.0)

        # Remove the last observation from each trajectory
        for idx, obs in enumerate(observation):
            observation[idx] = obs[:-1]


        # print('Design 0:', ''.join([str(x) for x in designs[0]]))
        # print('Design 0 Values:', all_actions_values[0])
        # print('Design 0 Full Values:', all_actions_values_full[0])

        # -------------------------------------
        # Evaluate Designs
        # -------------------------------------

        all_her_rewards = []
        all_her_weights = []
        children = []
        all_rewards_flat = []
        for idx, design in enumerate(designs):
            design_bitstr = ''.join([str(x) for x in design])
            epoch_designs.append(design_bitstr)
            reward, design_obj, her_rewards, her_weights = self.calc_reward(design_bitstr, weight_samples_all[idx])
            children.append(design_obj)
            all_rewards[idx][-1] = reward
            all_her_rewards.append(her_rewards)
            all_her_weights.append(her_weights)
            all_rewards_flat.append(reward)

        # -------------------------------------
        # Save to replay buffer
        # -------------------------------------

        memories = []
        for idx, design in enumerate(designs):
            buffer_entry = {
                'observation': observation[idx],
                'cross_obs': cross_obs_tensor[idx],
                'actions': all_actions[idx],
                'rewards': all_rewards[idx],

                'bitstr': epoch_designs[idx],
                'epoch': self.curr_epoch,
            }
            memories.append(buffer_entry)
            children[idx].memory = deepcopy(buffer_entry)

        her_memories = []  # Hindsight Experience Replay
        for idx, memory in enumerate(memories):
            # Recalculate reward for different weight samples
            # Update cross_obs_tensor to reflect new weight sample
            for her_reward, her_weight in zip(all_her_rewards[idx], all_her_weights[idx]):
                mem_copy = deepcopy(memory)
                mem_copy['rewards'][-1] = her_reward
                buffer_entry = {
                    'observation': mem_copy['observation'],
                    'cross_obs': tf.convert_to_tensor(her_weight, dtype=tf.float32),
                    'actions': mem_copy['actions'],
                    'rewards': mem_copy['rewards'],
                    'bitstr': mem_copy['bitstr'],
                    'epoch': mem_copy['epoch'],
                }
                her_memories.append(buffer_entry)


        self.replay_buffer.extend(memories)
        # self.replay_buffer.extend(her_memories)

        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer = sorted(self.replay_buffer, key=lambda x: x['epoch'], reverse=True)
            self.replay_buffer = self.replay_buffer[:self.replay_buffer_size]

        # -------------------------------------
        # Update population
        # -------------------------------------

        self.population += children

        return np.mean(all_rewards_flat)

    def sample_value_network(self, observation, cross_obs_tensor):
        inf_idx = len(observation[0]) - 1
        observation = tf.convert_to_tensor(observation, dtype=tf.float32)
        q_values = self._sample_value_network(observation, cross_obs_tensor)
        q_values_t = q_values[:, inf_idx, :]
        q_values = q_values_t.numpy().tolist()  # 2D list: (batch_element, action_values)

        epsilon = self.linear_decay()
        actions = []
        actions_values = []
        for sample_q_values in q_values:
            if random.random() < epsilon:
                action = random.randint(0, self.num_actions - 1)
            else:
                action = np.argmax(sample_q_values)
            actions.append(action)
            actions_values.append(sample_q_values[action])

        return actions, actions_values, q_values

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    ])
    def _sample_value_network(self, observation, cross_obs_tensor):
        q_values = self.value_network([observation, cross_obs_tensor])
        return q_values

    # -------------------------------------
    # Q Network Training
    # -------------------------------------

    def update_q_network(self, use_pop=False):
        if len(self.replay_buffer) < update_batch_size_min:
            return

        # Population memories

        # -------------------------------------
        # Sample from replay buffer
        # -------------------------------------

        # Determine if using pop
        if use_pop is True:
            pop_memories = [design.memory for design in self.population]
            update_batch_size = min(update_batch_size_max, len(pop_memories))
            update_batch = random.sample(pop_memories, update_batch_size)
        else:
            update_batch_size = min(update_batch_size_max, len(self.replay_buffer))
            self.replay_buffer = sorted(self.replay_buffer, key=lambda x: x['epoch'], reverse=True)
            update_batch = random.sample(self.replay_buffer, update_batch_size)


        observation_batch = [x['observation'] for x in update_batch]
        cross_obs_batch = [x['cross_obs'] for x in update_batch]
        actions_batch = [x['actions'] for x in update_batch]
        rewards_batch = [x['rewards'] for x in update_batch]

        observation_tensor = tf.convert_to_tensor(observation_batch, dtype=tf.float32)
        cross_obs_tensor = tf.convert_to_tensor(cross_obs_batch, dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(actions_batch, dtype=tf.int32)

        # -------------------------------------
        # Calculate Q Targets
        # -------------------------------------

        q_targets = self.sample_target_network(observation_tensor, cross_obs_tensor)
        q_targets = q_targets.numpy()
        # print('Target Network Q Values:', q_targets)

        # Calculate Target Values
        target_values = []
        for idx in range(len(update_batch)):
            rewards = np.array(rewards_batch[idx])
            targets = np.array(q_targets[idx])
            target_vals = rewards[:-1] + self.gamma * targets[1:]
            target_vals = target_vals.tolist()
            target_vals.append(rewards[-1])
            target_values.append(target_vals)
        target_values = tf.convert_to_tensor(target_values, dtype=tf.float32)

        # -------------------------------------
        # Train Q Network
        # -------------------------------------

        loss = self.train_q_network(observation_tensor, cross_obs_tensor, actions_tensor, target_values)
        loss = loss.numpy()

        epoch_info = {
            'loss': loss,
        }
        return epoch_info

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    ])
    def train_q_network(self, observation, cross_obs_tensor, actions, target_values):
        with tf.GradientTape() as tape:
            pred_q_values = self.value_network([observation, cross_obs_tensor])
            q_values = tf.reduce_sum(
                tf.one_hot(actions, self.num_actions) * pred_q_values, axis=-1
            )
            loss = tf.reduce_mean(tf.square(target_values - q_values))

        gradients = tape.gradient(loss, self.value_network.trainable_variables)
        self.value_optimizer.apply_gradients(zip(gradients, self.value_network.trainable_variables))
        return loss

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    ])
    def sample_target_network(self, observation, cross_obs_tensor):
        q_values = self.target_value_network([observation, cross_obs_tensor], training=False)
        q_values = tf.reduce_max(q_values, axis=-1)
        return q_values

    # -------------------------------------
    # Reward Calculation
    # -------------------------------------

    def calc_reward(self, bitstr, weight):
        obj_value, obj_weight = self.problem.evaluate(bitstr)

        # -------------------------------------
        # Calculate performance reward
        # -------------------------------------

        w1 = weight
        w2 = 1.0 - weight
        value_term = w1 * obj_value
        weight_term = w2 * (1.0 - obj_weight)
        performance_term = value_term + weight_term
        performance_term = performance_term * perf_term_weight

        reward = performance_term

        # Her Rewards
        her_rewards = []
        her_weights = []
        for w in self.objective_weights:
            her_reward = self.calc_her_reward(obj_value, obj_weight, w)
            her_rewards.append(her_reward)
            her_weights.append([w])

        # -------------------------------------
        # Create design
        # -------------------------------------
        design = Design(
            design_vector=[int(i) for i in bitstr], evaluator=self.problem, num_bits=self.steps_per_design,
            c_type=self.c_type, p_num=0, val=True, constraints=True
        )
        design.set_objectives(obj_value * -1.0, obj_weight)
        design.evaluated = True
        if bitstr not in self.unique_designs:
            self.unique_designs.add(bitstr)
            self.unique_designs_vals.append([obj_value * -1.0, obj_weight])
            self.unique_designs_feasible.append(design.is_feasible)
            self.unique_designs_epoch.append(self.curr_epoch)
            self.nfe += 1
        return reward, design, her_rewards, her_weights

    def calc_her_reward(self, obj_value, obj_weight, weight):
        w1 = weight
        w2 = 1.0 - weight
        value_term = w1 * obj_value
        weight_term = w2 * (1.0 - obj_weight)
        performance_term = value_term + weight_term
        performance_term = performance_term * perf_term_weight
        return performance_term

    # -------------------------------------
    # Recording / Plotting
    # -------------------------------------

    def record(self, epoch_info):
        if epoch_info is None:
            return

        self.hv.append(self.calc_pop_hv())
        self.nfes.append(self.nfe)
        self.additional_info['loss'].append(epoch_info['loss'])
        self.additional_info['epsilons'].append(self.linear_decay())
        self.additional_info['avg_rewards'].append(epoch_info['avg_reward'])

        # Record new epoch / print
        if self.debug is True:
            print(f"Q Transformer {self.curr_epoch} ", end=' ')
            for key, value in epoch_info.items():
                if isinstance(value, list):
                    print(f"{key}: {value}", end=' | ')
                else:
                    print("%s: %.5f" % (key, value), end=' | ')
            print('HV:', self.hv[-1])

        if self.curr_epoch % self.plot_freq == 0:
            self.plot_progress()

    def plot_progress(self):
        print('Plotting Progress')


        gs = gridspec.GridSpec(2, 3)
        fig = plt.figure(figsize=(12, 9))  # default [6.4, 4.8], W x H  9x6, 12x8
        fig.suptitle('Results', fontsize=16)

        # Plot HV
        plt.subplot(gs[0, 0])
        plt.plot(self.nfes, self.hv, label='HV')
        plt.plot([r[0] for r in self.uniform_ga], [r[1] for r in self.uniform_ga], label='Uniform GA HV')
        plt.xlabel('NFE')
        plt.ylabel('HV')
        plt.title('Hypervolume Plot')
        plt.legend()

        # Plot Loss
        plt.subplot(gs[0, 1])
        if len(self.additional_info['loss']) <= 50:
            plt.plot(self.additional_info['loss'], label='Loss')
        else:
            plt.plot(self.additional_info['loss'][50:], label='Loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Plot')
        plt.legend()

        # Plot Avg Reward
        plt.subplot(gs[0, 2])
        plt.plot(self.additional_info['avg_rewards'], label='Avg Reward')
        plt.xlabel('Epoch')
        plt.ylabel('Avg Reward')
        plt.title('Avg Reward Plot')
        plt.legend()

        # Plot Epsilon
        plt.subplot(gs[1, 0])
        plt.plot(self.additional_info['epsilons'], label='Epsilon')
        plt.xlabel('Epoch')
        plt.ylabel('Epsilon')
        plt.title('Epsilon Plot')
        plt.legend()

        # Design Plot
        plt.subplot(gs[1, 1])
        feasible_x, feasible_y = [], []
        infeasible_x, infeasible_y = [], []
        for idx, obj_vals in enumerate(self.unique_designs_vals):
            if self.unique_designs_feasible[idx] is True:
                feasible_x.append(obj_vals[0] * -1.0)
                feasible_y.append(obj_vals[1])
            else:
                infeasible_x.append(obj_vals[0] * -1.0)
                infeasible_y.append(obj_vals[1])
        plt.scatter(feasible_x, feasible_y, color='blue')
        plt.scatter(infeasible_x, infeasible_y, color='grey', alpha=0.3)
        plt.xlim(0, 1.1)
        plt.ylim(0, 1.1)
        plt.xlabel('Vertical Stiffness')
        plt.ylabel('Volume Fraction')
        plt.title('All Designs')

        # Design plot epoch
        plt.subplot(gs[1, 2])
        x_vals = []
        y_vals = []
        colors = []
        for idx, obj_vals in enumerate(self.unique_designs_vals):
            x_vals.append(obj_vals[0] * -1.0)
            y_vals.append(obj_vals[1])
            colors.append(self.unique_designs_epoch[idx])
        scatter = plt.scatter(x_vals, y_vals, c=colors, cmap='viridis')
        plt.colorbar(scatter, label='Epoch')
        plt.xlim(0, 1.1)
        plt.ylim(0, 1.1)
        plt.xlabel('Vertical Stiffness')
        plt.ylabel('Volume Fraction')
        plt.title('All Designs')




        plt.tight_layout()
        save_path = os.path.join(self.run_dir, 'plots_' + str(self.val_itr) + '.png')
        plt.savefig(save_path)
        plt.close('all')





# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------


from problem.knapsack.TwoObjectiveSimpleKP import TwoObjectiveSimpleKP

if __name__ == '__main__':
    problem = TwoObjectiveSimpleKP(
        n=config.num_vars,
        new_problem=False
    )


    # q_network_load_path = os.path.join(config.results_save_dir, 'run_' + str(run_dir), 'pretrained', 'actor_weights_2550')  # 600 worked best

    q_network_load_path = None

    alg = QLearning(
        run_num=run_dir,
        problem=problem,
        epochs=task_epochs,
        q_network_load_path=q_network_load_path,
        debug=True,
        c_type='uniform',
        run_val=True,
        val_itr=run_num,
        val_task=0
    )
    alg.run()



