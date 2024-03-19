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
from model import get_multi_task_decoder_v2 as get_model
from collections import OrderedDict
import tensorflow_addons as tfa
from problem import get_str_targets




class AbstractMultiTask(AbstractTask):

    def __init__(
            self,
            run_num=0,
            barrier=None,
            problem=None,
            epochs=50,
            actor_load_path=None,
            critic_load_path=None,
            save_init_weights=False,
    ):
        super(AbstractMultiTask, self).__init__(run_num, barrier, problem, epochs, actor_load_path, critic_load_path)

        # HV
        self.ref_point = np.array(config.hv_ref_point)  # value, weight
        self.hv_client = HV(self.ref_point)
        self.nds = NonDominatedSorting()
        self.unique_designs = set()
        self.unique_designs_vals = []
        self.unique_designs_feasible = []
        self.constraint_returns = []
        self.avg_str_values = []
        self.min_stiffness_delta = []

        # Population
        self.pop_size = 100
        self.population = []
        self.hv = []
        self.nfes = []

        # Pretrain save dir
        self.pretrain_save_dir = os.path.join(self.run_dir, 'pretrained')
        if not os.path.exists(self.pretrain_save_dir):
            os.makedirs(self.pretrain_save_dir)
        self.actor_pretrain_save_path = os.path.join(self.pretrain_save_dir, 'actor_weights')
        self.critic_pretrain_save_path = os.path.join(self.pretrain_save_dir, 'critic_weights')

        # Initialize weights
        self.save_init_weights = save_init_weights
        self.actor_init_save_path = os.path.join(self.pretrain_save_dir, 'actor_weights_init')
        self.critic_init_save_path = os.path.join(self.pretrain_save_dir, 'critic_weights_init')


    def build(self):

        # Optimizer parameters
        self.actor_learning_rate = 0.0001  # 0.0001
        self.critic_learning_rate = 0.0001  # 0.0001
        self.train_actor_iterations = 250  # was 250
        self.train_critic_iterations = 40  # was 40

        if self.actor_optimizer is None:
            self.actor_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.actor_learning_rate)
            # self.actor_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=self.actor_learning_rate)
        if self.critic_optimizer is None:
            self.critic_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.critic_learning_rate)
            # self.critic_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=self.critic_learning_rate)

        self.c_actor, self.c_critic = get_model(self.actor_load_path, self.critic_load_path)

        if not self.actor_load_path and not self.critic_load_path and self.save_init_weights is True:
            self.c_actor.save_weights(self.actor_init_save_path)
            self.c_critic.save_weights(self.critic_init_save_path)

        self.c_actor.summary()

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





















