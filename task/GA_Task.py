import numpy as np
import time
import tensorflow as tf
from copy import deepcopy
import matplotlib.gridspec as gridspec
import random
import json
import config
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.selection.tournament import compare
from task.AbstractTask import AbstractTask



import utils
from problem.TrussDesign import TrussDesign as Design
from problem.TrussProblem import TrussProblem as Problem




class GA_Task(AbstractTask):

    def __init__(
            self,
            run_num=0,
            barrier=None,
            problem=None,
            limit=50,
            actor_load_path=None,
            critic_load_path=None,
            debug=False,
            c_type='uniform',
            max_nfe=5000,
            clip_pop=False,
            problem_num=0,
            run_val=False,
            save_num=-1,
            pop_size=30,
            offspring_size=30,
            steps_per_design=config.num_vars,
    ):
        super(GA_Task, self).__init__(run_num, barrier, problem, limit, actor_load_path, critic_load_path)
        self.debug = debug
        self.c_type = c_type
        self.clip_pop = clip_pop
        self.problem_num = problem_num
        self.run_val = run_val
        self.save_num = save_num

        # HV
        self.ref_point = np.array(config.hv_ref_point)  # value, weight
        self.hv_client = HV(self.ref_point)
        self.unique_designs = set()
        self.unique_designs_lst = []
        self.unique_designs_vals = []
        self.unique_designs_in_window = []
        self.unique_designs_stiff_ratio = []

        # Algorithm parameters
        self.pop_size = pop_size
        self.offspring_size = offspring_size
        self.max_nfe = max_nfe
        self.nfe = 0
        self.limit = limit
        self.steps_per_design = steps_per_design  # 30 | 60

        # Population
        self.population = []
        self.hv = []         # hv progress over time
        self.hv_window = []  # hv progress over time
        self.nfes = []       # nfe progress over time

        # Files
        if self.save_num != -1:
            self.uniform_ga_file = os.path.join(self.run_dir, 'hv_uniform_ga_'+str(self.save_num)+'.json')
            self.uniform_ga_file_all = os.path.join(self.run_dir, 'hv_uniform_ga_all_'+str(self.save_num)+'.json')
        else:
            self.uniform_ga_file = os.path.join(self.run_dir, 'uniform_ga.json')
            self.uniform_ga_file_all = os.path.join(self.run_dir, 'uniform_ga_all.json')

    def run(self):
        if os.path.exists(self.uniform_ga_file):
            with open(self.uniform_ga_file, 'r') as f:
                performance = json.load(f)
            return performance

        print('--> RUNNING ALG TASK:', self.run_num)
        self.init_population()
        self.eval_population()
        terminated = False
        counter = 0
        progress_bar = tqdm(total=self.max_nfe)
        while terminated is False and self.nfe < self.max_nfe:
            curr_nfe = deepcopy(self.nfe)

            # 1. Create offspring
            curr_time = time.time()
            self.create_offspring()
            # print('\nTime to create offspring:', time.time() - curr_time)

            # 2. Evaluate offspring
            curr_time = time.time()
            self.eval_population()
            # print('Time to evaluate:', time.time() - curr_time)

            # 3. Prune population
            curr_time = time.time()
            self.prune_population()
            # print('Time to prune:', time.time() - curr_time)

            # 4. Log iteration
            curr_time = time.time()
            self.record(None)
            # print('Time to record:', time.time() - curr_time)
            # self.activate_barrier()
            counter += 1

            if counter >= self.limit:
                terminated = True

            update_delta = self.nfe - curr_nfe
            progress_bar.update(update_delta)

            # if counter % 10 == 0:
            #     self.plot_comp()

        # 5. Save population
        self.save_population()
        # self.plot_comp()

        # 6. Save performance if file does not exist
        performance = [[nfe, hv] for hv, nfe in zip(self.hv, self.nfes)]
        if not os.path.exists(self.uniform_ga_file):
            with open(self.uniform_ga_file, 'w') as f:
                json.dump(performance, f, indent=4)
        # performance_window = [[nfe, hv] for hv, nfe in zip(self.hv_window, self.nfes)]
        # if not os.path.exists(self.uniform_ga_file):
        #     with open(self.uniform_ga_file, 'w') as f:
        #         json.dump(performance_window, f, indent=4)

        # self.plot()

        # Return HV performance for comparison
        return performance

    # -------------------------------------
    # Population Functions
    # -------------------------------------

    def save_population(self):
        bit_strs = []
        for design in self.population:
            bit_strs.append(design.get_vector_str())
        bit_strs = list(set(bit_strs))
        with open(os.path.join(self.run_dir, 'population.json'), 'w') as f:
            json.dump(bit_strs, f, indent=4)

    def calc_pop_hv(self):
        objectives = self.eval_population()
        F = np.array(objectives)
        hv = self.hv_client.do(F)
        return hv

    def calc_pop_hv_feasible(self):
        objectives = []
        for design_objs, in_window in zip(self.unique_designs_vals, self.unique_designs_in_window):
            if in_window is True:
                objectives.append(design_objs)
        if len(objectives) == 0:
            return 0.0
        F = np.array(objectives)
        hv = self.hv_client.do(F)


        # objectives = self.eval_population()
        # feasible = [idx for idx, design in enumerate(self.population) if design.in_stiffness_window is True]
        # if len(feasible) == 0:
        #     return 0.0
        # feasible_objectives = [objectives[idx] for idx in feasible]
        # F = np.array(feasible_objectives)
        # hv = self.hv_client.do(F)
        return hv

    def init_population(self):
        self.population = []
        for x in range(self.pop_size):
            design = Design(evaluator=self.problem, num_bits=self.steps_per_design, p_num=self.problem_num, val=self.run_val)
            self.population.append(design)

    def eval_population(self):
        self.eval_population_batch()

        evals = []
        for design in self.population:
            design_str = design.get_vector_str()
            if design.evaluated is False and design_str not in self.unique_designs:
                self.nfe += 1
            if design.evaluated is False and design_str in self.unique_designs:
                d_idx = self.unique_designs_lst.index(design_str)
                design.set_objectives(self.unique_designs_vals[d_idx][0], self.unique_designs_vals[d_idx][1])
                design.evaluated = True
                design.is_feasible = self.unique_designs_in_window[d_idx]
                design.stiffness_ratio = self.unique_designs_stiff_ratio[d_idx]
            vals = design.evaluate()
            evals.append(vals)
            if design_str not in self.unique_designs:
                self.unique_designs.add(design_str)
                self.unique_designs_lst.append(design_str)
                self.unique_designs_vals.append(vals)
                self.unique_designs_in_window.append(deepcopy(design.is_feasible))
                self.unique_designs_stiff_ratio.append(deepcopy(design.stiffness_ratio))
        return evals

    def eval_population_batch(self):
        batch_evals = []
        for design in self.population:
            design_str = design.get_vector_str()
            if design.evaluated is False and design_str not in self.unique_designs:
                self.nfe += 1
                batch_evals.append(design)

        if len(batch_evals) > 0:
            # print('evaluating batch designs: ', len(batch_evals))
            eval_vectors = [design.vector for design in batch_evals]
            batch_vals = self.problem.evaluate_batch(eval_vectors, self.problem_num, self.run_val)
            for idx, design in enumerate(batch_evals):
                vals = design.set_evaluate(batch_vals[idx])
                design_str = design.get_vector_str()
                if design_str not in self.unique_designs:
                    self.unique_designs.add(design_str)
                    self.unique_designs_lst.append(design_str)
                    self.unique_designs_vals.append(vals)
                    self.unique_designs_in_window.append(deepcopy(design.is_feasible))
                    self.unique_designs_stiff_ratio.append(deepcopy(design.stiffness_ratio))

    def prune_population(self):

        # 0. Get Objectives
        objectives = self.eval_population()
        stiffness_ratios = [design.stiffness_ratio for design in self.population]
        in_stiffness_window = [design.is_feasible for design in self.population]

        # Design A and Design B
        # If design A and design B are in stiffness ratio window, determine dominance by objective values
        # If design A is in stiffness window and design B is not, design A dominates design B
        # If design A is and design B are not in stiffness window, determine dominance by stiffness ratio delta

        # 1. Determine survivors
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

        # 2. Update population (sometimes the first front is larger than pop_size)
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
        # - case: both designs are in stiffness window
        # If design A and design B are in stiffness ratio window, determine dominance by objective values
        if in_stiffness_window[i] is True and in_stiffness_window[j] is True:
            for obj_i, obj_j in zip(objectives[i], objectives[j]):
                if obj_i > obj_j:  # Since lower values are better, i dominates j if both objectives are smaller
                    return False
            return True

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



        raise ValueError('-- NO DOMINANCE CONDITIONS WERE MET, CHECK DOMINANCE FUNC')

    # -------------------------------------
    # Tournament Functions
    # -------------------------------------

    def binary_tournament(self, solutions=None):
        if solutions is None:
            solutions = self.population

        select_size = len(solutions)
        if self.clip_pop is True and select_size > self.pop_size:
            select_size = self.pop_size

        p1 = random.randrange(select_size)
        p2 = random.randrange(select_size)
        while p1 == p2:
            p2 = random.randrange(select_size)

        player1 = solutions[p1]
        player2 = solutions[p2]

        winner_idx = compare(
            p1, player1.rank,
            p2, player2.rank,
            'smaller_is_better',
            return_random_if_equal=False
        )
        if winner_idx is None:
            winner_idx = compare(
                p1, player1.crowding_dist,
                p2, player2.crowding_dist,
                'larger_is_better',
                return_random_if_equal=True
            )
        return winner_idx

    def create_offspring(self):
        objectives = self.eval_population()
        F = np.array(objectives)

        stiffness_ratios = [design.stiffness_ratio for design in self.population]
        in_stiffness_window = [design.is_feasible for design in self.population]

        # Set pareto rank and crowding distance
        fronts = self.custom_dominance_sorting(objectives, stiffness_ratios, in_stiffness_window)
        for k, front in enumerate(fronts, start=1):
            crowding_of_front = utils.calc_crowding_distance(F[front, :])
            for i, idx in enumerate(front):
                self.population[idx].crowding_dist = crowding_of_front[i]
                self.population[idx].rank = k

        # Get parent pairs
        pairs = []
        while len(pairs) < self.offspring_size:
            parent1_idx = self.binary_tournament()
            parent2_idx = self.binary_tournament()
            while parent2_idx == parent1_idx:
                parent2_idx = self.binary_tournament()
            pairs.append([
                self.population[parent1_idx],
                self.population[parent2_idx]
            ])

        # Create offspring
        offspring = self.crossover_parents(pairs)
        self.population.extend(offspring)

    def crossover_parents(self, parent_pairs):
        offspring = []
        for pair in parent_pairs:
            parent1 = pair[0]
            parent2 = pair[1]
            child = Design(evaluator=self.problem, c_type=self.c_type, p_num=self.problem_num, val=self.run_val)
            child.crossover(parent1, parent2)
            child.mutate()
            offspring.append(child)
        return offspring

    def record(self, epoch_info):
        self.hv.append(self.calc_pop_hv())
        # self.hv_window.append(self.calc_pop_hv_feasible())
        self.nfes.append(self.nfe)

    def plot_comp(self):

        gs = gridspec.GridSpec(3, 2)
        fig = plt.figure(figsize=(16, 8))  # default [6.4, 4.8], W x H  9x6, 12x8
        fig.suptitle('Results', fontsize=16)

        plt.subplot(gs[0, 0])
        plt.plot(self.nfes, self.hv)
        plt.xlabel('NFE')
        plt.ylabel('HV')
        plt.title('HV Plot')

        plt.subplot(gs[0, 1])
        plt.plot(self.nfes, self.hv)
        plt.xlabel('NFE')
        plt.ylabel('HV')
        plt.title('HV Plot')

        plt.subplot(gs[1, 0])
        plt.plot(self.nfes, self.hv)
        plt.xlabel('NFE')
        plt.ylabel('HV')
        plt.title('HV Plot')

        plt.subplot(gs[1, 1])
        plt.plot(self.nfes, self.hv)
        plt.xlabel('NFE')
        plt.ylabel('HV')
        plt.title('HV Plot')

        plt.subplot(gs[2, 0])
        plt.plot(self.nfes, self.hv)
        plt.xlabel('NFE')
        plt.ylabel('HV')
        plt.title('HV Plot')

        plt.subplot(gs[2, 1])
        for idx, obj_vals in enumerate(self.unique_designs_vals):
            if self.unique_designs_in_window[idx] is True:
                plt.scatter(obj_vals[0] * -1.0, obj_vals[1], color='blue')
        plt.xlim(0, 1.1)
        plt.ylim(0, 1.1)
        plt.xlabel('Vertical Stiffness')
        plt.ylabel('Volume Fraction')
        plt.title('All Designs')

        plt.tight_layout()
        if self.save_num != -1:
            save_path = os.path.join(self.run_dir, 'ga_comp_'+str(self.save_num)+'.png')
        else:
            save_path = os.path.join(self.run_dir, 'ga_comp.png')
        plt.savefig(save_path)
        plt.close('all')




    def plot(self):

        # 1. Plot HV all
        plt.figure(figsize=(8, 8))
        plt.plot(self.nfes, self.hv)
        plt.xlabel('NFE')
        plt.ylabel('HV')
        plt.title('HV Progress')
        plt.savefig(os.path.join(self.run_dir, self.c_type + '_hv_all.png'))
        plt.close('all')

        # 1.2 Plot HV feasible
        # plt.figure(figsize=(8, 8))
        # plt.plot(self.nfes, self.hv_window)
        # plt.xlabel('NFE')
        # plt.ylabel('HV')
        # plt.title('HV Progress')
        # plt.savefig(os.path.join(self.run_dir, self.c_type + '_hv.png'))

        # 2. Plot designs all
        # plt.figure(figsize=(8, 8))
        # for obj_vals in self.unique_designs_vals:
        #     plt.scatter(obj_vals[0] * -1.0, obj_vals[1], color='blue')
        # plt.xlabel('Vertical Stiffness')
        # plt.ylabel('Volume Fraction')
        # plt.title('Designs')
        # plt.savefig(os.path.join(self.run_dir, self.c_type + '_designs_all.png'))
        # plt.close('all')

        # 3. Plot designs feasible
        plt.figure(figsize=(8, 8))
        for idx, obj_vals in enumerate(self.unique_designs_vals):
            if self.unique_designs_in_window[idx] is True:
                plt.scatter(obj_vals[0] * -1.0, obj_vals[1], color='blue')
        plt.xlabel('Vertical Stiffness')
        plt.ylabel('Volume Fraction')
        plt.title('All Designs')
        if self.save_num != -1:
            plt.savefig(os.path.join(self.run_dir, self.c_type + '_designs_'+str(self.save_num)+'.png'))
        else:
            plt.savefig(os.path.join(self.run_dir, self.c_type + '_designs.png'))
        plt.close('all')



if __name__ == '__main__':
    # problem = ConstantTruss(n=30)
    problem = Problem(sidenum=config.sidenum)

    problem_num = 0
    run_val = True

    # task_runner = GA_Task(
    #     run_num=11,
    #     problem=problem,
    #     limit=100000,
    #     c_type='uniform',
    #     max_nfe=10000,
    #     problem_num=problem_num,
    #     run_val=run_val,
    #     pop_size=100,
    #     offspring_size=100
    # )
    # task_runner.run()
    # task_runner.plot()


    runs = 30
    for save_num in range(3, runs):
        task_runner = GA_Task(
            run_num=11,
            problem=problem,
            limit=100000,
            c_type='uniform',
            max_nfe=10000,
            problem_num=problem_num,
            run_val=run_val,
            save_num=save_num,
            pop_size=100,
            offspring_size=100
        )
        performance = task_runner.run()
        # task_runner.plot()




