import random
import numpy as np
import os
import pickle
from tqdm import tqdm

import config

from problem.knapsack.AbstractKnapsack import AbstractKnapsack



ratio_threshold = 0.0001



class TwoObjectiveSimpleKPC(AbstractKnapsack):

    def __init__(self, n=10, new_problem=False, id=''):
        super(TwoObjectiveSimpleKPC, self).__init__(n, new_problem, id)

        # --> Loss Landscape
        self.val_range = [0, 10]
        self.weight_range = [0, 10]

        # --> Items
        if new_problem is False:
            self.items = self.load_items()
        else:
            self.generate_new_problem()


    def generate_new_problem(self):
        prob_value = 1.0
        problem = {
            'values': [],
            'weights': [],
            'value_norm': 1,
            'weight_norm': 1,
            'max_weight': 0,
            'ratio': 1.0,
        }

        # Create weights / values
        for item_idx in range(self.n):

            # Assign Value
            item_value = 0
            if random.random() <= prob_value:
                item_value = random.uniform(self.val_range[0], self.val_range[1])

            # Assign Weight
            item_weight = random.uniform(self.weight_range[0], self.weight_range[1])
            problem['values'].append(item_value)
            problem['weights'].append(item_weight)

        # Set max weight
        problem['max_weight'] = np.sum(problem['weights']) / 2.0

        # Find Norms
        problem = self.find_norms(problem)

        # Save problem items
        self.items = problem
        self.save()

    def find_norms(self, problem):
        random_designs = [
            self.random_design() for _ in range(10000)
        ]

        weights, values = [], []
        for design in tqdm(random_designs):
            value, weight, constraint = self.evaluate(design, items=problem)
            if constraint == 0:
                weights.append(weight)
                values.append(value)

        problem['value_norm'] = np.max(values) * 1.2  # Add margin
        problem['weight_norm'] = np.max(weights) * 1.1  # Add margin
        return problem






    def evaluate(self, solution, items=None):
        if len(solution) != self.n:
            raise Exception('Solution must be of length n:', self.n)
        if type(solution) == str:
            solution = [int(i) for i in solution]

        # --> Solution is a list of n binary values
        if items is None:
            items = self.items

        # ---------------------------------------
        # Objectives
        # ---------------------------------------

        # 1. Calculate value
        value = 0
        for i in range(self.n):
            if solution[i] == 1:
                value += items['values'][i]

        # 2. Calculate weight
        weight = 0
        for i in range(self.n):
            if solution[i] == 1:
                weight += items['weights'][i]

        # 3. Calculate constraint
        constraint_val = 0
        # if weight > items['max_weight']:
        #     constraint_val = abs(weight - items['max_weight'])
        if weight > 0:
            ratio = value / weight
            ratio_delta = abs(ratio - items['ratio'])
            if ratio_delta > ratio_threshold:
                constraint_val = ratio_delta
            else:
                constraint_val = 0
        else:
            constraint_val = 10


        if constraint_val > 0:
            value = 0
            weight = 1

        # 4. Normalize
        # print(value, weight)
        if constraint_val == 0:
            weight = weight / items['weight_norm']
            value = value / items['value_norm']

        return value, weight, constraint_val


if __name__ == '__main__':
    n = config.num_vars

    kp = TwoObjectiveSimpleKPC(n=n, new_problem=True)

    # sol = ''.join([str(1) for _ in range(n)])

    feasible_found = False
    while not feasible_found:
        sol = kp.random_design()
        value, weight, constraint = kp.evaluate(sol)
        if constraint == 0:
            feasible_found = True

    print('Solution:', sol)
    print('Value:', value)
    print('Weight:', weight)
    print('Constraint:', constraint)

    kp.save()






