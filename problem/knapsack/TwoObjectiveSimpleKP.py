import random
import numpy as np
import os
import pickle

import config

from problem.knapsack.AbstractKnapsack import AbstractKnapsack



class TwoObjectiveSimpleKP(AbstractKnapsack):

    def __init__(self, n=10, new_problem=False, id=''):
        super(TwoObjectiveSimpleKP, self).__init__(n, new_problem, id)

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
            'value_norm': 0,
            'weight_norm': 0,
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

        # Find Norms
        problem['value_norm'] = np.sum(problem['values'])
        problem['weight_norm'] = np.sum(problem['weights'])

        # Save problem items
        self.items = problem
        self.save()


    def evaluate(self, solution):
        if len(solution) != self.n:
            raise Exception('Solution must be of length n:', self.n)
        if type(solution) == str:
            solution = [int(i) for i in solution]

        # --> Solution is a list of n binary values

        # ---------------------------------------
        # Objectives
        # ---------------------------------------

        # 1. Calculate value
        value = 0
        for i in range(self.n):
            if solution[i] == 1:
                value += self.items['values'][i]

        # 2. Calculate weight
        weight = 0
        for i in range(self.n):
            if solution[i] == 1:
                weight += self.items['weights'][i]

        # 3. Normalize
        # print(value, weight)
        weight = weight / self.items['weight_norm']
        value = value / self.items['value_norm']

        return value, weight


if __name__ == '__main__':
    n = config.num_vars

    kp = TwoObjectiveSimpleKP(n=n, new_problem=True)

    sol = ''.join([str(1) for _ in range(n)])
    # sol = kp.random_solution()

    value, weight = kp.evaluate(sol)

    print('Solution:', sol)
    print('Value:', value)
    print('Weight:', weight)

    kp.save()






