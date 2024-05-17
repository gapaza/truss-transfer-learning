import random
import numpy as np
import os
import pickle

import config

from problem.knapsack.AbstractKnapsack import AbstractKnapsack





class ManyObjectiveSimpleKP(AbstractKnapsack):

    def __init__(self, n=10, new_problem=False, id=''):
        super(ManyObjectiveSimpleKP, self).__init__(n, new_problem, id)

        # --> Loss Landscape
        self.obj1_range = [0, 10]
        self.obj2_range = [0, 10]
        # self.obj3_range = [0, 10]
        self.weight_range = [0, 10]

        # --> Items
        if new_problem is False:
            self.items = self.load_items()
        else:
            self.generate_new_problem()

    def generate_new_problem(self):
        prob_value = 1.0
        problem = {
            'values_1': [],
            'values_2': [],
            'weights': [],
            'value_1_norm': 0,
            'value_2_norm': 0,
            'weight_norm': 0,
        }

        # Create weights / values
        for item_idx in range(self.n):

            # Assign Value
            item_value_1 = 0
            if random.random() <= prob_value:
                item_value_1 = random.uniform(self.obj1_range[0], self.obj2_range[1])

            item_value_2 = 0
            if random.random() <= prob_value:
                item_value_2 = random.uniform(self.obj2_range[0], self.obj2_range[1])

            # Assign Weight
            item_weight = random.uniform(self.weight_range[0], self.weight_range[1])

            # Add to problem
            problem['values_1'].append(item_value_1)
            problem['values_2'].append(item_value_2)
            problem['weights'].append(item_weight)

        # Find Norms
        problem['value_1_norm'] = np.sum(problem['values_1'])
        problem['value_2_norm'] = np.sum(problem['values_2'])
        problem['weight_norm'] = np.sum(problem['weights'])

        # Save problem items
        self.items = problem
        self.save()



    def evaluate(self, solution):
        if len(solution) != self.n:
            raise Exception('Solution must be of length n:', self.n)
        if type(solution) == str:
            solution = [int(i) for i in solution]

        # ---------------------------------------
        # Objectives
        # ---------------------------------------

        # Objective 1
        obj1 = 0
        for i in range(self.n):
            obj1 += solution[i] * self.items['values_1'][i]

        # Objective 2
        obj2 = 0
        for i in range(self.n):
            obj2 += solution[i] * self.items['values_2'][i]

        # Objective 3
        # obj3 = 0
        # for i in range(self.n):
        #     obj3 += solution[i] * self.items['values_3'][i]

        # Weight
        weight = 0
        for i in range(self.n):
            weight += solution[i] * self.items['weights'][i]

        # Normalize
        obj1 = obj1 / self.items['value_1_norm']
        obj2 = obj2 / self.items['value_2_norm']
        # obj3 = obj3 / self.items['value_3_norm']
        weight = weight / self.items['weight_norm']

        return [obj1, obj2, weight]




if __name__ == '__main__':
    n = config.num_vars

    kp = ManyObjectiveSimpleKP(n=n, new_problem=True)

    sol = ''.join([str(1) for _ in range(n)])
    # sol = kp.random_solution()

    value_1, value_2, weight = kp.evaluate(sol)

    print('Solution:', sol)
    print('Value 1:', value_1)
    print('Value 2:', value_2)
    print('Weight:', weight)

    kp.save()






