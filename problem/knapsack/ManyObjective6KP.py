import random
import numpy as np
import os
import pickle

import config

from problem.knapsack.AbstractKnapsack import AbstractKnapsack





class ManyObjective6KP(AbstractKnapsack):

    def __init__(self, n=10, new_problem=False, id=''):
        super(ManyObjective6KP, self).__init__(n, new_problem, id)

        # --> Loss Landscape
        self.obj1_range = [0, 10]
        self.obj2_range = [0, 10]
        self.obj3_range = [0, 10]
        self.obj4_range = [0, 10]
        self.obj5_range = [0, 10]
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
            'values_3': [],
            'values_4': [],
            'values_5': [],
            'weights': [],
            'value_1_norm': 0,
            'value_2_norm': 0,
            'value_3_norm': 0,
            'value_4_norm': 0,
            'value_5_norm': 0,
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

            item_value_3 = 0
            if random.random() <= prob_value:
                item_value_3 = random.uniform(self.obj3_range[0], self.obj3_range[1])

            item_value_4 = 0
            if random.random() <= prob_value:
                item_value_4 = random.uniform(self.obj4_range[0], self.obj4_range[1])

            item_value_5 = 0
            if random.random() <= prob_value:
                item_value_5 = random.uniform(self.obj5_range[0], self.obj5_range[1])

            # Assign Weight
            item_weight = random.uniform(self.weight_range[0], self.weight_range[1])

            # Add to problem
            problem['values_1'].append(item_value_1)
            problem['values_2'].append(item_value_2)
            problem['values_3'].append(item_value_3)
            problem['values_4'].append(item_value_4)
            problem['values_5'].append(item_value_5)
            problem['weights'].append(item_weight)

        # Find Norms
        problem['value_1_norm'] = np.sum(problem['values_1'])
        problem['value_2_norm'] = np.sum(problem['values_2'])
        problem['value_3_norm'] = np.sum(problem['values_3'])
        problem['value_4_norm'] = np.sum(problem['values_4'])
        problem['value_5_norm'] = np.sum(problem['values_5'])
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
        obj3 = 0
        for i in range(self.n):
            obj3 += solution[i] * self.items['values_3'][i]

        # Objective 4
        obj4 = 0
        for i in range(self.n):
            obj4 += solution[i] * self.items['values_4'][i]

        # Objective 5
        obj5 = 0
        for i in range(self.n):
            obj5 += solution[i] * self.items['values_5'][i]

        # Weight
        weight = 0
        for i in range(self.n):
            weight += solution[i] * self.items['weights'][i]

        # Normalize
        obj1 = obj1 / self.items['value_1_norm']
        obj2 = obj2 / self.items['value_2_norm']
        obj3 = obj3 / self.items['value_3_norm']
        obj4 = obj4 / self.items['value_4_norm']
        obj5 = obj5 / self.items['value_5_norm']
        weight = weight / self.items['weight_norm']

        return [obj1, obj2, obj3, obj4, obj5, weight]




if __name__ == '__main__':
    n = config.num_vars

    kp = ManyObjective6KP(n=n, new_problem=True)

    sol = ''.join([str(1) for _ in range(n)])
    # sol = kp.random_solution()

    value_1, value_2, value_3, value_4, value_5, weight = kp.evaluate(sol)

    print('Solution:', sol)
    print('Value 1:', value_1)
    print('Value 2:', value_2)
    print('Value 3:', value_3)
    print('Value 4:', value_4)
    print('Value 5:', value_5)
    print('Weight:', weight)

    kp.save()






