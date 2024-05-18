import random
import numpy as np
import os
import pickle
import config







class AbstractKnapsack:

    def __init__(self, n=10, new_problem=False, id=''):
        # get name of class
        self.f_name = self.__class__.__name__ + '_'+str(n)+'_'+id+'.pkl'
        self.path = os.path.join(config.root_dir, 'problem', 'store', self.f_name)
        self.n = n
        self.new_problem = new_problem
        self.id = id

        # Problem Items
        self.items = None

    def generate_new_problem(self):
        pass

    def evaluate(self, solution):
        pass

    # --------------------------------
    # Common Methods
    # --------------------------------

    def random_design(self):
        bit_array = np.zeros(self.n, dtype=int)
        num_bits_to_flip = np.random.randint(1, self.n)
        indices_to_flip = np.random.choice(self.n, num_bits_to_flip, replace=False)
        for index in indices_to_flip:
            bit_array[index] = 1
        return list(bit_array)


    def save(self):
        with open(self.path, 'wb') as f:
            pickle.dump(self.items, f)
    def load_items(self):
        with open(self.path, 'rb') as f:
            items = pickle.load(f)
        return items



