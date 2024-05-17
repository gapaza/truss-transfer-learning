import numpy as np
import random
from copy import deepcopy

class KnapsackDesign:

    def __init__(self, design_vector=None, evaluator=None, num_bits=30, c_type='uniform', p_num=0, val=False, constraints=False):
        self.evaluator = evaluator
        self.num_bits = num_bits
        self.c_type = c_type  # values: random, point, uniform
        self.p_num = p_num
        self.val = val
        self.constraints = constraints

        self.vector = design_vector
        if not design_vector:
            self.vector = KnapsackDesign.random_design(num_bits)
            self.vector = list(self.vector)

        # Evaluation
        self.evaluated = False
        self.objectives = None

        self.is_feasible = True       # always assume true for now
        self.feasibility_score = 0.0  # negative, want to minimize this
        self.constraint_vals = None

        # Alg metrics
        self.crowding_dist = None
        self.rank = None

        self.memory = None

        # Constraints
        self.enforce_constraints()


    @staticmethod
    def random_design(n):
        bit_array = np.zeros(n, dtype=int)
        num_bits_to_flip = np.random.randint(1, n)
        indices_to_flip = np.random.choice(n, num_bits_to_flip, replace=False)
        for index in indices_to_flip:
            bit_array[index] = 1
        return list(bit_array)

    ##################
    ### Objectives ###
    ##################

    def set_objectives(self, value, weight):
        self.objectives = [value, weight]  # v_stiffness if negative
        self.evaluated = True

    def set_evaluate(self, results):
        value, weight = results
        value = float(value * -1.0)  # convert to negative value
        weight = float(weight)
        self.objectives = [value, weight]
        self.evaluated = True
        return deepcopy(self.objectives)

    def evaluate(self):
        if self.evaluated is True:
            return deepcopy(self.objectives)
        else:
            value, weight = self.evaluator.evaluate(self.vector)
            return self.set_evaluate([value, weight])

    #################
    ### Crossover ###
    #################

    def crossover(self, mother_d, father_d):
        c_type = self.c_type

        # Crossover
        child_vector = self.vector
        if c_type == 'point':
            crossover_point = random.randint(1, self.num_bits)
            child_vector = list(np.concatenate((mother_d.vector[:crossover_point], father_d.vector[crossover_point:])))
        elif c_type == 'uniform':
            child_vector = [random.choice([m_bit, f_bit]) for m_bit, f_bit in zip(mother_d.vector, father_d.vector)]

        # Set vector
        self.vector = deepcopy(child_vector)

        # Enforce Constraints
        self.enforce_constraints()


    ##############
    ### Mutate ###
    ##############

    def mutate(self):
        mut_prob = 1.0 / len(self.vector)
        for i in range(len(self.vector)):
            if random.random() < mut_prob:
                if self.vector[i] == 0:
                    self.vector[i] = 1
                else:
                    self.vector[i] = 0
        self.enforce_constraints()

    ###################
    ### Constraints ###
    ###################

    def enforce_constraints(self):
        return None

    def violates_constraints(self):
        if sum(self.vector) == 0:
            return True
        else:
            return False

    ###############
    ### Helpers ###
    ###############

    def get_vector_str(self):
        return self.vec_to_str(self.vector)

    def vec_to_str(self, vec):
        return ''.join([str(bit) for bit in vec])

    def vec_to_str_pnt(self, vec, pnt):
        mother_half = ''.join([str(bit) for bit in vec[:pnt]])
        father_half = ''.join([str(bit) for bit in vec[pnt:]])
        return mother_half + '|' + father_half

    def get_design_json(self):
        return {
            'design': self.get_vector_str(),
            'objectives': self.objectives,
        }



