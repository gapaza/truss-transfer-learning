import numpy as np
import random
from copy import deepcopy

class TrussDesign:

    def __init__(self, design_vector=None, evaluator=None, num_bits=30, c_type='uniform', p_num=0, val=False, constraints=False):
        self.evaluator = evaluator
        self.num_bits = num_bits
        self.c_type = c_type  # values: random, point, uniform
        self.p_num = p_num
        self.val = val
        self.constraints = constraints

        self.vector = design_vector
        if not design_vector:
            self.vector = TrussDesign.random_design(num_bits)
            self.vector = list(self.vector)

        # Evaluation
        self.evaluated = False
        self.objectives = None       # v_stiffness, vol_frac
        self.stiffness_ratio = None
        self.h_stiffness = None

        self.is_feasible = True  # always assume true for now
        self.feasibility_score = 0.0  # negative, want to minimize this
        self.constraint_vals = None

        # Alg metrics
        self.crowding_dist = None
        self.rank = None

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

    def set_objectives(self, v_stiffness, vol_frac):
        self.objectives = [v_stiffness, vol_frac]  # v_stiffness if negative
        self.evaluated = True

    ##################
    ### Evaluation ###
    ##################

    def set_evaluate(self, results):
        h_stiff, v_stiff, stiff_ratio, vol_frac, constraints = results
        v_stiff = float(v_stiff * -1.0)  # convert to negative value
        vol_frac = float(vol_frac)
        stiff_ratio = float(stiff_ratio)

        # Constraints
        if self.constraints is True:
            self.evaluate_constraints(constraints, stiff_ratio)

        self.objectives = [v_stiff, vol_frac]
        self.stiffness_ratio = stiff_ratio
        self.h_stiffness = h_stiff
        self.evaluated = True
        return deepcopy(self.objectives)


    def evaluate(self):
        if self.evaluated is True:
            return deepcopy(self.objectives)
        else:
            # print('--> LINEAR EVALUATION')
            # exit(0)
            h_stiff, v_stiff, stiff_ratio, vol_frac, constraints = self.evaluator.evaluate(self.vector, self.p_num, self.val)
            return self.set_evaluate([h_stiff, v_stiff, stiff_ratio, vol_frac, constraints])

    def evaluate_constraints(self, constraints, stiff_ratio):
        feasibility_constraint = float(constraints[0])
        connectivity_constraint = float(constraints[1])

        # Stiffness Ratio Constraint
        in_stiffness_window = False
        stiffness_ratio_delta = abs(self.evaluator.target_stiffness_ratio - stiff_ratio)
        if stiffness_ratio_delta < self.evaluator.feasible_stiffness_delta:
            in_stiffness_window = True

        # print('CONSTRAINTS:', feasibility_constraint, connectivity_constraint, stiffness_ratio_delta)


        if feasibility_constraint == 1.0 and connectivity_constraint == 1.0 and in_stiffness_window is True:
            self.is_feasible = True
        else:
            self.is_feasible = False

        self.constraint_vals = [feasibility_constraint, connectivity_constraint, stiffness_ratio_delta]

        self.feasibility_score = ((feasibility_constraint*10) + connectivity_constraint + (1.0 - stiffness_ratio_delta)) * -1.0  # convert to negative value
        # self.feasibility_score = stiffness_ratio_delta  # want to minimize, so keep positive


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



