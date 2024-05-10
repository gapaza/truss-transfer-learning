import numpy as np
import os
import config
import utils
import time
from tqdm import tqdm
import random
from copy import deepcopy
import pickle
from py4j.java_gateway import JavaGateway
from problem.TrussFeatures import TrussFeatures
from problem.eval.VolFrac import VolFrac
from multiprocessing import Pool


def calc_vol_frac(params):
    solution, sidenum, member_radius, side_length = params
    # volume_fraction = TrussFeatures(solution, sidenum, None).calculate_volume_fraction_5x5(member_radius, side_length)
    volume_fraction, num_intersections, features = VolFrac(sidenum, solution).evaluate(member_radius, side_length)
    return [volume_fraction, num_intersections]





class TrussProblem:

    def __init__(self, sidenum=3, calc_heuristics=False, calc_constraints=False, target_stiffness_ratio=1.0, feasible_stiffness_delta=0.1):
        self.f_name = self.__class__.__name__ + '_'+str(sidenum)+'.pkl'
        self.path = os.path.join(config.root_dir, 'problem', 'store', self.f_name)
        self.sidenum = sidenum
        if self.sidenum == 3:
            self.n = 30
        elif self.sidenum == 4:
            self.n = 108
        elif self.sidenum == 5:
            self.n = 280
        elif self.sidenum == 6:
            self.n = 600  # TODO: get the actual value
        else:
            raise ValueError('Invalid sidenum:', self.sidenum)
        self.calc_heuristics = calc_heuristics
        self.calc_constraints = calc_constraints
        self.target_stiffness_ratio = target_stiffness_ratio
        self.feasible_stiffness_delta = feasible_stiffness_delta

        # Java Gateway
        self.gateway = JavaGateway()
        self.my_java_object = self.gateway.entry_point

        # -----------------------------
        # Problem Configs
        # -----------------------------
        self.problem_types = [0, 1]  # 0 for truss stiffness, 1 for fibre stiffness

        self.member_radii = [300e-6, 250e-6, 200e-6, 150e-6, 100e-6]
        self.member_radii_norm = utils.normalize_list(self.member_radii)

        self.side_lengths = [5e-3, 10e-3, 15e-3, 30e-3, 45e-3]
        self.side_lengths_norm = utils.normalize_list(self.side_lengths)

        self.youngs_modulus = [1e5, 1.8162e6, 5e6, 20e6, 200e6]
        self.youngs_modulus_norm = utils.normalize_list(self.youngs_modulus)

        self.val_problems = [
            [0, 250e-6, 10e-3, 1.8162e6],
            [0, 275e-6, 35e-3, 100e6],
            [0, 400e-6, 100e-3, 300e6],
            [1, 250e-6, 10e-3, 1.8162e6],
            [1, 275e-6, 35e-3, 100e6],
            [1, 400e-6, 100e-3, 300e6],
        ]  # 3x3 val problems
        # self.val_problems = [
        #     [0, 250e-6, 10e-3, 1.8162e6],
        # ]  # 5x5 val problems
        self.val_problems_norm = []
        for val_problem in self.val_problems:
            self.val_problems_norm.append([
                val_problem[0],
                utils.normalize_val_from_list(self.member_radii, val_problem[1]),
                utils.normalize_val_from_list(self.side_lengths, val_problem[2]),
                utils.normalize_val_from_list(self.youngs_modulus, val_problem[3])
            ])
            # print('Val Problem:', val_problem, 'Norm:', self.val_problems_norm[-1])
        # exit(0)


        self.train_problems = []
        self.train_problems_norm = []
        for idx_pt, t in enumerate(self.problem_types):
            for idx_r, r in enumerate(self.member_radii):
                for idx_s, s in enumerate(self.side_lengths):
                    for idx_y, y in enumerate(self.youngs_modulus):
                        p_config = [0, r, s, y]
                        # TODO: FOR NOW, WE ARE ONLY USING TRUSS STIFFNESS DURING TRAINING
                        if p_config not in self.val_problems and t == 0:
                            self.train_problems.append([t, r, s, y])
                            self.train_problems_norm.append([
                                t,
                                self.member_radii_norm[idx_r],
                                self.side_lengths_norm[idx_s],
                                self.youngs_modulus_norm[idx_y]
                            ])
        print('Train Problems:', len(self.train_problems))
        print('Val Problems:', len(self.val_problems))

        # Shuffle train problems and train problems norm in the same way
        combined = list(zip(self.train_problems, self.train_problems_norm))
        random.seed(0)
        random.shuffle(combined)
        self.train_problems[:], self.train_problems_norm[:] = zip(*combined)

        # -----------------------------
        # Normalization Values
        # -----------------------------


        self.h_stiffness_norms_val = []
        self.v_stiffness_norms_val = []
        self.vol_frac_norms_val = []
        for problem_num in range(len(self.val_problems)):
            # print('Calculating norm values for val problem:', problem_num)
            vals = self.calc_norm_values(problem_num=problem_num, run_val=True)
            self.h_stiffness_norms_val.append(vals[0])
            self.v_stiffness_norms_val.append(vals[1])
            self.vol_frac_norms_val.append(vals[2])

        self.h_stiffness_norms_train = []
        self.v_stiffness_norms_train = []
        self.vol_frac_norms_train = []
        for problem_num in range(len(self.train_problems)):
            # print('Calculating norm values for problem:', problem_num)
            vals = self.calc_norm_values(problem_num=problem_num)
            print(vals[0], vals[1], vals[2])
            self.h_stiffness_norms_train.append(vals[0])
            self.v_stiffness_norms_train.append(vals[1])
            self.vol_frac_norms_train.append(vals[2])


        # --> Worker Pool
        self.pool = Pool(processes=6)


    # -----------------------------
    # Evaluate Batch
    # -----------------------------

    def evaluate_batch(self, solutions, problem_num=0, run_val=False):
        conv_solns = []
        for sol in solutions:
            if type(sol) == str:
                sol = [int(i) for i in sol]
            conv_solns.append(sol)
        solutions = conv_solns

        # 1. Get norm values
        if run_val:
            h_norm = self.h_stiffness_norms_val[problem_num]
            v_norm = self.v_stiffness_norms_val[problem_num]
            vol_norm = self.vol_frac_norms_val[problem_num]
        else:
            h_norm = self.h_stiffness_norms_train[problem_num]
            v_norm = self.v_stiffness_norms_train[problem_num]
            vol_norm = self.vol_frac_norms_train[problem_num]
        if h_norm == 0 or v_norm == 0 or vol_norm == 0:
            if run_val:
                params = self.val_problems[problem_num]
            else:
                params = self.train_problems[problem_num]
            s_fname = 'norm_values_' + '_'.join([str(i) for i in params]) + '_' + str(self.n) + '.pkl'
            print('Norm values are zero for file:', s_fname)
            exit(0)

        # 2. Evaluate
        batch_returns = self._evaluate_batch(solutions, problem_num=problem_num, run_val=run_val)
        # print('--------> BATCH:', batch_returns)

        # 3. Bath results
        batch_results = []
        for returns in batch_returns:
            h = returns[0]
            v = returns[1]
            sr = returns[2]
            vol = returns[3]
            constraints = None
            if self.calc_constraints:
                constraints = [returns[4], returns[5]]

            if h > h_norm:
                h = h_norm
            h = h / h_norm

            if v > v_norm:
                v = v_norm
            v = v / v_norm

            if vol > vol_norm:
                vol = vol_norm
            vol = vol / vol_norm

            # 4. Correct for training
            # if config.sidenum == 3:
            if v <= 0.0:
                vol = 1.0
            if vol >= 1.0:
                v = 0.0

            batch_results.append([h, v, sr, vol, constraints])

        return batch_results

    def _evaluate_batch(self, solutions, problem_num=0, run_val=False):
        conv_solns = []
        for sol in solutions:
            if type(sol) == str:
                sol = [int(i) for i in sol]
            conv_solns.append(sol)
        solutions = conv_solns

        # Prepopulation solution results
        if self.calc_constraints is False:
            def_results = [[0.0, 0.0, 0.0, 1.0] for i in range(len(solutions))]
        else:
            def_results = [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0] for i in range(len(solutions))]
        used_indices = []
        for soln in solutions:
            if sum(soln) == 0:
                used_indices.append(0)
            else:
                used_indices.append(1)

        # 1. Get problem parameters
        if run_val:
            params = self.val_problems[problem_num]
        else:
            params = self.train_problems[problem_num]
        java_list_designs = self.gateway.jvm.java.util.ArrayList()
        for used, sol in zip(used_indices, solutions):
            if used == 1:
                java_list_designs.append(self.convert_design(sol))

        # 2. Evaluate
        # --> TODO: Change constraints vs heuristics
        objectives = self.my_java_object.evaluateDesignsBatch(
            java_list_designs,
            params[0],  # problem type
            float(self.sidenum),
            params[1],  # member radius
            params[2],  # side length
            params[3],  # young's modulus

            # TODO: switch versions
            # Version 1
            self.calc_constraints,
            self.calc_heuristics,

            # Version 2
            # self.calc_heuristics,
            # self.calc_constraints
        )
        objectives = list(objectives)
        # print('---> BATCH SOURCE:', objectives)

        # Evaluate volume fractions
        vol_frac_params = []
        for sol in solutions:
            vol_frac_params.append([sol, self.sidenum, params[1], params[2]])
        vol_fracs = self.pool.map(calc_vol_frac, vol_frac_params)


        # Gather returns
        all_returns = []
        for idx, used in enumerate(used_indices):
            if used == 0:
                all_returns.append(def_results[idx])
            else:
                obj = objectives.pop(0)
                h_stiffness = float(obj[0])
                v_stiffness = float(obj[1])
                stiffness_ratio = float(obj[2])
                # volume_fraction = float(obj[3])
                volume_fraction = vol_fracs[idx][0]
                num_intersects = vol_fracs[idx][1]
                returns = [h_stiffness, v_stiffness, stiffness_ratio, volume_fraction]
                if self.calc_constraints is True:
                    # returns.append(float(obj[4]))
                    returns.append(float(num_intersects))
                    returns.append(float(obj[5]))
                all_returns.append(returns)
        return all_returns

    # -----------------------------
    # Evaluate
    # -----------------------------

    def evaluate(self, solution, problem_num=0, run_val=False):
        if type(solution) == str:
            solution = [int(i) for i in solution]

        # 1. Get norm values
        if run_val:
            h_norm = self.h_stiffness_norms_val[problem_num]
            v_norm = self.v_stiffness_norms_val[problem_num]
            vol_norm = self.vol_frac_norms_val[problem_num]
        else:
            h_norm = self.h_stiffness_norms_train[problem_num]
            v_norm = self.v_stiffness_norms_train[problem_num]
            vol_norm = self.vol_frac_norms_train[problem_num]

        # 2. Evaluate
        returns = self._evaluate(solution, problem_num=problem_num, run_val=run_val)
        # print('--------> SINGLE:', returns)

        h = returns[0]
        v = returns[1]
        sr = returns[2]
        vol = returns[3]
        constraints = None
        if self.calc_constraints:
            constraints = [returns[4], returns[5]]

        # 3. Normalize
        if h_norm == 0 or v_norm == 0 or vol_norm == 0:
            if run_val:
                params = self.val_problems[problem_num]
            else:
                params = self.train_problems[problem_num]
            s_fname = 'norm_values_' + '_'.join([str(i) for i in params]) + '_' + str(self.n) + '.pkl'
            print('Norm values are zero for file:', s_fname)
            exit(0)
        if h > h_norm:
            h = h_norm
        h = h / h_norm

        if v > v_norm:
            v = v_norm
        v = v / v_norm

        if vol > vol_norm:
            vol = vol_norm
        vol = vol / vol_norm

        # 4. Correct for training
        # if config.sidenum == 3:
        if v <= 0.0:
            vol = 1.0
        if vol >= 1.0:
            v = 0.0

        return h, v, sr, vol, constraints  # , returns[-1]

    def _evaluate(self, solution, problem_num=0, run_val=False):
        # problem_num: the train_problem instance to use
        # problem_type: 0 for truss stiffness, 1 for fibre stiffness

        # This special eval func only returns
        # 1. Horizontal Stiffness
        # 2. Vertical Stiffness
        # 3. Stiffness Ratio
        # 4. Volume Fraction

        if type(solution) == str:
            solution = [int(i) for i in solution]

        if sum(solution) == 0:
            if self.calc_constraints is False:
                return [0.0, 0.0, 10.0, 1.0]  # Make stiffness reatio large for penalty
            else:
                return [0.0, 0.0, 10.0, 1.0, 0.0, 0.0]

        # 1. Get problem parameters
        if run_val:
            params = self.val_problems[problem_num]
        else:
            params = self.train_problems[problem_num]
        java_list_design = self.convert_design(solution)

        # 2. Evaluate
        # print('Evaluating:', ''.join([str(int(i)) for i in solution]))
        # --> TODO: Change constraints vs heuristics
        objectives = self.my_java_object.evaluateDesign(
            java_list_design,
            params[0],  # problem type
            float(self.sidenum),
            params[1],  # member radius
            params[2],  # side length
            params[3],  # young's modulus

            # TODO: switch versions
            # Version 1
            self.calc_constraints,
            self.calc_heuristics,

            # Version 2
            # self.calc_heuristics,
            # self.calc_constraints
        )
        objectives = list(objectives)
        # print('---> SINGLE SOURCE:', ''.join([str(x) for x in solution]), objectives)

        h_stiffness = float(objectives[0])
        v_stiffness = float(objectives[1])
        stiffness_ratio = float(objectives[2])
        # volume_fraction = float(objectives[3])
        volume_fraction, num_intersections, features = VolFrac(self.sidenum, solution).evaluate(params[1], params[2])
        returns = [h_stiffness, v_stiffness, stiffness_ratio, volume_fraction]
        if self.calc_constraints is True:
            # returns.append(float(objectives[4]))
            returns.append(float(num_intersections))
            returns.append(float(objectives[5]))
        returns.append(features)
        return returns

    def convert_design(self, design):
        design_array = np.array(design, dtype=np.float64)
        java_list_design = self.gateway.jvm.java.util.ArrayList()
        for i in range(len(design_array)):
            java_list_design.append(design_array[i])
        return java_list_design

    def sample_random_designs(self, sample_size):
        num_ones = np.random.randint(1, self.n, sample_size)
        bitstrings = []
        for i in range(sample_size):
            design = np.zeros(self.n)
            design[:num_ones[i]] = 1
            np.random.shuffle(design)
            bitstrings.append(design)
        return bitstrings

    # ---------------------------------------
    # Normalization
    # ---------------------------------------

    def delete_norm_values(self, problem_num=0, run_val=False):
        if run_val:
            params = self.val_problems[problem_num]
        else:
            params = self.train_problems[problem_num]

        s_fname = 'norm_values_' + '_'.join([str(i) for i in params]) + '_' + str(self.n) + '.pkl'
        s_path = os.path.join(config.root_dir, 'problem', 'store', str(self.sidenum), s_fname)
        if os.path.exists(s_path):
            os.remove(s_path)
            print('Deleted norm values:', s_fname)

    def calc_norm_values(self, problem_num=0, run_val=False):
        if run_val:
            params = self.val_problems[problem_num]
        else:
            params = self.train_problems[problem_num]

        # -- Check if norm values have already been calculated --
        s_fname = 'norm_values_' + '_'.join([str(i) for i in params]) + '_' + str(self.n) + '.pkl'
        # print('Checking for saved norm values:', s_fname)
        s_path = os.path.join(config.root_dir, 'problem', 'store', str(self.sidenum), s_fname)
        if os.path.exists(s_path):
            # print('--> USING SAVED NORM VALUES')
            vals = pickle.load(open(s_path, 'rb'))
            # print('Norm values', params, vals)
            return vals

        # -- Get designs to calc norm values from --
        bit_lists = []
        bit_lists.append([1 for i in range(self.n)])  # Add all ones

        # sample_size = 2000
        # num_ones = np.random.randint(1, self.n, sample_size)
        # for i in range(sample_size):
        #     design = np.zeros(self.n)
        #     design[:num_ones[i]] = 1
        #     np.random.shuffle(design)
        #     bit_lists.append(design)
        # bit_lists.append([1 for i in range(self.n)])  # Add all ones

        # -- Calculate norm values --
        h_stiffness_vals = []
        v_stiffness_vals = []
        vol_frac_vals = []
        for i in tqdm(range(len(bit_lists)), desc='Calculating normalization values'):
            returns = self._evaluate(bit_lists[i], problem_num=problem_num, run_val=run_val)
            h = returns[0]
            v = returns[1]
            sr = returns[2]
            vol = returns[3]

            h_stiffness_vals.append(h)
            v_stiffness_vals.append(v)
            vol_frac_vals.append(vol)

        margin = 0.1
        h_stiffness_norm = np.max(h_stiffness_vals)  # + (np.max(h_stiffness_vals) * margin)
        v_stiffness_norm = np.max(v_stiffness_vals)  # + (np.max(v_stiffness_vals) * margin)
        vol_frac_norm = np.max(vol_frac_vals)  # Max volume fraction is 1.0 (or at least should be)
        vals = [h_stiffness_norm, v_stiffness_norm, vol_frac_norm]
        print('Norm values', params, vals)
        with open(s_path, 'wb') as f:
            pickle.dump(vals, f)
        return vals



####################################################################################################
# TESTING
####################################################################################################

import time
if __name__ == '__main__':
    truss = TrussProblem(
        sidenum=config.sidenum,
        calc_constraints=True,
        target_stiffness_ratio=1.0,
        feasible_stiffness_delta=0.01,
    )


    # design = [1 for x in range(truss.n)]
    # # design = '101000001110000010010100111001'
    # results = truss.evaluate(design, problem_num=0, run_val=True)
    # print('Results:', results)  # 3x3 94 max overlaps, 4x4 ~ 1304 max overlaps, 5x5 ~ 8942 max overlaps, 6x6 ~ 41397 max overlaps






    # design = '0000001010100000000100000000011011110001000000010001000100000000001000000001000000000001111001000000010010000011001010000010000000111100000111010000000000000111100100000100000000000000000010001000000000010000001000000000000000000000000001000000000010000010100110000000000010010000'
    # design = [int(i) for i in design]




    designs = truss.sample_random_designs(100)
    curr_time = time.time()
    results = truss.evaluate_batch(designs, problem_num=0, run_val=True)
    print('BATCH Time:', time.time() - curr_time)

    curr_time = time.time()
    single_results = []
    for design in designs:
        h, v, sr, vol, constraints = truss.evaluate(design, problem_num=0, run_val=True)
        single_results.append([h, v, sr, vol, constraints])
    print('SINGLE Time:', time.time() - curr_time)

    # Validate results are the same between batch and single
    for i in range(len(results)):
        if results[i] != single_results[i]:
            print('Results are different')
            print('Batch:', results[i])
            print('Single:', single_results[i])
            exit(0)
    print('SUCCESS')







    # design = [1 for x in range(truss.n)]
    # h, v, sr, vol, constraints = truss.evaluate(design, problem_num=0, run_val=True)
    # print('Horizontal Stiffness:', h)
    # print('Vertical Stiffness:', v)
    # print('Stiffness Ratio:', sr)
    # print('Volume Fraction:', vol)
    # print('Constraints:', constraints)

    #
    # designs = truss.sample_random_designs(3)
    # for design in designs:
    #     h, v, sr, vol, constraints = truss.evaluate(design, problem_num=0, run_val=True)
    #     feasibility_constraint = constraints[0]
    #     connectivity_constraint = constraints[1]
    #     print('------------------------')
    #     print('Horizontal Stiffness:', h)
    #     print('Vertical Stiffness:', v)
    #     print('Stiffness Ratio:', sr)
    #     print('Volume Fraction:', vol)
    #     print('Constraints:', constraints)
    #
    #










