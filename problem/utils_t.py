
from problem.TrussProblem import TrussProblem
from problem.TrussFeatures import TrussFeatures
from problem.eval.stiffness.evaluate import trussMetaCalc_NxN_1UC_rVar_AVar
import numpy as np





if __name__ == '__main__':
    member_radii, side_length, y_modulus = 250e-6, 10e-3, 1.8162e6
    p_num, val = 0, True


    design = [1 for x in range(30)]

    truss_design = TrussFeatures(design, 3, None)
    design_conn_array = np.array(truss_design.design_conn_array)
    radius_array = np.array([member_radii for x in range(len(design_conn_array))])

    print(design_conn_array)


    # Evaluate with truss problem
    problem = TrussProblem(sidenum=3, calc_constraints=False)
    result = problem._evaluate(design, p_num, run_val=val)
    print(result)  # [617817.8148655131, 617817.8148655131, 1.0, 0.44267458634709017]

    # Evaluate with trussMetaCalc
    result = trussMetaCalc_NxN_1UC_rVar_AVar(3, side_length, radius_array, y_modulus, design_conn_array)
    print(result)

















