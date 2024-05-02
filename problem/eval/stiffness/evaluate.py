

from problem.eval.stiffness.generateC import generateC
from problem.eval.stiffness.formK import formK
from problem.eval.stiffness.generateNC import generateNC
from problem.eval.stiffness.modifyAreas import modifyAreas
import numpy as np


def trussMetaCalc_NxN_1UC_rVar_AVar(sidenum, sel, rvar, E, CA):
    # Generate vector with nodal coordinates
    NC = generateNC(sel, sidenum)

    # Calculate Avar & modify for edge members
    Avar = np.pi * (rvar ** 2)  # Cross-sectional areas of truss members
    Avar = modifyAreas(Avar, CA, NC, sidenum)

    # Initialize C matrix
    C = np.zeros((3, 3))  # Assuming 3x3 for 2D truss analysis

    # Develop C-matrix from K-matrix
    C, _, _ = generateC(sel, rvar, NC, CA, Avar, E, C, sidenum)

    return C


def evaluate():
    # Sample values (similar to the MATLAB example)
    sidenum = 3  # For a 3x3 grid
    sel = 0.01  # Unit cell size
    CA = np.array([
        [1, 2], [2, 3], [1, 4], [1, 5], [2, 5], [3, 5], [3, 6], [4, 5], [5, 6],
        [4, 7], [5, 7], [5, 8], [5, 9], [6, 9], [7, 8], [8, 9]
    ])  # Connectivity array

    E = 1816200  # Young's modulus
    rvar = (250 * (10 ** -6)) * np.ones(CA.shape[0])  # Radii of truss elements


    print(rvar)
    exit(0)

    # Calculate the stiffness tensor
    C = trussMetaCalc_NxN_1UC_rVar_AVar(sidenum, sel, rvar, E, CA)

    # Print the result
    print("Stiffness Tensor (C):")
    print(C)



if __name__ == '__main__':
    print('Running Evaluation')
    evaluate()






