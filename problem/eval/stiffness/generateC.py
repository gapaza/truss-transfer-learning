import numpy as np

from problem.eval.stiffness.formK import formK


def generateC(sel, rvar, NC, CA, Avar, E, C, sidenum):
    # Initialize outputs
    uBasket = []
    FBasket = []

    # Iterate through once for each strain component
    for y in range(3):
        # Define vectors to hold indexes for output forces
        Fi_x = []
        Fi_y = []
        Fi_xy = []

        # Define strain vector: [e11, e22, e12]'
        strainvec = np.array([0, 0, 0], dtype=float)

        # set that component equal to a dummy value (0.01 strain),
        # set all other values to zero
        strainvec[y] = 0.01
        strainvec[2] = strainvec[2] * 2

        # use strain relations, BCs, and partitioned K-matrix to
        # solve for all unknowns
        e11, e22, e12 = strainvec

        K = formK(NC, CA, Avar, E)  # Assuming formK is defined elsewhere
        u_r = []
        F_q = []
        qvec = []
        rvec = []
        # Assigning Force/Displacement BCs for different nodes/DOFs
        for x in range(NC.shape[0]):  # looping through nodes by coordinate
            ND = NC / sel
            # Separating conditions for exterior nodes
            if (ND[x, 0] in [0, 1]) or (ND[x, 1] in [0, 1]):
                # Finding x-DOF
                if ND[x, 0] == 0:
                    u_r.append(0)
                    rvec.append((2 * x))
                elif ND[x, 0] == 1:
                    u_r.append(e11 * sel)
                    rvec.append((2 * x))
                    Fi_x.append((2 * x) - 1)  # Adjusted for zero-based indexing
                elif (ND[x, 1] == 0) and (e22 != 0):
                    u_r.append(0)
                    rvec.append((2 * x))
                else:
                    F_q.append(0)
                    qvec.append((2 * x))

                # Finding y-DOF
                if ND[x, 1] == 0:
                    u_r.append(0)
                    rvec.append((2 * x + 1))
                elif ND[x, 1] == 1:
                    u_r.append(e22 * sel)
                    rvec.append((2 * x + 1))
                    Fi_y.append((2 * x + 1) - 1)  # Adjusted for zero-based indexing
                    Fi_xy.append((2 * x) - 1)  # Adjusted for zero-based indexing
                elif (ND[x, 0] == 0) and (e11 != 0):
                    u_r.append(0)
                    rvec.append((2 * x))
                else:
                    F_q.append(0)
                    qvec.append((2 * x + 1))
            else:  # Condition for all interior nodes
                F_q.extend([0, 0])
                qvec.extend([(2 * x), (2 * x + 1)])

        # Adjusting for Python indexing
        u_r = np.array(u_r)
        F_q = np.array(F_q)
        qvec = np.array(qvec, dtype=int) - 1
        rvec = np.array(rvec, dtype=int) - 1

        qrvec = np.concatenate((qvec, rvec))
        newK = np.vstack((K[qvec, :], K[rvec, :]))
        newK = np.hstack((newK[:, qvec], newK[:, rvec]))
        K_qq = newK[:len(qvec), :len(qvec)]
        K_rq = newK[len(qvec):, :len(qvec)]
        K_qr = newK[:len(qvec), len(qvec):]
        K_rr = newK[len(qvec):, len(qvec):]
        u_q = np.linalg.solve(K_qq, F_q - np.dot(K_qr, u_r))
        F_r = np.dot(K_rq, u_q) + np.dot(K_rr, u_r)
        altu = np.concatenate((u_q, u_r))
        altF = np.concatenate((F_q, F_r))
        F = np.zeros(len(altF))
        u = np.zeros(len(altu))
        for x in range(len(qrvec)):
            F[qrvec[x]] = altF[x]
            u[qrvec[x]] = altu[x]

        # Finding average side "thicknesses" due to differing element radii
        horizrads = []
        for i in range(CA.shape[0]):
            if ((CA[i, 0] + sidenum) == CA[i, 1]) and (NC[CA[i, 0] - 1, 1] == sel):
                horizrads.append(rvar[i])
        vertrads = []
        for i in range(CA.shape[0]):
            if ((CA[i, 0] + 1) == CA[i, 1]) and (NC[CA[i, 0] - 1, 0] == sel):
                vertrads.append(rvar[i])
        horizmean = np.mean(horizrads)
        vertmean = np.mean(vertrads)

        # Correcting the calculation of forces for stress vector
        F_x = F_y = F_xy = 0
        for n in Fi_x:
            F_x += F[n]
        for n in Fi_y:
            F_y += F[n]
        for n in Fi_xy:
            F_xy += F[n]
        stressvec = np.array([F_x / (sel * 2 * vertmean), F_y / (sel * 2 * horizmean), F_xy / (sel * 2 * horizmean)])

        # use strain and stress vectors to solve for the corresponding
        # row of the C matrix
        Cdummy = stressvec / strainvec
        C[:, y] = Cdummy
        FBasket.append(F)
        uBasket.append(u)

    FBasket = np.column_stack(FBasket)
    uBasket = np.column_stack(uBasket)
    return C, uBasket, FBasket








if __name__ == '__main__':

    # Mock inputs for testing
    sel = 1.0  # Unit cell size
    rvar = np.array([0.1, 0.2, 0.1, 0.2])  # Radii of truss elements
    NC = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])  # Nodal coordinates
    CA = np.array([[1, 2], [2, 3], [3, 4], [4, 1], [1, 3], [2, 4]])  # Connectivity array
    Avar = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])  # Cross-sectional areas
    E = 210e9  # Young's modulus (e.g., steel in Pascals)
    C = np.zeros((3, 3))  # Initially empty stiffness tensor
    sidenum = 2  # Number of nodes along one side of the truss grid

    # Call the generateC function
    C_updated, uBasket, FBasket = generateC(sel, rvar, NC, CA, Avar, E, C, sidenum)

    # Print the outputs
    print("Updated Stiffness Tensor (C):")
    print(C_updated)
    print("\nDisplacement Vectors (uBasket):")
    print(uBasket)
    print("\nForce Vectors (FBasket):")
    print(FBasket)






