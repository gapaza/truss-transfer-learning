import numpy as np


def formK(NC, CA, Avar, E):
    # Forming Elemental Stiffness Matrices
    num_elements = CA.shape[0]
    Kbasket = np.zeros((4, 4, num_elements))
    for i in range(num_elements):
        x1, y1 = NC[CA[i, 0] - 1, :]
        x2, y2 = NC[CA[i, 1] - 1, :]
        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        c = (x2 - x1) / L
        s = (y2 - y1) / L
        c2 = c**2
        s2 = s**2
        ktemp = np.array([[c2, c*s, -c2, -c*s],
                          [c*s, s2, -c*s, -s2],
                          [-c2, -c*s, c2, c*s],
                          [-c*s, -s2, c*s, s2]])
        ke = ((Avar[i] * E) / L) * ktemp
        Kbasket[:, :, i] = ke

    # Global-to-local-coordinate-system Coordination
    GlobToLoc = np.zeros((num_elements, 4), dtype=int)
    for n in range(2):
        GN = CA[:, n]
        for d in range(2):
            GlobToLoc[:, n*2 + d] = (GN - 1) * 2 + d

    # Forming Global Truss Stiffness Matrix
    K = np.zeros((2 * NC.shape[0], 2 * NC.shape[0]))
    for e in range(num_elements):
        ke = Kbasket[:, :, e]
        for lr in range(4):
            gr = GlobToLoc[e, lr]
            for lc in range(4):
                gc = GlobToLoc[e, lc]
                K[gr, gc] += ke[lr, lc]

    return K




if __name__ == '__main__':
    # Define the nodal coordinates matrix (NC)
    # Each row represents a node: [x_coordinate, y_coordinate]
    NC = np.array([[0, 0],  # Node 1
                   [1, 0],  # Node 2
                   [0, 1]])  # Node 3

    # Define the connectivity array (CA)
    # Each row represents an element: [node1_index, node2_index]
    CA = np.array([[1, 2],  # Element connecting nodes 1 and 2
                   [1, 3]])  # Element connecting nodes 1 and 3

    # Define the cross-sectional areas of each element (Avar)
    Avar = np.array([0.01, 0.01])  # Arbitrary areas for both elements

    # Define Young's modulus (E)
    E = 210e9  # Young's modulus in Pascals (e.g., 210 GPa for steel)

    # Call the formK function
    K = formK(NC, CA, Avar, E)

    # Print the global stiffness matrix (K)
    print("Global Stiffness Matrix (K):")
    print(K)




