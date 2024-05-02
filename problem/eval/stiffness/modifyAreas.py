import numpy as np


def modifyAreas(Avar, CA, NC, sidenum):
    # Identify edge nodes
    edgenodes = np.concatenate([
        np.arange(1, sidenum + 1),
        np.arange(2 * sidenum, sidenum ** 2 - sidenum + 1, sidenum),
        np.arange(sidenum + 1, sidenum ** 2 - (2 * sidenum) + 2, sidenum),
        np.arange((sidenum ** 2) - sidenum + 1, sidenum ** 2 + 1)
    ])

    # Identify members connecting solely to edge nodes
    edgeconn1 = np.isin(CA[:, 0], edgenodes)
    edgeconn2 = np.isin(CA[:, 1], edgenodes)
    edgeconnectors = edgeconn1 & edgeconn2

    # Isolate edge members based on angle
    CAedgenodes = CA * edgeconnectors[:, np.newaxis]
    CAedgenodes = CAedgenodes[np.any(CAedgenodes, axis=1)]
    x1 = NC[CAedgenodes[:, 0] - 1, 0]
    x2 = NC[CAedgenodes[:, 1] - 1, 0]
    y1 = NC[CAedgenodes[:, 0] - 1, 1]
    y2 = NC[CAedgenodes[:, 1] - 1, 1]
    L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    angles = np.rad2deg(np.abs(np.arccos((x2 - x1) / L)))
    CAedgy = []
    for i in range(len(CAedgenodes)):
        if angles[i] == 0 or angles[i] == 90:
            CAedgy.append(CAedgenodes[i])
    CAedgy = np.array(CAedgy)

    # Find and modify areas belonging to edge members
    if CAedgy.size > 0:
        edgemembers = np.isin(CA, CAedgy).all(axis=1)
        selectAreas = Avar * edgemembers
        k = np.where(selectAreas)[0]
        Avar[k] = Avar[k] / 2

    return Avar






if __name__ == '__main__':
    Avar = np.array([1, 2, 3, 4, 5])  # Example cross-sectional areas
    CA = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 1]])  # Example connectivity array
    NC = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]])  # Example nodal coordinates
    sidenum = 2  # Example side number

    # Modify areas
    Avar_modified = modifyAreas(Avar, CA, NC, sidenum)
    print(Avar_modified)











