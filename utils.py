import numpy as np




def normalize_list(data):
    if not data:
        return []

    min_val = min(data)
    max_val = max(data)
    range_val = max_val - min_val

    # Avoid division by zero in case all values are the same
    if range_val == 0:
        return [0.5 for _ in data]  # or return data if you prefer

    normalized = [(x - min_val) / range_val for x in data]
    return normalized


def normalize_val_from_list(data, val):
    if not data:
        raise ValueError('Data list is empty')

    min_val = min(data)
    max_val = max(data)
    range_val = max_val - min_val

    # Avoid division by zero in case all values are the same
    if range_val == 0:
        return 0.5  # or return val if you prefer

    normalized = (val - min_val) / range_val
    return normalized





def calc_crowding_distance(F):
    n_points, n_obj = F.shape

    if n_points <= 2:
        return np.full(n_points, np.inf)
        # return np.full(n_points, 1.)
    else:

        is_unique = np.arange(n_points)

        # index the unique points of the array
        _F = F[is_unique]

        # sort each column and get index
        I = np.argsort(_F, axis=0, kind='mergesort')

        # sort the objective space values for the whole matrix
        _F = _F[I, np.arange(n_obj)]

        # calculate the distance from each point to the last and next
        dist = np.row_stack([_F, np.full(n_obj, np.inf)]) - np.row_stack([np.full(n_obj, -np.inf), _F])

        # calculate the norm for each objective - set to NaN if all values are equal
        norm = np.max(_F, axis=0) - np.min(_F, axis=0)
        norm[norm == 0] = np.nan

        # prepare the distance to last and next vectors
        dist_to_last, dist_to_next = dist, np.copy(dist)
        dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

        # if we divide by zero because all values in one columns are equal replace by none
        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0

        # sum up the distance to next and last and norm by objectives - also reorder from sorted list
        J = np.argsort(I, axis=0)
        _cd = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

        # save the final vector which sets the crowding distance for duplicates to zero to be eliminated
        crowding = np.zeros(n_points)
        crowding[is_unique] = _cd

    # crowding[np.isinf(crowding)] = 1e+14
    return crowding





if __name__ == '__main__':
    data = [300e-6, 250e-6, 200e-6, 150e-6, 100e-6]
    print(normalize_list(data))  # [0.0, 0.25, 0.5, 0.75, 1.0]


