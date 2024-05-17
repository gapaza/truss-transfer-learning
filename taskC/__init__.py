import numpy as np




def sample_many_obj_weights(n, total=1):
    combinations = list(generate_fixed_sum_combinations(n, total=total))
    combinations = [tuple(round(x, 1) for x in combination) for combination in combinations]
    combinations = [combination[1:] for combination in combinations]

    print(combinations)
    print(len(combinations))
    # get all first values
    first_values = [combination[0] for combination in combinations]
    second_values = [combination[1] for combination in combinations]
    return combinations



def generate_fixed_sum_combinations(n, total=1):
    if n == 1:
        yield (total,)
    else:
        for value in np.linspace(0, total, num=int(total*10+1)):
            for permutation in generate_fixed_sum_combinations(n - 1, total - value):
                yield (value,) + permutation

# Example usage for 3 weights summing to 1:
combinations = sample_many_obj_weights(3, total=1)







