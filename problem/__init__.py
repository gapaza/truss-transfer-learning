import numpy as np



# For Values: 0.8, 1.2, 0.01
# [0.8  0.82 0.84 0.86 0.88 0.9  0.92 0.94 0.96 0.98 1.   1.02 1.04 1.06 1.08 1.1  1.12 1.14 1.16 1.18 1.2 ]


def get_str_targets(min_tsr, max_tsr, sr_delta):
    bin_width = sr_delta * 2.0

    target_ratios = [min_tsr]
    while (target_ratios[-1] + bin_width) < max_tsr:
        target_ratios.append(target_ratios[-1] + bin_width)



    # num_bins = int((max_tsr - min_tsr) / (sr_delta*2.0)) + 2
    # target_ratios = np.linspace(min_tsr, max_tsr, num_bins)
    # target_ratios = np.round(target_ratios, 5)

    min_val = min(target_ratios)
    max_val = max(target_ratios)
    range = max_val - min_val
    target_ratios_norm = []
    for val in target_ratios:
        val_norm = (val - min_val) / range
        target_ratios_norm.append(val_norm)
    return list(target_ratios), list(target_ratios_norm)






if __name__ == '__main__':
    # str_delta = 0.01
    # min_str = 0.8
    # max_str = 1.2
    min_str, max_str, str_delta = 0.0, 2.0, 0.01
    target_ratios, target_ratios_norm = get_str_targets(min_str, max_str, str_delta)

    train_ratios = []
    val_ratios = []
    cnt = 1
    for x, y in zip(target_ratios, target_ratios_norm):
        if cnt % 10 == 0:
            val_ratios.append((x, y))
        else:
            train_ratios.append((x, y))
        cnt += 1

    print('All Ratios:', sorted(train_ratios + val_ratios, key=lambda x: x[0]))
    print('Train Ratios:', len(train_ratios), train_ratios)
    print('Val Ratios:', len(val_ratios), val_ratios)
















