import numpy as np



# For Values: 0.8, 1.2, 0.01
# [0.8  0.82 0.84 0.86 0.88 0.9  0.92 0.94 0.96 0.98 1.   1.02 1.04 1.06 1.08 1.1  1.12 1.14 1.16 1.18 1.2 ]


def get_str_targets(min_tsr, max_tsr, sr_delta):
    num_bins = int((max_tsr - min_tsr) / (sr_delta*2.0)) + 2
    target_ratios = np.linspace(min_tsr, max_tsr, num_bins)
    target_ratios = np.round(target_ratios, 5)

    min_val = min(target_ratios)
    max_val = max(target_ratios)
    range = max_val - min_val
    target_ratios_norm = []
    for val in target_ratios:
        val_norm = (val - min_val) / range
        target_ratios_norm.append(val_norm)
    return list(target_ratios), list(target_ratios_norm)






if __name__ == '__main__':
    sr_delta = 0.01
    min_tsr = 0.8
    max_tsr = 1.2
    target_ratios = get_str_targets(min_tsr, max_tsr, sr_delta)















