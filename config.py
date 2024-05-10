import os
import pickle
from datetime import datetime
import platform
import json

# CPU vs GPU
import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')
#
# num_threads = 50
# tf.config.threading.set_inter_op_parallelism_threads(num_threads)
# tf.config.threading.set_intra_op_parallelism_threads(num_threads)

# from tensorflow.keras import mixed_precision
# policy = mixed_precision.Policy('mixed_bfloat16')
# mixed_precision.set_global_policy(policy)
# import keras
# keras.mixed_precision.set_global_policy("mixed_bfloat16")



#
#       _____   _                   _                _
#      |  __ \ (_)                 | |              (_)
#      | |  | | _  _ __  ___   ___ | |_  ___   _ __  _   ___  ___
#      | |  | || || '__|/ _ \ / __|| __|/ _ \ | '__|| | / _ \/ __|
#      | |__| || || |  |  __/| (__ | |_| (_) || |   | ||  __/\__ \
#      |_____/ |_||_|   \___| \___| \__|\___/ |_|   |_| \___||___/
#
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_dir = os.path.join(parent_dir, 'truss-transfer-learning')

# Models
models_dir = os.path.join(root_dir, 'model', 'store')
baseline_dir = os.path.join(root_dir, 'model', 'baseline')

# Results
results_dir = os.path.join(root_dir, 'results')
results_save_dir = os.path.join(results_dir, 'RUN')
results_save_dir_2 = os.path.join(results_dir, 'RUN2')
results_save_dir_3 = os.path.join(results_dir, 'RUN3')



#######################
# --> Bit Decoder <-- #
#######################
bd_num_weight_vecs = 9
bd_embed_dim = 16

actor_embed_dim = 16
actor_heads = 8
actor_dense = 256
actor_dropout = 0.0

# 3x3 Actor Vals: 16, 16, 256

# Previous Actor Vals: 64, 32, 1024 (run_3, best so far)
# Actor Vals: 32, 16, 512 (run_4, not great)
# Actor Vals: 128, 32, 1024 (run_5)

critic_embed_dim = 16
critic_heads = 8
critic_dense = 256
critic_dropout = 0.0

num_conditioning_vars = 2  # 2 for multiple stiffness ratio constraints, 4 for without

fine_tune_actor = False
fine_tune_critic = False

# Both Actor and Critic 32-512-32 for multi-task heavy constraint




# ------------------------------------
# HV Config
# ------------------------------------
hv_ref_point = [0, 1]  # vertical stiffness, volume fraction  -- was [0, 1]


sidenum = 3  # 3 | 4 | 5 | 6

# num_vars = 30  # 30 | 108 | 280 | 600
if sidenum == 3:
    num_vars = 30
elif sidenum == 4:
    num_vars = 108
elif sidenum == 5:
    num_vars = 280
elif sidenum == 6:
    num_vars = 600
else:
    raise ValueError('Invalid sidenum')

# num_vars_nr (non-repeatable)
num_vars_nr = 36







