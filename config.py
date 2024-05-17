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



model_embed_dim = 32
model_heads = 32
model_dense = 512
model_dropout = 0.0

actor_embed_dim = model_embed_dim
actor_heads = model_heads
actor_dense = model_dense
actor_dropout = model_dropout

critic_embed_dim = model_embed_dim
critic_heads = model_heads
critic_dense = model_dense
critic_dropout = model_dropout

num_conditioning_vars = 1

fine_tune_actor = False
fine_tune_critic = False

# Both Actor and Critic 32-512-32 for multi-task heavy constraint




# ------------------------------------
# HV Config
# ------------------------------------
hv_ref_point = [0, 1]  # vertical stiffness, volume fraction  -- was [0, 1]
hv_ref_point_mo = [0, 0, 1]
hv_ref_point_mo4 = [0, 0, 0, 1]
hv_ref_point_mo6 = [0, 0, 0, 0, 0, 1]

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

# sidenum = 4
# num_vars = 60

# num_vars_nr (non-repeatable)
num_vars_nr = 36







