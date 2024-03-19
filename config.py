import os
import pickle
from datetime import datetime
import platform
import json

# CPU vs GPU
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

# num_threads = 8
# tf.config.threading.set_inter_op_parallelism_threads(num_threads)
# tf.config.threading.set_intra_op_parallelism_threads(num_threads)




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

actor_embed_dim = 64
actor_heads = 16
actor_dense = 2048
actor_dropout = 0.0

critic_embed_dim = 32
critic_heads = 16
critic_dense = 512
critic_dropout = 0.0

num_conditioning_vars = 4

fine_tune_actor = False
fine_tune_critic = False


# Both Actor and Critic 32-512-32 for multi-task heavy constraint




# ------------------------------------
# HV Config
# ------------------------------------
hv_ref_point = [0, 1]  # vertical stiffness, volume fraction


sidenum = 5  # 3 | 5
num_vars = 280  # 30 | 280









