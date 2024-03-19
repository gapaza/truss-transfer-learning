import numpy as np
import time
import tensorflow as tf
from copy import deepcopy
import random
import config
import multiprocessing as mp

import seaborn as sns
import matplotlib.pyplot as plt
import os

from model import get_multi_task_decoder as get_model


# class AbstractTask(mp.Process):
class AbstractTask:

    def __init__(self, run_num=0, barrier=None, problem=None, epochs=50, actor_load_path=None, critic_load_path=None):
        # super(AbstractTask, self).__init__()
        self.actor_load_path = actor_load_path
        self.critic_load_path = critic_load_path
        self.run_num = run_num
        self.barrier = barrier
        self.problem = problem
        self.epochs = epochs

        # Writing
        self.run_root = config.results_save_dir
        self.run_dir = os.path.join(self.run_root, 'run_' + str(self.run_num))
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

        # Save file (when finished)
        self.actor_save_path = os.path.join(self.run_dir, 'actor_weights')
        self.critic_save_path = os.path.join(self.run_dir, 'critic_weights')

        # Optimizers and models initialized when new process is created
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.c_actor = None
        self.c_critic = None

        # Logging
        self.returns = []
        self.c_loss = []
        self.p_loss = []
        self.p_iter = []
        self.entropy = []
        self.kl = []

    def activate_barrier(self):
        if self.barrier is not None:
            self.barrier.wait()

    def build(self):

        # Optimizer parameters
        self.actor_learning_rate = 0.0001
        self.critic_learning_rate = 0.0001
        self.train_actor_iterations = 250
        self.train_critic_iterations = 80
        if self.actor_optimizer is None:
            self.actor_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.actor_learning_rate)
        if self.critic_optimizer is None:
            self.critic_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.critic_learning_rate)

        self.c_actor, self.c_critic = get_model(self.actor_load_path, self.critic_load_path)

    def run(self):
        print('--> RUNNING ABSTRACT TASK')
        self.build()

    def record(self, epoch_info):
        # Record new epoch / print
        print(f"Proc {self.run_num}", end=' ')
        for key, value in epoch_info.items():
            print(f"{key}: {value}", end=' | ')
        print('')

        # Update metrics
        self.returns.append(epoch_info['mb_return'])
        self.c_loss.append(epoch_info['c_loss'])
        self.p_loss.append(epoch_info['p_loss'])
        self.p_iter.append(epoch_info['p_iter'])
        self.entropy.append(epoch_info['entropy'])
        self.kl.append(epoch_info['kl'])


