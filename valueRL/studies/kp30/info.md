











# Set random seed
    seed_num = 0
    random.seed(seed_num)
    tf.random.set_seed(seed_num)

# Run parameters
    save_init_weights = False
    load_init_weights = False
    run_dir = 10
    run_num = 36  # ------------------------- RUN NUM
    plot_freq = 50
    task_epochs = 10000

# Trajectory Sampling
    total_weight_samples = 9
    num_weight_samples = 9  # 4
    repeat_size = 1
    global_mini_batch_size = num_weight_samples * repeat_size

# Target Network Update
    value_learning_rate = 0.001
    use_warmup = False
    update_batch_size_max = 32
    update_batch_size_min = 32
    update_batch_iterations = 60
    update_target_network_freq = 5

# Replay Buffer
    replay_buffer_size = 1000

# Epsilon Greedy
    epsilon = 0.99
    epsilon_end = 0.01
    decay_steps =100 * config.num_vars

# Reward
    perf_term_weight = 1.0











