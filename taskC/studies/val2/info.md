
# Validation Study 2

The purpose of this study is to assess the ability of the pre-trained generative agent to transfer to a fibre stiffness model from a truss stiffness model.
This study focuses on the constrained 3x3 truss design problem.


### Problem 

    # Constraints
    target_stiffness_ratio = 1.0
    feasible_stiffness_delta = 0.01

    # Training Problems
    num_training_problems = 36


### Model

    # Actor Model (2x decoders)
    actor_embed_dim = 16
    actor_heads = 8
    actor_dense = 256
    actor_dropout = 0.0

    # Critic Model (1x decoders)
    critic_embed_dim = 16
    critic_heads = 8
    critic_dense = 256
    critic_dropout = 0.0


## Pretraining

The model is pretrained on 36 training tasks for a total of 600 epochs.

### PPO Hyperparameters

    # Batch Size
    num_weight_samples = 4
    num_task_samples = 6
    repeat_size = 1

    # Parameters
    clip_ratio = 0.2
    target_kl = 0.005
    entropy_coef = 0.1  # was 0.1

    # Learning Rates
    actor_learning_rate = 0.0001
    critic_learning_rate = 0.0001
    train_actor_iterations = 250
    train_critic_iterations = 40

### Reward

    # Performance Term
    perf_term_weight = 1.0

    # Constraint Term
    constraint_term_weight = 0.01
    str_multiplier = 10.0
    fea_multiplier = 1.0





## Fine-tuning

The pre-trained model is evaluated on three validation problems. 



### Problem 1

Details here...


### Problem 2

Details here...


### Problem 3

Details here...












