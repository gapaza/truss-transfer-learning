import tensorflow as tf
import config

# ---------------------------------------
# MultiTaskDecoder
# ---------------------------------------

from model.MultiTaskDecoder import MultiTaskDecoder, MultiTaskDecoderCritic

def get_multi_task_decoder(checkpoint_path_actor=None, checkpoint_path_critic=None):
    design_len = config.num_vars
    conditioning_values = config.num_conditioning_vars

    actor_model = MultiTaskDecoder()
    decisions = tf.zeros((1, design_len))
    weights = tf.zeros((1, conditioning_values))
    actor_model([decisions, weights])

    critic_model = MultiTaskDecoderCritic()
    decisions = tf.zeros((1, design_len + 1))
    weights = tf.zeros((1, conditioning_values))
    critic_model([decisions, weights])


    # Load Weights
    if checkpoint_path_actor:
        actor_model.load_weights(checkpoint_path_actor).expect_partial()
    if checkpoint_path_critic:
        critic_model.load_weights(checkpoint_path_critic).expect_partial()


    # actor_model.summary()

    return actor_model, critic_model



# ---------------------------------------
# StiffnessEncoder
# ---------------------------------------

from model.StiffnessEncoder import StiffnessEncoder, StiffnessEncoderCritic

def get_stiffness_encoder(checkpoint_path_actor=None, checkpoint_path_critic=None):
    design_len = config.num_vars

    actor_model = StiffnessEncoder()
    decisions = tf.zeros((1, design_len))
    actor_model(decisions)

    critic_model = StiffnessEncoderCritic()
    decisions = tf.zeros((1, design_len))
    critic_model(decisions)

    # Load Weights
    if checkpoint_path_actor:
        actor_model.load_weights(checkpoint_path_actor).expect_partial()
    if checkpoint_path_critic:
        critic_model.load_weights(checkpoint_path_critic).expect_partial()

    actor_model.summary()

    return actor_model, critic_model



# ---------------------------------------
# StiffnessDecoder
# ---------------------------------------

from model.StiffnessDecoder import StiffnessDecoder, StiffnessDecoderCritic

def get_stiffness_decoder(checkpoint_path_actor=None, checkpoint_path_critic=None):
    design_len = config.num_vars

    actor_model = StiffnessDecoder()
    decisions = tf.zeros((1, design_len))
    designs = tf.zeros((1, design_len))
    actor_model([decisions, designs])

    critic_model = StiffnessDecoderCritic()
    decisions = tf.zeros((1, design_len + 1))
    designs = tf.zeros((1, design_len))
    critic_model([decisions, designs])

    # Load Weights
    if checkpoint_path_actor:
        actor_model.load_weights(checkpoint_path_actor).expect_partial()
    if checkpoint_path_critic:
        critic_model.load_weights(checkpoint_path_critic).expect_partial()

    actor_model.summary()

    return actor_model, critic_model



# ---------------------------------------
# MultiTaskDecoderV2
# ---------------------------------------

from model.MultiTaskDecoderV2 import MultiTaskDecoderV2, MultiTaskDecoderV2Critic

def get_multi_task_decoder_v2(checkpoint_path_actor=None, checkpoint_path_critic=None):
    design_len = config.num_vars
    conditioning_values = config.num_conditioning_vars

    actor_model = MultiTaskDecoderV2()
    decisions = tf.zeros((1, design_len))
    weights = tf.zeros((1, conditioning_values))
    actor_model([decisions, weights])

    critic_model = MultiTaskDecoderV2Critic()
    decisions = tf.zeros((1, design_len + 1))
    weights = tf.zeros((1, conditioning_values))
    critic_model([decisions, weights])

    # Load Weights
    if checkpoint_path_actor:
        actor_model.load_weights(checkpoint_path_actor).expect_partial()
    if checkpoint_path_critic:
        critic_model.load_weights(checkpoint_path_critic).expect_partial()

    actor_model.summary()

    return actor_model, critic_model





















