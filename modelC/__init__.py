import tensorflow as tf
import config

# ---------------------------------------
# MultiTaskDecoder
# ---------------------------------------

from modelC.MultiTaskDecoder import MultiTaskDecoder, MultiTaskDecoderCritic

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
# MultiTaskDecoderMoe
# ---------------------------------------

from modelC.MultiTaskDecoderMoe import MultiTaskDecoderMoe, MultiTaskDecoderMoeCritic

def get_multi_task_decoder_moe(checkpoint_path_actor=None, checkpoint_path_critic=None):
    design_len = config.num_vars
    conditioning_values = config.num_conditioning_vars

    actor_model = MultiTaskDecoderMoe()
    decisions = tf.zeros((1, design_len))
    weights = tf.zeros((1, conditioning_values))
    actor_model([decisions, weights])

    critic_model = MultiTaskDecoderMoeCritic()
    decisions = tf.zeros((1, design_len + 1))
    weights = tf.zeros((1, conditioning_values))
    critic_model([decisions, weights])

    # Load Weights
    if checkpoint_path_actor:
        actor_model.load_weights(checkpoint_path_actor).expect_partial()
    if checkpoint_path_critic:
        critic_model.load_weights(checkpoint_path_critic).expect_partial()

    return actor_model, critic_model


# ---------------------------------------
# FeatureMTD
# ---------------------------------------

from modelC.FeatureMTD import FeatureMTD, FeatureMTDCritic

def get_feature_mtd(checkpoint_path_actor=None, checkpoint_path_critic=None):
    design_len = config.num_vars
    conditioning_values = config.num_conditioning_vars

    actor_model = FeatureMTD()
    decisions = tf.zeros((1, design_len))
    weights = tf.zeros((1, conditioning_values))
    actor_model([decisions, weights])

    critic_model = FeatureMTDCritic()
    decisions = tf.zeros((1, design_len + 1))
    weights = tf.zeros((1, conditioning_values))
    critic_model([decisions, weights])

    # Load Weights
    if checkpoint_path_actor:
        actor_model.load_weights(checkpoint_path_actor).expect_partial()
    if checkpoint_path_critic:
        critic_model.load_weights(checkpoint_path_critic).expect_partial()

    return actor_model, critic_model


# ---------------------------------------
# HierarchicalMTD
# ---------------------------------------


from modelC.HierarchicalMTD import HierarchicalMTD, HierarchicalMTDCritic

def get_hierarchical_mtd(checkpoint_path_actor=None, checkpoint_path_critic=None):
    design_len = config.num_vars
    conditioning_values = config.num_conditioning_vars

    actor_model = HierarchicalMTD()
    decisions = tf.zeros((1, design_len))
    weights = tf.zeros((1, conditioning_values))
    actor_model([decisions, weights])

    critic_model = HierarchicalMTDCritic()
    decisions = tf.zeros((1, design_len + 1))
    weights = tf.zeros((1, conditioning_values))
    critic_model([decisions, weights])

    critic_model_2 = HierarchicalMTDCritic()
    decisions = tf.zeros((1, design_len + 1))
    weights = tf.zeros((1, conditioning_values))
    critic_model_2([decisions, weights])

    # Load Weights
    if checkpoint_path_actor:
        actor_model.load_weights(checkpoint_path_actor).expect_partial()
    if checkpoint_path_critic:
        critic_model.load_weights(checkpoint_path_critic).expect_partial()

    return actor_model, critic_model, critic_model_2


# ---------------------------------------
# MultiTaskDecoderConstraint
# ---------------------------------------

from modelC.MultiTaskDecoderConstraint import MultiTaskDecoderConstraint, MultiTaskDecoderConstraintCritic

def get_multi_task_decoder_constraint(checkpoint_path_actor=None, checkpoint_path_critic=None):
    design_len = config.num_vars
    conditioning_values = config.num_conditioning_vars

    actor_model = MultiTaskDecoderConstraint()
    decisions = tf.zeros((1, design_len))
    weights = tf.zeros((1, conditioning_values))
    actor_model([decisions, weights])

    critic_model = MultiTaskDecoderConstraintCritic()
    decisions = tf.zeros((1, design_len + 1))
    weights = tf.zeros((1, conditioning_values))
    critic_model([decisions, weights])

    # Load Weights
    if checkpoint_path_actor:
        actor_model.load_weights(checkpoint_path_actor).expect_partial()
    if checkpoint_path_critic:
        critic_model.load_weights(checkpoint_path_critic).expect_partial()

    return actor_model, critic_model





