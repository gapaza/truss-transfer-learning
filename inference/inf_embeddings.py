import os
import config
from model import get_multi_task_decoder as get_model
import tensorflow as tf
import numpy as np
import tensorflow as tf
from keras_nlp.layers import RotaryEmbedding
import random



# random.seed(0)
# tf.random.set_seed(1)


def run():
    print('--> Embedding Inference')

    actor_save_path = os.path.join(config.results_save_dir, 'run_' + str(3), 'pretrained', 'actor_weights_4850')  # 4850
    critic_save_path = os.path.join(config.results_save_dir, 'run_' + str(3), 'pretrained', 'critic_weights_4850')  # 4850
    actor, critic = get_model(actor_save_path, critic_save_path)


    design_embedding = actor.design_embedding_layer
    token_embedding = design_embedding.token_embedding
    position_embedding = design_embedding.position_embedding



    model_input = [[0, 1, 2, 3]]
    weights_input = [[0.1, 0.2, 0.3, 0.4]]
    model_input = tf.convert_to_tensor(model_input, dtype=tf.float32)
    weights_input = tf.convert_to_tensor(weights_input, dtype=tf.float32)

    # model_output = actor(model_input)

    model_output, constraint_output = actor.call_constraint([model_input, weights_input])
    print('Model Output: ', model_output.shape)
    print('Constraint Output: ', constraint_output.shape, constraint_output)

    # embedding = token_embedding(model_input)
    # # print(embedding.numpy().tolist()[0][0])
    # # print(embedding.numpy().tolist()[0][1])
    # # print(embedding.numpy().tolist()[0][2])
    # print(embedding.numpy().tolist()[0][3])
    #
    # embedding = position_embedding(embedding)
    # # print(embedding.numpy().tolist()[0][0])
    # # print(embedding.numpy().tolist()[0][1])
    # # print(embedding.numpy().tolist()[0][2])
    # print(embedding.numpy().tolist()[0][3])





from model.moe_decoder.moe import MoeLayer
from model.moe_decoder.MoeDecoder import MoeDecoder


def run2():

    embedding_seq = [
        [1, 1, 1, 1],
        [1, 1, 2, 1],
        [1, 2, 1, 1]
    ]
    embedding_tensor = tf.convert_to_tensor(embedding_seq, dtype=tf.float32)
    embedding_tensor = tf.expand_dims(embedding_tensor, axis=0)
    print(embedding_tensor)

    # moe_layer = MoeLayer(4, 12, 4, 2)
    # output = moe_layer(embedding_tensor)
    # print(output)

    moe_decoder = MoeDecoder(
        intermediate_dim=32,
        num_heads=2,
        num_experts=4,
        num_experts_per_token=2
    )

    output = moe_decoder(embedding_tensor)
    print(output)


























if __name__ == '__main__':
    run2()










