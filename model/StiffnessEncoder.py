import keras
from keras import layers
import tensorflow as tf
import config
import keras_nlp
import math
from keras_nlp.layers import TransformerDecoder, TransformerEncoder
from keras_nlp.layers import TokenAndPositionEmbedding
from keras_nlp.layers import SinePositionEncoding

# Vocabulary
# 0: [pad]
# 1: [start]
# 2: 0-bit
# 3: 1-bit

use_actor_mlp = True
use_critic_mlp = True

actor_embed_dim = 32
actor_heads = 32
actor_dense = 512

critic_embed_dim = 32
critic_heads = 32
critic_dense = 512

# ------------------------------------
# Actor
# ------------------------------------

@keras.saving.register_keras_serializable(package="StiffnessEncoder", name="StiffnessEncoder")
class StiffnessEncoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

        # Variables
        self.vocab_size = 2
        self.vocab_output_size = 30      # selects which bit to flip
        self.gen_design_seq_length = 30  # 30-bit design vector
        self.embed_dim = actor_embed_dim
        self.num_heads = actor_heads
        self.dense_dim = actor_dense

        # Token + Position embedding
        self.design_embedding_layer = TokenAndPositionEmbedding(
            self.vocab_size,
            self.gen_design_seq_length,
            self.embed_dim,
            mask_zero=True
        )

        # Decoder Stack
        self.normalize_first = False
        self.encoder_1 = TransformerEncoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='encoder_1')
        self.encoder_2 = TransformerEncoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='encoder_2')
        # self.encoder_3 = TransformerEncoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='encoder_3')
        # self.encoder_4 = TransformerEncoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='encoder_4')
        # self.encoder_5 = TransformerEncoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='encoder_5')

        # Testing MLP
        self.input_layer = layers.Dense(self.gen_design_seq_length, activation='linear', name='input_layer')
        self.hidden_1 = layers.Dense(self.dense_dim, activation='relu', name='hidden_1')
        self.hidden_2 = layers.Dense(self.dense_dim, activation='relu', name='hidden_2')
        # self.hidden_3 = layers.Dense(self.dense_dim, activation='relu', name='hidden_3')



        # Design Prediction Head
        # self.token_pooling_layer = layers.GlobalAveragePooling1D()
        # self.token_pooling_layer = layers.GlobalMaxPooling1D()
        self.token_pooling_layer = layers.Dense(1, name='token_pooling_layer')
        self.action_prediction_head = layers.Dense(
            self.vocab_output_size,
            # kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.005),
            name="design_prediction_head"
        )
        self.activation = layers.Activation('softmax', dtype='float32')
        self.log_activation = layers.Activation('log_softmax', dtype='float32')

    def call(self, inputs, training=False, mask=None):
        design_sequences = inputs

        # --- Testing MLP ---
        if use_actor_mlp is True:
            return self.call_mlp(design_sequences)
        # return self.call_mlp(design_sequences)


        # 3. Embedding / Encoding
        design_sequences_embedded = self.design_embedding_layer(design_sequences, training=training)
        decoded_design = design_sequences_embedded
        decoded_design = self.encoder_1(decoded_design)
        decoded_design = self.encoder_2(decoded_design)
        # decoded_design = self.encoder_3(decoded_design)
        # decoded_design = self.encoder_4(decoded_design)
        # decoded_design = self.encoder_5(decoded_design)

        # 4. Design Prediction Head
        decoded_design = self.token_pooling_layer(decoded_design)
        decoded_design = tf.squeeze(decoded_design, axis=-1)
        design_prediction_logits = self.action_prediction_head(decoded_design)
        design_prediction = self.activation(design_prediction_logits)
        design_prediction_log = self.log_activation(design_prediction_logits)

        return design_prediction, design_prediction_log  # For training

    def call_mlp(self, inputs):
        # --- Testing MLP ---
        x = self.input_layer(inputs)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        # x = self.hidden_3(x)
        design_prediction_logits = self.action_prediction_head(x)
        design_prediction = self.activation(design_prediction_logits)
        design_prediction_log = self.log_activation(design_prediction_logits)
        return design_prediction, design_prediction_log  # For training


    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)













# ------------------------------------
# Critic
# ------------------------------------

@keras.saving.register_keras_serializable(package="StiffnessEncoderCritic", name="StiffnessEncoderCritic")
class StiffnessEncoderCritic(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

        # Variables
        self.num_objectives = 1
        self.vocab_size = 2
        self.gen_design_seq_length = 30
        self.embed_dim = critic_embed_dim
        self.num_heads = critic_heads
        self.dense_dim = critic_dense

        # Token + Position embedding
        self.design_embedding_layer = TokenAndPositionEmbedding(
            self.vocab_size,
            self.gen_design_seq_length,
            self.embed_dim,
            mask_zero=True
        )

        # Encoder Stack
        self.normalize_first = False
        self.encoder_1 = TransformerEncoder(self.dense_dim, self.num_heads)
        self.encoder_2 = TransformerEncoder(self.dense_dim, self.num_heads)


        # Testing MLP
        self.input_layer = layers.Dense(self.gen_design_seq_length, activation='linear', name='input_layer')
        self.hidden_1 = layers.Dense(self.dense_dim, activation='relu', name='hidden_1')
        self.hidden_2 = layers.Dense(self.dense_dim, activation='relu', name='hidden_2')
        # self.hidden_3 = layers.Dense(self.dense_dim, activation='relu', name='hidden_3')

        # Output Prediction Head
        # self.token_pooling_layer = layers.GlobalAveragePooling1D()
        self.token_pooling_layer = layers.Dense(1, name='token_pooling_layer')
        self.output_modeling_head = layers.Dense(self.num_objectives, name='output_modeling_head')
        self.activation = layers.Activation('linear', dtype='float32')


    def call(self, inputs, training=False, mask=None):
        design_sequences = inputs

        # --- Testing MLP ---
        if use_critic_mlp is True:
            return self.call_mlp(design_sequences)

        # 3. Decoder Stack
        design_sequences_embedded = self.design_embedding_layer(design_sequences, training=training)
        decoded_design = design_sequences_embedded
        decoded_design = self.encoder_1(decoded_design)
        decoded_design = self.encoder_2(decoded_design)

        # 4. Output Prediction Head
        decoded_design = self.token_pooling_layer(decoded_design)  # output shape (batch, 1)
        decoded_design = tf.squeeze(decoded_design, axis=-1)
        output_prediction_logits = self.output_modeling_head(decoded_design)
        output_prediction = self.activation(output_prediction_logits)

        return output_prediction  # For training


    def call_mlp(self, inputs):
        # --- Testing MLP ---
        x = self.input_layer(inputs)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        # x = self.hidden_3(x)
        output_prediction_logits = self.output_modeling_head(x)
        output_prediction = self.activation(output_prediction_logits)
        return output_prediction


    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)





















