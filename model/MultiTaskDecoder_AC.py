import keras
from keras import layers
import tensorflow as tf
import config
import keras_nlp
import math
from keras_nlp.layers import TransformerDecoder
from keras_nlp.layers import TokenAndPositionEmbedding
from keras_nlp.layers import SinePositionEncoding

# Vocabulary
# 0: [pad]
# 1: [start]
# 2: 0-bit
# 3: 1-bit

fine_tune_actor = config.fine_tune_actor
fine_tune_critic = config.fine_tune_critic

actor_embed_dim = config.actor_embed_dim
actor_heads = config.actor_heads
actor_dense = config.actor_dense
actor_dropout = config.actor_dropout

# ------------------------------------
# Actor / Critic
# ------------------------------------

@keras.saving.register_keras_serializable(package="MultiTaskDecoder_AC", name="MultiTaskDecoder_AC")
class MultiTaskDecoder_AC(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

        # Variables
        self.num_objectives = 1
        self.vocab_size = 4
        self.vocab_output_size = 2
        self.gen_design_seq_length = config.num_vars + 1
        self.embed_dim = actor_embed_dim
        self.num_heads = actor_heads
        self.dense_dim = actor_dense

        # Conditioning Vector Positional Encoding
        self.positional_encoding = SinePositionEncoding(name='positional_encoding')
        # self.conditioning_values = 4
        # self.positional_encoding = self.add_weight(
        #     shape=(self.conditioning_values, 1),  # for learning across embedding dim use (self.conditioning_values, self.embed_dim)
        #     initializer='zeros',                  # Start with zeros
        #     trainable=True,
        #     name='positional_encoding'
        # )

        # Token + Position embedding
        self.design_embedding_layer = TokenAndPositionEmbedding(
            self.vocab_size,
            self.gen_design_seq_length,
            self.embed_dim,
            mask_zero=True
        )

        # Decoder Stack
        self.normalize_first = False
        self.decoder_1 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_1', dropout=actor_dropout)
        self.decoder_2 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_2', dropout=actor_dropout)
        # self.decoder_3 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_3', dropout=actor_dropout)
        # self.decoder_4 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_4', dropout=actor_dropout)
        # self.decoder_5 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_5', dropout=actor_dropout)


        # Design Prediction Head
        self.design_prediction_head = layers.Dense(
            self.vocab_output_size,
            # kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.005),
            name="design_prediction_head"
        )
        self.design_activation = layers.Activation('softmax', dtype='float32')

        # Output Prediction Head
        self.output_modeling_head = layers.Dense(self.num_objectives, name='output_modeling_head')
        self.output_activation = layers.Activation('linear', dtype='float32')

    def call(self, inputs, training=False, mask=None):
        design_sequences, weights = inputs

        # 1. Weights
        weight_seq = self.add_positional_encoding(weights)  # (batch, num_weights, embed_dim)
        # weight_seq = weight_seq[:, 0:1, :]

        # 2. Embed design_sequences
        design_sequences_embedded = self.design_embedding_layer(design_sequences, training=training)

        # 3. Decoder Stack
        decoded_design = design_sequences_embedded
        decoder_1_output = self.decoder_1(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True, training=training)
        decoder_2_output = self.decoder_2(decoder_1_output, encoder_sequence=weight_seq, use_causal_mask=True, training=training)
        # decoded_design = self.decoder_3(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True, training=training)
        # decoded_design = self.decoder_4(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True, training=training)
        # decoded_design = self.decoder_5(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True)

        # 3. Decoder Stack (no cross attention)
        # decoded_design = self.decoder_1(decoded_design, use_causal_mask=True, training=training)
        # decoded_design = self.decoder_2(decoded_design, use_causal_mask=True, training=training)

        # 4. Design Prediction Head
        design_prediction_logits = self.design_prediction_head(decoder_2_output)
        design_prediction = self.design_activation(design_prediction_logits)

        # 5. Output Prediction Head
        output_prediction_logits = self.output_modeling_head(decoder_1_output)
        output_prediction = self.output_activation(output_prediction_logits)

        return design_prediction, output_prediction

    def add_positional_encoding(self, weights):

        # Tile conditioning weights across embedding dimension
        weight_seq = tf.expand_dims(weights, axis=-1)
        weight_seq = tf.tile(weight_seq, [1, 1, self.embed_dim])

        # For sine positional encoding
        pos_enc = self.positional_encoding(weight_seq)
        weight_seq = weight_seq + pos_enc

        # Add learned positional encoding constants
        # position_bias_broadcasted = tf.reshape(self.positional_encoding, [1, self.conditioning_values, 1])  # Make it broadcastable: (1, 2, 1)
        # position_bias_broadcasted = tf.tile(position_bias_broadcasted, [1, 1, self.embed_dim])  # Shape: (1, 2, embed_dim)
        # weight_seq += position_bias_broadcasted

        return weight_seq

    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


























