import keras
from keras import layers
import tensorflow as tf
import config
import keras_nlp
import math
from keras_nlp.layers import TransformerDecoder
from model.alibi_decoder.AlibiDecoder import AlibiDecoder
from keras_nlp.layers import TokenAndPositionEmbedding
from keras_nlp.layers import SinePositionEncoding
from keras_nlp.layers import RotaryEmbedding

# Vocabulary
# 0: [pad]
# 1: [start]
# 2: 0-bit
# 3: 1-bit

fine_tune_actor = False
fine_tune_critic = False

actor_embed_dim = 64
actor_heads = 16
actor_dense = 512
actor_dropout = 0.0

critic_embed_dim = 32
critic_heads = 16
critic_dense = 512
critic_dropout = 0.0

# ------------------------------------
# Actor
# ------------------------------------

@keras.saving.register_keras_serializable(package="UniversalMultiTaskDecoder", name="UniversalMultiTaskDecoder")
class UniversalMultiTaskDecoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

        # Variables
        self.vocab_size = 4
        self.vocab_output_size = 2
        self.gen_design_seq_length = config.num_vars_nr
        self.embed_dim = actor_embed_dim
        self.num_heads = actor_heads
        self.dense_dim = actor_dense

        # Conditioning Vector Positional Encoding
        # self.positional_encoding = SinePositionEncoding(name='positional_encoding')
        self.positional_encoding = RotaryEmbedding(name='positional_encoding')

        # Token + Position embedding
        # self.design_embedding_layer = TokenAndPositionEmbedding(
        #     self.vocab_size,
        #     self.gen_design_seq_length,
        #     self.embed_dim,
        #     mask_zero=True
        # )
        self.design_embedding = layers.Embedding(
            self.vocab_size,
            self.embed_dim,
            mask_zero=True,
            name='design_embedding'
        )
        self.design_positional_embedding = RotaryEmbedding(name='design_positional_embedding_rotary')



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
            name="design_prediction_head"
        )
        self.activation = layers.Activation('softmax', dtype='float32')
        self.log_activation = layers.Activation('log_softmax', dtype='float32')

    def call(self, inputs, training=False, mask=None):
        design_sequences, weights = inputs

        # 1. Weights
        weight_seq = self.add_positional_encoding(weights)  # (batch, num_weights, embed_dim)

        # 2. Embed design_sequences
        # design_sequences_embedded = self.design_embedding_layer(design_sequences, training=training)
        design_sequences_embedded = self.design_embedding(design_sequences)
        design_sequences_embedded = self.design_positional_embedding(design_sequences_embedded)

        # 3. Decoder Stack
        decoded_design = design_sequences_embedded
        decoded_design = self.decoder_1(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True, training=training)
        decoded_design = self.decoder_2(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True, training=training)
        # decoded_design = self.decoder_3(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True, training=training)
        # decoded_design = self.decoder_4(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True, training=training)
        # decoded_design = self.decoder_5(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True)

        design_prediction_logits = self.design_prediction_head(decoded_design)
        design_prediction = self.activation(design_prediction_logits)

        return design_prediction  # For training

    def add_positional_encoding(self, weights):
        # Weights has shape (batch, weight + num_decisions, 2)

        # Tile conditioning weights across embedding dimension
        # weight_seq = tf.expand_dims(weights, axis=-1)
        weight_seq = tf.tile(weights, [1, 1, int(self.embed_dim / 4)])

        # For sine positional encoding
        # pos_enc = self.positional_encoding(weight_seq)
        # weight_seq = weight_seq + pos_enc
        weight_seq = self.positional_encoding(weight_seq)

        return weight_seq

    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



# ------------------------------------
# Critic
# ------------------------------------

@keras.saving.register_keras_serializable(package="UniversalMultiTaskDecoderCritic", name="UniversalMultiTaskDecoderCritic")
class UniversalMultiTaskDecoderCritic(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

        # Variables
        self.num_objectives = 1
        self.vocab_size = 4
        self.gen_design_seq_length = config.num_vars_nr + 1
        self.embed_dim = critic_embed_dim
        self.num_heads = critic_heads
        self.dense_dim = critic_dense

        # Conditioning Vector Positional Encoding
        # self.positional_encoding = SinePositionEncoding(name='positional_encoding')
        self.positional_encoding = RotaryEmbedding(name='positional_encoding')

        # Token + Position embedding
        # self.design_embedding_layer = TokenAndPositionEmbedding(
        #     self.vocab_size,
        #     self.gen_design_seq_length,
        #     self.embed_dim,
        #     mask_zero=True
        # )
        self.design_embedding = layers.Embedding(
            self.vocab_size,
            self.embed_dim,
            mask_zero=True,
            name='design_embedding'
        )
        self.design_positional_embedding = RotaryEmbedding(name='design_positional_embedding_rotary')

        # Decoder Stack
        self.normalize_first = False
        self.decoder_1 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_1', dropout=critic_dropout)
        self.decoder_2 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_2')

        # Output Prediction Head
        self.output_modeling_head = layers.Dense(self.num_objectives, name='output_modeling_head')
        self.activation = layers.Activation('linear', dtype='float32')


    def call(self, inputs, training=False, mask=None):
        design_sequences, weights = inputs

        # 1. Weights
        weight_seq = self.add_positional_encoding(weights)

        # 2. Embed design_sequences
        # design_sequences_embedded = self.design_embedding_layer(design_sequences, training=training)
        design_sequences_embedded = self.design_embedding(design_sequences)
        design_sequences_embedded = self.design_positional_embedding(design_sequences_embedded)

        # 3. Decoder Stack
        decoded_design = design_sequences_embedded
        decoded_design = self.decoder_1(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True, training=training)
        decoded_design = self.decoder_2(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True, training=training)

        # 4. Output Prediction Head
        output_prediction_logits = self.output_modeling_head(decoded_design)
        output_prediction = self.activation(output_prediction_logits)

        return output_prediction  # For training

    def add_positional_encoding(self, weights):
        # Tile conditioning weights across embedding dimension
        # weight_seq = tf.expand_dims(weights, axis=-1)
        weight_seq = tf.tile(weights, [1, 1, int(self.embed_dim / 4)])

        # For sine positional encoding
        # pos_enc = self.positional_encoding(weight_seq)
        # weight_seq = weight_seq + pos_enc
        weight_seq = self.positional_encoding(weight_seq)

        return weight_seq

    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)





















