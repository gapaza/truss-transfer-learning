import keras
from keras import layers
import tensorflow as tf
import config
import keras_nlp
import math
from keras_nlp.layers import TransformerDecoder
from keras_nlp.layers import TokenAndPositionEmbedding
from keras_nlp.layers import SinePositionEncoding
from keras_nlp.layers import RotaryEmbedding

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

critic_embed_dim = config.critic_embed_dim
critic_heads = config.critic_heads
critic_dense = config.critic_dense
critic_dropout = config.critic_dropout

# ------------------------------------
# Actor
# ------------------------------------

@keras.saving.register_keras_serializable(package="MultiTaskDecoder", name="MultiTaskDecoder")
class MultiTaskDecoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

        # Variables
        self.vocab_size = 4
        self.vocab_output_size = 2
        self.gen_design_seq_length = config.num_vars
        self.embed_dim = actor_embed_dim
        self.num_heads = actor_heads
        self.dense_dim = actor_dense

        # Conditioning Vector Positional Encoding
        # self.positional_encoding = SinePositionEncoding(name='positional_encoding')
        self.positional_encoding = RotaryEmbedding(name='positional_encoding')

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
            name="design_prediction_head"
        )
        self.activation = layers.Activation('softmax', dtype='float32')
        self.log_activation = layers.Activation('log_softmax', dtype='float32')

        # --------------------------
        # Constraint Head
        # --------------------------

        # self.constraint_pooling = tf.keras.layers.GlobalAveragePooling1D()
        # self.constraint_hidden = layers.Dense(
        #     self.embed_dim / 2.0,
        #     activation='relu',
        #     name="constraint_hidden"
        # )
        # self.constraint_prediction_head = layers.Dense(
        #     1,
        #     activation='linear',
        #     name="constraint_prediction_head"
        # )

        # --------------------------
        # Fine-tuning
        # --------------------------

        self.ft_decoder = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='ft_decoder')
        self.ft_design_prediction_head = layers.Dense(
            self.vocab_output_size,
            # kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.005),
            name="ft_design_prediction_head"
        )
        self.ft_activation = layers.Activation('softmax', dtype='float32')


    def call(self, inputs, training=False, mask=None):
        design_sequences, weights = inputs

        # 1. Weights
        # split off last weight
        # constraint_weight = weights[:, -1:]
        # constraint_weight = tf.expand_dims(constraint_weight, axis=-1)
        # constraint_weight = tf.tile(constraint_weight, [1, 1, self.embed_dim])
        # weights = weights[:, :-1]

        weight_seq = self.add_positional_encoding(weights)  # (batch, num_weights, embed_dim)

        # 2. Embed design_sequences
        design_sequences_embedded = self.design_embedding_layer(design_sequences, training=training)

        # 3. Decoder Stack
        decoded_design = design_sequences_embedded
        decoded_design = self.decoder_1(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True, training=training)
        decoded_design = self.decoder_2(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True, training=training)
        # decoded_design = self.decoder_3(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True, training=training)
        # decoded_design = self.decoder_4(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True, training=training)
        # decoded_design = self.decoder_5(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True)


        # 4. Design Prediction Head
        if fine_tune_actor is False:
            design_prediction_logits = self.design_prediction_head(decoded_design)
            design_prediction = self.activation(design_prediction_logits)
        else:
            obj_weight_seq = weight_seq[:, 0:1, :]
            decoded_design = self.ft_decoder(decoded_design, encoder_sequence=obj_weight_seq, use_causal_mask=True, training=training)
            design_prediction_logits = self.ft_design_prediction_head(decoded_design)
            design_prediction = self.ft_activation(design_prediction_logits)

        return design_prediction  # For training

    def add_positional_encoding(self, weights):

        # Tile conditioning weights across embedding dimension
        weight_seq = tf.expand_dims(weights, axis=-1)
        weight_seq = tf.tile(weight_seq, [1, 1, self.embed_dim])

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

@keras.saving.register_keras_serializable(package="MultiTaskDecoderCritic", name="MultiTaskDecoderCritic")
class MultiTaskDecoderCritic(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

        # Variables
        self.num_objectives = 1
        self.vocab_size = 4
        self.gen_design_seq_length = config.num_vars + 1
        self.embed_dim = critic_embed_dim
        self.num_heads = critic_heads
        self.dense_dim = critic_dense

        # Conditioning Vector Positional Encoding
        # self.positional_encoding = SinePositionEncoding(name='positional_encoding')
        self.positional_encoding = RotaryEmbedding(name='positional_encoding')

        # Token + Position embedding
        self.design_embedding_layer = TokenAndPositionEmbedding(
            self.vocab_size,
            self.gen_design_seq_length,
            self.embed_dim,
            mask_zero=True
        )

        # Decoder Stack
        self.normalize_first = False
        self.decoder_1 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_1', dropout=critic_dropout)
        # self.decoder_2 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_2')
        # self.decoder_3 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_3')

        # Output Prediction Head
        self.output_modeling_head = layers.Dense(self.num_objectives, name='output_modeling_head')
        self.activation = layers.Activation('linear', dtype='float32')

        # --------------------------
        # Fine-tuning
        # --------------------------

        # FT Decoder Stack
        self.ft_decoder = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='ft_decoder')

        # FT Output Prediction Head
        self.ft_output_modeling_head = layers.Dense(self.num_objectives, name='ft_output_modeling_head')
        self.ft_activation = layers.Activation('linear', dtype='float32')


    def call(self, inputs, training=False, mask=None):
        design_sequences, weights = inputs

        # 1. Weights
        # constraint_weight = weights[:, -1:]
        # constraint_weight = tf.expand_dims(constraint_weight, axis=-1)
        # constraint_weight = tf.tile(constraint_weight, [1, 1, self.embed_dim])
        # weights = weights[:, :-1]

        weight_seq = self.add_positional_encoding(weights)

        # 2. Embed design_sequences
        design_sequences_embedded = self.design_embedding_layer(design_sequences, training=training)

        # 3. Decoder Stack
        decoded_design = design_sequences_embedded
        decoded_design = self.decoder_1(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True, training=training)
        # decoded_design = self.decoder_2(decoded_design, encoder_sequence=constraint_weight, use_causal_mask=True, training=training)
        # decoded_design = self.decoder_3(decoded_design, encoder_sequence=constraint_weight, use_causal_mask=True, training=training)

        # 4. Output Prediction Head
        if fine_tune_critic is False:
            output_prediction_logits = self.output_modeling_head(decoded_design)
            output_prediction = self.activation(output_prediction_logits)
        else:
            decoded_design = self.ft_decoder(decoded_design, use_causal_mask=True)
            output_prediction_logits = self.ft_output_modeling_head(decoded_design)
            output_prediction = self.ft_activation(output_prediction_logits)

        return output_prediction  # For training

    def add_positional_encoding(self, weights):
        # Tile conditioning weights across embedding dimension
        weight_seq = tf.expand_dims(weights, axis=-1)
        weight_seq = tf.tile(weight_seq, [1, 1, self.embed_dim])

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





















