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

critic_embed_dim = config.critic_embed_dim
critic_heads = config.critic_heads
critic_dense = config.critic_dense
critic_dropout = config.critic_dropout

# ------------------------------------
# Actor
# ------------------------------------

@keras.saving.register_keras_serializable(package="MultiTaskDecoderV2", name="MultiTaskDecoderV2")
class MultiTaskDecoderV2(tf.keras.Model):
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
        self.decoder_3 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_3', dropout=actor_dropout)
        # self.decoder_4 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_4')
        # self.decoder_5 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_5')

        # FT Decoder Stack
        self.ft_decoder = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='ft_decoder')


        # Design Prediction Head
        self.design_prediction_head = layers.Dense(
            self.vocab_output_size,
            # kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.005),
            name="design_prediction_head"
        )
        self.activation = layers.Activation('softmax', dtype='float32')
        self.log_activation = layers.Activation('log_softmax', dtype='float32')

        # FT Design Prediction Head
        self.ft_design_prediction_head = layers.Dense(
            self.vocab_output_size,
            # kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.005),
            name="ft_design_prediction_head"
        )
        self.ft_activation = layers.Activation('softmax', dtype='float32')


    def call2(self, inputs, training=False, mask=None):
        design_sequences, weights, weights_mask = inputs

        # 1. Weights shape (batch, 2)
        # split weights into two separate conditioning values
        weight_seq_mask = weights_mask[:, 0:1]
        weight_seq = weights[:, 0:1]
        weight_seq = tf.expand_dims(weights, axis=-1)
        weight_seq = tf.tile(weight_seq, [1, 1, self.embed_dim])

        stiff_seq_mask = weights_mask[:, 1:2]
        stiff_seq = weights[:, 1:2]
        stiff_seq = tf.expand_dims(stiff_seq, axis=-1)
        stiff_seq = tf.tile(stiff_seq, [1, 1, self.embed_dim])

        # weight_seq = self.add_positional_encoding(weights)  # (batch, num_weights, embed_dim)
        # weight_seq = weight_seq[:, 0:1, :]

        # 2. Embed design_sequences
        design_sequences_embedded = self.design_embedding_layer(design_sequences, training=training)

        # 3. Decoder Stack
        decoded_design = design_sequences_embedded
        decoded_design = self.decoder_1(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True, encoder_padding_mask=weights_mask,  training=training)
        decoded_design = self.decoder_2(decoded_design, encoder_sequence=stiff_seq, use_causal_mask=True, encoder_padding_mask=stiff_seq_mask, training=training)
        decoded_design = self.decoder_3(decoded_design, encoder_sequence=stiff_seq, use_causal_mask=True, encoder_padding_mask=stiff_seq_mask, training=training)
        # decoded_design = self.decoder_4(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True)
        # decoded_design = self.decoder_5(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True)

        # 4. Design Prediction Head
        if fine_tune_actor is False:
            design_prediction_logits = self.design_prediction_head(decoded_design)
            design_prediction = self.activation(design_prediction_logits)
        else:
            decoded_design = self.ft_decoder(decoded_design, use_causal_mask=True)
            design_prediction_logits = self.ft_design_prediction_head(decoded_design)
            design_prediction = self.ft_activation(design_prediction_logits)

        return design_prediction  # For training


    def call(self, inputs, training=False, mask=None):
        design_sequences, weights = inputs

        # 1. Weights
        # weight_seq = self.add_positional_encoding(weights)  # (batch, num_weights, embed_dim)
        # weight_seq = weight_seq[:, 0:1, :]

        weight_seq = weights[:, 0:1]
        weight_seq = tf.expand_dims(weights, axis=-1)
        weight_seq = tf.tile(weight_seq, [1, 1, self.embed_dim])

        stiff_seq = weights[:, 1:2]
        stiff_seq = tf.expand_dims(stiff_seq, axis=-1)
        stiff_seq = tf.tile(stiff_seq, [1, 1, self.embed_dim])







        # 2. Embed design_sequences
        design_sequences_embedded = self.design_embedding_layer(design_sequences, training=training)

        # 3. Decoder Stack
        decoded_design = design_sequences_embedded
        decoded_design = self.decoder_1(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True, training=training)
        decoded_design = self.decoder_2(decoded_design, encoder_sequence=stiff_seq, use_causal_mask=True, training=training)
        decoded_design = self.decoder_3(decoded_design, encoder_sequence=stiff_seq, use_causal_mask=True, training=training)
        # decoded_design = self.decoder_4(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True)
        # decoded_design = self.decoder_5(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True)

        # 4. Design Prediction Head
        if fine_tune_actor is False:
            design_prediction_logits = self.design_prediction_head(decoded_design)
            design_prediction = self.activation(design_prediction_logits)
        else:
            decoded_design = self.ft_decoder(decoded_design, use_causal_mask=True)
            design_prediction_logits = self.ft_design_prediction_head(decoded_design)
            design_prediction = self.ft_activation(design_prediction_logits)

        return design_prediction  # For training

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













# ------------------------------------
# Critic
# ------------------------------------

@keras.saving.register_keras_serializable(package="MultiTaskDecoderV2Critic", name="MultiTaskDecoderV2Critic")
class MultiTaskDecoderV2Critic(tf.keras.Model):
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
        self.positional_encoding = SinePositionEncoding(name='positional_encoding')
        # self.conditioning_values = 4
        # self.positional_encoding = self.add_weight(
        #     # for learning across embedding dim use (self.conditioning_values, self.embed_dim)
        #     shape=(self.conditioning_values, 1),
        #     initializer='zeros',  # Start with zeros
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
        self.decoder_1 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_1', dropout=critic_dropout)
        # self.decoder_2 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_2')

        # FT Decoder Stack
        self.ft_decoder = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='ft_decoder')

        # Output Prediction Head
        self.output_modeling_head = layers.Dense(self.num_objectives, name='output_modeling_head')
        self.activation = layers.Activation('linear', dtype='float32')

        # FT Output Prediction Head
        self.ft_output_modeling_head = layers.Dense(self.num_objectives, name='ft_output_modeling_head')
        self.ft_activation = layers.Activation('linear', dtype='float32')


    def call2(self, inputs, training=False, mask=None):
        design_sequences, weights, weights_mask = inputs

        # 1. Weights
        weight_seq = self.add_positional_encoding(weights)
        # weight_seq = weight_seq[:, 0:1, :]  # Only use first weight

        # 2. Embed design_sequences
        design_sequences_embedded = self.design_embedding_layer(design_sequences, training=training)

        # 3. Decoder Stack
        decoded_design = design_sequences_embedded
        decoded_design = self.decoder_1(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True, encoder_padding_mask=weights_mask, training=training)
        # decoded_design = self.decoder_2(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True, training=training)

        # 4. Output Prediction Head
        if fine_tune_critic is False:
            output_prediction_logits = self.output_modeling_head(decoded_design)
            output_prediction = self.activation(output_prediction_logits)
        else:
            decoded_design = self.ft_decoder(decoded_design, use_causal_mask=True)
            output_prediction_logits = self.ft_output_modeling_head(decoded_design)
            output_prediction = self.ft_activation(output_prediction_logits)

        return output_prediction


    def call(self, inputs, training=False, mask=None):
        design_sequences, weights = inputs

        # 1. Weights
        weight_seq = self.add_positional_encoding(weights)
        # weight_seq = weight_seq[:, 0:1, :]  # Only use first weight

        # 2. Embed design_sequences
        design_sequences_embedded = self.design_embedding_layer(design_sequences, training=training)

        # 3. Decoder Stack
        decoded_design = design_sequences_embedded
        decoded_design = self.decoder_1(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True, training=training)
        # decoded_design = self.decoder_2(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True, training=training)

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





















