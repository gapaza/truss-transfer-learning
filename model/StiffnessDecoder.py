import keras
from keras import layers
import tensorflow as tf
import config
import keras_nlp
import math
from keras_nlp.layers import TransformerDecoder
from keras_nlp.layers import TokenAndPositionEmbedding
from keras_nlp.layers import SinePositionEncoding


actor_embed_dim = 16
actor_heads = 16
actor_dense = 256
actor_dropout = 0.0


critic_embed_dim = 16
critic_heads = 16
critic_dense = 256
critic_dropout = 0.0


# ------------------------------------
# Actor
# ------------------------------------

@keras.saving.register_keras_serializable(package="StiffnessDecoder", name="StiffnessDecoder")
class StiffnessDecoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

        # Variables
        self.vocab_size = 4
        self.vocab_output_size = 2
        self.gen_design_seq_length = 30
        self.embed_dim = actor_embed_dim
        self.num_heads = actor_heads
        self.dense_dim = actor_dense

        # Bit flip embeddings
        self.decision_embedding_layer = TokenAndPositionEmbedding(
            self.vocab_size,
            self.gen_design_seq_length,
            self.embed_dim,
            mask_zero=True
        )

        # Design embedding
        self.design_embedding_layer = TokenAndPositionEmbedding(
            2,
            self.gen_design_seq_length,
            self.embed_dim,
            mask_zero=False
        )

        # Decoder Stack
        self.normalize_first = False
        self.decoder_1 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, dropout=actor_dropout)
        self.decoder_2 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, dropout=actor_dropout)
        # self.decoder_3 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first)

        # Decision Prediction Head
        self.decision_prediction_head = layers.Dense(
            self.vocab_output_size,
            name="decision_prediction_head"
        )
        self.activation = layers.Activation('softmax', dtype='float32')
        self.log_activation = layers.Activation('log_softmax', dtype='float32')


    def call(self, inputs, training=False, mask=None):
        decision_sequences, design_sequences = inputs

        # Bit flip embeddings
        decision_embeddings = self.decision_embedding_layer(decision_sequences, training=training)

        # Design embeddings
        design_embeddings = self.design_embedding_layer(design_sequences, training=training)

        # Decoder Stack
        decoder_1_output = self.decoder_1(decision_embeddings, encoder_sequence=design_embeddings, use_causal_mask=True, training=training)
        decoder_2_output = self.decoder_2(decoder_1_output, encoder_sequence=design_embeddings, use_causal_mask=True, training=training)
        # decoder_3_output = self.decoder_3(decoder_2_output, encoder_sequence=design_embeddings, use_causal_mask=True, training=training)

        # Decision Prediction Head
        decision_prediction_logits = self.decision_prediction_head(decoder_2_output)
        decision_prediction_probs = self.activation(decision_prediction_logits)

        return decision_prediction_probs


    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)




# ------------------------------------
# Critic
# ------------------------------------

@keras.saving.register_keras_serializable(package="StiffnessDecoderCritic", name="StiffnessDecoderCritic")
class StiffnessDecoderCritic(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

        # Variables
        self.num_objectives = 1
        self.vocab_size = 4
        self.gen_design_seq_length = 31
        self.embed_dim = critic_embed_dim
        self.num_heads = critic_heads
        self.dense_dim = critic_dense

        # Bit flip embeddings
        self.decision_embedding_layer = TokenAndPositionEmbedding(
            self.vocab_size,
            self.gen_design_seq_length,
            self.embed_dim,
            mask_zero=True
        )

        # Design embedding
        self.design_embedding_layer = TokenAndPositionEmbedding(
            2,
            self.gen_design_seq_length - 1,
            self.embed_dim,
            mask_zero=False
        )

        # Decoder Stack
        self.normalize_first = False
        self.decoder_1 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, dropout=critic_dropout)
        self.decoder_2 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, dropout=critic_dropout)

        # Decision Prediction Head
        self.decision_prediction_head = layers.Dense(
            self.num_objectives,
            name="decision_prediction_head"
        )
        self.activation = layers.Activation('linear', dtype='float32')


    def call(self, inputs, training=False, mask=None):
        decision_sequences, design_sequences = inputs

        # Bit flip embeddings
        decision_embeddings = self.decision_embedding_layer(decision_sequences, training=training)

        # Design embeddings
        design_embeddings = self.design_embedding_layer(design_sequences, training=training)

        # Decoder Stack
        decoder_1_output = self.decoder_1(decision_embeddings, encoder_sequence=design_embeddings, use_causal_mask=True, training=training)
        decoder_2_output = self.decoder_2(decoder_1_output, encoder_sequence=design_embeddings, use_causal_mask=True, training=training)

        # Decision Prediction Head
        decision_prediction_logits = self.decision_prediction_head(decoder_2_output)
        decision_value = self.activation(decision_prediction_logits)

        return decision_value



    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



