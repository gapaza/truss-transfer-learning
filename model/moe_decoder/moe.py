import math
import tensorflow as tf
import keras
from keras import layers



@keras.saving.register_keras_serializable(package="MoeLayer", name="MoeLayer")
class MoeLayer(tf.keras.layers.Layer):

    def __init__(self, embed_dim, intermediate_dim, num_experts, num_experts_per_token, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.embed_dim = embed_dim
        self.intermediate_dim = intermediate_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_token

        # 1. Experts
        self.experts = []
        for i in range(self.num_experts):
            expert_name = 'expert_{}'.format(i)
            self.experts.append(
                MoeFF(embed_dim=self.embed_dim, hidden_dim=self.intermediate_dim, name=expert_name)
            )

        # 2. Gate
        self.gate_layer = layers.Dense(self.num_experts, activation='linear', use_bias=False)


    def call(self, inputs, **kwargs):
        x = inputs

        # 2.1. Apply gating (selects an expert for each token)
        gate_output = self.gate_layer(x)  # (2, 3, 8) -> (2, 3, 4)
        top_values, top_indices = tf.math.top_k(gate_output, k=self.num_experts_per_tok)

        # 2.2. Apply softmax to get selected expert weights
        selected_expert_weights = tf.nn.softmax(top_values, axis=-1)

        # 2.3 Fill the expert weights with zeros and scatter in selected expert weights
        expert_weights = tf.zeros_like(gate_output)  # (2, 3, 4)

        # 2.4. Get the coordinates of the top_indices
        coords = tf.where(tf.ones_like(top_values, dtype=tf.bool))
        flat_tensor2 = tf.reshape(top_indices, [-1])
        flat_tensor2 = tf.cast(flat_tensor2, dtype=tf.int64)
        tensor1_first_two_columns = tf.cast(coords[:, :2], dtype=tf.int64)
        new_coords = tf.concat([tensor1_first_two_columns, tf.reshape(flat_tensor2, (-1, 1))], axis=1)

        # 2.5. Scatter the selected expert weights
        expert_weights = tf.tensor_scatter_nd_update(expert_weights, new_coords, tf.reshape(selected_expert_weights, [-1]))
        expert_weights_flat = tf.reshape(expert_weights, (-1, self.num_experts))

        # ---------------
        # --- Experts ---
        # ---------------
        final_output = tf.zeros_like(x)
        batch_dim, seq_dim, embed_dim = x.shape
        final_output_flat = tf.reshape(final_output, (-1, self.embed_dim))
        x_input = tf.reshape(x, (-1, self.embed_dim))

        for i, expert in enumerate(self.experts):
            target_expert_tensor = tf.fill(tf.shape(top_indices), value=i)


            # Create a boolean mask where elements equal to the expert index are True
            expert_mask = tf.equal(top_indices, target_expert_tensor)
            expert_mask_reduced = tf.reduce_any(expert_mask, axis=-1, keepdims=True)
            # expert_mask_reduced = tf.flatten(expert_mask_reduced)
            expert_mask_reduced = tf.reshape(expert_mask_reduced, [-1])

            # if any of the elements in the expert_mask_reduced is True
            if tf.reduce_any(expert_mask_reduced):
                expert_input = tf.boolean_mask(x_input, expert_mask_reduced)
                expert_output = expert(expert_input)

                # Extract gating scores for the expert
                expert_weights_expanded = tf.expand_dims(expert_weights_flat[:, i], axis=-1)
                expert_weights_expanded = tf.boolean_mask(expert_weights_expanded, expert_mask_reduced)

                weighted_output = expert_output * expert_weights_expanded

                # Scatter add the expert output back to the final_output tensor
                true_indices = tf.where(expert_mask_reduced)[:, 0]

                final_output_flat = tf.tensor_scatter_nd_add(final_output_flat, tf.reshape(true_indices, (-1, 1)), weighted_output)

        final_output = tf.reshape(final_output_flat, tf.shape(inputs))

        return final_output



@keras.saving.register_keras_serializable(package="MoeFF", name="MoeFF")
class MoeFF(tf.keras.layers.Layer):

    def __init__(self, embed_dim, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.w1 = layers.Dense(self.hidden_dim, activation='swish', use_bias=False)
        self.w2 = layers.Dense(self.embed_dim, activation='linear', use_bias=False)
        self.w3 = layers.Dense(self.hidden_dim, activation='linear', use_bias=False)

    def call(self, inputs, **kwargs):
        x1 = self.w1(inputs)
        x2 = self.w3(inputs)
        x = x1 * x2
        x = self.w2(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "hidden_dim": self.hidden_dim,
            }
        )
        return config


















