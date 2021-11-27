# import tenserflow, layers and keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, attention=None, dense_proj=None, layernorm_1=None, layernorm_2=None, supports_masking=None, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        if attention is None:
            self.attention = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim
            )
        else:
            self.attention = attention
        if dense_proj is None:
            self.dense_proj = keras.Sequential(
                [layers.Dense(dense_dim, activation="relu"),
                 layers.Dense(embed_dim), ]
            )
        else:
            self.dense_proj = dense_proj
        if layernorm_1 is None:
            self.layernorm_1 = layers.LayerNormalization()
        else:
            self.layernorm_1 = layernorm_1
        if layernorm_2 is None:
            self.layernorm_2 = layers.LayerNormalization()
        else:
            self.layernorm_2 = layernorm_2
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(
                mask[:, tf.newaxis, tf.newaxis, :], dtype="int32")
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super(TransformerEncoder, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'dense_dim': self.dense_dim,
            'num_heads': self.num_heads,
            'attention': self.attention,
            'dense_proj': self.dense_proj,
            'layernorm_1': self.layernorm_1,
            'layernorm_2': self.layernorm_2,
            'supports_masking': self.supports_masking

        })
        return config


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, token_embeddings=None, position_embeddings=None, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        if token_embeddings is None:
            self.token_embeddings = layers.Embedding(
                input_dim=vocab_size, output_dim=embed_dim
            )
        else:
            self.token_embeddings = token_embeddings
        if position_embeddings is None:
            self.position_embeddings = layers.Embedding(
                input_dim=sequence_length, output_dim=embed_dim
            )
        else:
            self.position_embeddings = position_embeddings
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        # if inputs is of type RaggedTensor then convert to Tensor
        if isinstance(inputs, tf.RaggedTensor):
            inputs = tf.cast(inputs.to_tensor(), tf.int32)
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({
            'token_embeddings': self.token_embeddings,
            'position_embeddings': self.position_embeddings,
            'sequence_length': self.sequence_length,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim
        })
        return config


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, attention_1=None, attention_2=None, dense_proj=None, layernorm_1=None, layernorm_2=None, layernorm_3=None, supports_masking=None, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        if attention_1 is None:
            self.attention_1 = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim
            )
        else:
            self.attention_1 = attention_1
        if attention_2 is None:
            self.attention_2 = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim
            )
        else:
            self.attention_2 = attention_2
        if dense_proj is None:
            self.dense_proj = keras.Sequential(
                [layers.Dense(latent_dim, activation="relu"),
                 layers.Dense(embed_dim), ]
            )
        else:
            self.dense_proj = dense_proj
        if layernorm_1 is None:
            self.layernorm_1 = layers.LayerNormalization()
        else:
            self.layernorm_1 = layernorm_1
        if layernorm_2 is None:
            self.layernorm_2 = layers.LayerNormalization()
        else:
            self.layernorm_2 = layernorm_2
        if layernorm_3 is None:
            self.layernorm_3 = layers.LayerNormalization()
        else:
            self.layernorm_3 = layernorm_3
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1),
             tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)

    def get_config(self):
        config = super(TransformerDecoder, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'latent_dim': self.latent_dim,
            'num_heads': self.num_heads,
            'attention_1': self.attention_1,
            'attention_2': self.attention_2,
            'dense_proj': self.dense_proj,
            'layernorm_1': self.layernorm_1,
            'layernorm_2': self.layernorm_2,
            'layernorm_3': self.layernorm_3,
            'supports_masking': self.supports_masking,

        })
        return config
