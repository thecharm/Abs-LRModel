import tensorflow as tf

from attention_layers import positional_encoding, EncoderLayer


class TransformerEncoder(tf.keras.models.Model):
    def __init__(self, num_layers, depth, num_heads, dff, input_vocab_size, seq_len, out_depth, rate=0.1):
        super(TransformerEncoder, self).__init__()

        self.depth = depth
        self.num_layers = num_layers
        self.seq_len = seq_len

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, depth, mask_zero=True)
        self.pos_encoding = positional_encoding(input_vocab_size, self.depth)

        self.encoder_layers = [EncoderLayer(depth, num_heads, dff, rate)
                               for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

        self.fc = tf.keras.layers.Dense(out_depth)

    def call(self, x, training=None, mask=None):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.depth, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.encoder_layers[i](x, training, mask)

        return x, self.fc(tf.reshape(x, [-1, self.seq_len * self.depth]))


class SimpleEncoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, h_len):
        super(SimpleEncoder, self).__init__()
        self.h_len = h_len
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.gru = tf.keras.layers.GRU(self.h_len,
                                       dropout=0.2,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, **kwargs):
        """

        :param x: (batch_sz, seq_len)
        :return:
        """

        x = self.embedding(x)
        output, state = self.gru(x)
        return output, state


class BiLSTM(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, h_len):
        super(BiLSTM, self).__init__()
        assert h_len % 2 == 0
        self.h_len = h_len // 2
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        forward_layer = tf.keras.layers.GRU(self.h_len,
                                            dropout=0.2,
                                            return_sequences=True,
                                            return_state=True,
                                            recurrent_initializer='glorot_uniform')
        backward_layer = tf.keras.layers.GRU(self.h_len,
                                             dropout=0.2,
                                             return_sequences=True,
                                             return_state=True,
                                             go_backwards=True,
                                             recurrent_initializer='glorot_uniform')
        self.bi_dir = tf.keras.layers.Bidirectional(layer=forward_layer,
                                                    backward_layer=backward_layer)

    def call(self, x, **kwargs):
        """

        :param x: (batch_sz, seq_len)
        :return: output: (batch_sz, seq_len, h_len), state: (batch_sz, h_len)
        """

        x = self.embedding(x)
        output, stateL, stateR = self.bi_dir(x)
        state = tf.concat([stateR, stateL], axis=-1)
        return output, state
