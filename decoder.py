import tensorflow as tf

from attention_layers import DecoderAttention


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, h_len):
        super(Decoder, self).__init__()
        self.h_len = h_len
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=1, mask_zero=True)
        self.rnn = tf.keras.layers.GRU(self.h_len,
                                       dropout=0.2,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

        self.fc = tf.keras.layers.Dense(vocab_size)

        self.attention = DecoderAttention(self.h_len)

    def call(self, x, hidden, enc_output, coverage_vector):
        """input one element

        :param x: (batch_sz, 1)
        :param hidden: (batch_sz, h_len)
        :param enc_output: (batch_sz, max_len)
        :param coverage_vector: (batch_sz, max_len)
        :return output: (batch_sz, vocab_size + 1)
        """
        # (batch_sz, 1) -> (batch_sz, 1, embedding_len)
        x = self.embedding(x)

        # (batch_sz, 1, h_len)
        output, state = self.rnn(x, initial_state=hidden)

        context_vector, attention_weights = self.attention(output[0],
                                                           enc_output,
                                                           coverage_vector)
        attention_weights = tf.reshape(attention_weights, [-1, attention_weights.shape[1]])
        coverage_vector = coverage_vector + attention_weights

        output = tf.concat([tf.expand_dims(context_vector, 1), output], axis=-1)

        # (batch_sz, 1, h_len) -> (batch_sz, h_len)
        output = tf.reshape(output, (-1, output.shape[2]))

        # (batch_sz, h_len) -> (batch_sz, vocab_size)
        output = tf.nn.softmax(self.fc(output))

        return output, state, attention_weights, coverage_vector
