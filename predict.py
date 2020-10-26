import config
import tensorflow as tf


def forward(inputs, decoder, enc_output, hidden, coverage_vector):
    for i in range(len(inputs)):
        predictions, hidden, _, coverage_vector = \
            decoder(tf.expand_dims([inputs[i]], 1), hidden, enc_output, coverage_vector)
    predictions = tf.math.log(predictions)
    predictions = tf.reshape(predictions, [predictions.shape[-1]])  # (1, vocab_sz) -> (vocab_sz,)
    return predictions


def beam_search(inputs, decoder, enc_output, hidden, coverage_vector, end_token_id, top_k: int, maxlen: int):
    """

    :param inputs: (len, ), 解码前对decoder的输入，在left part中是objects，在right part中是left part的输出
    :param decoder:
    :param enc_output:
    :param hidden:
    :param coverage_vector:
    :param end_token_id: 标记解码结束的token的id
    :param top_k:
    :param maxlen: 解码的最大长度
    :return: 最优的输出
    """
    predictions = forward(inputs, decoder, enc_output, hidden, coverage_vector)

    outputs = [[t] for t in tf.argsort(predictions)[-top_k:]]  # 候选输出
    scores = [predictions[t] for t in tf.argsort(predictions)[-top_k:]]  # 候选得分

    for i in range(maxlen - 1):
        _outputs = []
        _scores = []
        _att_wights = []
        for j in range(top_k):  # 遍历现有的候选输出
            if outputs[j][-1] == end_token_id:  # 已经到结尾结果的不再搜索
                _outputs.append(outputs[j])
                _scores.append(scores[j])
                continue

            predictions = forward(tf.concat([inputs, outputs[j]], axis=0), decoder, enc_output, hidden, coverage_vector)

            # 将生成的结果中前 top_k 好的添加到候选
            _outputs.extend([outputs[j] + [t] for t in tf.argsort(predictions)[-top_k:]])
            # 加上本次生成的得分，并按照序列长度标准化(添加前的序列长度为 i+1)
            _scores.extend(
                [scores[j] + predictions[t] for t in tf.argsort(predictions)[-top_k:]])

        # 在得到的至多 top_k * top_k 个结果中选择最好的 top_k 个
        _arg_top_k = tf.argsort(_scores)[-top_k:]
        outputs = [_outputs[t] for t in _arg_top_k]
        scores = [_scores[t] for t in _arg_top_k]

    # standardization
    scores = [scores[i] / len(o) for i, o in enumerate(outputs)]

    return outputs[tf.argmax(scores)]


def predict(article, guiding_object, object_sequence, encoder, decoder_left, decoder_right, tokenizer, beam_size):
    """

    Parameters:
        article: (articles_maxlen, )
        guiding_object:
        object_sequence:
        encoder:
        decoder_left:
        decoder_right:
        tokenizer:
        beam_size:

    Returns:
    """
    article = tf.expand_dims(article, 0)
    enc_output, enc_hidden = encoder(article)

    dec_left_hidden = enc_hidden
    dec_right_hidden = enc_hidden

    coverage_vector_left = tf.zeros([1, config.articles_maxlen])
    coverage_vector_right = tf.zeros([1, config.articles_maxlen])

    # left part
    left_input = tf.reverse(tf.concat([guiding_object, object_sequence], 0), [0])
    result_left = beam_search(left_input, decoder_left, enc_output,
                              dec_left_hidden, coverage_vector_left,
                              tokenizer.word_index[config.start_token], beam_size, config.left_abstracts_maxlen)

    result_left = tf.reverse(result_left, [0])
    result_left = tf.concat([result_left, guiding_object], 0)

    # right part
    result_right = beam_search(result_left, decoder_right, enc_output,
                               dec_right_hidden, coverage_vector_right,
                               tokenizer.word_index[config.end_token], beam_size, config.right_abstracts_maxlen)

    return tf.concat([result_left, result_right], 0).numpy()
