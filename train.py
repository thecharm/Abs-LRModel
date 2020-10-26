import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
import time
import logging
import numpy as np
import tensorflow as tf
import config
from tqdm import trange
from rouge import Rouge
from predict import predict
from decoder import Decoder
from encoder import BiLSTM
from prepare_data import make_train_dataset
checkpoint_prefix = os.path.join(config.checkpoint_dir, "ckpt")

assert tf.__version__.startswith('2.')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

dataset_train, dataset_val, tokenizer, steps_per_epoch = make_train_dataset('train', 'val')

mirrored_strategy = tf.distribute.MirroredStrategy()
print('Number of devices for distributed training: {}'.format(mirrored_strategy.num_replicas_in_sync))
with mirrored_strategy.scope():
    # BiLSTM encoder
    encoder = BiLSTM(config.vocab_size, config.embedding_len, config.encoder_feature_len)
    decoder_left = Decoder(config.vocab_size, config.embedding_len, config.decoder_hidden_len)
    decoder_right = Decoder(config.vocab_size, config.embedding_len, config.decoder_hidden_len)
    optimizer = tf.keras.optimizers.Adam()

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    def loss_function(target, pred):
        mask = tf.math.logical_not(tf.math.equal(target, 0))
        loss_ = loss_object(target, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.nn.compute_average_loss(loss_, global_batch_size=config.batch_size)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder_left=decoder_left,
                                     decoder_right=decoder_right)

dist_dataset_train = mirrored_strategy.experimental_distribute_dataset(dataset_train)
dataset_val = dataset_val.shuffle(1000).repeat()

file_path = './log/'
# 创建一个logger
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)

# 创建一个handler，
timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
fh = logging.FileHandler(file_path + 'cnndm_log_' + timestamp + '.txt')
fh.setLevel(logging.DEBUG)

# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# 定义handler的输出格式
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# 给logger添加handler
logger.addHandler(fh)
logger.addHandler(ch)

with mirrored_strategy.scope():
    @tf.function
    def train_step(articles, left_input, left_mask, right_input, right_mask):
        """

        Parameters:
            articles: (batch_sz, articles_maxlen)
            left_input: (batch_sz, left_abstracts_maxlen + objects_max_len), post padding, reversed
            left_mask: (batch_sz, left_abstracts_maxlen + objects_max_len)
            right_input: (batch_sz, abstracts_max_len), start with start_token, post padding
            right_mask: (batch_sz, abstracts_max_len)

        Returns: batch_loss
        """
        left_loss = 0
        right_loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(articles)

            # left part
            dec_hidden_l = enc_hidden
            coverage_vec_l = tf.zeros([articles.shape[0], articles.shape[1]])

            # predict with teaching-force
            for i in range(left_input.shape[1] - 1):
                predictions, dec_hidden_l, attention_wights_left, coverage_vec_l = \
                    decoder_left(tf.expand_dims(left_input[:, i], axis=1), dec_hidden_l, enc_output, coverage_vec_l)
                left_loss += loss_function(tf.multiply(left_input[:, i + 1], left_mask[:, i + 1]), predictions)

            # right part
            dec_hidden_r = enc_hidden
            coverage_vec_r = tf.zeros([articles.shape[0], articles.shape[1]])

            # predict with teaching-force
            for i in range(right_input.shape[1] - 1):
                predictions, dec_hidden_r, attention_wights_right, coverage_vec_r = \
                    decoder_right(tf.expand_dims(right_input[:, i], axis=1), dec_hidden_r, enc_output, coverage_vec_r)
                right_loss += loss_function(tf.multiply(right_input[:, i + 1], right_mask[:, i + 1]), predictions)

            regular_loss = tf.nn.scale_regularization_loss(
                              config.coverage_lambda * tf.reduce_sum(
                                  tf.minimum(attention_wights_left, coverage_vec_l)))
            regular_loss += tf.nn.scale_regularization_loss(
                               config.coverage_lambda * tf.reduce_sum(
                                   tf.minimum(attention_wights_right, coverage_vec_r)))
            loss = left_loss + right_loss + regular_loss

        # 按长度缩放
        batch_loss = loss / int(right_input.shape[1] - 1)

        variables = encoder.trainable_variables + decoder_left.trainable_variables + decoder_right.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss, left_loss, right_loss, regular_loss

    def distributed_train_step(articles, left_input, left_mask, right_input, right_mask):
        batch_loss_pr, left_loss_pr, right_loss_pr, regular_loss_pr = mirrored_strategy.run(
            train_step, args=(articles, left_input, left_mask, right_input, right_mask))
        batch_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, batch_loss_pr, axis=None)
        left_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, left_loss_pr, axis=None)
        right_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, right_loss_pr, axis=None)
        regular_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, regular_loss_pr, axis=None)
        return batch_loss, left_loss, right_loss, regular_loss


def val(data_size):
    def seq_to_str(seq):
        output = ' '.join(tokenizer.sequences_to_texts([np.array(seq)]))
        return output.replace(config.start_token, ' ').replace(config.end_token, ' ').replace('<s>', ' ').replace('</s>', ' ')

    logger.info('Start validating')
    hypothesis = []
    reference = []
    val_iter = iter(dataset_val)
    for _ in trange(data_size):
        article, abstract, guiding_objects, object_sequence = next(val_iter)
        hypothesis.append(seq_to_str(
            predict(article, guiding_objects, object_sequence, encoder, decoder_left, decoder_right, tokenizer, config.beam_top_k)))
        reference.append(seq_to_str(abstract))
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference, avg=True)
    logger.info(str(scores) + '\n')


def train():
    logger.info('Start training')
    with mirrored_strategy.scope():
        for epoch in range(0, config.epochs):
            epoch_start = time.time()
            batch100_start = time.time()

            total_loss = 0.0
            num_batches = 0
            train_iter = iter(dist_dataset_train)

            for _ in range(steps_per_epoch):
                articles, left_mask, left_inp, right_mask, right_inp = next(train_iter)
                batch_loss, left_loss, right_loss, regular_loss = distributed_train_step(
                    articles, left_inp, left_mask, right_inp, right_mask)
                total_loss += batch_loss
                num_batches += 1

                if num_batches % 100 == 0:
                    template = 'Epoch {} Batch {} Loss(scaled) {:.6f} Left part {:.6f} Right part {:.6f} Regularization {:.6f}  {:.2f} sec'
                    logger.info(template.format(epoch + 1, num_batches, batch_loss, left_loss, right_loss, regular_loss,
                                                time.time() - batch100_start))
                    batch100_start = time.time()

            pred_test()
            logger.info('Epoch {} Loss {:.6f}'.format(epoch + 1, total_loss / steps_per_epoch))
            logger.info('Time taken for 1 epoch {:.2f} sec\n'.format(time.time() - epoch_start))
            checkpoint.save(file_prefix=checkpoint_prefix)
            val(100)


def pred_test(load_ckpt=False):
    def seq_to_str(seq):
        return ' '.join(tokenizer.sequences_to_texts([np.array(seq)]))

    if load_ckpt:
        checkpoint.restore(tf.train.latest_checkpoint(config.checkpoint_dir))
    for (article, abstract, guiding_objects, object_sequence) in dataset_val.take(5):
        pred_g = predict(article, guiding_objects, object_sequence, encoder, decoder_left, decoder_right, tokenizer, 1)
        pred_b = predict(article, guiding_objects, object_sequence, encoder, decoder_left, decoder_right, tokenizer, 5)
        logger.info('Target: ' + seq_to_str(abstract))
        logger.info('Guiding object: ' + seq_to_str(guiding_objects))
        logger.info('Object sequence: ' + seq_to_str(object_sequence))
        logger.info('Greedy search: ' + seq_to_str(pred_g))
        logger.info('Beam search: ' + seq_to_str(pred_b) + '\n')


if __name__ == '__main__':
    train()
