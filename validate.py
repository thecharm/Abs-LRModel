import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import config
import numpy as np
from tqdm import tqdm
from rouge import Rouge
import tensorflow as tf
from decoder import Decoder
from encoder import BiLSTM
from predict import predict
from prepare_data import make_val_dataset
from multiprocessing.dummy import Pool as ThreadPool

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

dataset_val, tokenizer, data_size = make_val_dataset('val')


def seq_to_str(seq):
    output = ' '.join(tokenizer.sequences_to_texts([np.array(seq)]))
    return output.replace(config.start_token, ' ').replace(config.end_token, ' ').replace('<s>', ' ').replace('</s>', ' ')


encoder = BiLSTM(config.vocab_size, config.embedding_len, config.encoder_feature_len)
decoder_left = Decoder(config.vocab_size, config.embedding_len, config.decoder_hidden_len)
decoder_right = Decoder(config.vocab_size, config.embedding_len, config.decoder_hidden_len)
optimizer = tf.keras.optimizers.Adam()
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder_left=decoder_left,
                                 decoder_right=decoder_right)
checkpoint.restore(tf.train.latest_checkpoint(config.checkpoint_dir))

hypothesis = []
reference = []
pbar = tqdm(total=data_size)
for article, abstract, guiding_objects, object_sequence in dataset_val:
    hypothesis.append(predict(article, guiding_objects, object_sequence, encoder, decoder_left, decoder_right, tokenizer, config.beam_top_k))
    reference.append(abstract)
    pbar.update()


pool = ThreadPool(6)
hypothesis_ = pool.map(seq_to_str, hypothesis)
reference_ = pool.map(seq_to_str, reference)
pool.close()
pool.join()

rouge = Rouge()
scores = rouge.get_scores(hypothesis_, reference_, avg=True)
print(scores)
