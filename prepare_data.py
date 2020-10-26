import tensorflow as tf

import config
from tokenizer import tokenize
from utils import list_gather


def make_val_dataset(keyword='val'):
    articles, abstracts, guiding_obj, obj_seq, _, _, _, _, tokenizer = tokenize(keyword)
    size = len(articles)
    idx = tf.random.shuffle(tf.range(size))
    articles = tf.gather(articles, idx)
    abstracts = list_gather(abstracts, idx)
    guiding_objects = list_gather(guiding_obj, idx)
    object_sequences = list_gather(obj_seq, idx)

    def generator():
        for data in zip(articles, abstracts, guiding_objects, object_sequences):
            yield data

    dataset = tf.data.Dataset.from_generator(generator, (tf.int32, tf.int32, tf.int32, tf.int32))

    return dataset, tokenizer, size


def make_train_dataset(train='train', val='val'):
    articles_train, _, _, _, l_mask_train, l_inp_train, r_mask_train, r_inp_train, tokenizer = tokenize(train)

    train_sz = len(articles_train)
    train_idx = tf.random.shuffle(tf.range(train_sz))

    train_articles = list_gather(articles_train, train_idx)  # to avoid OOM
    train_left_mask = tf.gather(l_mask_train, train_idx)
    train_left_input = tf.gather(l_inp_train, train_idx)
    train_right_mask = tf.gather(r_mask_train, train_idx)
    train_right_input = tf.gather(r_inp_train, train_idx)

    dataset_train = tf.data.Dataset.from_tensor_slices(
        (train_articles, train_left_mask, train_left_input, train_right_mask, train_right_input))
    dataset_train = dataset_train.repeat().batch(config.batch_size, drop_remainder=True)

    dataset_val, _, val_size = make_val_dataset(val)
    print('train data: {}\nval data: {}\n'.format(len(train_articles), val_size))

    return dataset_train, dataset_val, tokenizer, len(train_articles) // config.batch_size
