import os
import re
import nltk
import struct
import pickle
import tensorflow as tf
from collections import Counter
from tensorflow.core.example import example_pb2
from tensorflow.keras.preprocessing.sequence import pad_sequences

import config
from utils import list_split


def _gen_data_from_bin_file(data_path):
    with open(data_path, 'rb') as reader:
        while True:
            len_bytes = reader.read(8)
            if not len_bytes:
                break
            str_len = struct.unpack('q', len_bytes)[0]
            example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
            yield example_pb2.Example.FromString(example_str)


def train_tokenizer():
    articles_abstracts_data = \
        [data for data in _gen_data_from_bin_file('D:/Datasets/DailymailCNN_finished_files/train.bin')]
    articles_data_all = [article.features.feature['article'].bytes_list.value[0].decode()
                         for article in articles_abstracts_data]
    abstracts_data_all = [article.features.feature['abstract'].bytes_list.value[0].decode()
                          for article in articles_abstracts_data]
    abstracts_data_all = [config.start_token + ' ' + abstract + ' ' + config.end_token
                          for abstract in abstracts_data_all]

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=config.vocab_size,
                                                      filters='',
                                                      oov_token=config.oov_token)
    tokenizer.fit_on_texts(articles_data_all)
    tokenizer.fit_on_texts(abstracts_data_all)
    pickle.dump(tokenizer, open('tokenizer.pickle', 'wb'))


def tokenize(keyword='train'):
    """
    Setup input pipeline

    Returns: articles_int, abstracts_int, guiding_objects, object_sequences,
              left_mask, left_input, right_mask, right_input, tokenizer
    """
    print('Start tokenize', keyword, 'data...')
    a_path = [s for s in os.listdir(config.articles_path) if keyword in s]
    e_path = [s for s in os.listdir(config.objects_path) if keyword in s]

    print('\tLoading articles, abstracts and objects...')
    articles_abstracts_data = []
    for f in a_path:
        articles_abstracts_data.extend([data for data in _gen_data_from_bin_file(config.articles_path + f)])
    articles_data_all = [article.features.feature['article'].bytes_list.value[0].decode()
                         for article in articles_abstracts_data]
    abstracts_data_all = [article.features.feature['abstract'].bytes_list.value[0].decode()
                          for article in articles_abstracts_data]
    abstracts_data_all = [config.start_token + ' ' + abstracts + ' ' + config.end_token
                          for abstracts in abstracts_data_all]

    objects_data_all = []
    for f in e_path:
        objects_data_all.extend([line for line in open(config.objects_path + f, 'r', encoding='UTF-8')][3::4])
    objects_data_all = [([re.split(r'\s?<sep>\s?', s)[:-1]
                          for s in re.split(r'^<s>\s?|\s?</s>\s?<s>\s?|\s?</s>$', objects.strip())[1:-1]])
                        for objects in objects_data_all]
    objects_data_all = [[obj for objs in objects for obj in objs] for objects in objects_data_all]

    articles_data_all = articles_data_all[:config.data_size]
    abstracts_data_all = abstracts_data_all[:config.data_size]
    objects_data_all = objects_data_all[:config.data_size]

    print('\tDividing sentences...')
    nltk_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences_data_all = [nltk_tokenizer.tokenize(article) for article in articles_data_all]

    print('\tTokenizing...')
    tokenizer = pickle.load(open('tokenizer.pickle', 'rb'))
    tokenizer.num_words = config.vocab_size
    tokenizer.oov_token = config.oov_token
    oov_index = tokenizer.word_index[config.oov_token]

    sentences_int_all = [tokenizer.texts_to_sequences(sentences) for sentences in sentences_data_all]
    abstracts_int_all = tokenizer.texts_to_sequences(abstracts_data_all)
    objects_int_all = [tokenizer.texts_to_sequences(objs) for objs in objects_data_all]

    print('\tGenerating partial articles...')
    articles_int, abstracts_int, guiding_objects, object_sequences, left_int, right_int, origin_idx = [], [], [], [], [], [], []
    for i in range(len(sentences_int_all)):  # for each article
        # 不足objects_max_num个非oov object的，排除掉
        non_oov_obj = []
        for obj in objects_int_all[i]:
            if obj != [oov_index] and obj != [oov_index, oov_index] and obj != [oov_index, oov_index, oov_index]:
                non_oov_obj.append(obj)
        if len(set([tuple(obj) for obj in non_oov_obj])) < config.objects_max_num:
            continue
        if [1] in non_oov_obj:
            print(non_oov_obj)
        object_words = [word for obj in non_oov_obj for word in obj]

        # 对article中的每一句话，如果摘要的所有object中有词在这句话中出现，将这句话添加到part_article中
        part_article = []
        for sentence in sentences_int_all[i]:
            for word in object_words:
                if word in sentence:
                    part_article.extend(sentence)
                    break
        if not part_article:
            continue

        # 选出guiding_object和object_sequence
        candidates = [tuple(t) for t in non_oov_obj] + [tuple([word]) for o in non_oov_obj for word in o if len(o) > 1]
        for word in candidates:
            if word == [oov_index]:
                candidates.remove(word)
        candidates = Counter(candidates)
        candidates, times = zip(*sorted(candidates.items(), key=lambda x: x[1], reverse=True))
        if times[0] == 1:  # 如果都只出现过一次，就只从原始实体里取
            candidates = non_oov_obj
        else:
            candidates = [list(obj) for obj in candidates]

        max_times = times[0]
        min_diff = 99999
        left_abstract, right_abstract = [], []
        for k, obj in enumerate(candidates):
            if times[k] < max_times:
                if not left_abstract:  # 如果前面的objects全部分割失败
                    max_times = times[k]
                else:  # 如果已经分割成功，退出循环
                    break
            if len(obj) == 1 and obj not in non_oov_obj:  # 对于不是原始实体的单个单词
                if obj[0] < 400 or nltk.pos_tag(tokenizer.index_word[obj[0]])[0][1] not in ['NN', 'NNS', 'NNP', 'NNPS']:  # 如果单词是高频词或不是名词，排除掉
                    candidates.remove(obj)
                    continue
            try:
                left_abstract, right_abstract = list_split(abstracts_int_all[i], obj)
                if abs(len(left_abstract) - len(right_abstract)) < min_diff:  # 选择分割得到的结果两边长度差异最小的
                    guiding_object = obj
                    min_diff = abs(len(left_abstract) - len(right_abstract))
            except ValueError:  # 分割失败
                continue
        if not left_abstract:  # 如果全部objects分割失败
            continue

        candidates.remove(guiding_object)
        object_sequence = [word for obj in candidates[:config.objects_max_num - 1] for word in obj]

        articles_int.append(part_article)
        abstracts_int.append(abstracts_int_all[i])
        guiding_objects.append(guiding_object)
        object_sequences.append(object_sequence)
        left_int.append(left_abstract)
        right_int.append(right_abstract)
        origin_idx.append(i)

    print('\tGenerating input data and mask...')
    left_mask = [[1] * len(left) + [0] * len(guiding_objects[i] + object_sequences[i]) for i, left in enumerate(left_int)]
    left_input = [left + guiding_objects[i] + object_sequences[i] for i, left in enumerate(left_int)]
    right_mask = [[0] * len(left + guiding_objects[i]) + [1] * len(right_int[i]) for i, left in enumerate(left_int)]
    right_input = [left + guiding_objects[i] + right_int[i] for i, left in enumerate(left_int)]

    # padding
    articles_int = pad_sequences(articles_int, maxlen=config.articles_maxlen, padding='post', truncating='post')
    left_mask = pad_sequences(left_mask, maxlen=config.left_abstracts_maxlen + 5, padding='pre', truncating='pre')
    left_input = pad_sequences(left_input, maxlen=config.left_abstracts_maxlen + 5, padding='pre', truncating='pre')
    right_mask = pad_sequences(right_mask, maxlen=config.left_abstracts_maxlen + config.right_abstracts_maxlen + 5, padding='post', truncating='post')
    right_input = pad_sequences(right_input, maxlen=config.left_abstracts_maxlen + config.right_abstracts_maxlen + 5, padding='post', truncating='post')

    # reverse left
    left_mask = left_mask[:, ::-1]
    left_input = left_input[:, ::-1]

    return articles_int, abstracts_int, guiding_objects, object_sequences, left_mask, left_input, right_mask, right_input, tokenizer


# def tokenize_val(keyword='val'):
#     print('Start tokenize', keyword, 'data...')
#     a_path = [s for s in os.listdir(config.articles_path) if keyword in s]
#     e_path = [s for s in os.listdir(config.objects_path) if keyword in s]
#
#     print('\tLoading articles, abstracts and objects...')
#     articles_abstracts_data = []
#     for f in a_path:
#         articles_abstracts_data.extend([data for data in _gen_data_from_bin_file(config.articles_path + f)])
#     articles_data_all = [article.features.feature['article'].bytes_list.value[0].decode()
#                          for article in articles_abstracts_data]
#     abstracts_data_all = [article.features.feature['abstract'].bytes_list.value[0].decode()
#                           for article in articles_abstracts_data]
#     abstracts_data_all = [re.split(r'^<s>\s?|\s?</s> <s>\s?|\s?</s>$', abstract)[1:-1]
#                           for abstract in abstracts_data_all]
#     abstracts_data_all = [[config.start_token + ' ' + abstract + ' ' + config.end_token for abstract in abstracts]
#                           for abstracts in abstracts_data_all]
#
#     objects_data_all = []
#     for f in e_path:
#         objects_data_all.extend([line for line in open(config.objects_path + f, 'r', encoding='UTF-8')][1::4])
#     objects_data_all = [re.split(r'\s?<sep>\s?', s)[:-1] for s in objects_data_all]
#
#     print('\tDividing sentences...')
#     nltk_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#     sentences_data_all = [nltk_tokenizer.tokenize(article) for article in articles_data_all]
#
#     print('\tTokenizing...')
#     tokenizer = pickle.load(open('tokenizer.pickle', 'rb'))
#     tokenizer.num_words = config.vocab_size
#     tokenizer.oov_token = config.oov_token
#     oov_index = tokenizer.word_index[config.oov_token]
#
#     sentences_int_all = [tokenizer.texts_to_sequences(sentences) for sentences in sentences_data_all]
#     abstracts_int_all = [tokenizer.texts_to_sequences(abstracts) for abstracts in abstracts_data_all]
#     objects_int_all = [tokenizer.texts_to_sequences(a) for a in objects_data_all]
#
#     # 将object和object中的单词按照频率排序
#     def high_freq(inp):
#         if not inp:
#             return []
#         inp = [tuple(t) for t in inp]
#         inp_flatten = [tuple([word]) for o in inp for word in o]
#         result = Counter(inp + inp_flatten)
#         result, _ = zip(*sorted(result.items(), key=lambda x: x[1], reverse=True))
#         result = [list(t) for t in result]
#         try:
#             result.remove([oov_index])  # remove useless object
#         except ValueError:
#             pass
#         try:
#             result.remove([tokenizer.word_index['cnn']])  # remove useless object
#         except ValueError:
#             pass
#         return result
#
#     pool = ThreadPool(6)
#     objects_int_all = pool.map(high_freq, objects_int_all)
#     pool.close()
#     pool.join()
#
#     print('\tGenerating partial articles...')
#     articles_int, abstracts_int, guiding_objects, object_sequences = [], [], [], []
#     for i in range(len(sentences_int_all)):  # for each article
#         # 不足objects_max_num个非oov object的，排除掉
#         object_sequence = []  # non-oov objects
#         for obj in objects_int_all[i]:
#             if obj[0] != oov_index:
#                 object_sequence.append(obj)
#         if len(object_sequence) < config.objects_max_num:
#             continue
#
#         # 将object_sequence中的第一个取出作为guiding_object，并将剩余的展开为一维
#         guiding_object = object_sequence[0]
#         object_sequence = [word for obj in object_sequence[1:config.objects_max_num] for word in obj]
#
#         # 对article中的每一句话，如果object_sequence中有词在这句话中出现，将这句话添加到part_article中
#         part_article = []
#         for sentence in sentences_int_all[i]:
#             for word in guiding_object + object_sequence:
#                 if word in sentence:
#                     part_article.extend(sentence)
#                     break
#         if not part_article:
#             continue
#
#         # 一篇article的多个abstract
#         abstracts_int.extend(abstracts_int_all[i])
#         articles_int.extend([part_article] * len(abstracts_int_all[i]))
#         guiding_objects.extend([guiding_object] * len(abstracts_int_all[i]))
#         object_sequences.extend([object_sequence] * len(abstracts_int_all[i]))
#
#     # padding
#     articles_int = pad_sequences(articles_int, maxlen=config.articles_maxlen, padding='post')
#
#     return articles_int, abstracts_int, guiding_objects, object_sequences, tokenizer


if __name__ == '__main__':
    train_tokenizer()
