start_token = '<start>'
end_token = '<end>'
mid_token = '<mid>'
oov_token = 'UNK'
src_path = 'data/Gigaword/org_data/{}.src.txt'
tgt_path = 'data/Gigaword/org_data/{}.tgt.txt'
objects_path = 'data/Gigaword/object/{}.src.txt'
checkpoint_dir = 'Gigaword_training_checkpoints'

epochs = 10000

data_size = 500

# hyperparameters
vocab_size = 30000
articles_maxlen = 50
left_abstracts_maxlen = 25
right_abstracts_maxlen = 25
objects_max_num = 3

batch_size = 4
embedding_len = 128

# these two should be equal
encoder_feature_len = 256
decoder_hidden_len = 256

# Transformer
encoder_num_layers = 6
encoder_num_heads = 8
encoder_dff = 2048

beam_top_k = 5
coverage_lambda = 0
