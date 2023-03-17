import nltk
import numpy as np
import re
import shutil
import tensorflow as tf
import os
import unicodedata
import zipfile

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def clean_up_logs(data_dir):
    checkpoint_dir = os.path.join(data_dir, "checkpoints")
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        os.makedirs(checkpoint_dir)
    return checkpoint_dir


def preprocess_sentence(sent):
    sent = "".join([c for c in unicodedata.normalize("NFD", sent) 
        if unicodedata.category(c) != "Mn"])
    sent = re.sub(r"([!.?])", r" \1", sent)
    sent = re.sub(r"[^a-zA-Z!.?]+", r" ", sent)
    sent = re.sub(r"\s+", " ", sent)
    sent = sent.lower()
    return sent


def download_and_read(url, num_sent_pairs=30000):
    local_file = url.split('/')[-1]
    if not os.path.exists(local_file):
        os.system("wget -O {:s} {:s}".format(local_file, url))
        with zipfile.ZipFile(local_file, "r") as zip_ref:
            zip_ref.extractall(".")
    local_file = os.path.join(".", "fra.txt")
    en_sents, fr_sents_in, fr_sents_out = [], [], []
    with open(local_file, "r") as fin:
        for i, line in enumerate(fin):
            en_sent, fr_sent = line.strip().split('\t')
            en_sent = [w for w in preprocess_sentence(en_sent).split()]
            fr_sent = preprocess_sentence(fr_sent)
            fr_sent_in = [w for w in ("BOS " + fr_sent).split()]
            fr_sent_out = [w for w in (fr_sent + " EOS").split()]
            en_sents.append(en_sent)
            fr_sents_in.append(fr_sent_in)
            fr_sents_out.append(fr_sent_out)
            if i >= num_sent_pairs - 1:
                break
    return en_sents, fr_sents_in, fr_sents_out


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, num_units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(num_units)
        self.W2 = tf.keras.layers.Dense(num_units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query is the decoder state at time step j
        # query.shape: (batch_size, num_units)
        # values are encoder states at every timestep i
        # values.shape: (batch_size, num_timesteps, num_units)

        # add time axis to query: (batch_size, 1, num_units)
        query_with_time_axis = tf.expand_dims(query, axis=1)
        # compute score:
        score = self.V(tf.keras.activations.tanh(
            self.W1(values) + self.W2(query_with_time_axis)))
        # compute softmax
        alignment = tf.nn.softmax(score, axis=1)
        # compute attended output
        context = tf.reduce_sum(
            tf.linalg.matmul(
                tf.linalg.matrix_transpose(alignment),
                values
            ), axis=1
        )
        context = tf.expand_dims(context, axis=1)
        return context, alignment


class LuongAttention(tf.keras.layers.Layer):
    def __init__(self, num_units):
        super(LuongAttention, self).__init__()
        self.W = tf.keras.layers.Dense(num_units)

    def call(self, query, values):
        # add time axis to query
        query_with_time_axis = tf.expand_dims(query, axis=1)
        # compute score
        score = tf.linalg.matmul(
            query_with_time_axis, self.W(values), transpose_b=True)
        # compute softmax
        alignment = tf.nn.softmax(score, axis=2)
        # compute attended output
        context = tf.matmul(alignment, values)
        return context, alignment


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, num_timesteps, 
            embedding_dim, encoder_dim, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.encoder_dim = encoder_dim
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, input_length=num_timesteps)
        self.rnn = tf.keras.layers.GRU(
            encoder_dim, return_sequences=True, return_state=True)

    def call(self, x, state):
        x = self.embedding(x)
        x, state = self.rnn(x, initial_state=state)
        return x, state

    def init_state(self, batch_size):
        return tf.zeros((batch_size, self.encoder_dim))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, num_timesteps,
            decoder_dim, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.decoder_dim = decoder_dim

        # self.attention = LuongAttention(embedding_dim)
        self.attention = BahdanauAttention(embedding_dim)

        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, input_length=num_timesteps)
        self.rnn = tf.keras.layers.GRU(
            decoder_dim, return_sequences=True, return_state=True)

        self.Wc = tf.keras.layers.Dense(decoder_dim, activation="tanh")
        self.Ws = tf.keras.layers.Dense(vocab_size)

    def call(self, x, state, encoder_out):
        x = self.embedding(x)
        context, alignment = self.attention(x, encoder_out)
        x = tf.expand_dims(
                tf.concat([
                    x, tf.squeeze(context, axis=1)
                ], axis=1), 
            axis=1)
        x, state = self.rnn(x, state)
        x = self.Wc(x)
        x = self.Ws(x)
        return x, state, alignment


def loss_fn(ytrue, ypred):
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(ytrue, 0))
    mask = tf.cast(mask, dtype=tf.int64)
    loss = scce(ytrue, ypred, sample_weight=mask)
    return loss


@tf.function
def train_step(encoder_in, decoder_in, decoder_out, encoder_state):
    with tf.GradientTape() as tape:
        encoder_out, encoder_state = encoder(encoder_in, encoder_state)
        decoder_state = encoder_state

        loss = 0
        for t in range(decoder_out.shape[1]):
            decoder_in_t = decoder_in[:, t]
            decoder_pred_t, decoder_state, _ = decoder(decoder_in_t,
                decoder_state, encoder_out)
            loss += loss_fn(decoder_out[:, t], decoder_pred_t)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss / decoder_out.shape[1]


def predict(encoder, decoder, batch_size, 
        sents_en, data_en, sents_fr_out, 
        word2idx_fr, idx2word_fr):
    random_id = np.random.choice(len(sents_en))
    print("input    : ",  " ".join(sents_en[random_id]))
    print("label    : ", " ".join(sents_fr_out[random_id]))

    encoder_in = tf.expand_dims(data_en[random_id], axis=0)
    decoder_out = tf.expand_dims(sents_fr_out[random_id], axis=0)

    encoder_state = encoder.init_state(1)
    encoder_out, encoder_state = encoder(encoder_in, encoder_state)
    decoder_state = encoder_state

    pred_sent_fr = []
    decoder_in = tf.expand_dims(
        tf.constant(word2idx_fr["BOS"]), axis=0)

    while True:
        decoder_pred, decoder_state, _ = decoder(
            decoder_in, decoder_state, encoder_out)
        decoder_pred = tf.argmax(decoder_pred, axis=-1)
        pred_word = idx2word_fr[decoder_pred.numpy()[0][0]]
        pred_sent_fr.append(pred_word)
        if pred_word == "EOS":
            break
        decoder_in = tf.squeeze(decoder_pred, axis=1)

    print("predicted: ", " ".join(pred_sent_fr))


def evaluate_bleu_score(encoder, decoder, test_dataset, 
        word2idx_fr, idx2word_fr):

    bleu_scores = []
    smooth_fn = SmoothingFunction()

    for encoder_in, decoder_in, decoder_out in test_dataset:
        encoder_state = encoder.init_state(batch_size)
        encoder_out, encoder_state = encoder(encoder_in, encoder_state)
        decoder_state = encoder_state

        ref_sent_ids = np.zeros_like(decoder_out)
        hyp_sent_ids = np.zeros_like(decoder_out)
        for t in range(decoder_out.shape[1]):
            decoder_out_t = decoder_out[:, t]
            decoder_in_t = decoder_in[:, t]
            decoder_pred_t, decoder_state, _ = decoder(
                decoder_in_t, decoder_state, encoder_out)
            decoder_pred_t = tf.argmax(decoder_pred_t, axis=-1)
            for b in range(decoder_pred_t.shape[0]):
                ref_sent_ids[b, t] = decoder_out_t.numpy()[0]
                hyp_sent_ids[b, t] = decoder_pred_t.numpy()[0][0]
        
        for b in range(ref_sent_ids.shape[0]):
            ref_sent = [idx2word_fr[i] for i in ref_sent_ids[b] if i > 0]
            hyp_sent = [idx2word_fr[i] for i in hyp_sent_ids[b] if i > 0]
            # remove trailing EOS
            ref_sent = ref_sent[0:-1]
            hyp_sent = hyp_sent[0:-1]
            bleu_score = sentence_bleu([ref_sent], hyp_sent,
                smoothing_function=smooth_fn.method1)
            bleu_scores.append(bleu_score)

    return np.mean(np.array(bleu_scores))


# NUM_SENT_PAIRS = 100
# EMBEDDING_DIM = 32
# ENCODER_DIM, DECODER_DIM = 64, 64
# BATCH_SIZE = 8
# NUM_EPOCHS = 3

NUM_SENT_PAIRS = 30000
EMBEDDING_DIM = 256
ENCODER_DIM, DECODER_DIM = 1024, 1024
BATCH_SIZE = 64
NUM_EPOCHS = 30

tf.random.set_seed(42)

data_dir = "./data"
checkpoint_dir = clean_up_logs(data_dir)

# Test code for attention classes
# batch_size = BATCH_SIZE
# num_timesteps = MAXLEN_EN
# num_units = ENCODER_DIM

# query = np.random.random(size=(batch_size, num_units))
# values = np.random.random(size=(batch_size, num_timesteps, num_units))

# # check out dimensions for Bahdanau attention
# b_attn = BahdanauAttention(num_units)
# context, alignments = b_attn(query, values)
# print("Bahdanau: context.shape:", context.shape, "alignments.shape:", alignments.shape)
# # Bahdanau: context.shape: (64, 1024) alignments.shape: (64, 8, 1)

# # check out dimensions for Luong attention
# l_attn = LuongAttention(num_units)
# context, alignments = l_attn(query, values)
# print("Luong: context.shape:", context.shape, "alignments.shape:", alignments.shape)
# # Luong: context.shape: (64, 1024) alignments.shape: (64, 8, 1)
# End test code for attention classes

# data preparation
download_url = "http://www.manythings.org/anki/fra-eng.zip"
sents_en, sents_fr_in, sents_fr_out = download_and_read(
    download_url, num_sent_pairs=NUM_SENT_PAIRS)

tokenizer_en = tf.keras.preprocessing.text.Tokenizer(
    filters="", lower=False)
tokenizer_en.fit_on_texts(sents_en)
data_en = tokenizer_en.texts_to_sequences(sents_en)
data_en = tf.keras.preprocessing.sequence.pad_sequences(data_en, padding="post")

tokenizer_fr = tf.keras.preprocessing.text.Tokenizer(
    filters="", lower=False)
tokenizer_fr.fit_on_texts(sents_fr_in)
tokenizer_fr.fit_on_texts(sents_fr_out)
data_fr_in = tokenizer_fr.texts_to_sequences(sents_fr_in)
data_fr_in = tf.keras.preprocessing.sequence.pad_sequences(data_fr_in, padding="post")
data_fr_out = tokenizer_fr.texts_to_sequences(sents_fr_out)
data_fr_out = tf.keras.preprocessing.sequence.pad_sequences(data_fr_out, padding="post")

maxlen_en = data_en.shape[1]
maxlen_fr = data_fr_out.shape[1]
print("seqlen (en): {:d}, (fr): {:d}".format(maxlen_en, maxlen_fr))

batch_size = BATCH_SIZE
dataset = tf.data.Dataset.from_tensor_slices((data_en, data_fr_in, data_fr_out))
dataset = dataset.shuffle(10000)
test_size = NUM_SENT_PAIRS // 4
test_dataset = dataset.take(test_size).batch(batch_size, drop_remainder=True)
train_dataset = dataset.skip(test_size).batch(batch_size, drop_remainder=True)

vocab_size_en = len(tokenizer_en.word_index)
vocab_size_fr = len(tokenizer_fr.word_index)
word2idx_en = tokenizer_en.word_index
idx2word_en = {v:k for k, v in word2idx_en.items()}
word2idx_fr = tokenizer_fr.word_index
idx2word_fr = {v:k for k, v in word2idx_fr.items()}
print("vocab size (en): {:d}, vocab size (fr): {:d}".format(
    vocab_size_en, vocab_size_fr))
# vocab size (en): 57, vocab size (fr): 123

# check encoder/decoder dimensions
embedding_dim = EMBEDDING_DIM
encoder_dim, decoder_dim = ENCODER_DIM, DECODER_DIM

encoder = Encoder(vocab_size_en+1, embedding_dim, maxlen_en, encoder_dim)
decoder = Decoder(vocab_size_fr+1, embedding_dim, maxlen_fr, decoder_dim)

# # Test code for encoder and decoder with attention
# for encoder_in, decoder_in, decoder_out in train_dataset:
#     print("inputs:", encoder_in.shape, decoder_in.shape, decoder_out.shape)
#     encoder_state = encoder.init_state(batch_size)
#     encoder_out, encoder_state = encoder(encoder_in, encoder_state)
#     decoder_state = encoder_state
#     decoder_pred = []
#     for t in range(decoder_out.shape[1]):
#         decoder_in_t = decoder_in[:, t]
#         decoder_pred_t, decoder_state, _ = decoder(decoder_in_t,
#             decoder_state, encoder_out)
#         decoder_pred.append(decoder_pred_t.numpy())
#     decoder_pred = tf.squeeze(np.array(decoder_pred), axis=2)
#     break
# print("encoder input          :", encoder_in.shape)
# print("encoder output         :", encoder_out.shape, "state:", encoder_state.shape)
# print("decoder output (logits):", decoder_pred.shape, "state:", decoder_state.shape)
# print("decoder output (labels):", decoder_out.shape)

# Bahdanau:
# encoder input          : (64, 8)
# encoder output         : (64, 8, 1024) state: (64, 1024)
# decoder output (logits): (8, 64, 7658) state: (64, 1024)
# decoder output (labels): (64, 16)
# 
# Luong:
# encoder input          : (64, 8)
# encoder output         : (64, 8, 1024) state: (64, 1024)
# decoder output (logits): (8, 64, 7658) state: (64, 1024)
# decoder output (labels): (64, 16)


optimizer = tf.keras.optimizers.Adam()
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

num_epochs = NUM_EPOCHS
eval_scores = []

for e in range(num_epochs):
    encoder_state = encoder.init_state(batch_size)

    for batch, data in enumerate(train_dataset):
        encoder_in, decoder_in, decoder_out = data
        # print(encoder_in.shape, decoder_in.shape, decoder_out.shape)
        loss = train_step(
            encoder_in, decoder_in, decoder_out, encoder_state)
    
    print("Epoch: {}, Loss: {:.4f}".format(e + 1, loss.numpy()))

    if e % 10 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)
    
    predict(encoder, decoder, batch_size, sents_en, data_en,
        sents_fr_out, word2idx_fr, idx2word_fr)

    eval_score = evaluate_bleu_score(encoder, decoder, test_dataset, word2idx_fr, idx2word_fr)
    print("Eval Score (BLEU): {:.3e}".format(eval_score))
    # eval_scores.append(eval_score)

checkpoint.save(file_prefix=checkpoint_prefix)