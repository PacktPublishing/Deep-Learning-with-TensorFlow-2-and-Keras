import gensim
import logging
import numpy as np
import os
import shutil
import tensorflow as tf

from scipy.sparse import csr_matrix
# from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

DATA_DIR = "./data"
UCI_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00371/NIPS_1987-2015.csv"

NUM_WALKS_PER_VERTEX = 32
MAX_PATH_LENGTH = 40
RESTART_PROB = 0.15

RANDOM_WALKS_FILE = os.path.join(DATA_DIR, "random-walks.txt")
W2V_MODEL_FILE = os.path.join(DATA_DIR, "w2v-neurips-papers.model")

def download_and_read(url):
    local_file = url.split('/')[-1]
    p = tf.keras.utils.get_file(local_file, url, cache_dir=".")
    row_ids, col_ids, data = [], [], []
    rid = 0
    f = open(p, "r")
    for line in f:
        line = line.strip()
        if line.startswith("\"\","):
            # header
            continue
        if rid % 100 == 0:
            print("{:d} rows read".format(rid))
        # compute non-zero elements for current row
        counts = np.array([int(x) for x in line.split(',')[1:]])
        nz_col_ids = np.nonzero(counts)[0]
        nz_data = counts[nz_col_ids]
        nz_row_ids = np.repeat(rid, len(nz_col_ids))
        rid += 1
        # add data to big lists
        row_ids.extend(nz_row_ids.tolist())
        col_ids.extend(nz_col_ids.tolist())
        data.extend(nz_data.tolist())
    print("{:d} rows read, COMPLETE".format(rid))
    f.close()
    TD = csr_matrix((
        np.array(data), (
            np.array(row_ids), np.array(col_ids)
            )
        ),
        shape=(rid, counts.shape[0]))
    return TD


def construct_random_walks(E, n, alpha, l, ofile):
    """ NOTE: takes a long time to do, consider using some parallelization
        for larger problems.
    """
    if os.path.exists(ofile):
        print("random walks generated already, skipping")
        return
    f = open(ofile, "w")
    for i in range(E.shape[0]):  # for each vertex
        if i % 100 == 0:
            print("{:d} random walks generated from {:d} starting vertices"
                .format(n * i, i))
        if i <= 3273:
            continue
        for j in range(n):       # construct n random walks
            curr = i
            walk = [curr]
            target_nodes = np.nonzero(E[curr])[1]
            for k in range(l):   # each of max length l, restart prob alpha
                # should we restart?
                if np.random.random() < alpha and len(walk) > 5:
                    break
                # choose one outgoing edge and append to walk
                try:
                    curr = np.random.choice(target_nodes)
                    walk.append(curr)
                    target_nodes = np.nonzero(E[curr])[1]
                except ValueError:
                    continue
            f.write("{:s}\n".format(" ".join([str(x) for x in walk])))

    print("{:d} random walks generated from {:d} starting vertices, COMPLETE"
        .format(n * i, i))
    f.close()


class Documents(object):
    def __init__(self, input_file):
        self.input_file = input_file

    def __iter__(self):
        with open(self.input_file, "r") as f:
            for i, line in enumerate(f):
                if i % 1000 == 0:
                    if i % 1000 == 0:
                        logging.info("{:d} random walks extracted".format(i))
                yield line.strip().split()


def train_word2vec_model(random_walks_file, model_file):
    if os.path.exists(model_file):
        print("Model file {:s} already present, skipping training"
            .format(model_file))
        return
    docs = Documents(random_walks_file)
    model = gensim.models.Word2Vec(
        docs,
        size=128,    # size of embedding vector
        window=10,   # window size
        sg=1,        # skip-gram model
        min_count=2,
        workers=4
    )
    model.train(
        docs, 
        total_examples=model.corpus_count,
        epochs=50)
    model.save(model_file)


# def evaluate_model_file(td_matrix, model_file, source_node_ids):
#     model = gensim.models.Word2Vec.load(model_file).wv
#     for source_node_id in source_node_ids:
#         most_similar = model.most_similar(str(source_node_id))
#         scores = [x[1] for x in most_similar]
#         target_ids = [x[0] for x in most_similar]
#         X = np.repeat(td_matrix[source_node_id].todense(), 10, axis=0)
#         Y = td_matrix[target_ids].todense()
#         cosims = [cosine_similarity(X[i], Y[i])[0, 0] for i in range(X.shape[0])]
#         rank_corr = spearmanr(scores, cosims, axis=0)[0]
#         print("{:d}\t{:.5f}".format(source_node_id, rank_corr))

def evaluate_model(td_matrix, model_file, source_id):
    model = gensim.models.Word2Vec.load(model_file).wv
    most_similar = model.most_similar(str(source_id))
    scores = [x[1] for x in most_similar]
    target_ids = [x[0] for x in most_similar]
    # compare top 10 scores with cosine similarity between source and each target
    X = np.repeat(td_matrix[source_id].todense(), 10, axis=0)
    Y = td_matrix[target_ids].todense()
    cosims = [cosine_similarity(X[i], Y[i])[0, 0] for i in range(10)]
    for i in range(10):
        print("{:d} {:s} {:.3f} {:.3f}".format(
            source_id, target_ids[i], cosims[i], scores[i]))


# read data and convert to Term-Document matrix
TD = download_and_read(UCI_DATA_URL)
# compute undirected, unweighted edge matrix
E = TD.T * TD
# binarize
E[E > 0] = 1
print(E.shape)

# construct random walks (caution: long process!)
construct_random_walks(E, NUM_WALKS_PER_VERTEX, RESTART_PROB, 
    MAX_PATH_LENGTH, RANDOM_WALKS_FILE)

# train model
train_word2vec_model(RANDOM_WALKS_FILE, W2V_MODEL_FILE)

# evaluate
source_id = np.random.choice(E.shape[0])
evaluate_model(TD, W2V_MODEL_FILE, source_id)

