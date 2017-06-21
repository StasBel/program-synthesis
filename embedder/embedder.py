import collections
import numpy as np
import tensorflow as tf
import math
from collections import Counter, namedtuple
from itertools import chain


class Embedder:
    def __init__(self, embeddings, dictionary, reverse_dictionary):
        self.emb = embeddings
        self.dic = dictionary
        self.rdic = reverse_dictionary

    def vec(self, clazz):
        return self.emb[self.dic[clazz]]


def substitute(edges):
    flattern = list(chain(*edges))
    counter = Counter(flattern)
    i2count, c2index, i2class = {}, {}, {}
    for index, (clazz, count) in enumerate(counter.most_common()):
        i2count[index], c2index[clazz], i2class[index] = count, index, clazz
    data = []
    for a, b in edges:
        data.append((c2index[a], c2index[b]))
    return data, i2count, c2index, i2class


def gen_batch(n, batch_size):
    return list(np.random.choice(n, batch_size, replace=False))


def make_embedding(edges, embedding_size, num_steps=10000, batch_size=128,
                   valid_size=16, valid_window=100, num_sampled=10):
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)

    data, count, dictionary, reverse_dictionary = substitute(edges)
    vocabulary_size = len(dictionary)

    graph = tf.Graph()

    with graph.as_default():
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        with tf.device("/cpu:0"):
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                          stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=train_labels,
                                             inputs=embed, num_sampled=num_sampled, num_classes=vocabulary_size))

        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as session:
        init.run()

        average_loss = 0
        for step in range(num_steps):
            batch_ind = gen_batch(len(data), batch_size)
            batch_inputs, batch_labels = zip(*(data[i] for i in batch_ind))
            batch = np.array(batch_inputs, dtype=np.int32)
            batch = batch.reshape((batch_size,))
            labels = np.array(batch_labels, dtype=np.int32)
            labels = labels.reshape((batch_size, 1))
            feed_dict = {train_inputs: batch, train_labels: labels}
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

        final_embeddings = normalized_embeddings.eval()

    return Embedder(final_embeddings, dictionary, reverse_dictionary)
