'''
BPR
@author: ZTY
@created: 7/18/2018
'''

import numpy as np
import tensorflow as tf
import csv
import random
from load_data import load_data
from evaluate import evaluate


class BPR(object):
    def __init__(self, num_user, num_item, num_factor, reg_rate, lr):
        print("num_factor:", num_factor, "regularization_rate:", reg_rate, "learning_rate:", lr)
        print("model preparing...")
        self.num_user = num_user
        self.num_item = num_item
        self.num_factor = num_factor
        self.reg_rate = reg_rate
        self.lr = lr

        self.u = tf.placeholder(tf.int32, [None], name="uid")
        self.i = tf.placeholder(tf.int32, [None], name="iid")
        self.j = tf.placeholder(tf.int32, [None], name="jid")

        self.W_u = tf.Variable(tf.truncated_normal([self.num_user, self.num_factor], stddev=0.01), name="W_u")
        self.W_i = tf.Variable(tf.truncated_normal([self.num_item, self.num_factor], stddev=0.01), name="W_i")
        self.b_i = tf.Variable(tf.zeros([self.num_item, 1]), name="b_i")

        self.u_emb = tf.nn.embedding_lookup(self.W_u, self.u)
        self.i_emb = tf.nn.embedding_lookup(self.W_i, self.i)
        self.j_emb = tf.nn.embedding_lookup(self.W_i, self.j)
        self.i_b = tf.nn.embedding_lookup(self.b_i, self.i)
        self.j_b = tf.nn.embedding_lookup(self.b_i, self.j)

        self.regularization = tf.nn.l2_loss(self.W_u) + tf.nn.l2_loss(self.W_i) + tf.nn.l2_loss(self.b_i)
        self.loss = -tf.reduce_mean(tf.log_sigmoid(tf.reduce_sum(self.u_emb * (self.i_emb - self.j_emb), 1, True) + self.i_b - self.j_b)) + self.reg_rate * self.regularization
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.name_scope("test"):
            self.test_u = tf.placeholder(tf.int32, [None], name="test_uid")
            self.test_i = tf.placeholder(tf.int32, [None], name="test_iid")

            self.test_u_emb = tf.nn.embedding_lookup(self.W_u, self.test_u)
            self.test_i_emb = tf.nn.embedding_lookup(self.W_i, self.test_i)
            self.test_i_b = tf.nn.embedding_lookup(self.b_i, self.test_i)

            self.score = tf.reshape(tf.reduce_sum(self.test_u_emb * self.test_i_emb, 1, True) + self.test_i_b, [-1])
            self.rank_list = tf.nn.top_k(self.score, 100)[1]

        print("model prepared...")


def generate_train_data(train_dict, validate_dict, test_dict, num_item, num_neg_sample):
    train_data = []
    for u in train_dict:
        for i in train_dict[u]:
            neg_sample = random.sample(list(set(range(num_item)) - set(train_dict[u]) - {validate_dict[u]} - {test_dict[u]}), num_neg_sample)
            for neg in neg_sample:
                train_data.append([u, i, neg])
    return train_data


def generate_train_batch(train_data, batch_size):
    train_batch = []
    np.random.shuffle(train_data)
    i = 0
    while i + batch_size < len(train_data):
        train_batch.append(np.asarray(train_data[i:i+batch_size]))
        i += batch_size
    train_batch.append(np.asarray(train_data[i:i+batch_size]))
    return train_batch


def generate_test_data(test_dict, negative_dict):
    test_data = []
    for u in test_dict:
        candidate = [test_dict[u]]
        candidate.extend(negative_dict[u])
        test_data.append([u, np.asarray(candidate)])
    return test_data


if __name__ == '__main__':
    dataset = ''
    # Model hyperparameters
    num_factor = 8
    reg_rate = 0
    num_neg_sample = 1
    # Training parameters
    batch_size = 256
    num_epoch = 30
    lr = 1e-3
    random_seed = 2018

    with tf.device('/cpu:0'), tf.Graph().as_default(), tf.Session() as sess:
        random.seed(random_seed)
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)

        num_user, num_item, trust_dict, train_dict, validate_dict, test_dict, negative_dict = load_data(''.format(dataset=dataset))
        model = BPR(num_user, num_item, num_factor, reg_rate, lr)
        sess.run(tf.global_variables_initializer())
        #train_data = generate_train_data(train_dict, validate_dict, test_dict, num_item, num_neg_sample)
        validate_data = generate_test_data(validate_dict, negative_dict)
        test_data = generate_test_data(test_dict, negative_dict)

        result = []
        for epoch in range(num_epoch):
            print("epoch:", epoch)
            train_loss = []
            train_data = generate_train_data(train_dict, validate_dict, test_dict, num_item, num_neg_sample)
            train_batch = generate_train_batch(train_data, batch_size)
            for batch in train_batch:
                batch_loss, _ = sess.run([model.loss, model.train_op], feed_dict={model.u: batch[:, 0], model.i: batch[:, 1], model.j: batch[:, 2]})
                train_loss.append(batch_loss)
            train_loss = sum(train_loss) / len(train_data)

            rank_list = []
            for line in validate_data:
                rank_u = sess.run(model.rank_list, feed_dict={model.test_u: np.asarray([line[0] for _ in range(100)]), model.test_i: line[1]})
                rank_list.append(rank_u.tolist())
            validate_hr, validate_ndcg = evaluate(rank_list, 0, 10)
            print("train loss:", train_loss, "validate hit ratio:", validate_hr, "validate ndcg:", validate_ndcg)
            result.append([epoch, train_loss, validate_hr, validate_ndcg])

        rank_list = []
        for line in test_data:
            rank_u = sess.run(model.rank_list, feed_dict={model.test_u: np.asarray([line[0] for _ in range(100)]), model.test_i: line[1]})
            rank_list.append(rank_u.tolist())
        test_hr5, test_ndcg5 = evaluate(rank_list, 0, 5)
        test_hr10, test_ndcg10 = evaluate(rank_list, 0, 10)
        test_hr20, test_ndcg20 = evaluate(rank_list, 0, 20)
        print("test hit ratio@5:", test_hr5, "test ndcg@5:", test_ndcg5)
        print("test hit ratio@10:", test_hr10, "test ndcg@10:", test_ndcg10)
        print("test hit ratio@20:", test_hr20, "test ndcg@20:", test_ndcg20)
        result.append(["test set@5", "", test_hr5, test_ndcg5])
        result.append(["test set@10", "", test_hr10, test_ndcg10])
        result.append(["test set@20", "", test_hr20, test_ndcg20])
    print("over!")
    with open('BPR_{dataset}.csv'.format(dataset=dataset), 'w', newline='') as f:
        writer = csv.writer(f)
        for line in result:
            writer.writerow(line)
