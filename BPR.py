'''
BPR
@author: Tianyu Zhu
@created: 2019/4/18
'''

import numpy as np
import tensorflow as tf
import csv
import random
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

        self.W_u = tf.Variable(tf.random_normal([self.num_user, self.num_factor], stddev=0.01), name="W_u")
        self.W_i = tf.Variable(tf.random_normal([self.num_item, self.num_factor], stddev=0.01), name="W_i")

        self.u_emb = tf.nn.embedding_lookup(self.W_u, self.u)
        self.i_emb = tf.nn.embedding_lookup(self.W_i, self.i)
        self.j_emb = tf.nn.embedding_lookup(self.W_i, self.j)
        self.r_hat_ui = tf.reduce_sum(self.u_emb * self.i_emb, 1, True)
        self.r_hat_uj = tf.reduce_sum(self.u_emb * self.j_emb, 1, True)
        self.bpr_loss = -tf.reduce_mean(tf.log_sigmoid(self.r_hat_ui - self.r_hat_uj))

        self.regularization = tf.nn.l2_loss(self.W_u) + tf.nn.l2_loss(self.W_i)
        self.loss = self.bpr_loss + self.reg_rate * self.regularization
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        print("model prepared...")


def generate_train_data(train_dict, num_item, num_neg_sample):
    train_data = []
    for u in train_dict:
        for i in train_dict[u]:
            for _ in range(num_neg_sample):
                neg_sample = random.sample(list(range(num_item)), 1)[0]
                while neg_sample in train_dict[u]:
                    neg_sample = random.sample(list(range(num_item)), 1)[0]
                train_data.append([u, i, neg_sample])
    return train_data


def generate_train_batch(train_data, batch_size):
    train_batch = []
    np.random.shuffle(train_data)
    i = 0
    while i < len(train_data):
        train_batch.append(np.asarray(train_data[i:i+batch_size]))
        i += batch_size
    return train_batch


def generate_test_data(test_dict, negative_dict):
    test_data = []
    for u in test_dict:
        test_data.append([u, test_dict[u]])
        for neg in negative_dict[u]:
            test_data.append([u, neg])
    test_data = np.asarray(test_data)
    return test_data


def get_feed_dict(model, batch):
    feed_dict = dict()
    feed_dict[model.u] = batch[:, 0]
    feed_dict[model.i] = batch[:, 1]
    feed_dict[model.j] = batch[:, 2]
    return feed_dict


if __name__ == '__main__':
    dataset = 'ml-1m'
    # Model hyperparameters
    num_factor = 64
    reg_rate = 1e-5
    num_neg_sample = 1
    # Training parameters
    batch_size = 256
    num_epoch = 100
    lr = 1e-3
    random_seed = 2019

    with tf.device('/cpu:0'), tf.Graph().as_default(), tf.Session() as sess:
        random.seed(random_seed)
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)

        [train_dict, validate_dict, test_dict, negative_dict, num_user, num_item] = np.load('data/{dataset}/{dataset}.npy'.format(dataset=dataset))
        print('dataset: {dataset}, num_user: {num_user}, num_item: {num_item}'.format(dataset=dataset, num_user=num_user, num_item=num_item))

        model = BPR(num_user, num_item, num_factor, reg_rate, lr)
        sess.run(tf.global_variables_initializer())
        validate_data = generate_test_data(validate_dict, negative_dict)
        test_data = generate_test_data(test_dict, negative_dict)

        result = []
        for epoch in range(num_epoch):
            print("epoch:", epoch)
            train_bpr_loss = []
            train_loss = []
            train_data = generate_train_data(train_dict, num_item, num_neg_sample)
            train_batch = generate_train_batch(train_data, batch_size)
            for batch in train_batch:
                bpr_loss, loss, _ = sess.run([model.bpr_loss, model.loss, model.train_op], feed_dict=get_feed_dict(model, batch))
                train_bpr_loss.append(bpr_loss)
                train_loss.append(loss)
            train_bpr_loss = sum(train_bpr_loss) / len(train_data)
            train_loss = sum(train_loss) / len(train_data)
            print("train bpr loss:", train_bpr_loss, "train loss:", train_loss)

            r_hat = sess.run(model.r_hat_ui, feed_dict={model.u: validate_data[:, 0], model.i: validate_data[:, 1]})
            rank_list = np.reshape(r_hat, [-1, 100]).argsort()[:, ::-1].tolist()
            validate_hr, validate_ndcg = evaluate(rank_list, 0, 10)
            print("validate hit ratio:", validate_hr, "validate ndcg:", validate_ndcg)
            result.append([epoch, train_loss, validate_hr, validate_ndcg])

            r_hat = sess.run(model.r_hat_ui, feed_dict={model.u: test_data[:, 0], model.i: test_data[:, 1]})
            rank_list = np.reshape(r_hat, [-1, 100]).argsort()[:, ::-1].tolist()
            test_hr, test_ndcg = evaluate(rank_list, 0, 10)
            print("test hit ratio:", test_hr, "test ndcg:", test_ndcg)

    print("over!")
    with open('BPR_{dataset}_{num_factor}_{reg_rate}.csv'.format(dataset=dataset, num_factor=num_factor, reg_rate=reg_rate), 'w', newline='') as f:
        writer = csv.writer(f)
        for line in result:
            writer.writerow(line)
