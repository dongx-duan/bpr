# coding: utf-8

import tensorflow as tf
import os
import lmdb
import cPickle as pickle
import numpy
import random
from multiprocessing import Process, Queue

with open("man20k.pkl", "r") as f:
    user_id_mapping, item_id_mapping, train_ratings, test_ratings = pickle.load(f)

image_features = {}
db = lmdb.open('./features20k')
with db.begin(write=False) as ctx:
    for iid in item_id_mapping.values():
        image_features[iid] = numpy.fromstring(ctx.get(str(iid)), dtype=numpy.float32)

assert(len(item_id_mapping) == len(image_features))
print len(user_id_mapping)
print len(item_id_mapping)


train_queue = Queue(4)
def uniform_sample_batch(train_ratings, item_count, image_features, sample_count=20000, batch_size=512):
    for i in range(sample_count):
        t = []
        iv = []
        jv = []
        for b in xrange(batch_size):
            u = random.sample(train_ratings.keys(), 1)[0]
            i = random.sample(train_ratings[u], 1)[0]
            j = random.randint(0, item_count-1)
            while j in train_ratings[u]:
                j = random.randint(0, item_count-1)
            t.append([u, i, j])
            iv.append(image_features[i])
            jv.append(image_features[j])
        # block if queue is full
        train_queue.put( (numpy.asarray(t), numpy.vstack(tuple(iv)), numpy.vstack(tuple(jv))), True )
    train_queue.put(None)

def train_data_process(sample_count=20000, batch_size=512):
    p = Process(target=uniform_sample_batch, args=(train_ratings, len(item_id_mapping), image_features, sample_count, batch_size))
    return p


def test_batch_generator_by_user(train_ratings, test_ratings, item_count, image_features):
    # using leave one cv
    for u in test_ratings.keys():
        i = test_ratings[u]
        t = []
        ilist = []
        jlist = []
        for j in range(item_count):
            if j != test_ratings[u] and not (j in train_ratings[u]):
                # find item not in test[u] and train[u]
                t.append([u, i, j])
                ilist.append(image_features[i])
                jlist.append(image_features[j])
        yield numpy.asarray(t), numpy.vstack(tuple(ilist)), numpy.vstack(tuple(jlist))


def vbpr(user_count, item_count, hidden_dim=20, hidden_img_dim=128, 
         learning_rate = 0.001,
         l2_regulization = 0.01, 
         bias_regulization=1.0):
    """
    user_count: total number of users
    item_count: total number of items
    hidden_dim: hidden feature size of MF
    hidden_img_dim: [4096, hidden_img_dim]
    """
    u = tf.placeholder(tf.int32, [None])
    i = tf.placeholder(tf.int32, [None])
    j = tf.placeholder(tf.int32, [None])
    iv = tf.placeholder(tf.float32, [None, 4096])
    jv = tf.placeholder(tf.float32, [None, 4096])
    
    with tf.device("/gpu:1"):
        user_emb_w = tf.get_variable("user_emb_w", [user_count+1, hidden_dim], 
                                    initializer=tf.random_normal_initializer(0, 0.1))
        user_img_w = tf.get_variable("user_img_w", [user_count+1, hidden_img_dim],
                                    initializer=tf.random_normal_initializer(0, 0.1))
        item_emb_w = tf.get_variable("item_emb_w", [item_count+1, hidden_dim], 
                                    initializer=tf.random_normal_initializer(0, 0.1))
        item_b = tf.get_variable("item_b", [item_count+1, 1], 
                                    initializer=tf.constant_initializer(0.0))
        
        u_emb = tf.nn.embedding_lookup(user_emb_w, u)
        u_img = tf.nn.embedding_lookup(user_img_w, u)
        
        i_emb = tf.nn.embedding_lookup(item_emb_w, i)
        i_b = tf.nn.embedding_lookup(item_b, i)
        j_emb = tf.nn.embedding_lookup(item_emb_w, j)
        j_b = tf.nn.embedding_lookup(item_b, j)
    
    with tf.device("/gpu:1"):
        img_emb_w = tf.get_variable("image_embedding_weights", [4096, hidden_img_dim], 
                                   initializer=tf.random_normal_initializer(0, 0.1))

        img_i_j = tf.matmul(iv - jv,  img_emb_w)

        # MF predict: u_i > u_j
        x = i_b - j_b + tf.reduce_sum(tf.mul(u_emb, (i_emb - j_emb)), 1, keep_dims=True) +             tf.reduce_sum(tf.mul(u_img, img_i_j),1, keep_dims=True)

        # auc score is used in test/cv
        # reduce_mean is reasonable BECAUSE
        # all test (i, j) pairs of one user is in ONE batch
        auc = tf.reduce_mean(tf.to_float(x > 0))

        l2_norm = tf.add_n([
                tf.reduce_sum(tf.mul(u_emb, u_emb)), 
                tf.reduce_sum(tf.mul(u_img, u_img)),
                tf.reduce_sum(tf.mul(i_emb, i_emb)),
                tf.reduce_sum(tf.mul(j_emb, j_emb)),
                tf.reduce_sum(tf.mul(img_emb_w, img_emb_w)),
                bias_regulization * tf.reduce_sum(tf.mul(i_b, i_b)),
                bias_regulization * tf.reduce_sum(tf.mul(j_b, j_b))
            ])

        loss = l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(x)))
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return u, i, j, iv, jv, loss, auc, train_op


# In[17]:

user_count = len(user_id_mapping)
item_count = len(item_id_mapping)

with tf.Graph().as_default(), tf.Session() as session:
    with tf.variable_scope('vbpr'):
        u, i, j, iv, jv, loss, auc, train_op = vbpr(user_count, item_count)
    
    session.run(tf.initialize_all_variables())
    
    for epoch in range(1, 20):
        print "epoch ", epoch
        _loss_train = 0.0
        sample_count = 20000
        batch_size = 512
        p = train_data_process(sample_count, batch_size)
        p.start()
        data = train_queue.get(True) #block if queue is empty
        while data:
            d, _iv, _jv = data
            _loss, _ = session.run([loss, train_op], feed_dict={
                    u:d[:,0], i:d[:,1], j:d[:,2], iv:_iv, jv:_jv
                })
            _loss_train += _loss
            data = train_queue.get(True)
        p.join()
        print "train_loss:", _loss_train/sample_count

        _auc_all = 0
        _loss_test = 0.0
        _test_user_count = len(test_ratings)
        for d, _iv, _jv in test_batch_generator_by_user(train_ratings, 
                                                      test_ratings, item_count, image_features):
            _loss, _auc = session.run([loss, auc], feed_dict={
                    u:d[:,0], i:d[:,1], j:d[:,2], iv:_iv, jv:_jv
                })
            _loss_test += _loss
            _auc_all += _auc
        print "test_loss: ", _loss_test/_test_user_count, " auc: ", _auc_all/_test_user_count
        print ""

