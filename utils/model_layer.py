import sys

import numpy as np
# import tensorflow as tf

import tensorflow as tf
# in tf 2.* take use of the tf 1.* api
if tf.__version__ >= '2.0.0':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()


def emb_init(name, feat_num, embedding_size, zero_first_row=True, pre_trained=False, trained_emb_path=None):
    if not pre_trained:
        with tf.variable_scope("weight_matrix"):
            embeddings = tf.get_variable(name=name,
                                         dtype=tf.float32,
                                         shape=(feat_num, embedding_size),
                                         initializer=tf.keras.initializers.glorot_normal())

                                         #initializer=tf.contrib.layers.xavier_initializer())

        if zero_first_row:  # The first row of initialization is zero
            embeddings = tf.concat((tf.zeros(shape=[1, embedding_size]), embeddings[1:]), 0)
    else:
        pass
        with tf.variable_scope("pre-trained_weight_matrix"):
            load_emb = np.load(tf.gfile.GFile(trained_emb_path, "rb"))
            embeddings = tf.constant(load_emb, dtype=tf.float32, name=name)
            sys.stdout.flush()

    return embeddings


def nonzero_reduce_mean(emb):
    axis_2_sum = tf.reduce_sum(emb, axis=2)
    multi_cate_nonzero = tf.count_nonzero(axis_2_sum, 1, keepdims=True, dtype=float)
    multi_cate_sum = tf.reduce_sum(emb, axis=1)
    reduce_mean_emb = tf.div_no_nan(multi_cate_sum, multi_cate_nonzero)
    return reduce_mean_emb


def multi_cate_emb(multi_feats_range, init_emb, multi_cate_feats):
    all_multi_emb = None
    for item in multi_feats_range:
        index_start = int(item[0])
        index_end = int(item[1])

        multi_emb = tf.nn.embedding_lookup(init_emb, ids=multi_cate_feats[:, index_start:index_end])
        # multi_emb = tf.reduce_sum(multi_emb, axis=1)  # sum-pool
        multi_emb = nonzero_reduce_mean(multi_emb)  # mean-pool

        if int(item[0]) == 0:
            all_multi_emb = multi_emb
        else:
            all_multi_emb = tf.concat([all_multi_emb, multi_emb], axis=1)

    return all_multi_emb


class InteractingLayer:
    def __init__(self, num_layer, att_emb_size=32, seed=2020, head_num=3, use_res=1):
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        self.num_layer = num_layer
        self.att_emb_size = att_emb_size
        self.seed = seed
        self.head_num = head_num
        self.use_res = use_res

    def __call__(self, inputs):
        input_shape = inputs.get_shape().as_list()
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 3 dimensions" % len(input_shape))

        embedding_size = int(input_shape[-1])
        self.w_query = tf.get_variable(name=str(self.num_layer) + '_query',
                                       dtype=tf.float32,
                                       shape=(embedding_size, self.att_emb_size * self.head_num),
                                       initializer=tf.keras.initializers.glorot_normal(seed=self.seed))
                                       #initializer=tf.contrib.layers.xavier_initializer(seed=self.seed))
        self.w_key = tf.get_variable(name=str(self.num_layer) + '_key',
                                     dtype=tf.float32,
                                     shape=(embedding_size, self.att_emb_size * self.head_num),
                                     initializer=tf.keras.initializers.glorot_normal(seed=self.seed))
                                     #initializer=tf.contrib.layers.xavier_initializer(seed=self.seed + 1))
        self.w_value = tf.get_variable(name=str(self.num_layer) + '_value',
                                       dtype=tf.float32,
                                       shape=(embedding_size, self.att_emb_size * self.head_num),
                                       initializer=tf.keras.initializers.glorot_normal(seed=self.seed))
                                       #initializer=tf.contrib.layers.xavier_initializer(seed=self.seed + 2))
        if self.use_res:
            self.w_res = tf.get_variable(name=str(self.num_layer) + '_res',
                                         dtype=tf.float32,
                                         shape=(embedding_size, self.att_emb_size * self.head_num),
                                         initializer=tf.keras.initializers.glorot_normal(seed=self.seed))
                                         #initializer=tf.contrib.layers.xavier_initializer(seed=self.seed))

        querys = tf.tensordot(inputs, self.w_query, axes=1)  # None F D*head_num
        keys = tf.tensordot(inputs, self.w_key, axes=1)
        values = tf.tensordot(inputs, self.w_value, axes=1)

        # head_num None F D
        querys = tf.stack(tf.split(querys, self.head_num, axis=2))
        keys = tf.stack(tf.split(keys, self.head_num, axis=2))
        values = tf.stack(tf.split(values, self.head_num, axis=2))

        inner_product = tf.matmul(querys, keys, transpose_b=True)  # head_num None F F
        # Scale
        inner_product = inner_product / (keys.get_shape().as_list()[-1] ** 0.5)
        # Activation
        self.normalized_att_scores = tf.nn.softmax(inner_product)

        result = tf.matmul(self.normalized_att_scores, values)  # head_num None F D
        result = tf.concat(tf.split(result, self.head_num, axis=0), axis=-1)  # 1 None F D*head_num
        result = tf.squeeze(result, axis=0)  # None F D*head_num

        if self.use_res:
            result += tf.tensordot(inputs, self.w_res, axes=1)
        result = tf.nn.relu(result)

        return result
