#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf

import tensorflow as tf
# in tf 2.* take use of the tf 1.* api
if tf.__version__ >= '2.0.0':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()


import utils.model_layer as my_layer


def model_fn(features, labels, mode, params):
    tf.set_random_seed(2019)

    cont_feats = features["cont_feats"]
    cate_feats = features["cate_feats"]
    vector_feats = features["vector_feats"]

    single_cate_feats = cate_feats[:, 0:params.cate_field_size]
    multi_cate_feats = cate_feats[:, params.cate_field_size:]
    cont_feats_index = tf.Variable([[i for i in range(params.cont_field_size)]], trainable=False, dtype=tf.int64,
                                   name="cont_feats_index")

    cont_index_add = tf.add(cont_feats_index, params.cate_feats_size)

    index_max_size = params.cont_field_size + params.cate_feats_size
    feats_emb = my_layer.emb_init(name='feats_emb', feat_num=index_max_size, embedding_size=params.embedding_size)

    # cont_feats -> Embedding
    with tf.name_scope("cont_feat_emb"):
        ori_cont_emb = tf.nn.embedding_lookup(feats_emb, ids=cont_index_add, name="ori_cont_emb")
        cont_value = tf.reshape(cont_feats, shape=[-1, params.cont_field_size, 1], name="cont_value")
        cont_emb = tf.multiply(ori_cont_emb, cont_value)
        cont_emb = tf.reshape(cont_emb, shape=[-1, params.cont_field_size * params.embedding_size], name="cont_emb")

    # single_category -> Embedding
    with tf.name_scope("single_cate_emb"):
        cate_emb = tf.nn.embedding_lookup(feats_emb, ids=single_cate_feats)
        cate_emb = tf.reshape(cate_emb, shape=[-1, params.cate_field_size * params.embedding_size])

    # multi_category -> Embedding
    # with tf.name_scope("multi_cate_emb"):
    #     multi_cate_emb = my_layer.multi_cate_emb(params.multi_feats_range, feats_emb, multi_cate_feats)

    # deep input dense
    #dense_input = tf.concat([cont_emb, vector_feats, cate_emb, multi_cate_emb], axis=1, name='dense_vector')
    # dense_input = tf.concat([cont_emb, vector_feats, cate_emb], axis=1, name='dense_vector')
    dense_input = tf.concat([cont_emb, vector_feats, cate_emb], axis=1, name='dense_vector')

    # experts
    experts_weight = tf.get_variable(name='experts_weight',
                                     dtype=tf.float32,
                                     shape=(dense_input.get_shape()[1], params.experts_units, params.experts_num),
                                     initializer=tf.keras.initializers.glorot_normal())
                                     #initializer=tf.contrib.layers.xavier_initializer())
    experts_bias = tf.get_variable(name='expert_bias',
                                   dtype=tf.float32,
                                   shape=(params.experts_units, params.experts_num),
                                   initializer=tf.keras.initializers.glorot_normal())
                                   #initializer=tf.contrib.layers.xavier_initializer())

    # gates
    gate1_weight = tf.get_variable(name='gate1_weight',
                                   dtype=tf.float32,
                                   shape=(dense_input.get_shape()[1], params.experts_num),
                                   initializer=tf.keras.initializers.glorot_normal())
                                   #initializer=tf.contrib.layers.xavier_initializer())
    gate1_bias = tf.get_variable(name='gate1_bias',
                                 dtype=tf.float32,
                                 shape=(params.experts_num,),
                                 initializer=tf.keras.initializers.glorot_normal())
                                 #initializer=tf.contrib.layers.xavier_initializer())
    gate2_weight = tf.get_variable(name='gate2_weight',
                                   dtype=tf.float32,
                                   shape=(dense_input.get_shape()[1], params.experts_num),
                                   initializer=tf.keras.initializers.glorot_normal())
                                   #initializer=tf.contrib.layers.xavier_initializer())
    gate2_bias = tf.get_variable(name='gate2_bias',
                                 dtype=tf.float32,
                                 shape=(params.experts_num,),
                                 initializer=tf.keras.initializers.glorot_normal())
                                 #initializer=tf.contrib.layers.xavier_initializer())

    # f_{i}(x) = activation(W_{i} * x + b), where activation is ReLU according to the paper
    experts_output = tf.tensordot(dense_input, experts_weight, axes=1)
    use_experts_bias = True
    if use_experts_bias:
        experts_output = tf.add(experts_output, experts_bias)
    experts_output = tf.nn.relu(experts_output)

    # g^{k}(x) = activation(W_{gk} * x + b), where activation is softmax according to the paper
    gate1_output = tf.matmul(dense_input, gate1_weight)
    gate2_output = tf.matmul(dense_input, gate2_weight)
    user_gate_bias = True
    if user_gate_bias:
        gate1_output = tf.add(gate1_output, gate1_bias)
        gate2_output = tf.add(gate2_output, gate2_bias)
    gate1_output = tf.nn.softmax(gate1_output)
    gate2_output = tf.nn.softmax(gate2_output)

    # f^{k}(x) = sum_{i=1}^{n}(g^{k}(x)_{i} * f_{i}(x))
    label1_input = tf.multiply(experts_output, tf.expand_dims(gate1_output, axis=1))
    label1_input = tf.reduce_sum(label1_input, axis=2)
    label1_input = tf.reshape(label1_input, [-1, params.experts_units])
    label2_input = tf.multiply(experts_output, tf.expand_dims(gate2_output, axis=1))
    label2_input = tf.reduce_sum(label2_input, axis=2)
    label2_input = tf.reshape(label2_input, [-1, params.experts_units])

    len_layers = len(params.hidden_units)
    with tf.variable_scope('ctr_deep'):
        dense_ctr = tf.layers.dense(inputs=label1_input, units=params.hidden_units[0], activation=tf.nn.relu)
        for i in range(1, len_layers):
            dense_ctr = tf.layers.dense(inputs=dense_ctr, units=params.hidden_units[i], activation=tf.nn.relu)
        ctr_out = tf.layers.dense(inputs=dense_ctr, units=1)
    with tf.variable_scope('cvr_deep'):
        dense_cvr = tf.layers.dense(inputs=label2_input, units=params.hidden_units[0], activation=tf.nn.relu)
        for i in range(1, len_layers):
            dense_cvr = tf.layers.dense(inputs=dense_cvr, units=params.hidden_units[i], activation=tf.nn.relu)
        cvr_out = tf.layers.dense(inputs=dense_cvr, units=1)

    ctr_score = tf.identity(tf.nn.sigmoid(ctr_out), name='ctr_score')
    cvr_score = tf.identity(tf.nn.sigmoid(cvr_out), name='cvr_score')
    ctcvr_score = ctr_score * cvr_score
    ctcvr_score = tf.identity(ctcvr_score, name='ctcvr_score')

    score = tf.add(ctr_score * params.label1_weight, cvr_score * params.label2_weight)
    score = tf.identity(score, name='score')

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=score)

    else:
        ctr_labels = tf.identity(labels['label'], name='ctr_labels')
        ctcvr_labels = tf.identity(labels['label2'], name='ctcvr_labels')
        ctr_auc = tf.metrics.auc(labels=ctr_labels, predictions=ctr_score, name='auc')
        ctcvr_auc = tf.metrics.auc(labels=ctcvr_labels, predictions=ctcvr_score, name='auc')
        metrics = {
            'ctr_auc': ctr_auc,
            'ctcvr_auc': ctcvr_auc
        }
        # ctr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=ctr_labels, logits=ctr_out))
        ctr_loss = tf.reduce_mean(tf.losses.log_loss(labels=ctr_labels, predictions=ctr_score))
        ctcvr_loss = tf.reduce_mean(tf.losses.log_loss(labels=ctcvr_labels, predictions=ctcvr_score))
        loss = ctr_loss + ctcvr_loss

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(params.learning_rate)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        else:
            train_op = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=metrics,
        train_op=train_op)


def model_estimator(params):
    # shutil.rmtree(conf.model_dir, ignore_errors=True)
    tf.reset_default_graph()
    config = tf.estimator.RunConfig() \
        .replace(session_config=tf.ConfigProto(device_count={'GPU': params.is_GPU}),
                 log_step_count_steps=params.log_step_count_steps,
                 save_checkpoints_steps=params.save_checkpoints_steps,
                 keep_checkpoint_max=params.keep_checkpoint_max,
                 save_summary_steps=params.save_summary_steps)

    model = tf.estimator.Estimator(
        model_fn=model_fn,
        config=config,
        model_dir=params.model_dir,
        params=params,
    )
    return model
