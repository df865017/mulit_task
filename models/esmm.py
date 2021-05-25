#!/usr/bin/env python
# coding=utf-8

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
    #print(single_cate_feats)
    # multi_cate_feats = cate_feats[:, params.cate_field_size:]

    feats_emb = my_layer.emb_init(name='feats_emb', feat_num=params.cate_feats_size,
                                  embedding_size=params.embedding_size)


    # single_category -> Embedding
    cate_emb = tf.nn.embedding_lookup(feats_emb, ids=single_cate_feats)
    #print(cate_emb)
    cate_emb = tf.reshape(cate_emb, shape=[-1, params.cate_field_size * params.embedding_size])
    #print(cate_emb)
    # multi_category -> Embedding
    # multi_cate_emb = my_layer.multi_cate_emb(params.multi_feats_range, feats_emb, multi_cate_feats)

    # deep input dense
    dense_input = tf.concat([cont_feats, vector_feats, cate_emb], axis=1, name='dense_vector') #, multi_cate_emb

    len_layers = len(params.hidden_units)
    with tf.variable_scope('ctr_deep'):
        dense_ctr = tf.layers.dense(inputs=dense_input, units=params.hidden_units[0], activation=tf.nn.relu)
        for i in range(1, len_layers):
            dense_ctr = tf.layers.dense(inputs=dense_ctr, units=params.hidden_units[i], activation=tf.nn.relu)
        ctr_out = tf.layers.dense(inputs=dense_ctr, units=1)
    with tf.variable_scope('cvr_deep'):
        dense_cvr = tf.layers.dense(inputs=dense_input, units=params.hidden_units[0], activation=tf.nn.relu)
        for i in range(1, len_layers):
            dense_cvr = tf.layers.dense(inputs=dense_cvr, units=params.hidden_units[i], activation=tf.nn.relu)
        cvr_out = tf.layers.dense(inputs=dense_cvr, units=1)

    ctr_score = tf.identity(tf.nn.sigmoid(ctr_out), name='ctr_score')
    cvr_score = tf.identity(tf.nn.sigmoid(cvr_out), name='cvr_score')
    ctcvr_score = ctr_score * cvr_score
    ctcvr_score = tf.identity(ctcvr_score, name='ctcvr_score')

    ctr_pow = 0.5
    cvr_pow = 1
    score = tf.multiply(tf.pow(ctr_score, ctr_pow), tf.pow(cvr_score, cvr_pow))
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
