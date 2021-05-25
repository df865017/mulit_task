#!/usr/bin/env python
# coding=utf-8
import sys
import time

import tensorflow as tf
# in tf 2.* take use of the tf 1.* api
if tf.__version__ >= '2.0.0':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

from sklearn.metrics import roc_auc_score

import utils.data_loader as data_loader


def model_optimizer(params, mode, labels, out):
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=out)

    else:
        labels = tf.identity(labels, name='labels')
        auc = tf.metrics.auc(labels=labels, predictions=out, name='auc')
        metrics = {
            'auc': auc
        }
        loss = tf.reduce_mean(tf.losses.log_loss(labels=labels, predictions=out))

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


# def model_optimizer(params, mode, labels, out):
#     if mode == tf.estimator.ModeKeys.PREDICT:
#         return tf.estimator.EstimatorSpec(mode=mode, predictions=out)
#
#     labels = tf.where(labels >= params.rate, x=tf.ones_like(labels), y=tf.zeros_like(labels))
#     labels = tf.identity(labels, name='labels')
#     output = {'out': tf.estimator.export.PredictOutput(out)}
#
#     loss = tf.reduce_mean(tf.losses.log_loss(labels=labels, predictions=out))
#     auc = tf.metrics.auc(labels=labels, predictions=out, name='auc')
#
#     metrics = {
#         'auc': auc
#     }
#     if mode == tf.estimator.ModeKeys.TRAIN:
#         # logging_hook = tf.train.LoggingTensorHook(every_n_iter=1,
#         #                                           tensors={'auc': 'layer_0'})
#         optimizer = tf.train.AdamOptimizer(params.learning_rate)
#         train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
#         return tf.estimator.EstimatorSpec(mode=mode,
#                                           loss=loss,
#                                           train_op=train_op,
#                                           eval_metric_ops=metrics,
#                                           export_outputs=output,
#                                           # training_hooks = [logging_hook]
#                                           )
#
#     if mode == tf.estimator.ModeKeys.EVAL:
#         return tf.estimator.EstimatorSpec(mode=mode,
#                                           loss=loss,
#                                           eval_metric_ops=metrics,
#                                           export_outputs=output)
#     sys.stdout.flush()

def model_early_stop(valid_metric_list, num=1):
    max_metric = max(valid_metric_list)
    sum_list = len(valid_metric_list)
    if sum_list > 3:
        count_early_stop = 0
        for i in range(num):
            if valid_metric_list[-1 * (i + 1)] < max_metric:
                count_early_stop += 1
        if count_early_stop == num:
            return 1
    return 0


def model_fit(model, params, train_files, predict_files):
    valid_metric_list = []
    for epoch in range(params.epochs):
        st = time.time()
        # hook = tf.train.ProfilerHook(save_steps=10, output_dir=params.model_dir, show_memory=True)
        # model.train(input_fn=lambda: data_loader.input_fn(train_files, params), hooks=[hook])
        model.train(input_fn=lambda: data_loader.input_fn(train_files, params))
        results = model.evaluate(input_fn=lambda: data_loader.input_fn(predict_files, params))
        end_time = time.time()
        print('[%s] eval-ctr_auc=%.5f\t eval-ctcvr_auc=%.5f\t loss=%.5f\t [%.2f s]'
              % (epoch + 1, results['ctr_auc'], results['ctcvr_auc'], results['loss'], end_time - st))
        sys.stdout.flush()
        valid_metric_list.append(results['ctcvr_auc'])

        if model_early_stop(valid_metric_list):
            print("early_stop and save_model_pb")
            trained_model_path = model_save_pb(params, model)
            return trained_model_path

    print("save_model_pb")
    trained_model_path = model_save_pb(params, model)
    return trained_model_path


def model_save_pb(params, model):
    feature_spec = {
        'cont_feats': tf.placeholder(shape=[None, params.cont_field_size],
                                     dtype=tf.float32, name='cont_feats'),
        'vector_feats': tf.placeholder(shape=[None, params.vector_feats_size],
                                       dtype=tf.float32, name='vector_feats'),
        'cate_feats': tf.placeholder(
            shape=[None, params.cate_field_size + params.multi_feats_size + params.attention_feats_size],
            dtype=tf.int64, name='cate_feats')
    }
    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
    return model.export_savedmodel(params.model_pb, serving_input_receiver_fn)


def model_predict(trained_model_path, predict_files, params):
    """
        加载pb模型,预测tfrecord类型的数据
    """
    with tf.Session() as sess:
        model_sess = tf.Session(graph=tf.Graph())
        tf.saved_model.loader.load(model_sess, [tf.saved_model.tag_constants.SERVING], trained_model_path)
        ctr_list = []
        ctcvr_list = []
        label_list = []
        label2_list = []
        feature_dict, labels = data_loader.input_fn(predict_files, params=params)
        try:
            while True:
                cont_value, cate_index, vector_value, label = sess.run(
                    [
                        feature_dict["cont_feats"],
                        feature_dict["cate_feats"],
                        feature_dict["vector_feats"],
                        labels
                    ])
                feed_dict_map = {
                    "cont_feats:0": cont_value,
                    "cate_feats:0": cate_index,
                    "vector_feats:0": vector_value
                }
                # cont_test =[[0.621,0.375,0,0.38,0.30833334,0.043055557,0.041666668,0,0.174035,0.382218,0.03514557,0.23204666,0.63703,0.03514557,0.050364286,0.06662625,0.050253894,0.02934125,0.059127625,0.04727403,0.06085,0.09191433,0.042265452,0.5071667,1,0.03393404,0.07569143,0.110406,0.045789823,0.040384,0.0966312,0.04011137,0.2793,0.0315538,0.034192897,0.343572,0.2280928,0.07531,0,0.36833334,0.11033333,0.11383643,0.085734904,0.18099865,0.21649487,0.056297183,0.27,0.34,0.05682154,0.11,0.144,0.04008595,0.02,0.026,0.012721886,0.47867215,0.057008315,0.13144019,0.0037747615,0.095062,0.1534344,0.098546,0.1609098,0,0,0.155,0.10666667,0.13,0.14,0.51428574,0.7,1,1,1,1,0.3,0.53333336,0.73333335,1,1,0.13333334,0.23333333,0.26666668,1,0.56666666,0.033333335,0.033333335,0.033333335,0.1,0.1,0.32184276,0.32184276,0.33333334,1,1,1,1,0.6166667,0.99462366,1,1,1,1,0.49353868,0.09548176,0.071426734,0,0,0,0,0.006666667,0.008069219,0.004227895,0.0026879262,0,0.042424,0.1216411,0.0789144,0.23721413,0.1716417,0.47509292,0.114118196,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.22226538,0,0.823365,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.01,0.009945186,0.02,0.027486755,0.010286209,0.011420415,0.116666,0.14482799,0.0999952,0.0874112,0.22702645,0.5826725,0.064432114,1,0,0,0,0,0.02,0.036771044,0.19,0.12602167,0.058334,0.11996801,0.3220512,0.25538442,0.6264494,0.33808243,1,0.4785506,0.045197222,0.1015862,0.025940944,0,0,0.36164692,1,0.05025091,0.56904125,1,0.062496196,0,0,0,0,0,0,0.36867902,0.37824163,0.7594071,0.7192402,0.7406788,0.07774272,0.029975014,0,0]]
                # vector_test = [[0.127938,-0.024507,-0.034775,0.139589,-0.022194,-0.02504,0.057595,-0.061542,0.06617,0.001776,-0.006895,0.139986,-0.017825,-0.017082,0.036326,-0.095608,-0.060825,-0.050974,-0.016836,0.005366,-0.050041,0.029888,-0.07281,0.103261,0.052575,0.016638,0.054495,0.053722,0.027912,0.085459,-0.001956,-0.034916,-0.009175,0.055843,0.030169,-0.020012,0.002558,0.016575,-0.153872,0.095729,0.06477,0.001007,0.173617,-0.069311,0.073834,-0.107755,-0.036436,-0.151322,-0.037306,-0.085394,-0.084992,0.028416,-0.150947,-0.092988,0.033267,0.090444,-0.007677,0.087557,-0.011084,0.02188,-0.001829,0.14647,0.070584,0.015246,-0.00453,-0.007661,-0.044511,0.115343,0.08314,-0.051419,-0.144845,0.114681,0.090421,0.02182,-0.026956,0.055011,0.020883,-0.024194,0.064304,-0.006628,-0.139222,-0.014742,-0.048755,-0.006714,0.068525,-0.088183,-0.154155,0.126314,-0.067481,-0.029496,-0.010659,-0.000116,-0.00286,-0.088313,0.065189,-0.067735,0.054731,-0.018271,-0.058961,0.035818,-0.107415,-0.019752,-0.04989,0.028519,-0.008824,0.036169,-0.078459,-0.084623,-0.033223,-0.049004,-0.049906,-0.011349,0.061328,0.03971,0.05572,0.037039,0.037346,0.015967,0.03596,-0.079549,0.05562,-0.003787,0.007553,0.048134,0.110212,0.035251,-0.012426,0.040189,-0.065934,0.024965,0.085998,-0.026156,0.021003,0.000415,-0.063335,-0.107018,-0.007636,-0.103743,0.008674,-0.007018,-0.12172,-0.155141,-0.078171,-0.081563,0.151277,0.001085,0.041309,0.094566,-0.069134,0.050369,-0.078747,0.0068,0.034038,0.02044,-0.055907,0.015149,0.046706,0.146128,0.122049,-0.006767,0.115887,-0.01826,0.048786,0.064711,0.092596,0.022173,-0.009475,0.059026,0.083564,-0.12157,0.038314,-0.021612,-0.036716,0.116524,-0.020439,0.045493,-0.033245,0.036567,0.045419,0.075688,-0.032745,-0.08414,0.039893,0.014466,-0.103956,0.045408,-0.111776,0.211391,-0.013373,0.081905,0.009883,0.06048,0.047331,0.075289,0.119356,0.164626,0.014725,0.004545,0.026799,-0.013875,0.03173,-0.04764,0.11624,0.05617,0.0609,0.0088,0.09928,0.10932,-0.05369,-0.10526,-0.00934,-0.07239,-0.01698,0.00998,0.01661,0.01856,-0.08258,-0.14472,0.06151,0.0038,0.01439,-0.04005,-0.02914,0.08814,-0.02706,0.00733,0.01833,0.11295,-0.0137,0.00797,0.02565,0.06427,0.01334,0.00929,-0.0937,-0.03042,0.04774,-0.01829,0.01017,-0.01039,0.05372,0.02682,0.13912,0.06827,0.03203,-0.03733,-0.0733,-0.16059,0.04544,0.07142,-0.05724,-0.07453,-0.0187,0.05412,0.08372,-0.11121,-0.01417,0.05114,0.00201,0.09613,0.07423,0.1121,0.04529,0.0052,-0.03421,-0.06474,-0.11707,0.03178,0.03488,0.04669,-0.15074,-0.02883,-0.00233,0.04622,-0.03758,-0.02559,0.04718,0.01477,0.11349,0.15037,-0.04432,0.08662,0.0148,0.08444,0.0287,0.00347,0.04632,0.06091,-0.08416,-0.17196,-0.00572,0.11432,0.03388,0.00546,-0.00283,-0.00208,0.05473,0.05898,-0.10507,-0.06091,0.11098,-0.03147,-0.15943,0.08633,-0.08334,0.16045,0.0452,0.02949,-0.0516,0.02115,-0.02425,-0.0016,0.06263,0.03739,0.0054,-0.02781,-0.00391,-0.05766,0.01439,-0.0453,0.03632,-0.03023,-0.0886,-0.07373,0.04632,0.14353,-0.07548,-0.0202,0.05804,-0.10784,0.05408,-0.02444,0.0049,0.04196,-0.09448,-0.09732,-0.00776,-0.10054,0.00979,0.00593,-0.0748,-0.13526,-0.14669,-0.15313,0.03741,-0.01002,-0.00194,0.15395,-0.03904,-0.02199,-0.04173,0.07804,-0.13163,0.0251,-0.12257,-0.06535,0.01643,0.01842,0.12227,-0.00883,0.12123,-0.01791,0.04368,0.00911,0.18353,-0.02261,-0.03283,0.09697,-0.0655,-0.08288,0.05012,-0.09717,0.03696,0.06595,-0.0035,-0.06674,0.0287,-0.07235,0.00897,0.01434,-0.00641,0.04007,-0.09726,0.11026,0.02953,0.00657,-0.00197,0.0888,0.04348,-0.00879,0.05464,0.18432,-0.01622,-0.00909,0.0248,0.09874,-0.02908,-0.09137,0.04715,0.04164]]
                # cate_test =[[896,184,674,875,765,645,696,1141,452,899,395,1386,395,1386,1386,717,1032,1386,1103,179,273,1058,174,1231,22,1051]]
                # feed_dict_map = {
                #     "cont_feats:0": cont_test,
                #     "cate_feats:0": cate_test,
                #     "vector_feats:0": vector_test
                # }
                ctr = model_sess.run("ctr_score:0", feed_dict=feed_dict_map)
                ctcvr = model_sess.run("ctcvr_score:0", feed_dict=feed_dict_map)
                ctr_score = ctr[:, 0]
                ctcvr_score = ctcvr[:, 0]
                label_list.extend(label['label'])
                ctr_list.extend(ctr_score)
                label2_list.extend(label['label2'])
                ctcvr_list.extend(ctcvr_score)
                # break
        except tf.errors.OutOfRangeError:
            print("val of ctr_auc:%.5f" % roc_auc_score(label_list, ctr_list))
            print("val of cvr_auc:%.5f" % roc_auc_score(label2_list, ctcvr_list))
            sys.stdout.flush()
            print('---end---')
