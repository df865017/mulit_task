#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
# in tf 2.* take use of the tf 1.* api
if tf.__version__ >= '2.0.0':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()


def parse_function(example_proto, params):
    features = {
        # 'label': tf.FixedLenFeature([1], tf.string),
        # 'label2': tf.FixedLenFeature([1], tf.string),
        # 'cont_feats': tf.FixedLenFeature([params.cont_field_size], tf.string),
        # 'vector_feats': tf.FixedLenFeature([params.vector_feats_size], tf.string),
        # 'cate_feats': tf.FixedLenFeature(
        #     [params.cate_field_size + params.multi_feats_size + params.attention_feats_size], tf.string),

        'label': tf.FixedLenFeature([1], tf.float32),
        'label2': tf.FixedLenFeature([1], tf.float32),
        'cont_feats': tf.FixedLenFeature([params.cont_field_size], tf.float32),
        'vector_feats': tf.FixedLenFeature([params.vector_feats_size], tf.float32),
        'cate_feats': tf.FixedLenFeature([ 2], tf.int64),
        #params.cate_field_size + params.multi_feats_size + params.attention_feats_size

        # 'cate_feats': tf.FixedLenFeature([params.cate_field_size + params.multi_feats_size + params.attention_feats_size], tf.int64),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    labels = dict()
    label = parsed_features['label']
    label2 = parsed_features['label2']
    parsed_features.pop('label')
    parsed_features.pop('label2')
    labels['label'] = label
    labels['label2'] = label2
    return parsed_features, labels


def input_fn(file_dir_list, params):
    # data_set = tf.data.TFRecordDataset(file_dir_list, buffer_size=params.batch_size * params.batch_size) \
    #     .map(lambda x: parse_function(x, params), num_parallel_calls=1) \
    #     .shuffle(buffer_size=params.batch_size * 10) \
    #     .batch(params.batch_size, drop_remainder=True)

    files = tf.data.Dataset.list_files(file_dir_list, shuffle=False)

    #data_set = files.apply(tf.contrib.data.parallel_interleave(
    data_set=files.apply(tf.data.experimental.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=10, block_length=1, sloppy=False)) \
        .map(lambda x: parse_function(x, params), num_parallel_calls=4) \
        .batch(params.batch_size) \
        .prefetch(4000)

    iterator = data_set.make_one_shot_iterator()
    feature_dict, labels = iterator.get_next()
    return feature_dict,labels


def get_file_list(input_path):
    file_list = tf.gfile.ListDirectory(input_path)
    print("file_list_len:", len(file_list))
    file_dir_list = []
    for file in file_list:
        if file[:4] == "part":
            file_path = input_path + file
            file_dir_list.append(file_path)

    return file_dir_list


if __name__ == '__main__':
    a = dict()
    a['liu'] = 'an'
    for key, value in a.items():
        print(key, " = ", value)
