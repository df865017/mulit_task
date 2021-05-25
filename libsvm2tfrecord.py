# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random

import tensorflow as tf
# in tf 2.* take use of the tf 1.* api
if tf.__version__ >= '2.0.0':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

SINGLE_CATE_NUM = 5



def _bytes_feature(value):
	"""Returns a bytes_list from a string / byte."""
	if isinstance(value, type(tf.constant(0))):
		value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
	"""Returns a float_list from a float / double."""
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
	"""Returns an int64_list from a bool / enum / int / uint."""
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


class TrainingInstance(object):
	"""A single training instance"""

	def __init__(self, label_click, label_conv, cont_feats, vector_feats, single_cate_feats, single_cate_feats_int,
				 mutil_cate_feats, mutil_cate_feats_int,
				 mutil_cate_feats_w):
		self.label_click = label_click
		self.label_conv = label_conv
		self.cont_feats = cont_feats
		self.vector_feats = vector_feats
		self.single_cate_feats = single_cate_feats
		self.single_cate_feats_int = single_cate_feats_int
		self.mutil_cate_feats = mutil_cate_feats
		self.mutil_cate_feats_int = mutil_cate_feats_int
		self.mutil_cate_feats_w = mutil_cate_feats_w


def create_training_instances(input_files, rng):
	"""Create `TrainingInstance`s from raw text."""
	instances = []
	for input_file in input_files:
		with tf.gfile.GFile(input_file, "r") as reader:
			while True:
				line = reader.readline()
				if not line:
					break
				info = line.strip().split(" ")
				if len(info) != 1457:
					continue
				cont_feats, vector_feats, single_cate_feats, single_cate_feats_int, \
				mutil_cate_feats, mutil_cate_feats_int, mutil_cate_feats_w = [], [], [], [], [], [], []
				label_click = [float(info[0].split("_")[0])]
				label_conv = [float(info[0].split("_")[1])]
				for i in range(1, 287):
					cont_feats.append(float(info[i].split(":")[1]))
				for i in range(287, 292):
					single_cate_feats.append([info[i].split(":")[0]])
					single_cate_feats_int.append([int(info[i].split(":")[0])])
				for i in range(292, 432):
					mutil_cate_feats.append(info[i].split(":")[0])
					mutil_cate_feats_int.append(int(info[i].split(":")[0]))
					mutil_cate_feats_w.append(float(info[i].split(":")[1]))
				for i in range(433, 1457):
					vector_feats.append(float(info[i].split(":")[1]))
				instance = TrainingInstance(label_click, label_conv, cont_feats, vector_feats, single_cate_feats,
											single_cate_feats_int,
											mutil_cate_feats, mutil_cate_feats_int, mutil_cate_feats_w)
				instances.append(instance)
	rng.shuffle(instances)
	return instances


def write_instance_to_example_files(output_files, instances):
	writers = []
	for output_file in output_files:
		writers.append(tf.python_io.TFRecordWriter(output_file))
	writer_index = 0
	total_written = 0
	for (inst_index, instance) in enumerate(instances):
		if len(instance.single_cate_feats) != SINGLE_CATE_NUM:
			continue
		features = collections.OrderedDict()
		features["label_click"] = _float_feature(instance.label_click)
		features["label_conv"] = _float_feature(instance.label_conv)
		features["cont_feats"] = _float_feature(instance.cont_feats)
		features["vector_feats"] = _float_feature(instance.vector_feats)
		for i in range(SINGLE_CATE_NUM):
			features["single_cate_feats_%d" % i] = _bytes_feature(instance.single_cate_feats[i])
			features["single_cate_feats_int_%d" % i] = _int64_feature(instance.single_cate_feats_int[i])
		features["mutil_cate_feats"] = _bytes_feature(instance.mutil_cate_feats)
		features["mutil_cate_feats_int"] = _int64_feature(instance.mutil_cate_feats_int)
		features["mutil_cate_feats_w"] = _float_feature(instance.mutil_cate_feats_w)
		tf_example = tf.train.Example(features=tf.train.Features(feature=features))
		writers[writer_index].write(tf_example.SerializeToString())
		writer_index = (writer_index + 1) % len(writers)
		total_written += 1
	tf.logging.info("get %d total sample" % total_written)
	for writer in writers:
		writer.close()


def transformdata(args):
	tf.logging.set_verbosity(tf.logging.INFO)
	rng = random.Random(args["random_seed"])
	input_files = []
	for input_pattern in args["input_file"].split(","):
		input_files.extend(tf.gfile.Glob(input_pattern))
	tf.logging.info("*** Reading from input files ***")
	for input_file in input_files:
		tf.logging.info("  %s", input_file)
	output_files = args["output_file"].split(",")
	tf.logging.info("*** Writing to output files ***")
	for output_file in output_files:
		tf.logging.info("  %s", output_file)
	instances = create_training_instances(input_files, rng)
	write_instance_to_example_files(output_files, instances)


def parse_args(arguments):
	import argparse
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--input_file",
		default=None,
		type=str,
		help="Input raw text file (or comma-separated list of files).")

	parser.add_argument(
		"--output_file",
		default=None,
		type=str,
		help="Output TF example file (or comma-separated list of files).")

	parser.add_argument(
		"--random_seed",
		default=12345,
		type=int,
		help="Random seed for data generation.")

	args = parser.parse_args(arguments)
	return args


if __name__ == "__main__":
	import sys

	# args = parse_args(sys.argv[1:])
	args={"input_file":"./traindatasets/libsvm_in.txt","random_seed":12345,"output_file":"./traindatasets/part-00000"}
	transformdata(args)
