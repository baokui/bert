import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
def int64List(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[values]))
# 生成字符串型属性
def bytesList(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
# 读取
max_seq_length = 128
reader = tf.TFRecordReader()
file_queue = tf.train.string_input_producer(['data/vpa_tfrecord/vpaTrn.tfrecord'])
_, example_parse = reader.read(file_queue)
example = tf.parse_single_example(example_parse,
                                  features={
                                      "input_ids":tf.FixedLenFeature([max_seq_length], tf.int64),
                                      "tip_ids": tf.FixedLenFeature([4], tf.int64)
                                  })
image_data = example['input_ids']
# 数据解码
image = tf.cast(image_data, tf.float32)

tips = example['tip_ids']
tips = tf.cast(tips,tf.float32)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    thresds = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(10):
        images,tips1 = sess.run([image,tips])
