# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
import json
import numpy as np
import sys

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_ids,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class XnliProcessor(DataProcessor):
    """Processor for the XNLI data set."""

    def __init__(self):
        self.language = "zh"

    def get_train_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(
            os.path.join(data_dir, "multinli",
                         "multinli.train.%s.tsv" % self.language))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "train-%d" % (i)
            text_a = tokenization.convert_to_unicode(line[0])
            text_b = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[2])
            if label == tokenization.convert_to_unicode("contradictory"):
                label = tokenization.convert_to_unicode("contradiction")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "xnli.dev.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "dev-%d" % (i)
            language = tokenization.convert_to_unicode(line[0])
            if language != tokenization.convert_to_unicode(self.language):
                continue
            text_a = tokenization.convert_to_unicode(line[6])
            text_b = tokenization.convert_to_unicode(line[7])
            label = tokenization.convert_to_unicode(line[1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
            text_a = tokenization.convert_to_unicode(line[8])
            text_b = tokenization.convert_to_unicode(line[9])
            if set_type == "test":
                label = "contradiction"
            else:
                label = tokenization.convert_to_unicode(line[-1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[3])
            text_b = tokenization.convert_to_unicode(line[4])
            if set_type == "test":
                label = "0"
            else:
                label = tokenization.convert_to_unicode(line[0])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # Only the test set has a header
            if set_type == "test" and i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_a = tokenization.convert_to_unicode(line[1])
                label = "0"
            else:
                text_a = tokenization.convert_to_unicode(line[3])
                label = tokenization.convert_to_unicode(line[1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class sortProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def _read_txt(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
            lines = [t.split('\t') for t in lines]
        return lines

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir, idx_label):
        """See base class."""
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, "dev.txt")), "dev", idx_label)

    def get_test_examples(self, data_dir, idx_label):
        """See base class."""
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, "test.txt")), "test", idx_label)

    def get_labels(self, D_label):
        """See base class."""
        return [[str(i) for i in range(len(D_label[k]))] for k in D_label]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[0])
            text_b = tokenization.convert_to_unicode(line[1])
            if set_type == "test":
                labels = "0"
            else:
                labels = tokenization.convert_to_unicode(line[-1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=labels))
        return examples


def convert_single_example(ex_index, example, label_lists, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_ids=[0],
            is_real_example=False)

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokensA = []
    tokensB = []
    segment_ids = [0 for _ in range(max_seq_length)]
    tokensA.append("[CLS]")
    tokensB.append("[CLS]")
    for token in tokens_a:
        tokensA.append(token)
    tokensA.append("[SEP]")
    for token in tokens_b:
        tokensB.append(token)
    tokensB.append("[SEP]")

    input_ids = [tokenizer.convert_tokens_to_ids(tokensA),tokenizer.convert_tokens_to_ids(tokensB)]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_maskA = [1] * len(input_ids[0])
    input_maskB = [1] * len(input_ids[1])

    # Zero-pad up to the sequence length.
    while len(input_ids[0]) < max_seq_length:
        input_ids[0].append(0)
        input_maskA.append(0)
    while len(input_ids[1]) < max_seq_length:
        input_ids[1].append(0)
        input_maskB.append(0)

    assert len(input_ids[0]) == max_seq_length
    assert len(input_ids[1]) == max_seq_length
    assert len(input_maskA) == max_seq_length
    assert len(input_maskB) == max_seq_length
    assert len(segment_ids) == max_seq_length
    label_ids = int(example.label)
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=[input_maskA,input_maskB],
        segment_ids=segment_ids,
        label_ids=label_ids,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(
        examples, label_lists, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""
    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, label_lists,
                                         max_seq_length, tokenizer)
        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f
        features = collections.OrderedDict()
        features["input_idsA"] = create_int_feature(feature.input_ids[0])
        features["input_idsB"] = create_int_feature(feature.input_ids[1])
        features["input_maskA"] = create_int_feature(feature.input_mask[0])
        features["input_maskB"] = create_int_feature(feature.input_mask[1])
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_ids])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_idsA": tf.FixedLenFeature([seq_length], tf.int64),
        "input_idsB": tf.FixedLenFeature([seq_length], tf.int64),
        "input_maskA": tf.FixedLenFeature([seq_length], tf.int64),
        "input_maskB": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }
    name_to_features["label_ids"] = tf.FixedLenFeature([], tf.int64)

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def cosine(q,a):
    pooled_len_1 = tf.sqrt(tf.reduce_sum(q * q, 1))
    pooled_len_2 = tf.sqrt(tf.reduce_sum(a * a, 1))
    pooled_mul_12 = tf.reduce_sum(q * a, 1)
    score = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 +1e-8, name="scores")
    return score

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, use_one_hot_embeddings):
    """Creates a classification model."""
    output_layer = []
    for k in range(len(input_ids)):
        with tf.variable_scope('lm', reuse=k > 0):
            model = modeling.BertModel(
                config=bert_config,
                is_training=is_training,
                input_ids=input_ids[k],
                input_mask=input_mask[k],
                token_type_ids=segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings)
            output_layer.append(model.get_pooled_output())
    feature0 = output_layer[0]
    feature1 = output_layer[1]
    feature_qr = tf.layers.dense(inputs=output_layer[0], units = 256, activation = tf.nn.relu)
    feature_dc = tf.layers.dense(inputs=output_layer[1], units = 256, activation = tf.nn.relu)
    print("TEST:")
    score = cosine(feature_qr,feature_dc)
    print(score,labels)
    c = tf.square(score - tf.cast(labels, dtype=tf.float32))
    per_example_loss = tf.reduce_mean(c,axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss,score,per_example_loss,feature_qr,feature_dc,feature0,feature1)

def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  import re,collections
  """Compute the union of the current variables and checkpoint variables."""
  initialized_variable_names = {}
  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var
  init_vars = tf.train.list_variables(init_checkpoint)
  print('init_variable:',init_vars)
  print('train_variable:',tvars)
  print('name_to_variable',name_to_variable)
  assignment_map = collections.OrderedDict()
  vars_others = []
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if 'lm/'+name in name_to_variable:
      assignment_map[name] = 'lm/'+name
    elif name in name_to_variable:
      assignment_map[name] = name
    else:
      vars_others.append(name)
      continue
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1
  return (assignment_map, initialized_variable_names,vars_others)

def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        print(features)
        input_ids = [features["input_idsA"],features["input_idsB"]]
        input_mask = [features["input_maskA"],features["input_maskB"]]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        (loss,score,per_example_loss,feature_qr,feature_dc,feature0,feature1) = create_model(bert_config, is_training, input_ids, input_mask, segment_ids,label_ids, use_one_hot_embeddings)
        total_loss = loss
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            print('TEST:assignment_map0',assignment_map)
            if len(assignment_map)==0:
                (assignment_map, initialized_variable_names, vars_others
                 ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
                print("assignment_map", assignment_map)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        return output_spec
    return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder,Num_labels):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d0 = {
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
        }
        d0["label_ids"] = tf.constant(all_label_ids, shape=[num_examples], dtype=tf.float32)
        d = tf.data.Dataset.from_tensor_slices(d0)

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return features

def netConfig():
    import numpy as np
    print('参数量：%d'%(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
    tvars = [v for v in tf.trainable_variables()]
    for t in tvars:
        print(t)
def main(_):
    FLAGS.data_dir = "/search/odin/guobk/vpa/vpa-studio-research/sort/data"
    FLAGS.task_name = "sort"
    FLAGS.bert_config_file = "/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/bert_config.json"
    FLAGS.vocab_file = "/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/vocab.txt"
    FLAGS.init_checkpoint = "/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/bert_model.ckpt"
    FLAGS.output_dir = "model/" + FLAGS.task_name
    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
        "xnli": XnliProcessor,
        "sort": sortProcessor,
    }
    label_lists = ['0','1']
    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_lists,FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True
        )
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

class bert_cls:
    def __init__(self,task_name='all',gpu='5'):
        #os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        data_dir = "/search/odin/guobk/vpa/vpa-studio-research/labelClassify/DataLabel"
        bert_config_file = "/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/bert_config.json"
        vocab_file = "/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/vocab.txt"
        self.max_seq_length = 128
        output_dir = "model/label/" + task_name
        path_map = os.path.join(data_dir, 'map_index.json')
        path_alpha = os.path.join(data_dir, 'label_alpha.json')
        D_map = json.load(open(path_map, 'r'))
        D_alpha0 = json.load(open(path_alpha, 'r'))
        D_alpha = {k: [D_alpha0[k][kk] for kk in D_map[k]] for k in D_map}
        self.L0 = ['使用场景P0', '表达对象P0', '表达者性别倾向P0', '文字风格']
        self.D_map_inv = [{D_map[self.L0[i]][t]:t for t in D_map[self.L0[i]]} for i in range(len(self.L0))]
        tf.logging.set_verbosity(tf.logging.INFO)
        is_training = False
        processors = {
            "cola": ColaProcessor,
            "mnli": MnliProcessor,
            "mrpc": MrpcProcessor,
            "xnli": XnliProcessor,
            "all": LabelClass
        }
        bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        processor = processors[task_name.lower()]()
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)
        tf.reset_default_graph()
        self.input_ids = tf.placeholder(tf.int32, shape=[None, self.max_seq_length], name='input_ids')
        self.input_mask = tf.placeholder(tf.int32, shape=[None, self.max_seq_length], name='input_mask')
        self.segment_ids = tf.placeholder(tf.int32, shape=[None, self.max_seq_length], name='segment_ids')
        self.label_lists = processor.get_labels(D_alpha)
        self.Num_labels = [len(label_list) for label_list in self.label_lists]
        self.Label_ids = [tf.placeholder(tf.int32, shape=[None, ], name="label_ids_" + str(i)) for i in
                          range(len(self.Num_labels))]
        (Loss, LossList, LogitsList, self.ProbList, Per_example_loss) = create_model(
            bert_config, is_training, self.input_ids, self.input_mask, self.segment_ids, self.Label_ids,
            self.Num_labels, use_one_hot_embeddings=False)
        self.saver = tf.train.Saver(max_to_keep=None)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        init_checkpoint = tf.train.latest_checkpoint(output_dir)
        self.saver.restore(self.session, init_checkpoint)
    def predict(self,inputStr):
        example = InputExample(guid='guid', text_a=inputStr, label=['0' for _ in range(len(self.Num_labels))])
        feature = convert_single_example(10, example, self.label_lists, self.max_seq_length, self.tokenizer)
        X_input_ids = feature.input_ids
        X_segment_ids = feature.segment_ids
        X_input_mask = feature.input_mask
        feed_dict = {self.input_ids: [X_input_ids], self.segment_ids: [X_segment_ids],
                     self.input_mask: [X_input_mask]}
        P = self.session.run(self.ProbList, feed_dict=feed_dict)
        P = [p[0] for p in P]
        R = {}
        for i in range(len(P)):
            p = P[i]
            idx = np.argmax(p)
            label = self.D_map_inv[i][idx]
            score = np.float(p[idx])
            p = list(p)
            result = {self.D_map_inv[i][j]: np.float(p[j]) for j in range(len(p))}
            R[self.L0[i]] = [label,score,result]
        return R,P
def test(S,init='bert'):
    tf.reset_default_graph()
    FLAGS.data_dir = "/search/odin/guobk/vpa/vpa-studio-research/sort/data"
    FLAGS.task_name = "sort"
    FLAGS.bert_config_file = "/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/bert_config.json"
    FLAGS.vocab_file = "/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/vocab.txt"
    FLAGS.init_checkpoint = "/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/bert_model.ckpt"
    FLAGS.output_dir = "model/" + FLAGS.task_name
    label_lists = ['0', '1']
    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,FLAGS.init_checkpoint)
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    is_training = False
    max_seq_length = FLAGS.max_seq_length
    input_ids = [tf.placeholder(tf.int32, shape=[None, max_seq_length], name='input_idsA'),tf.placeholder(tf.int32, shape=[None, max_seq_length], name='input_idsB')]
    input_mask = [tf.placeholder(tf.int32, shape=[None, max_seq_length], name='input_maskA'),tf.placeholder(tf.int32, shape=[None, max_seq_length], name='input_maskB')]
    segment_ids = tf.placeholder(tf.int32, shape=[None, max_seq_length], name='segment_ids')
    labels = tf.placeholder(tf.int32, shape=[None, ], name='labels')
    loss,score,per_example_loss,feature_qr,feature_dc,feature0,feature1 = create_model(bert_config, is_training, input_ids, input_mask, segment_ids,labels, use_one_hot_embeddings=False)
    sess = tf.Session()
    saver = tf.train.Saver()
    model_file = tf.train.latest_checkpoint(FLAGS.output_dir)
    if model_file and init!='bert':
        print('load model from checkpoint')
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_file)
    else:
        print('load model from bert init')
        tvars = tf.trainable_variables()
        (assignment_map, initialized_variable_names, vars_others
         ) = get_assignment_map_from_checkpoint(tvars, FLAGS.init_checkpoint)
        tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)
        sess.run(tf.global_variables_initializer())
    # with open(os.path.join(FLAGS.data_dir,'test.txt'),'r') as f:
    #     S = f.read().strip().split('\n')
    # loss0 = []
    # score0 = []
    # for i in range(len(S)):
    #     s = S[i].split('\t')
    #     text_a = s[0]
    #     text_b = s[1]
    #     label = int(s[2])
    #     example = InputExample(guid='guid', text_a=text_a, text_b = text_b, label='0')
    #     feature = convert_single_example(10, example, label_lists, max_seq_length,tokenizer)
    #     feed_dict = {input_ids[0]: [feature.input_ids[0]],
    #                  input_ids[1]: [feature.input_ids[1]],
    #                  segment_ids: [feature.segment_ids],
    #                  input_mask[0]: [feature.input_mask[0]],
    #                  input_mask[1]: [feature.input_mask[1]],
    #                  labels:[label]}
    #     y_qr,y_dc,y_score,loss_ = sess.run([feature_qr,feature_dc,score,loss], feed_dict=feed_dict)
    #     score0.append(y_score[0])
    #     loss0.append(loss_)
    #     if i%100==0:
    #         print(i,len(S),np.mean(loss0))
    f0 = feature0 if init=='bert' else feature_qr
    f1 = feature1 if init=='bert' else feature_dc
    Q = [s.split('\t')[0] for s in S]
    D = [s.split('\t')[1] for s in S]
    Q = list(set(Q))
    D = list(set(D))
    Y_dc = []
    for i in range(len(D)):
        text_a = '我'
        text_b = D[i]
        example = InputExample(guid='guid', text_a=text_a, text_b=text_b, label='0')
        feature = convert_single_example(10, example, label_lists, max_seq_length, tokenizer)
        feed_dict = {input_ids[0]: [feature.input_ids[0]],
                     input_ids[1]: [feature.input_ids[1]],
                     segment_ids: [feature.segment_ids],
                     input_mask[0]: [feature.input_mask[0]],
                     input_mask[1]: [feature.input_mask[1]]}
        y_dc = sess.run(f1, feed_dict=feed_dict)
        y_dc = y_dc[0]
        Y_dc.append(y_dc)
        if i % 100 == 0:
            print(i, len(D))
    Y_qr = []
    for i in range(len(Q)):
        text_b = '我'
        text_a = Q[i]
        example = InputExample(guid='guid', text_a=text_a, text_b=text_b, label='0')
        feature = convert_single_example(10, example, label_lists, max_seq_length, tokenizer)
        feed_dict = {input_ids[0]: [feature.input_ids[0]],
                     input_ids[1]: [feature.input_ids[1]],
                     segment_ids: [feature.segment_ids],
                     input_mask[0]: [feature.input_mask[0]],
                     input_mask[1]: [feature.input_mask[1]]}
        y_qr = sess.run(f0, feed_dict=feed_dict)
        y_qr = y_qr[0]
        Y_qr.append(y_qr)
        if i % 100 == 0:
            print(i, len(Q))
    return Y_qr,Y_dc
def demo():
    FLAGS.data_dir = "/search/odin/guobk/vpa/vpa-studio-research/sort/data"
    with open(os.path.join(FLAGS.data_dir,'test.txt'),'r') as f:
        S = f.read().strip().split('\n')
    Y_qr0, Y_dc0 = test(S,init='bert')
    Y_qr,Y_dc = test(S,init='train')
    np.save('model/sort/Y_qr0.npy',Y_qr0)
    np.save('model/sort/Y_qr.npy', Y_qr)
    np.save('model/sort/Y_dc0.npy', Y_dc0)
    np.save('model/sort/Y_dc.npy', Y_dc)

if __name__ == "__main__":
    # flags.mark_flag_as_required("data_dir")
    # flags.mark_flag_as_required("task_name")
    # flags.mark_flag_as_required("vocab_file")
    # flags.mark_flag_as_required("bert_config_file")
    # flags.mark_flag_as_required("output_dir")
    #tf.app.run()
    if sys.argv[1]=='test':
        demo()
    else:
        main(0)
