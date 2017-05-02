# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

class TextCNN(object):
	def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
		# Placeholders for input, output and dropout
		self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
		self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
		self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

		# Keeping track of l2 regularization loss (optional)
		l2_loss = tf.constant(0.0)

		# Embedding layer，随机embedding
		with tf.device('/cpu:0'), tf.name_scope('embedding'):
			W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='W')
			self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
			#加一维，变成4维
			self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

		#卷积＋池化层 Create a convolution + maxpool layer for each filter size
		pooled_outputs = []
		for i, filter_size in enumerate(filter_sizes):
			with tf.name_scope('conv-maxpool-%s' % filter_size):
				# Convolution Layer，filter也是4维
				filter_shape = [filter_size, embedding_size, 1, num_filters]
				W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
				b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
				#二维卷积
				conv = tf.nn.conv2d(
					self.embedded_chars_expanded,
					W,
					strides=[1, 1, 1, 1],
					padding='VALID',
					name='conv')

				# Apply nonlinearity
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

				# Maxpooling over the outputs，也是四维
				pooled = tf.nn.max_pool(
					h,
					ksize=[1, sequence_length - filter_size + 1, 1, 1],
					strides=[1, 1, 1, 1],
					padding='VALID',
					name='pool')
				pooled_outputs.append(pooled)

		# Combine all the pooled features
		num_filters_total = num_filters * len(filter_sizes)
		#最后一维上连接，然后一维化
		self.h_pool = tf.concat(3, pooled_outputs)
		self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

		# Add dropout
		with tf.name_scope('dropout'):
			self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

		# Final (unnormalized) scores and predictions
		with tf.name_scope('output'):
			W = tf.get_variable(
				'W',
				shape=[num_filters_total, num_classes],
				initializer=tf.contrib.layers.xavier_initializer())
			b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
			l2_loss += tf.nn.l2_loss(W)
			l2_loss += tf.nn.l2_loss(b)
			self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')
			self.predictions = tf.argmax(self.scores, 1, name='predictions')

		# 计算交叉熵损失Calculate mean cross-entropy loss
		with tf.name_scope('loss'):
			losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
			self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

		# Accuracy
		with tf.name_scope('accuracy'):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')

		with tf.name_scope('num_correct'):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.num_correct = tf.reduce_sum(tf.cast(correct_predictions, 'float'), name='num_correct')