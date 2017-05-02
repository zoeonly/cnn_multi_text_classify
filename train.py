# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import logging
import data_helper
import numpy as np
import tensorflow as tf
from text_cnn import TextCNN
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split

logging.getLogger().setLevel(logging.INFO)

def train_cnn():
	"""Step 0: 加载数据和参数"""
	train_file = sys.argv[1]
	x_raw, y_raw, _, labels = data_helper.load_data_and_labels(train_file)

	parameter_file = sys.argv[2]
	params = json.loads(open(parameter_file).read())

	"""Step 1: 完成单词到ID的映射,一行为一个sequence"""
	max_document_length = max([len(x.split(' ')) for x in x_raw])
	logging.info('The maximum length of all sentences: {}'.format(max_document_length))
	vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
	x = np.array(list(vocab_processor.fit_transform(x_raw)))
	y = np.array(y_raw)

	"""Step 2: _x和测试集"""
	x_, x_test, y_, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

	"""Step 3: 将_x分为训练集和验证集"""
	shuffle_indices = np.random.permutation(np.arange(len(y_)))
	x_shuffled = x_[shuffle_indices]
	y_shuffled = y_[shuffle_indices]
	x_train, x_dev, y_train, y_dev = train_test_split(x_shuffled, y_shuffled, test_size=0.1)

	"""Step 4: save the labels into labels.json since predict.py needs it"""
	with open('./labels.json', 'w') as outfile:
		json.dump(labels, outfile, indent=4)

	logging.info('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
	logging.info('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))

#-------------------------------------------------------------------------------------------------------------------

	"""Step 5: build a graph and cnn object"""
	graph = tf.Graph()
	with graph.as_default():
		# tf.ConfigProto一般用在创建session的时候。用来对session进行参数配置
		#log_device_placement=True : 是否打印设备分配日志
		#allow_soft_placement=True ： 如果你指定的设备不存在，允许TF自动分配设备
		session_conf = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
		sess = tf.Session(config = session_conf)
		with sess.as_default():
			cnn = TextCNN(
				sequence_length = x_train.shape[1],
				num_classes = y_train.shape[1],
				vocab_size = len(vocab_processor.vocabulary_),
				embedding_size = params['embedding_dim'],
				filter_sizes = list(map(int, params['filter_sizes'].split(","))),
				num_filters = params['num_filters'],
				l2_reg_lambda = params['l2_reg_lambda'])

			#学习率
			global_step = tf.Variable(0, name="global_step", trainable=False)
			learning_rate = tf.train.exponential_decay(1e-3, global_step, 1000, 0.99, staircase=True) 
			
			#优化器,选用最优的adam,速度快，效果稳定 可以直接这样用：train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)
			optimizer = tf.train.AdamOptimizer(learning_rate)
			grads_and_vars = optimizer.compute_gradients(cnn.loss)
			train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
			
			#保存模型和断点的路径
			timestamp = str(int(time.time()))
			out_dir = os.path.abspath(os.path.join(os.path.curdir, "trained_model_" + timestamp))
		
			checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
			checkpoint_prefix = os.path.join(checkpoint_dir, "model")
			if not os.path.exists(checkpoint_dir):
				os.makedirs(checkpoint_dir)

			saver = tf.train.Saver(tf.global_variables())

			# One training step: train the model with one batch
			def train_step(x_batch, y_batch, train_summary_op):
				feed_dict = {
					cnn.input_x: x_batch,
					cnn.input_y: y_batch,
					cnn.dropout_keep_prob: params['dropout_keep_prob']}
				_, step, train_summary = sess.run([train_op, global_step, train_summary_op], feed_dict)
				return train_summary

			# One evaluation step: evaluate the model with one batch
			def dev_step(x_batch, y_batch, dev_summary_op):
				feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: 1.0}
				step, dev_summary, num_correct = sess.run([global_step, dev_summary_op, cnn.num_correct], feed_dict)
				return dev_summary, num_correct

			# 可视化
			loss_summary = tf.summary.scalar("loss", cnn.loss) 
			acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
			train_summary_op = tf.summary.merge([loss_summary, acc_summary])
			dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
			train_summary_dir = os.path.join(out_dir, "logs", "train")
			dev_summary_dir = os.path.join(out_dir, "logs", "dev")

			sess.run(tf.global_variables_initializer())
			# 保存单词表用于预测Save the word_to_id map since predict.py needs it
			vocab_processor.save(os.path.join(out_dir, "vocab.pickle"))
			# 可视化
			train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
			dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

			# Training starts here
			train_batches = data_helper.batch_iter(list(zip(x_train, y_train)), params['batch_size'], params['num_epochs'])
			best_accuracy, best_at_step = 0, 0

			"""Step 6: 训练train the cnn model with x_train and y_train (batch by batch)"""
			for train_batch in train_batches:
				x_train_batch, y_train_batch = zip(*train_batch)
				train_summary = train_step(x_train_batch, y_train_batch, train_summary_op)
			
				current_step = tf.train.global_step(sess, global_step)

				if current_step % 100 == 0:
					train_summary_writer.add_summary(train_summary, current_step)

				"""Step 6.1: 用验证集评价模型evaluate the model with x_dev and y_dev (batch by batch)"""
				if current_step % params['evaluate_every'] == 0:
					dev_batches = data_helper.batch_iter(list(zip(x_dev, y_dev)), params['batch_size'], 1)
					total_dev_correct = 0
					for dev_batch in dev_batches:
						x_dev_batch, y_dev_batch = zip(*dev_batch)
						dev_summary, num_dev_correct = dev_step(x_dev_batch, y_dev_batch, dev_summary_op)
						total_dev_correct += num_dev_correct
						dev_summary_writer.add_summary(dev_summary, current_step)

					dev_accuracy = float(total_dev_correct) / len(y_dev)
					logging.critical('Accuracy on dev set: {}'.format(dev_accuracy))

					"""Step 6.2:保存模型save the model if it is the best based on accuracy of the dev set"""
					if dev_accuracy >= best_accuracy:
						best_accuracy, best_at_step = dev_accuracy, current_step
						path = saver.save(sess, checkpoint_prefix, global_step=current_step)
						logging.critical('Saved model {} at step {}'.format(path, best_at_step))
						logging.critical('Best accuracy {} at step {}'.format(best_accuracy, best_at_step))

			"""Step 7: 预测predict x_test (batch by batch)"""
			test_batches = data_helper.batch_iter(list(zip(x_test, y_test)), params['batch_size'], 1)
			total_test_correct = 0
			for test_batch in test_batches:
				x_test_batch, y_test_batch = zip(*test_batch)
				_, num_test_correct = dev_step(x_test_batch, y_test_batch, dev_summary_op)
				total_test_correct += num_test_correct

			test_accuracy = float(total_test_correct) / len(y_test)
			logging.critical('Accuracy on test set is {} based on the best model {}'.format(test_accuracy, path))
			logging.critical('complete!')

if __name__ == '__main__':
	# python train.py ./mr.p ./parameters.json
	train_cnn()
