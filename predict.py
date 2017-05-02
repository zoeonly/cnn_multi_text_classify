import os
import sys
import json
import logging
import data_helper
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import learn

logging.getLogger().setLevel(logging.INFO)

def predict_unseen_data():
	"""Step 0: load trained model and parameters"""
	params = json.loads(open('./parameters.json').read())
	checkpoint_dir = sys.argv[1]
	if not checkpoint_dir.endswith('/'):
		checkpoint_dir += '/'
	checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir + 'checkpoints')
	logging.critical('Loaded the trained model: {}'.format(checkpoint_file))

	"""Step 1: load data for prediction"""
	test_file = sys.argv[2]
	x_test, y_test, revs, labels = data_helper.load_data_and_labels(test_file)
	logging.info('The number of x_test: {}'.format(len(x_test)))
	logging.info('The number of y_test: {}'.format(len(y_test)))

	vocab_path = os.path.join(checkpoint_dir, "vocab.pickle")
	vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
	x_test = np.array(list(vocab_processor.transform(x_test)))

	"""Step 2: compute the predictions"""
	graph = tf.Graph()
	with graph.as_default():
		#配置并初始化
		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		sess = tf.Session(config=session_conf)

		with sess.as_default():
			saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
			saver.restore(sess, checkpoint_file)

			# Get the placeholders from the graph by name
			input_x = graph.get_operation_by_name("input_x").outputs[0]
			dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

			# Tensors we want to evaluate
			predictions = graph.get_operation_by_name("output/predictions").outputs[0]

			# Generate batches for one epoch
			batches = data_helper.batch_iter(list(x_test), params['batch_size'], 1, shuffle=False)
			
			# Collect the predictions here
			all_predictions = []
			for x_test_batch in batches:
				batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
				all_predictions = np.concatenate([all_predictions, batch_predictions])

	# Print accuracy if y_test is defined
	if y_test is not None:
		y_test = np.argmax(y_test, axis=1)
		correct_predictions = sum(all_predictions == y_test)
		logging.critical('The accuracy is: {}'.format(correct_predictions / float(len(y_test))))

	# Save the evaluation to a csv
	predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
	out_path = os.path.join(checkpoint_dir, "..", "prediction.csv")
	logging.critical("Saving evaluation to {0}".format(out_path))
	with open(out_path, 'w') as f:
		csv.writer(f).writerows(predictions_human_readable)

if __name__ == '__main__':
	# python predict.py ./trained_model_1488201627/ mr_pre.p
	predict_unseen_data()
