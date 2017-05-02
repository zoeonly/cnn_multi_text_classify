# -*- coding: utf-8 -*-
import re
import logging
import numpy as np
import pandas as pd
from collections import Counter
import cPickle
import json

def load_data_and_labels(filename):
	"""Load sentences and labels"""
	revs = cPickle.load(open(filename,"rb"))
	# 读取出预处理后的数据 revs {"y":label,"text":"word1 word2 ...","num_words":..,"split":随机数（0-cv）}
	print "data loaded!"
	revs = np.random.permutation(revs) #原始的sample正负样本是分别聚在一起的，这里随机打散

	# Map the actual labels to one hot labels
	labels = json.loads(open('./labels.json').read())
	one_hot = np.zeros((len(labels), len(labels)), int)
	np.fill_diagonal(one_hot, 1)
	label_dict = dict(zip(labels, one_hot))
	x_raw = []
	y_raw = []
	for i in range(len(revs)):
		#print len(revs[i]["text"].split(" "))
		if len(revs[i]["text"].split(" ")) < 500:
			x_raw.append(revs[i]["text"])
			y_raw.append(label_dict[revs[i]["y"]])
	return x_raw, y_raw, revs, labels

def batch_iter(data, batch_size, num_epochs, shuffle=True):
	"""Iterate the data batch by batch"""
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int(data_size / batch_size) + 1

	for epoch in range(num_epochs):
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data

		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]

if __name__ == '__main__':
	input_file = './mr.p'
	load_data_and_labels(input_file)
