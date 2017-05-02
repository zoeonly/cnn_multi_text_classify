# -*- coding: utf-8 -*-
import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
import os
import json

# from tensorflow.contrib import learn

vector_size = 300 #一个单词向量的长度
 
def build_data_cv_multi(data_folder, clean_string = False):
    """
    多分类数据预处理，y为类标签
    """
    revs = []
    labels = json.loads(open('./labels.json').read())
    for sub_folder in os.listdir(data_folder):
        if sub_folder in labels:
            print "now process %s"%sub_folder
            print "%s has %s files"%(sub_folder,str(len(os.listdir(data_folder+"/"+sub_folder))))
            for filename in os.listdir(data_folder+"/"+sub_folder):    		
                with open(data_folder + "/" + sub_folder + "/" + filename, "rb") as f:
                    rev = []
                    for line in f: 
                        rev.append(line.strip())
                    if clean_string:
                        orig_rev = clean_str(" ".join(rev))
                    else:
                        orig_rev = " ".join(rev).lower()
                    datum  = {"y": sub_folder, 
                    "text": orig_rev
                    }
                    revs.append(datum)
            print "process %s finished" % sub_folder
    return revs

def add_unknown_words(word_vecs, vocab, min_revs=1, k=vector_size):
    """
    For words that occur in at least min_revs documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    未登录单词的向量随机初始化，-0.25,0.25
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_revs:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


if __name__=="__main__":    
    w2v_file = ""     
    data_folder = "data/20news-18828"
    revs = build_data_cv_multi(data_folder, clean_string=True) 
    print "data loaded!"
    cPickle.dump(revs, open("mr.p", "wb"))
    print "dataset created!"
    
