# cnn_multi_text_classify

dataset: 20-Newsgroup

pretrain: $ python process_data.py 

to generate a file named mr.p

train: $ python train.py mr.p parameters.json

predict: $ python predict.py ./trained_model_directory/ new_data.file
