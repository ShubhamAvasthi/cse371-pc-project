import tflearn
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
import tensorflow as tf
import os
os.environ['KERAS_BACKEND']='tensorflow'
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, concatenate, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model,Sequential
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers, optimizers

# Main Model Compile
def blstm(inp_dim,vocab_size, embed_size, num_classes, learn_rate):   
    model = Sequential()
    model.add(Embedding(vocab_size, embed_size, input_length=inp_dim, trainable=True))
    model.add(Dropout(0.25))
    model.add(Bidirectional(LSTM(embed_size, activation='tanh', recurrent_activation='sigmoid', return_sequences = True)))
    model.add(Bidirectional(LSTM(embed_size, activation='tanh', recurrent_activation='sigmoid', return_sequences = True)))
    model.add(Bidirectional(LSTM(embed_size, activation='tanh', recurrent_activation='sigmoid')))
    model.add(Dropout(0.50))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    model.summary()
    return model


# Model Call
def get_model(m_type,inp_dim, vocab_size, embed_size, num_classes, learn_rate):
    model = blstm(inp_dim, vocab_size, embed_size, num_classes, learn_rate)
    return model


# Extracting the intermidiate output of the BiLSTM
def feature(model, layer_idx, X_batch):
	get_activations = K.function([model.layers[0].input, K.learning_phase()],[model.layers[layer_idx].output,])
	activations = get_activations([X_batch,0])
	return activations 


