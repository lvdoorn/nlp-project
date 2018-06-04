#!/usr/bin/env python
# Basic language model for tweets without location data

import numpy as np
import sys # To run on the slurm cluster
sys.path.append("/home/xbt504/nlp-project/src")

import preprocessing
from keras.layers import Input, Dense, LSTM, Embedding, TimeDistributed, Masking
from keras.models import Model

# Parameters
embedding_size = 64
lstm_size_1 = 128
lstm_size_2 = 128
max_tweet_length = 149
sequence_length = max_tweet_length + 1 

def getModel():
    char_mapping = preprocessing.loadCharMapping()
    vocab_size = len(char_mapping)
    in_layer = Input(shape=(sequence_length,))
    mask = Masking(mask_value=char_mapping['<pad>'])(in_layer)
    embedding = Embedding(output_dim=embedding_size, input_dim=vocab_size)(mask)
    lstm = LSTM(lstm_size_1, return_sequences=True)(embedding)
    lstm = LSTM(lstm_size_2, return_sequences=True)(lstm)
    predictions = TimeDistributed(Dense(vocab_size, activation='tanh'))(lstm)
    model = Model(inputs=in_layer, outputs=predictions)
    return model

def getGenerator(trainX, trainY, tweets):
    vocab_size = 10 #len(char_mapping)
    def generator(x, y):
        i = 0
        while True:
            i = (i + 1) % len(x)
            yield (np.array([x[i]]), to_categorical(np.array([y[i]]), num_classes=vocab_size))
    return generator
