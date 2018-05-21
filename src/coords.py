#!/usr/bin/env python
# Basic language model for tweets with coordinates as location data

import numpy as np
import sys # To run on the slurm cluster
sys.path.append("/home/xbt504/nlp-project/src")

import preprocessing
from keras.layers import Input, Dense, LSTM, concatenate, Embedding, TimeDistributed, Masking
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
    sequence_input = Input(shape=(sequence_length,), name='sequence_input')
    mask = Masking(mask_value=char_mapping['<pad>'])(sequence_input)
    embedding = Embedding(output_dim=embedding_size, input_dim=vocab_size)(mask)

    location_input = Input(shape=(sequence_length,2), name='location_input')

    merge = concatenate([embedding, location_input])
    lstm = LSTM(lstm_size_1, return_sequences=True)(merge)
    lstm = LSTM(lstm_size_2, return_sequences=True)(lstm)
    predictions = TimeDistributed(Dense(vocab_size, activation='tanh'))(lstm)
    model = Model(inputs=[sequence_input, location_input], outputs=predictions)
    return model
