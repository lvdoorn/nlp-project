#!/usr/bin/env python
# Basic language model for tweets with full names as location data

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
def padLocations(tweets):
    trainLoc = np.zeros(sequences.shape)
    for i, sequence in enumerate(sequences):
        for j in range(len(sequence)):
            if sequences[i, j] == char_mapping['<pad>']:
                trainLoc[i, j] = name_mapping['<pad>']
            else:
                trainLoc[i, j] = locations[i]
    return trainLoc



def getModel():
    char_mapping = preprocessing.loadCharMapping()
    vocab_size = len(char_mapping)
    name_mapping = preprocessing.loadNameMapping()
    loc_size = len(name_mapping)
    sequence_input = Input(shape=(sequence_length,), name='sequence_input')
    mask = Masking(mask_value=char_mapping['<pad>'])(sequence_input)
    embedding = Embedding(output_dim=embedding_size, input_dim=vocab_size)(mask)

    location_input = Input(shape=(sequence_length,), name='location_input')
    mask = Masking(mask_value=name_mapping['<pad>'])(location_input)
    loc_embedding = Embedding(output_dim=embedding_size, input_dim=loc_size)(mask)

    merge = concatenate([embedding, loc_embedding])
    lstm = LSTM(lstm_size_1, return_sequences=True)(merge)
    lstm = LSTM(lstm_size_2, return_sequences=True)(lstm)
    predictions = TimeDistributed(Dense(vocab_size, activation='tanh'))(lstm)
    model = Model(inputs=[sequence_input, location_input], outputs=predictions)
    return model

def getGenerator(X, Y, tweets):
    vocab_size = len(preprocessing.getCharMapping())
    locations = map(lambda x: x.getFullName(), tweets)
    
    def generator():
        i = 0
        while True:
            i = (i + 1) % len(X)
            yield ([np.array([X[i]]), np.array([loc[i]])], to_categorical(np.array([Y[i]]), num_classes=vocab_size))






