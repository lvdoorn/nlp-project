#!/usr/bin/env python
# Basic language model for tweets without location data

from __future__ import division
import json
import numpy as np
import random
import sys
sys.path.append("/home/xbt504/nlp-project/src")

from pickle import load
from keras.layers import Input, Dense, LSTM, concatenate, Embedding, TimeDistributed, Flatten, Reshape, Masking
from keras.models import Model
from keras.utils import to_categorical, plot_model
from util import perplexity, decode, prepareData, getText
from loaddata import getCharMapping, getNameMapping, loadData

# Parameters
embedding_size = 64
lstm_size_1 = 128
lstm_size_2 = 128
epochs = 100
saveModel = True
batch_size = 100
sequence_length = 281

# Load data and mappings from file
data = loadData()
char_mapping = getCharMapping()
vocab_size = len(char_mapping)
print 'Vocabulary size: %d characters' % vocab_size
name_mapping = getNameMapping()
loc_size = len(name_mapping)
print 'Amount of locations: %d' % loc_size
trainSet = data['trainSet']
testSet = data['testSet']
print "Size of training set: %d" % len(trainSet)
print "Size of test     set: %d" % len(testSet)

trainSequences = np.array(map(lambda x: x.getText(), trainSet))
trainX = trainSequences[:,:-1]
trainLoc = np.array(map(lambda x: x.getFullName(), trainSet))
trainLoc = np.vstack([[trainLoc] for _ in range(sequence_length)]).T #TODO: This runs, but needs to be masked
trainY = np.roll(trainSequences, -1)[:,:-1].reshape(len(trainSet), sequence_length, 1)

testSequences = np.array(map(lambda x: x.getText(), testSet))
testX = testSequences[:,:-1]
testLoc = np.array(map(lambda x: x.getFullName(), testSet))
testLoc = np.vstack([[testLoc] for _ in range(sequence_length)]).T #TODO: This runs, but needs to be masked
testY = np.roll(testSequences, -1)[:,:-1].reshape(len(testSet), sequence_length, 1)

sequence_input = Input(shape=(sequence_length,), name='sequence_input')
mask = Masking(mask_value=char_mapping['<pad>'])(sequence_input)
embedding = Embedding(output_dim=embedding_size, input_dim=vocab_size)(mask)

location_input = Input(shape=(sequence_length,), name='location_input') #TODO: Make sure this is masked similarly to normal input (currently not fixed)
loc_embedding = Embedding(output_dim=embedding_size, input_dim=vocab_size)(location_input)

merge = concatenate([embedding, loc_embedding])
lstm = LSTM(lstm_size_1, return_sequences=True)(merge)
lstm = LSTM(lstm_size_2, return_sequences=True)(lstm)
predictions = TimeDistributed(Dense(vocab_size, activation='relu'))(lstm)
model = Model(inputs=[sequence_input, location_input], outputs=predictions)
print model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[perplexity])

# Train and evaluate performance
model.fit([trainX, trainLoc], to_categorical(trainY), epochs=10, verbose=2)
result = model.evaluate([testX, testLoc], to_categorical(testY))
print str(model.metrics_names[0]) + ": " + str(result[0])
print str(model.metrics_names[1]) + ": " + str(result[1])

# Save model and mapping to file
if (saveModel):
    model.save('model_full_name.h5')

