#!/usr/bin/env python
# Basic language model for tweets without location data

import json
import numpy as np
import random
import sys
sys.path.append("/home/xbt504/nlp-project/src")

from pickle import load
from keras.layers import Input, Dense, LSTM, concatenate, Embedding
from keras.models import Model
from keras.utils import to_categorical, plot_model
from util import perplexity, decode, prepareData

embedding_size = 64
lstm_size_1 = 30
lstm_size_2 = 30
epochs = 100
saveModel = True
batch_size = 100
test_set_perc = 10.0
sequence_filename = 'sequences.txt'
sequence_length = 10

# Load data
data = map(decode, open(sequence_filename, 'r').read().splitlines())
print 'Loaded %d sequences' % len(data)

# Load mapping from file
char_mapping = load(open('char_mapping.pkl', 'rb'))
vocab_size = len(char_mapping)
print 'Vocabulary size: %d characters' % vocab_size

# Split into test and training data
testIndex = int(round(len(data) * ((100 - test_set_perc) / 100)))
#random.shuffle(data)
testSet = data[testIndex:] # Test data
trainSet = data[:testIndex] # Training data
print "Size of training set: %d" % len(trainSet)
print "Size of test     set: %d" % len(testSet)

# Make model
sequence_input = Input(shape=(sequence_length,), name='sequence_input')
embedding = Embedding(output_dim=embedding_size, input_dim=vocab_size)(sequence_input)
lstm = LSTM(lstm_size_1, return_sequences=True)(embedding)
lstm = LSTM(lstm_size_2)(lstm)
x = Dense(32, activation='relu')(lstm)
predictions = Dense(vocab_size, activation='softmax')(x)
model = Model(inputs=sequence_input, outputs=predictions)
print model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[perplexity])

# Train and evaluate performance
trainData = prepareData(trainSet, vocab_size)
model.fit(trainData['X'], trainData['y'], epochs=5, verbose=2)

testData = prepareData(testSet, vocab_size)
result = model.evaluate(testData['X'], testData['y'])
print str(model.metrics_names[0]) + ": " + str(result[0])
print str(model.metrics_names[1]) + ": " + str(result[1])

# Save model and mapping to file
if (saveModel):
    model.save('model_no_location.h5')
