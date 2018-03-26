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
from split import splitData

embedding_size = 64
lstm_size_1 = 128
lstm_size_2 = 128
epochs = 100
saveModel = True
batch_size = 100
test_set_perc = 10.0
sequence_filename = 'sequences.txt'
sequence_length = 281

# Load data
data = map(decode, open(sequence_filename, 'r').read().splitlines())
print 'Loaded %d sequences' % len(data)

# Load mapping from file
char_mapping = load(open('char_mapping.pkl', 'rb'))
vocab_size = len(char_mapping)
print 'Vocabulary size: %d characters' % vocab_size

# Split into test and training data
testIndex = int(round(len(data) * ((100 - test_set_perc) / 100)))
random.shuffle(data)
testSet = data[testIndex:] # Test data
trainSet = data[:testIndex] # Training data
print "Size of training set: %d" % len(trainSet)
print "Size of test     set: %d" % len(testSet)

#ts = splitData(data, True, 0.1)

#print 'test set length: %d' % len(ts)
#print 'ratio: %f' % (len(ts) / len(data))

trainX = np.array(map(lambda x: x['sequence'], trainSet))[:,:-1]
trainY = np.roll(np.array(map(lambda x: x['sequence'], trainSet)), -1)[:,:-1].reshape(len(trainSet), sequence_length, 1)
testX = np.array(map(lambda x: x['sequence'], testSet))[:,:-1]
testY = np.roll(np.array(map(lambda x: x['sequence'], testSet)), -1)[:,:-1].reshape(len(testSet), sequence_length, 1)

in_layer = Input(shape=(sequence_length,))
mask = Masking(mask_value=char_mapping['<pad>'])(in_layer)
embedding = Embedding(output_dim=embedding_size, input_dim=vocab_size)(mask)
lstm = LSTM(lstm_size_1, return_sequences=True)(embedding)
lstm = LSTM(lstm_size_2, return_sequences=True)(lstm)
predictions = TimeDistributed(Dense(vocab_size, activation='relu'))(lstm)
model = Model(inputs=in_layer, outputs=predictions)
print model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[perplexity])

# Train and evaluate performance
model.fit(trainX, to_categorical(trainY), epochs=10, verbose=2)
result = model.evaluate(testX, to_categorical(testY))
print str(model.metrics_names[0]) + ": " + str(result[0])
print str(model.metrics_names[1]) + ": " + str(result[1])

# Save model and mapping to file
if (saveModel):
    model.save('model_no_location.h5')

