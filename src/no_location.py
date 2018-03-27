#!/usr/bin/env python
# Basic language model for tweets without location data

from __future__ import division
import json
import numpy as np
import random
import sys
sys.path.append("/home/xbt504/nlp-project/src")

from pickle import load
from keras.layers import Input, Dense, LSTM, concatenate, Embedding, TimeDistributed, Masking
from keras.models import Model
from keras.utils import to_categorical, plot_model
from util import perplexity, decode
from tweet import Tweet

# Parameters
embedding_size = 64
lstm_size_1 = 128
lstm_size_2 = 128
epochs = 10
saveModel = True
batch_size = 100
test_set_ratio = 0.1
sequence_filename = 'sequences.txt'
max_tweet_length = 280
sequence_length = max_tweet_length + 1 

# Load data and mapping
data = map(decode, open(sequence_filename, 'r').read().splitlines())
data = map(lambda x: Tweet(x['sequence'], x['name'], ['lat'], ['lon']), data)
print 'Loaded %d sequences' % len(data)
char_mapping = load(open('char_mapping.pkl', 'rb'))
vocab_size = len(char_mapping)
print 'Vocabulary size: %d characters' % vocab_size

# Split into test and training data
testIndex = int(round(len(data) * (1 - test_set_ratio)))
random.shuffle(data)
testSet = data[testIndex:] # Test data
trainSet = data[:testIndex] # Training data
print "Size of training set: %d" % len(trainSet)
print "Size of test     set: %d" % len(testSet)

trainX = np.array(map(lambda x: x.getText(), trainSet))[:,:-1]
trainY = np.roll(np.array(map(lambda x: x.getText(), trainSet)), -1)[:,:-1].reshape(len(trainSet), sequence_length, 1)
testX = np.array(map(lambda x: x.getText(), testSet))[:,:-1]
testY = np.roll(np.array(map(lambda x: x.getText(), testSet)), -1)[:,:-1].reshape(len(testSet), sequence_length, 1)

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
def generator(x, y):
    i = 0
    while True:
        i = i + 1 % len(x)
        yield (np.array([x[i]]), to_categorical(np.array([y[i]]), num_classes=vocab_size))
model.fit_generator(generator(trainX, trainY), steps_per_epoch=len(trainSet), nb_epoch=epochs)
result = model.evaluate_generator(generator(testX, testY), steps=len(testX))
print str(model.metrics_names[0]) + ": " + str(result[0])
print str(model.metrics_names[1]) + ": " + str(result[1])

# Save model and mapping to file
if (saveModel):
    model.save('model_no_location.h5')

