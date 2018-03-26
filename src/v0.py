#!/usr/bin/env python
# Basic language model for tweets without location data

import json
import numpy as np
import random

from pickle import dump
from keras.layers import Input, Dense, LSTM, concatenate, Embedding
from keras.models import Model
from keras.utils import to_categorical, plot_model

from preprocessing import removeAutomated, removeEnglish
from util import getText, getMapping

sequence_length = 10
lstm_size_1 = 30
lstm_size_2 = 30
epochs = 10
visualization = True
visualizationFilename = 'model_v0.png'
saveModelAndMapping = True
batch_size = 100
test_set_perc = 10.0

# Load data
file_da = '../data/tweets_collected_da.jsonl'
file_sv = '../data/tweets_collected_sv.jsonl'
file_no = '../data/tweets_collected_no.jsonl'
da = open(file_da, 'r').read().splitlines()
sv = open(file_sv, 'r').read().splitlines()
no = open(file_no, 'r').read().splitlines()
allTweets = da + sv + no
allTweets = allTweets[:100]

# Remove unwanted data
allTweets = removeAutomated(allTweets)
allTweets = removeEnglish(allTweets)

# Get mapping
mapping = getMapping(allTweets)
vocab_size = len(mapping)
print 'Vocabulary size: %d characters' % vocab_size

# Split into test and training data
testIndex = int(round(len(allTweets) * ((100 - test_set_perc) / 100)))
random.shuffle(allTweets)
testSet = allTweets[testIndex:] # Test data
trainSet = allTweets[:testIndex] # Training data
print "Size of training set: %d" % len(trainSet)
print "Size of test     set: %d" % len(testSet)

def prepareData(tweets):
    # Create sequences
    textonly = map(getText, tweets)
    sequences = []
    for index, text in enumerate(textonly):
        for i in range(sequence_length, len(text)):
            seq = text[i - sequence_length:i + 1]
            sequences.append(seq)

    # Encode sequences
    encoded = []
    for text in sequences:
        encoded_seq = [mapping[char] for char in text]
        encoded.append(encoded_seq)
    sequences = np.array(encoded)
    X, y = sequences[:,:-1], sequences[:,-1]
    sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
    X = np.array(sequences)
    y = to_categorical(y, num_classes=vocab_size)
    return (X, y)

#def generate_input(path):
#    while 1:
#        f = open(path)
#        for line in f:
#            x, y = 
# Make model
sequence_input = Input(shape=(sequence_length,), name='sequence_input')
embedding = Embedding(output_dim=64, input_dim=vocab_size)(sequence_input)
lstm = LSTM(lstm_size_1, return_sequences=True)(embedding)
lstm = LSTM(lstm_size_2)(lstm)
x = Dense(32, activation='relu')(lstm)
predictions = Dense(32, activation='softmax')(x)
model = Model(inputs=sequence_input, outputs=predictions)
print model.summary()
if (visualization):
    plot_model(model, to_file=visualizationFilename)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Train and evaluate performance
trainData = prepareData(trainSet)
X, y = trainData[0], trainData[1] 
model.fit(X, y, epochs=epochs, verbose=2)

testData = prepareData(testSet)
testX, testY = testData[0], testData[1]
result = model.evaluate(testX, testY)
print str(model.metrics_names[0]) + ": " + str(result[0])
print str(model.metrics_names[1]) + ": " + str(result[1])

# Save model and mapping to file
if (saveModelAndMapping):
    model.save('model_v0.h5')
    dump(mapping, open('mapping_v0.pkl', 'wb'))

