#!/usr/bin/env python
# Model with full place names as location data

import json
import numpy as np
import random
import sys
sys.path.append("/home/xbt504/nlp-project/src")

from pickle import dump
from keras.layers import Input, Dense, LSTM, concatenate, Embedding
from keras.models import Model
from keras.utils import to_categorical, plot_model
from math import pow
import keras.backend as K

from preprocessing import removeAutomated, removeEnglish
from util import getText, getMapping, getFullName

sequence_length = 10
lstm_size_1 = 30
lstm_size_2 = 30
epochs = 10 
test_set_perc = 10.0

file_da = '../data/tweets_collected_da.jsonl'
file_sv = '../data/tweets_collected_sv.jsonl'
file_no = '../data/tweets_collected_no.jsonl'

da = open(file_da, 'r').read().splitlines()
sv = open(file_sv, 'r').read().splitlines()
no = open(file_no, 'r').read().splitlines()
allTweets = da + sv + no
allTweets = allTweets[:100]
allTweets = removeEnglish(allTweets)
allTweets = removeAutomated(allTweets)
random.shuffle(allTweets)
print "Amount of tweets: %d" % len(allTweets)

testIndex = int(round(len(allTweets) * ((100 - test_set_perc) / 100)))
testSet = allTweets[testIndex:] # Test data
trainSet = allTweets[:testIndex] # Training data
print "Size of training set: %d" % len(trainSet)
print "Size of test     set: %d" % len(testSet)

mapping = getMapping(allTweets)
vocab_size = len(mapping)
print 'Vocabulary size: %d' % vocab_size
nameMapping = dict((c, i) for i, c in enumerate(set(map(getFullName, allTweets))))
nameno = len(nameMapping)
print 'Amount of names: %d' % nameno

def prepareData(tweets):
    textonly = map(getText, tweets)
    nameonly = map(getFullName, tweets)

    encoded = []
    places = []
    for index, text in enumerate(textonly):
        for i in range(sequence_length, len(text)):
            places.append(nameonly[index])
            seq = text[i - sequence_length:i + 1]
            encoded_seq = [mapping[char] for char in seq]
            encoded.append(encoded_seq)
    
    encodedNames = []
    for name in places:
        encoded_name = nameMapping[name]
        encodedNames.append(encoded_name)
    names = [to_categorical(name, num_classes=nameno) for name in encodedNames]
    names = np.array(names)

    sequences = np.array(encoded)
    X, y = sequences[:,:-1], sequences[:,-1]
    #sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
    sequences = X
    X = np.array(sequences)
    y = to_categorical(y, num_classes=vocab_size)
    return { 'X': X, 'names': names, 'y': y }

def perplexity(y_true, y_pred):
    cross_entropy = K.categorical_crossentropy(y_true, y_pred)
    perplexity = K.pow(2.0, cross_entropy)
    return perplexity

# Make model
sequence_input = Input(shape=(sequence_length,), name='sequence_input')
print "sequence input shape: %s" % sequence_input.shape
embedding = Embedding(output_dim=64, input_dim=vocab_size)(sequence_input)
print embedding.shape
lstm = LSTM(lstm_size_1, return_sequences=True)(embedding)
lstm = LSTM(lstm_size_2)(lstm)
#lstm = Embedding(output_dim=30, input_dim=vocab_size)(lstm)
loc_input = Input(shape=(nameno,), name='location_input')
print 'location input shape: ' % loc_input.shape
#loc_embedding = Embedding(output_dim=64, input_dim=nameno)(loc_input)
loc_embedding = loc_input
#loc_embedding = Dense(64, activation='relu')(loc_input)
x = concatenate([lstm, loc_embedding])
x = Dense(64, activation='relu')(x)
predictions = Dense(vocab_size, activation='softmax')(x)
model = Model(inputs=[sequence_input, loc_input], outputs=predictions)
print model.summary()
plot_model(model, to_file='model.png')
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[perplexity])

# Train and evaluate model
data = prepareData(trainSet)
print data['X']
model.fit([data['X'], data['names']], data['y'], epochs=epochs, verbose=2)
testData = prepareData(testSet)

result = model.evaluate([testData['X'], testData['names']], testData['y'])
cross_entropy = result[0]
print "Perplexity: %s" % str(pow(2.0, cross_entropy))
print str(model.metrics_names[0]) + ": " + str(result[0])
print str(model.metrics_names[1]) + ": " + str(result[1])

# Save model and mapping to file
modelName = '_' + str(sequence_length) + '_' + str(lstm_size_1) + str(lstm_size_2) + '_' + str(epochs)
model.save('model_v1' + modelName + '.h5')
dump(mapping, open('mapping' + modelName + '.pkl', 'wb'))

