#!/usr/bin/env python

import sys
sys.path.append("/home/xbt504/nlp-project/src") # To run on cluster
import os
import numpy as np
import collections

import preprocessing
import keras
import no_location
import full_name
import coords

from keras.utils import to_categorical
from util import perplexity

# Process command line parameters
if (len(sys.argv) < 2):
    print "Usage: main.py modeltype(1,2,3) epochs"
    sys.exit(0)

# Parameters
sequence_length = 150
modelType = sys.argv[1]
if (modelType < '1' or modelType > '3'):
    print "Not a valid model type"
    sys.exit(1)
epochs = int(sys.argv[2])
modelTypes = {'1': "No location", '2': "Full name", '3': "Coordinates"}

print "Running with this configuration:"
print "Model type: %s" % modelTypes[modelType]
print "Epochs: %d" % epochs

output_dir = '../out'
noDuplicatesFilename = '%s/no_duplicates.txt' % output_dir
trainSetFilename = '%s/train_set.txt' % output_dir
testSetNormalFilename = '%s/test_set_normal.txt' % output_dir
testSetLocationFilename = '%s/test_set_location.txt' % output_dir

# We need to do preprocessing if the files do not exist yet
if not os.path.exists('../out/no_duplicates.txt'):
    preprocessing.removeDuplicates()
    preprocessing.createTrainTestSets()
else:
    print "Found files, no preprocessing needed"

# Load data and mappings from file
trainSet = preprocessing.loadTweets(trainSetFilename)
testSetNormal = preprocessing.loadTweets(testSetNormalFilename)
testSetLocation = preprocessing.loadTweets(testSetLocationFilename)

char_mapping = preprocessing.loadCharMapping()
name_mapping = preprocessing.loadNameMapping()

# test is a boolean indicating whether it is a test set tweet or not
def tweetToSequence(tweet, test):
    text = list(tweet.getText())
    fullSentence = ['<s>'] + text + ['</s>']
    sequence = None
    if test:
        sequence = map(lambda x: preprocessing.getCharTest(x, char_mapping), fullSentence)
    else:
        sequence = map(lambda x: preprocessing.getCharTrain(x, char_mapping), fullSentence)
    sequence = keras.preprocessing.sequence.pad_sequences([sequence], maxlen=sequence_length+1, value=preprocessing.getCharTrain('<pad>', char_mapping), padding='post')[0]
    return np.array(sequence)

# Arrange data
trainSequences = np.array(map(lambda x: tweetToSequence(x, False), trainSet))
trainX = trainSequences[:,:-1]
trainY = np.roll(trainSequences, -1)[:,:-1].reshape(len(trainSet), sequence_length, 1)
testSequencesNormal = np.array(map(lambda x: tweetToSequence(x, True), testSetNormal))
testXN = testSequencesNormal[:,:-1]
testYN = np.roll(testSequencesNormal, -1)[:,:-1].reshape(len(testSetNormal), sequence_length, 1)
testSequencesLocation = np.array(map(lambda x: tweetToSequence(x, True), testSetLocation))
testXL = testSequencesLocation[:,:-1]
testYL = np.roll(testSequencesLocation, -1)[:,:-1].reshape(len(testSetLocation), sequence_length, 1)

vocab_size = len(char_mapping)
# Load model
model = None
if (modelType == '1'):
    model = no_location.getModel()
elif (modelType == '2'):
    model = full_name.getModel()
elif (modelType == '3'):
    model = coords.getModel()
else:
    print "Not a valid model type"
    sys.exit(1)
print model.summary()

adam = keras.optimizers.Adam(clipnorm=1, clipvalue=0.5)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=[perplexity])

def padLocations(locations, sequences):
    trainLoc = np.zeros(sequences.shape)
    for i, sequence in enumerate(sequences):
        for j in range(len(sequence)):
            if sequences[i, j] == char_mapping['<pad>']:
                trainLoc[i, j] = name_mapping['<pad>']
            else:
                trainLoc[i, j] = locations[i]
    return trainLoc

def padCoords(lat, lon, sequences):
    trainLat = np.zeros(sequences.shape)
    trainLon = np.zeros(sequences.shape)
    for i, sequence in enumerate(sequences):
        for j in range(len(sequence)):
            if sequences[i, j] == char_mapping['<pad>']:
                trainLat[i, j] = name_mapping['<pad>']
                trainLon[i, j] = name_mapping['<pad>']
            else:
                trainLat[i, j] = lat[i]
                trainLon[i, j] = lon[i]
    return trainLat, trainLon

locations = map(lambda x: name_mapping[x.getFullName()], trainSet)
trainLoc = padLocations(locations, trainX)
locations = map(lambda x : name_mapping[x.getFullName()], testSetNormal)
testLocNormal = padLocations(locations, testXN)
locations = map(lambda x : name_mapping[x.getFullName()], testSetLocation)
testLocLocation = padLocations(locations, testXL)

latitudes = map(lambda x: x.getLat(), trainSet)
longitudes = map(lambda x: x.getLon(), trainSet)
trainLat, trainLon = padCoords(latitudes, longitudes, trainX)
latitudes = map(lambda x: x.getLat(), testSetNormal)
longitudes = map(lambda x: x.getLon(), testSetNormal)
testLatNormal, testLonNormal = padCoords(latitudes, longitudes, testXN)
latitudes = map(lambda x: x.getLat(), testSetLocation)
longitudes = map(lambda x: x.getLon(), testSetLocation)
testLatLocation, testLonLocation = padCoords(latitudes, longitudes, testXL)

if modelType == '1':
    print "Fitting model type 1"
    model.fit(trainX, to_categorical(trainY, num_classes=vocab_size), epochs=epochs, verbose=2)
    result = model.evaluate(testXN, to_categorical(testYN, num_classes=vocab_size), verbose=2)
    print "Evaluation on normal test set:"
    print str(model.metrics_names[0]) + ": " + str(result[0])
    print str(model.metrics_names[1]) + ": " + str(result[1])

    result = model.evaluate(testXL, to_categorical(testYL, num_classes=vocab_size), verbose=2)
    print "Evaluation on location-based test set:"
    print str(model.metrics_names[0]) + ": " + str(result[0])
    print str(model.metrics_names[1]) + ": " + str(result[1])

if modelType == '2':
    print "Fitting model type 2"
    model.fit([trainX, trainLoc], to_categorical(trainY, num_classes=vocab_size), epochs=epochs, verbose=2)
    result = model.evaluate([testXN, testLocNormal], to_categorical(testYN, num_classes=vocab_size), verbose=2)
    print "Evaluation on normal test set:"
    print str(model.metrics_names[0]) + ": " + str(result[0])
    print str(model.metrics_names[1]) + ": " + str(result[1])

    result = model.evaluate([testXL, testLocLocation], to_categorical(testYL, num_classes=vocab_size), verbose=2)
    print "Evaluation on location-based test set:"
    print str(model.metrics_names[0]) + ": " + str(result[0])
    print str(model.metrics_names[1]) + ": " + str(result[1])

if modelType == '3':
    def getLocInput(lat, lon):
        res = np.zeros((len(lat), sequence_length, 2))
        for i in range(len(lat)):
            for j in range(sequence_length):
                cell = np.array([lat[i, j], lon[i, j]])
            res[i, j] = cell
        return res

    print "Fitting model type 3"
    model.fit([trainX, getLocInput(trainLat, trainLon)], to_categorical(trainY, num_classes=vocab_size), epochs=epochs, verbose=2)
    result = model.evaluate([testXN, getLocInput(testLatNormal, testLonNormal)], to_categorical(testYN, num_classes=vocab_size), verbose=2)
    print "Evaluation on normal test set:"
    print str(model.metrics_names[0]) + ": " + str(result[0])
    print str(model.metrics_names[1]) + ": " + str(result[1])

    result = model.evaluate([testXL, getLocInput(testLatLocation, testLonLocation)], to_categorical(testYL, num_classes=vocab_size), verbose=2)
    print "Evaluation on location-based test set:"
    print str(model.metrics_names[0]) + ": " + str(result[0])
    print str(model.metrics_names[1]) + ": " + str(result[1])
