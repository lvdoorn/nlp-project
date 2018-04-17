#!/usr/bin/env python
# Basic language model for tweets without location data

import numpy as np
import sys # To run on the slurm cluster
sys.path.append("/home/xbt504/nlp-project/src")

from keras.layers import Input, Dense, LSTM, concatenate, Embedding, TimeDistributed, Masking
from keras.models import Model
from keras.utils import to_categorical, plot_model
from util import perplexity
from loaddata import getCharMapping, getNameMapping, loadData

# Parameters
embedding_size = 64
lstm_size_1 = 128
lstm_size_2 = 128
epochs = 100
if len(sys.argv) > 1:
    epochs = int(sys.argv[1])
print "epochs: %d" % epochs
saveModel = True
max_tweet_length = 280
sequence_length = max_tweet_length + 1

# Load data and mappings from file
data = loadData()
char_mapping = getCharMapping()
vocab_size = len(char_mapping)
print 'Vocabulary size: %d characters' % vocab_size
trainSet = data['trainSet']
testSet = data['testSet']
print "Size of training set: %d" % len(trainSet)
print "Size of test     set: %d" % len(testSet)

trainLat = map(lambda x: x.getLat(), trainSet)
trainLon = map(lambda x: x.getLon(), trainSet)
trainSequences = np.array(map(lambda x: x.getText(), trainSet))
trainX = trainSequences[:,:-1]
trainY = np.roll(trainSequences, -1)[:,:-1].reshape(len(trainSet), sequence_length, 1)
a = np.array([1, 2])
a = np.stack([a for _ in range(sequence_length)])
print a.shape
testLat = map(lambda x: x.getLat(), testSet)
testLon = map(lambda x: x.getLon(), testSet)
testSequences = np.array(map(lambda x: x.getText(), testSet))
testX = testSequences[:,:-1]
testY = np.roll(testSequences, -1)[:,:-1].reshape(len(testSet), sequence_length, 1)

sequence_input = Input(shape=(sequence_length,), name='sequence_input')
mask = Masking(mask_value=char_mapping['<pad>'])(sequence_input)
embedding = Embedding(output_dim=embedding_size, input_dim=vocab_size)(mask)

location_input = Input(shape=(sequence_length,2), name='location_input')

merge = concatenate([embedding, location_input])
lstm = LSTM(lstm_size_1, return_sequences=True)(merge)
lstm = LSTM(lstm_size_2, return_sequences=True)(lstm)
predictions = TimeDistributed(Dense(vocab_size, activation='relu'))(lstm)
model = Model(inputs=[sequence_input, location_input], outputs=predictions)
print model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[perplexity])

# Train and evaluate performance
def generator(x, y, lat, lon):
    i = 0
    while True:
        i = (i + 1) % len(x)
        resX = np.array([x[i]])
        a = np.array([lat[i], lon[i]])
        resLoc = np.array([np.stack([a for _ in range(sequence_length)])])
        resY = to_categorical(np.array([y[i]]), num_classes=vocab_size)
        yield ([resX, resLoc], resY)
model.fit_generator(generator(trainX, trainY, trainLat, trainLon), steps_per_epoch=9, nb_epoch=1, verbose=1)
result = model.evaluate_generator(generator(testX, testY, testLat, testLon), steps=1)

print str(model.metrics_names[0]) + ": " + str(result[0])
print str(model.metrics_names[1]) + ": " + str(result[1])

# Save model and mapping to file
if (saveModel):
    model.save('model_full_name.h5')

