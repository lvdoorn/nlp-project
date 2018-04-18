#!/usr/bin/env python
# Basic language model for tweets without location data

import numpy as np
import sys # To run on the slurm cluster
sys.path.append("/home/xbt504/nlp-project/src")

from keras.layers import Input, Dense, LSTM, concatenate, Embedding, TimeDistributed, Masking
from keras.models import Model
from keras.utils import to_categorical, plot_model
from keras import optimizers
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
name_mapping = getNameMapping()
name_mapping['<pad>'] = len(name_mapping)
loc_size = len(name_mapping)
print 'Amount of locations: %d' % loc_size
trainSet = data['trainSet']
testSet = data['testSet']
print "Size of training set: %d" % len(trainSet)
print "Size of test     set: %d" % len(testSet)

def padLocations(locations, sequences):
    trainLoc = np.zeros(sequences.shape)
    for i, sequence in enumerate(sequences):
        for j in range(len(sequence)):
            if sequences[i, j] == char_mapping['<pad>']:
                trainLoc[i, j] = name_mapping['<pad>']
            else:
                trainLoc[i, j] = locations[i]
    return trainLoc

locations = map(lambda x: x.getFullName(), trainSet)
trainSequences = np.array(map(lambda x: x.getText(), trainSet))
trainX = trainSequences[:,:-1]
trainLoc = padLocations(locations, trainX)
trainY = np.roll(trainSequences, -1)[:,:-1].reshape(len(trainSet), sequence_length, 1)

testSequences = np.array(map(lambda x: x.getText(), testSet))
testX = testSequences[:,:-1]
testLoc = padLocations(locations, testX)
testY = np.roll(testSequences, -1)[:,:-1].reshape(len(testSet), sequence_length, 1)

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
print model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[perplexity])

# Train and evaluate performance
def generator(x, loc, y):
    i = 0
    while True:
        i = (i + 1) % len(x)
        yield ([np.array([x[i]]), np.array([loc[i]])], to_categorical(np.array([y[i]]), num_classes=vocab_size))
steps = 750
model.fit_generator(generator(trainX, trainLoc, trainY), steps_per_epoch=steps, nb_epoch=epochs, verbose=2)
result = model.evaluate_generator(generator(testX, testLoc, testY), steps=len(testSet))

print str(model.metrics_names[0]) + ": " + str(result[0])
print str(model.metrics_names[1]) + ": " + str(result[1])

# Save model and mapping to file
if (saveModel):
    model.save('model_full_name.h5')

