import json
import numpy as np
import random

from pickle import dump
from keras.layers import Input, Dense, LSTM, concatenate
from keras.models import Model
from keras.utils import to_categorical, plot_model

from preprocessing import removeAutomated, removeEnglish

sequence_length = 10
lstm_size = 30
epochs = 25
visualization = True
visualizationFilename = 'model_v0.png'
saveModelAndMapping = True

# Load data
file_da = 'data/tweets_collected_da.jsonl'
file_sv = 'data/tweets_collected_sv.jsonl'
file_no = 'data/tweets_collected_no.jsonl'
da = open(file_da, 'r').read().splitlines()
sv = open(file_sv, 'r').read().splitlines()
no = open(file_no, 'r').read().splitlines()
allTweets = da + sv + no
allTweets = removeAutomated(allTweets)
allTweets = removeEnglish(allTweets)
print "Amount of tweets: %d" % len(allTweets)

def getText(tweet):
    return json.loads(tweet)['text']

# Create sequences
textonly = map(getText, allTweets)
sequences = []
for index, text in enumerate(textonly):
    for i in range(sequence_length, len(text)):
        seq = text[i - sequence_length:i + 1]
        sequences.append(seq)
print 'Amount of sequences: %d' % len(sequences)

# Create mapping
allChars = []
for tweet in textonly:
    for letter in tweet:
        allChars.append(letter)
chars = sorted(list(set(allChars)))
mapping = dict((c, i) for i, c in enumerate(chars))
vocab_size = len(mapping)
print 'Vocabulary size: %d' % vocab_size

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

# Make model
sequence_input = Input(shape=(sequence_length, vocab_size))
lstm = LSTM(lstm_size)(sequence_input)
location_input = Input(shape=(nameno,))
x = concatenate([lstm, location_input])
x = Dense(64, activation='relu')(x)
predictions = Dense(vocab_size, activation='softmax')(x)
model = Model(inputs=[sequence_input, location_input], outputs=predictions)
print model.summary()
if (visualization):
    plot_model(model, to_file=visualizationFilename)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([X, names], y, epochs=epochs, verbose=2)

# Save model and mapping to file
if (saveModelAndMapping):
    model.save('model_v0.h5')
    dump(mapping, open('mapping_v0.pkl', 'wb'))

