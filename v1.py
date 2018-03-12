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
epochs = 100 

file_da = 'data/tweets_collected_da.jsonl'
file_sv = 'data/tweets_collected_sv.jsonl'
file_no = 'data/tweets_collected_no.jsonl'

da = open(file_da, 'r').read().splitlines()
sv = open(file_sv, 'r').read().splitlines()
no = open(file_no, 'r').read().splitlines()
allTweets = da + sv + no
allTweets = removeEnglish(allTweets)
allTweets = removeAutomated(allTweets)
random.shuffle(allTweets)
#allTweets = allTweets[:100]
print "Amount of tweets: %d" % len(allTweets)

def getText(tweet):
    return json.loads(tweet)['text']

def getFullName(tweet):
    js = json.loads(tweet)
    if 'place' in js and js['place'] != None:
        if 'full_name' in js['place']:
            return js['place']['full_name']
    return ''

textonly = map(getText, allTweets)
nameonly = map(getFullName, allTweets)

sequences = []
places = []
for index, text in enumerate(textonly):
    for i in range(sequence_length, len(text)):
        places.append(nameonly[index])
        seq = text[i - sequence_length:i + 1]
        sequences.append(seq)
print 'Total sequences: %d' % len(sequences)
print 'Total places: %d' % len(set(places))
flattext = textonly[0]
for item in textonly:
    flattext += item

chars = sorted(list(set(flattext)))
mapping = dict((c, i) for i, c in enumerate(chars))
vocab_size = len(mapping)
print 'Vocabulary size: %d' % vocab_size
nameno = len(nameonly)
print 'Amount of names: %d' % nameno

encoded = []
for text in sequences:
    encoded_seq = [mapping[char] for char in text]
    encoded.append(encoded_seq)

sequences = np.array(encoded)
X, y = sequences[:,:-1], sequences[:,-1]
sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = np.array(sequences)
y = to_categorical(y, num_classes=vocab_size)

sequence_input = Input(shape=(sequence_length, vocab_size))
lstm = LSTM(lstm_size)(sequence_input)
location_input = Input(shape=(nameno,))
x = concatenate([lstm, location_input])
x = Dense(64, activation='relu')(x)

predictions = Dense(vocab_size, activation='softmax')(x)
model = Model(inputs=[sequence_input, location_input], outputs=predictions)

nameMapping = dict((c, i) for i, c in enumerate(nameonly))
encodedNames = []
for name in places:
    encoded_name = nameMapping[name]
    encodedNames.append(encoded_name)
names = [to_categorical(name, num_classes=nameno) for name in encodedNames]
names = np.array(names)
print model.summary()

plot_model(model, to_file='model.png')
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([X, names], y, epochs=epochs, verbose=2)

# Save model and mapping to file
modelName = '_' + str(sequence_length) + '_' + str(lstm_size) + '_' + str(epochs)
model.save('model' + modelName + '.h5')
dump(mapping, open('mapping' + modelName + '.pkl', 'wb'))

