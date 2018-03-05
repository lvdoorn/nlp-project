import json
import numpy as np

from pickle import dump
from keras.layers import Input, Dense, LSTM
from keras.models import Model
from keras.utils import to_categorical

sequence_length = 10
lstm_size = 75
epochs = 25

lines = open('data/tweets_collected_da.jsonl', 'r').read().splitlines()
textonly = map(lambda x: json.loads(x)["text"], lines)
print "Amount of tweets: %d" % len(textonly)

sequences = []
for text in textonly:
    for i in range(sequence_length, len(text)):
        seq = text[i - sequence_length:i + 1]
        sequences.append(seq)
print 'Total sequences: %d' % len(sequences)

flattext = textonly[0]
for item in textonly:
    flattext += item

chars = sorted(list(set(flattext)))
mapping = dict((c, i) for i, c in enumerate(chars))
vocab_size = len(mapping)

encoded = []
for text in sequences:
    encoded_seq = [mapping[char] for char in text]
    encoded.append(encoded_seq)
print 'Vocabulary size: %d' % vocab_size

sequences = np.array(encoded)
X, y = sequences[:,:-1], sequences[:,-1]
sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = np.array(sequences)
y = to_categorical(y, num_classes=vocab_size)

inputs = Input(shape=(sequence_length, vocab_size))
lstm = LSTM(lstm_size)(inputs)
predictions = Dense(vocab_size, activation='softmax')(lstm)
model = Model(inputs=inputs, outputs=predictions)

print model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=epochs, verbose=2)

# Save model and mapping to file
model.save('model.h5')
dump(mapping, open('mapping.pkl', 'wb'))

