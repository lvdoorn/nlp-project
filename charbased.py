import numpy as np
import json

from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from keras.preprocessing.sequence import pad_sequences
from pickle import dump, load

# Create model
lines = open('data/tweets_collected_da.jsonl', 'r').read().splitlines()
textonly = map(lambda x: json.loads(x)["text"], lines)[:100]
coordsonly = map(lambda x: json.loads(x)["coordinates"], lines)
placeonly = map(lambda x: json.loads(x)["place"], lines)

length = 10
sequences = []
for text in textonly:
    for i in range(length, len(text)):
        seq = text[i - length:i + 1]
        sequences.append(seq)
print 'Total sequences: %d' % len(sequences)

text = [y for x in textonly for y in x]
chars = sorted(list(set(text)))
mapping = dict((c, i) for i, c in enumerate(chars))
vocab_size = len(mapping)
print 'Vocabulary size: %d' % vocab_size

encoded = []
for line in sequences:
    encoded_seq = [mapping[char] for char in line]
    encoded.append(encoded_seq)

encoded = np.array(encoded)
X, y = encoded[:,:-1], encoded[:,-1]
print y 

sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = np.array(sequences)
y = to_categorical(y, num_classes=vocab_size)

model = Sequential()
model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(vocab_size, activation='softmax'))
print model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100, verbose=2)

# Save model and mapping
model.save('model.h5')
dump(mapping, open('mapping.pkl', 'wb'))

model = load_model('model.h5')
mapping = load(open('mapping.pkl', 'rb'))

# Generate new texts based on data
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
    in_text = seed_text
    for _ in range(n_chars):
        encoded = [mapping[char] for char in in_text]
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        encoded = to_categorical(encoded, num_classes=len(mapping))
        yhat = model.predict_classes(encoded, verbose=0)
        out_char = ''
        for char, index in mapping.items():
            if index == yhat:
                out_char = char
                break
        in_text += char 
    return in_text

print generate_seq(model, mapping, 10, 'was out ru', 20)
print generate_seq(model, mapping, 10, 'Fest og Fa', 20)
print generate_seq(model, mapping, 10, 'tur i Tils', 20)
print textonly

