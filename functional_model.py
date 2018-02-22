from keras.layers import Input, Dense, Embedding, LSTM
from keras.models import Model 
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

import json

import numpy as np

lines = open('data/tweets_collected_da.jsonl', 'r').read().splitlines()
textonly = map(lambda x: json.loads(x)["text"], lines)[:10]
print "Amount of tweets: %d" % len(textonly)

# Encode text as integers
tokenizer = Tokenizer()
tokenizer.fit_on_texts(textonly)
encoded = tokenizer.texts_to_sequences(textonly)
vocab_size = len(tokenizer.word_index) + 1
print "Vocabulary size: %d" % vocab_size

sequences = []
for i in range(0, len(encoded)):
    for j in range(1, len(encoded[i])):
        sequence = encoded[i][j-1:j+1]
        sequences.append(sequence)

sequences = np.array(sequences)
X, y = sequences[:,0], sequences[:,1]
X = to_categorical(X, num_classes=vocab_size)
y = to_categorical(y, num_classes=vocab_size)
print X.shape
print y.shape

inputs = Input(shape=(vocab_size,))
x = Embedding(vocab_size, 10, input_length=1)(inputs)
x = LSTM(50)(x)
predictions = Dense(vocab_size, activation='softmax')(x)
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=50, verbose=2)


in_texts = ["jeg", "er", "har", "en"] 
print in_texts
for in_text in in_texts:
    print in_text
    encoded = np.array(tokenizer.texts_to_sequences([in_text])[0])
    yhat = model.predict(encoded, verbose=0)
    for word, index in tokenizer.word_index.items():
        if index == yhat:
            print word

