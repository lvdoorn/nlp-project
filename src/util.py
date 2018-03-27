import json
import keras.backend as K
import numpy as np
from keras.utils import to_categorical

def getText(tweet):
    tweet['text']

def getCharMapping(tweets):
    allChars = []
    textonly = map(getText, tweets)
    print textonly
    for tweet in textonly:
        if tweet != None:
            for letter in tweet:
                allChars.append(letter)
    chars = sorted(list(set(allChars)))
    mapping = dict((c, i) for i, c in enumerate(chars))
    # Add padding, start and end characters
    mapping['<s>'] = len(chars)
    mapping['</s>'] = len(chars) + 1
    mapping['<pad>'] = len(chars) + 2
    return mapping

def getNameMapping(tweets):
    allNames = []
    nameonly = map(getFullName, tweets)
    for name in nameonly:
        allNames.append(name)
    names = sorted(list(set(allNames)))
    mapping = dict((c, i) for i, c in enumerate(names))
    return mapping

def getFullName(tweet):
    js = tweet
    if 'place' in js and js['place'] != None:
        if 'full_name' in js['place']:
            return js['place']['full_name']
    return ''

def getCoords(tweet):
    js = tweet
    if 'geo' in js and js['geo'] != None:
        if 'coordinates' in js['geo']:
            return js['geo']['coordinates']
    return ''

def perplexity(y_true, y_pred):
    cross_entropy = K.categorical_crossentropy(y_true, y_pred)
    perplexity = K.pow(2.0, cross_entropy)
    return perplexity

def decode(line):
    split = line.split(' ')
    sequence = map(int, split[0].split(','))
    name = int(split[1])
    lat = float(split[2])
    lon = float(split[3])
    return {
        'sequence': sequence,
        'name': name,
        'lat': lat,
        'lon': lon
    }
    
def prepareData(data, vocab_size):
    X = []
    y = []
    name = []
    lat = []
    lon = []
    for seq in data:
        sequence = seq['sequence']
        X.append(sequence[:-1])
        y.append(to_categorical(sequence[-1], num_classes=vocab_size))
        name.append(seq['name'])
        lat.append(seq['lat'])
        lon.append(seq['lon'])
    return {
        'X': np.array(X),
        'y': np.array(y),
        'name': np.array(name),
        'lat': np.array(lat),
        'lon': np.array(lon),
    }

