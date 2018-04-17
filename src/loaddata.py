import random

from pickle import load
from tweet import Tweet

sequence_filename = 'sequences.txt'
char_mapping_filename = 'char_mapping.pkl'
name_mapping_filename = 'name_mapping.pkl'
test_set_ratio = 0.1

def loadData():
    raw_data = open(sequence_filename, 'r').read().splitlines() 
    tweets = []
    for line in raw_data:
        split = line.split(' ')
        tweets.append(Tweet(map(int, split[0].split(',')), int(split[1]), float(split[2]), float(split[3])))
    testIndex = int(round(len(tweets) * (1 - test_set_ratio)))
    random.shuffle(tweets)
    trainSet = tweets[:testIndex]
    testSet = tweets[testIndex:]
    return {
        'trainSet': trainSet,
        'testSet': testSet
    }

def getCharMapping():
    return load(open(char_mapping_filename, 'rb'))

def getNameMapping():
    return load(open(name_mapping_filename, 'rb'))

