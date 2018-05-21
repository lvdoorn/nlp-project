import random

from pickle import load
from util import decode
from tweet import Tweet
from collections import Counter

sequence_filename = '../out/sequences.txt'
char_mapping_filename = '../out/char_mapping.pkl'
name_mapping_filename = '../out/name_mapping.pkl'
test_set_ratio = 0.1

def getCharMapping():
    return load(open(char_mapping_filename, 'rb'))

def getNameMapping():
    return load(open(name_mapping_filename, 'rb'))

def loadData():
    raw_data = open(sequence_filename, 'r').read().splitlines() 
    tweets = []
    for line in raw_data:
        split = line.split(' ' )
        tweets.append(Tweet(map(int, split[0].split(',')), int(split[1]), float(split[2]), float(split[3])))
    testIndex = int(round(len(tweets) * (1 - test_set_ratio)))
    random.seed(1)
    random.shuffle(tweets)
    trainSet = tweets[:testIndex]
    testSet = tweets[testIndex:]
    # Location based holdout set
    mapping = getNameMapping()
    holdout = []
    rest = []
    # Jonkoping = 224
    holdoutLocations = [224]
    names = []
    for tweet in tweets:
        names.append(tweet.getFullName())
        if tweet.getFullName() in holdoutLocations:
            holdout.append(tweet)
        else:
            rest.append(tweet)
    print names
    print Counter(names)
    print mapping.get(u'Tj\xf6rn, Sverige')
    #print len(names)
    #print sorted(mapping.items())
    #print len(holdout)
    #print len(rest)
    #print len(tweets)
    return {
        'trainSet': trainSet,
        'testSet': testSet,
        'locationTrainSet': rest,
        'locationTestSet': holdout
    }
loadData()
