#!/usr/bin/env python
# Preprocessing code to create a file with all necessary data

import json
import collections
import sys
sys.path.append("/home/xbt504/nlp-project/src/")

from langdetect import detect
from pickle import dump
from keras.preprocessing.sequence import pad_sequences 
from tweet import Tweet

# Preprocessing parameters
max_tweet_len = 280
max_chars = 250
output_dir = '../out'
output_filename = "%s/sequences.txt" % output_dir
char_mapping_filename = "%s/char_mapping.pkl" % output_dir
name_mapping_filename = "%s/name_mapping.pkl" % output_dir
country_filename = "%s/countries.txt" % output_dir
data_dir = '../raw_data'
file_da = '%s/tweets_collected_da.jsonl' % data_dir
file_sv = '%s/tweets_collected_sv.jsonl' % data_dir
file_no = '%s/tweets_collected_no.jsonl' % data_dir

# Reads raw input files, removes duplicates and returns a list of Tweet objects
def loadTweets():
    da = open(file_da, 'r').read().splitlines()
    sv = open(file_sv, 'r').read().splitlines()
    no = open(file_no, 'r').read().splitlines()
    tweets = (da + sv + no)
    original_length = len(tweets)
    print "Loaded %d tweets from file" % original_length
    print "Removing duplicates ..."
    tweets = list(set(tweets))
    print "%d duplicates removed" % (original_length - len(tweets))
    print "Converting %d tweets to json format ..." % len(tweets)
    tweetObjects = []
    for tweet in tweets:
        try:
            js = json.loads(tweet)
            if (not ('place' in js)) or js['place'] == None:
                continue
            elif (not ('full_name' in js['place'])):
                continue
            elif (not ('geo' in js)) or js['geo'] == None:
                continue
            elif (not ('coordinates' in js['geo'])):
                continue
            coords = js['geo']['coordinates']
            place = js['place']
            tweetObject = Tweet(js['text'], place['full_name'], coords[0], coords[1], place['country'], js['created_at'])
            tweetObjects.append(tweetObject)
        except ValueError:
            print 'ValueError'
    return tweetObjects

# Not used right now
def filterTweets(data):
    print "Filtering %d tweets based on language and automation ..." % len(tweetObjects)
    to_remove = []
    for index, tweet in enumerate(data):
        try:
            timestamp = tweet.getCreatedAt()
            # Check if it's likely to be an hourly weather tweet
            if (timestamp[14] == '0' and
                timestamp[15] == '0' and
                timestamp[17] == '0' and
                timestamp[18] < '5'):
                to_remove.append(index)
            elif (detect(tweet.getText()) == 'en'): # Check if the tweet is in english
                to_remove.append(index)
        except ValueError:
            print index
    for index in to_remove[::-1]:
        data.pop(index)
    return data

#tweets = filterTweets(tweetObjects)
#print "%d tweets left after all preprocessing" % len(tweets)

# Create mappings and write them to file
def getCharMapping(charset):
    # Add padding, start, end and unknown characters
    mapping = dict((c, i) for i, c in enumerate(charset + ['<s>', '</s>', '<pad>', '<unknown>']))
    return mapping

def getNameMapping(tweets):
    allNames = []
    for tweet in tweets:
        allNames.append(tweet.getFullName())
    names = sorted(list(set(allNames)))
    mapping = dict((c, i) for i, c in enumerate(names))
    return mapping

def getChar(c, mapping):
    unknown_value = mapping['<unknown>']
    return mapping.get(c, unknown_value)

def processRawData():
    tweets = loadTweets()
    max_tweet_len = max(map(lambda x: len(x.getText()), tweets))
    print 'Maximum tweet length: %s' % max_tweet_len
    text = map(lambda x: x.getText(), tweets)
    flat = [c for s in text for c in s]
    total_chars = len(flat)
    print 'Total amount of characters: %d' % total_chars
    print 'Unique characters: %d' % len(set(flat))
    x = collections.Counter(flat)
    chars_used = x.most_common()[:max_chars]
    counts = map(lambda x: x[1], chars_used)
    charset = map(lambda x: x[0], chars_used)
    print 'Using only the %d most used characters' % max_chars
    print 'Dropping %d of %d characters' % (total_chars - sum(counts), total_chars)
    char_mapping = getCharMapping(charset)
    name_mapping = getNameMapping(tweets)
    dump(char_mapping, open(char_mapping_filename, 'wb'))
    dump(name_mapping, open(name_mapping_filename, 'wb'))
    # Pad sequences up to maximum length, add start and end marks and write to file
    seqList = []
    for tweet in tweets:
        sequence = [getChar(c, char_mapping) for c in tweet.getText()]
        sequence = [getChar('<s>', char_mapping)] + sequence + [getChar('</s>', char_mapping)]
        sequence = pad_sequences([sequence], maxlen=max_tweet_len+2, value=getChar('<pad>', char_mapping), padding='post')[0]
        encoded_name = name_mapping[tweet.getFullName()]
        seqList.append('%s %d %f %f' % (','.join(str(n) for n in sequence), encoded_name, tweet.getLat(), tweet.getLon()))
    outstring = '\n'.join(seqList)
    outfile = open(output_filename, "w")
    print 'Writing sequences to file ...'
    outfile.write(outstring)
    print '%d sequences written to file %s' % (len(tweets), output_filename)
    
    # Compute country counts
    countries = map(lambda x: x.getCountry(), tweets)
    countryString = '\n'.join([
            'Country counts:',
            'Denmark: %d' % countries.count('Denmark'),
            'Sweden: %d' % countries.count('Sweden'),
            'Norway: %d' % countries.count('Norway')
        ])
    countryFile = open(country_file_name, "w")
    countryFile.write(countryString)

