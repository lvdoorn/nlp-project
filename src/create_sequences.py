#!/usr/bin/env python
# Preprocessing code to create a file with all necessary data

import json
import sys
sys.path.append("/home/xbt504/nlp-project/src/")

from langdetect import detect
from pickle import dump
from keras.preprocessing.sequence import pad_sequences 
from tweet import Tweet

# Preprocessing parameters
max_tweet_len = 280
output_filename = "sequences.txt"
char_mapping_filename = "char_mapping.pkl"
name_mapping_filename = "name_mapping.pkl"
file_da = '../collected_data/tweets_collected_da.jsonl'
file_sv = '../collected_data/tweets_collected_sv.jsonl'
file_no = '../collected_data/tweets_collected_no.jsonl'
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
        print 'Got a value error'

print "Filtering %d tweets based on language and automation ..." % len(tweetObjects)
def filterTweets(data):
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

tweets = filterTweets(tweetObjects)
print "%d tweets left after all preprocessing" % len(tweets)

# Create mappings and write them to file
def getCharMapping(tweets):
    allChars = []
    for tweet in tweets:
        for letter in tweet.getText():
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
    for tweet in tweets:
        allNames.append(tweet.getFullName())
    names = sorted(list(set(allNames)))
    mapping = dict((c, i) for i, c in enumerate(names))
    return mapping

char_mapping = getCharMapping(tweets)
name_mapping = getNameMapping(tweets)
print "vocab size: %d" % len(char_mapping)
dump(char_mapping, open(char_mapping_filename, 'wb'))
dump(name_mapping, open(name_mapping_filename, 'wb'))

# Pad sequences up to maximum length, add start and end marks and write to file
print 'Writing sequences to file ...'
seqList = []
for tweet in tweets:
    sequence = [char_mapping[char] for char in tweet.getText()]
    sequence = [char_mapping['<s>']] + sequence + [char_mapping['</s>']]
    sequence = pad_sequences([sequence], maxlen=max_tweet_len+2, value=char_mapping['<pad>'], padding='post')[0]
    encoded_name = name_mapping[tweet.getFullName()]
    seqList.append('%s %d %f %f' % (','.join(str(n) for n in sequence), encoded_name, tweet.getLat(), tweet.getLon()))
outstring = '\n'.join(seqList)
outfile = open(output_filename, "w")
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
print countryString
countryFile = open("countries.txt", "w")
countryFile.write(countryString)

