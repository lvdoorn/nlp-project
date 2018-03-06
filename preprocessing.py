import json
from langdetect import detect

def removeAutomated(filename):
    data = open(filename, 'r').read().splitlines()
    print 'Filename: %s' % filename
    print 'Original data size: %d' % len(data)
    to_remove = []
    for index, item in enumerate(data):
        tweet = json.loads(item)
        timestamp = tweet['created_at']
        # Check if it's likely to be an hourly weather tweet
        if (timestamp[14] == '0' and timestamp[15] == '0' and timestamp[17] == '0' and timestamp[18] < '5'):
            to_remove.append(index)

    print 'Removing %d automated tweets' % len(to_remove)
    for index in to_remove[::-1]:
        data.pop(index)
    print 'New data size: %d' % len(data)
    print ''

def removeEnglish(filename):
    data = open(filename, 'r').read().splitlines()
    print 'Filename: %s' % filename
    print 'Original data size: %d' % len(data)
    to_remove = []
    for index, item in enumerate(data):
        tweet = json.loads(item)
        text = tweet['text']
        # Check if tweet is in English
        if (detect(text) == 'en'):
            to_remove.append(index)
            print text
    print 'Removing %d English tweets' % len(to_remove)
    for index in to_remove[::-1]:
        data.pop(index)
    print 'New data size: %d' % len(data)
    print ''   

data_no = removeEnglish('data/tweets_collected_no.jsonl')
data_da = removeEnglish('data/tweets_collected_da.jsonl')
data_sv = removeEnglish('data/tweets_collected_sv.jsonl')

