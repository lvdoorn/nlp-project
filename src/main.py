from preprocessing import processRawData, loadTweets
import collections

#tweets = loadTweets()
#text = map(lambda x: x.getText(), tweets)
##print text
#
#flat = [c for s in text for c in s]
#print len(flat)
#x = collections.Counter(flat)
#chars_used = x.most_common()[:max_chars]
#print x.most_common()[:250]
processRawData()
