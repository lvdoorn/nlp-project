#import json
#
#objects_da = open('data/tweets_collected_da.jsonl', 'r').read().splitlines()
#objects_sv = open('data/tweets_collected_sv.jsonl', 'r').read().splitlines()
#objects_no = open('data/tweets_collected_no.jsonl', 'r').read().splitlines()
#allTweets = objects_da + objects_no + objects_sv
#
#def pp(string):
#    print json.dumps(string, sort_keys=True, indent=4)
#
#def getFullName(tweet):
#    js = json.loads(tweet)
#    if 'place' in js and js['place'] != None:
#        if 'full_name' in js['place']:
#            return js['place']['full_name']
#    return ''
#
## Full names
#res = map(getFullName, allTweets)
#d = {x:res.count(x) for x in res}
##pp(d)
#
## Coordinates
#def getCoordinates(tweet):
#    js = json.loads(tweet)
#    return js['geo']['coordinates']
#
#coords = map(getCoordinates, allTweets)
#pp(coords)
##pp(json.loads(allTweets[0]))
#lats = map(lambda pair: pair[0], coords)
#lons = map(lambda pair: pair[1], coords)
#print lats
#print lons
#
from numpy import array
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM, Input, Embedding
# prepare sequence
length = 100
seq = array(range(0,100))
seq2 = array(range(0, 50))
X = [seq.reshape(1, length), seq2.reshape(1, 50)]
y = seq.reshape(1, length, 1)
print X
print y
# define LSTM configuration
n_neurons = length
n_batch = 1
n_epoch = 10
# create LSTM
input_layer = Input(shape=(length,))
embedding = Embedding(output_dim=64, input_dim=length)(input_layer)
print embedding.shape
lstm = LSTM(128, return_sequences=True)(embedding)
lstm = LSTM(128, return_sequences=True)(lstm)
predictions = TimeDistributed(Dense(1, activation='relu'))(lstm)
model = Model(inputs=input_layer, outputs=predictions)
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
model.train_on_batch(array(seq2), seq2)
model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=2)
# evaluate
result = model.predict(X, batch_size=n_batch, verbose=0)
#for value in result[0,:,0]:
	#print('%.1f' % value)

