import numpy as np

import data.load

from metrics.accuracy import conlleval

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers import Convolution1D, MaxPooling1D

### Load Data
train_set, valid_set, dicts = data.load.atisfull()
w2idx, ne2idx, labels2idx = dicts['words2idx'], dicts['tables2idx'], dicts['labels2idx']

# Create index to word/label dicts
idx2w = {w2idx[k]: k for k in w2idx}
idx2ne = {ne2idx[k]: k for k in ne2idx}
idx2la = {labels2idx[k]: k for k in labels2idx}

### Model
n_classes = len(idx2la)
n_vocab = len(idx2w)

### Define model
model = Sequential()
model.add(Embedding(n_vocab, 100))
model.add(Convolution1D(64, 5, padding='same', activation='relu'))
model.add(Dropout(0.25))
model.add(GRU(100, return_sequences=True))
model.add(TimeDistributed(Dense(n_classes, activation='softmax')))
model.compile('rmsprop', 'categorical_crossentropy')
model.load_weights("best_model_weights.h5")

### Ground truths etc for conlleval
train_x, train_ne, train_label = train_set
val_x, val_ne, val_label = valid_set

words_val = [list(map(lambda x: idx2w[x], w)) for w in val_x]
groundtruth_val = [list(map(lambda x: idx2la[x], y)) for y in val_label]
words_train = [list(map(lambda x: idx2w[x], w)) for w in train_x]
groundtruth_train = [list(map(lambda x: idx2la[x], y)) for y in train_label]
while (True):
    try:
        sentence = input("please input the sentence\n")
        w2idxfunc = lambda x: w2idx[x]
        sentence_vec = list(w2idxfunc(w) for w in sentence.split(" "))
        print(sentence_vec)

        pred = model.predict(sentence_vec)
        pred = np.argmax(pred, -1)
        result = list([list(map(lambda x: idx2la[x], y)) for y in pred])
        print(result)
    except KeyError:
        print("KeyError happened, please check the words")
