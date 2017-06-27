# -*- coding: utf-8 -*-

import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.initializers import RandomNormal

from tqdm import tqdm

import io

from utils import write_to_output, tokenize_text, nearest_word

INPUT_SIZE = 25 # sequence/word length
SLIDE = 25 # SLIDE=1: Mary had a (little) & had a little (lamb)
LEARNING_RATE = 0.001
GAMMA = 1e-6
NUM_VECTORIZED_WORDS = 22000


def load_model(model_file, weight_file):
  with io.open(model_file, 'r', encoding='utf-8') as json_file:
    loaded_file = json_file.read()
  loaded_model = model_from_json(loaded_file)
  loaded_model.load_weights(weight_file)
  print("Succesfully loaded model and weights from drive! \n")
  return loaded_model

def load_test_file(test_file):
  listitified_text = []
  with io.open(test_file, 'r', encoding='utf-8') as testfile:
    for line in testfile:
      line = line.lstrip().rstrip().lower().split(" ")
      listitified_text.extend(line)
  return listitified_text

def splice_up_text(tokenized_txt, vec_dimensions):
  X_train = []
  y_train = []
  last_idx = len(tokenized_txt) - INPUT_SIZE
  print("Splitting Data...")
  for i in tqdm(xrange(0, last_idx, SLIDE)):
    X = tokenized_txt[i:i+INPUT_SIZE]
    y = tokenized_txt[i+INPUT_SIZE]
    X_train.append(X)
    y_train.append(y)
  X_train, y_train = np.asarray(X_train), np.asarray(y_train)
  # adjust dimensions
  print("Adjusting Training Data Dimensions...")
  # X_train = np.reshape(X_train, (len(X_train), INPUT_SIZE, NUM_VECTORIZED_WORDS))
  # cache as doubly nested numpy array
  return X_train, y_train

def train_lstm(X_train, y_train, vec_dimensions, model_dest, weight_dest):
  print("Building LSTM Model... \n")
  # initializer for bias
  initializer = RandomNormal(mean=0.0, stddev=0.05, seed=None)
  # construct an LSTM model
  model = Sequential()
  model.add(LSTM(256, input_shape=(INPUT_SIZE, vec_dimensions),
                      bias_initializer=initializer,
                      return_sequences=True))
  model.add(Dropout(0.2)) # prevent overfitting
  model.add(LSTM(256))
  model.add(Dropout(0.2))
  model.add(Dense(NUM_VECTORIZED_WORDS))
  model.add(Activation('softmax'))
  #optimizer = RMSprop(lr=LEARNING_RATE)
  optimizer = SGD(lr=LEARNING_RATE, decay=GAMMA, clipnorm=1.)
  #optimizer = Adam(lr=LEARNING_RATE, decay=GAMMA)
  model.compile(loss='categorical_crossentropy', optimizer=optimizer)
  # define checkpoint for future
  fpath = "weight-improvement={epoch:02d}-{loss:.4f}.hdf5"
  checkpoint = ModelCheckpoint(fpath, monitor='loss', verbose=1, save_best_only=True, mode='min')
  callbacks_list = [checkpoint]
  # train model with data
  print("Training in progress... \n")
  model.fit(X_train, y_train, batch_size=256, epochs=10, callbacks=callbacks_list)
  # save model as json
  model_json = unicode(model.to_json(), 'utf-8')
  with io.open(model_dest, 'w', encoding='utf-8') as jfile:
    jfile.write(model_json)
    # serialize weights to HDF5
    model.save_weights(weight_dest)
    print("Successfully saved model to drive! \n")

def test_lstm(tokenized_txt, model_file, weight_file):
  X_test, y_test, y_hat, y = [], [], [], []
  model = load_model(model_file, weight_file)
  last_idx = len(tokenized_txt) - INPUT_SIZE
  for i in xrange(0, last_idx, 1):
    X_test.append(tokenized_txt[i:i+INPUT_SIZE])
    y_test.append(tokenized_txt[i+INPUT_SIZE])
  X_test, y_test = np.asarray(X_test), np.asarray(y_test)
  print("Prediction in progress...")
  predicted_tokens = model.predict(X_test, batch_size=256, verbose=1)
  for t in predicted_tokens:
    y_hat.append(np.argmax(t))
  for _y in y_test:
    y.append(np.argmax(_y))
  return y_hat, y
