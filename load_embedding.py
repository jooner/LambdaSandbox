# -*- coding: utf-8 -*-

"""
First, we load up GloVe1.2's 6B corpus that encodes 400,000 words. We use the 50-dimensional version as default.

The function evaluate_scores allows us to measure distances of words from an input text file of the
format >> keyword compareword1 compareword2 ..., writing the results to output1.txt.


Using naive_next_word, which simply gets the closest embedded word from the GloVe vectorization,
we get the following result (quite catastrophic, for obvious reasons):
input sentence [true answer <> 50d corpus <> 100d corpus <> 300d corpus]
> four scores and seven years [ago <> since <> ago]
> his dressing gown always hung in the [bedroom <> which <> part]
> There's something common, vulgar, in flirting with one's [governess <> those <> indeed]
> Then a good-humored and rather pitiful smile showed itself on his handsome [face <> unassuming <> charming]
> She was just attempting to do what she had attempted to do ten times already in these last three [days <> four <> four]

Next, we do a vanilla version of next_word generation by training an LSTM
that takes the first 25 words as input and returns a word whose euclidean distance
is closest to its output.


"""

from __future__ import absolute_import
from __future__ import print_function
from utils import *
from lstm_net import *

import os
import io
import argparse
from tqdm import tqdm
import numpy as np

LARGE_NUM = 1e10


def parse_args():
  parser = argparse.ArgumentParser()
  pwd = os.path.expanduser(".")
  input_dir = os.path.join(pwd, "input")
  output_dir = os.path.join(pwd, "output")
  glove_dir = os.path.join(pwd, "../lambda/preprocess/GloVe-1.2/glove.6B")
  parser.add_argument("--glove_dir", default=glove_dir)
  parser.add_argument("--glove_corpus", default="6B")
  parser.add_argument("--glove_vec_size", default=100, type=int)
  parser.add_argument("--wordbank_file", default=os.path.join(input_dir, "wordbank.txt"))
  parser.add_argument("--input_file", default=os.path.join(input_dir, "input.txt"))
  parser.add_argument("--target_file1", default=os.path.join(output_dir, "output1.txt"))
  parser.add_argument("--target_file2", default=os.path.join(output_dir, "output2.txt"))
  parser.add_argument("--train_dir", default=os.path.join(input_dir, "traintext"))
  parser.add_argument("--model_file", default=os.path.join(pwd, "model.json"))
  parser.add_argument("--weight_file", default=os.path.join(pwd, "model.h5"))
  return parser.parse_args()


def main():
  args = parse_args()
  word2vec_dict, vec2word_dict = get_encoders(args.glove_dir, args.glove_corpus, args.glove_vec_size)
  listified_text = works_by_author(args.train_dir, "tolstoy")
  tokenized_text, common_dict = tokenize_text(listified_text, word2vec_dict, args.glove_vec_size)
  tokenized_text = one_hot_embed(tokenized_text)

  def train():
    # tokenized by index for one hot vectors
    X_train, y_train = splice_up_text(tokenized_text, args.glove_vec_size)
    train_lstm(X_train, y_train, MOST_COMMON_SIZE, args.model_file, args.weight_file)

  def test():
    test_text = load_test_file(args.input_file)
    tokenized_text, _ = tokenize_text(test_text, word2vec_dict, args.glove_vec_size)
    tokenized_text = one_hot_embed(tokenized_text)
    reverse_common = dict((index, word) for word, index in common_dict.iteritems())
    y_hat, true_y = test_lstm(tokenized_text, args.model_file, args.weight_file)
    word_pred, num_tokens = [], len(y_hat)
    for i, idx in enumerate(tqdm(y_hat)):
      print("Translating predicted token {0}/{1} to word...".format(i+1, num_tokens))
      #raise ValueError(token, len(token))
      word_pred.append(reverse_common[int(idx)])
      #word_pred.append(nearest_word(vec2word_dict, token))
      print(test_text[i:i+25], word_pred[i])

  train()
  test()

if __name__ == "__main__":
  main()
