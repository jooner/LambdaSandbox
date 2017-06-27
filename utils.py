# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function

from tqdm import tqdm
from keras.preprocessing import text
from scipy.spatial import distance

import numpy as np

import io
import os
import operator

NU = 0.005
EMBEDDING_SIZE = 100
MOST_COMMON_SIZE = 22000
LARGE_NUM = 1e10

def get_words(file_path):
  wordbank = {}
  with io.open(file_path, 'r', encoding='utf-8') as wb:
    for line in wb:
      line = line.lstrip().rstrip().split(" ")
      wordbank[line[0]] = line[1:]
  return wordbank

def get_single_score(keyword, complist, worddict):
  keyvec = worddict[keyword]
  score_list = []
  for w in complist:
    score_list.append(get_dist(keyvec, worddict[w]))
  return score_list

def evaluate_scores(target_file, wordbank, word2vec_dict):
  for word in wordbank:
    dis = word + str(get_single_score(word, wordbank[word], word2vec_dict)) + '\n'
    write_to_output(target_file, dis)

def get_last_words(input_file):
  last_word_list = []
  with io.open(input_file, 'r', encoding='utf-8') as rf:
    for line in rf:
      last_word = line.split(" ")[-1].strip("\n")
      last_word_list.append(last_word)
  return last_word_list

def one_hot_embed(indexed_text):
  """Converts list of indexed text to numpy array of one-hot vectors"""
  indexed_text = np.array(indexed_text)
  one_hot = np.zeros((len(indexed_text), MOST_COMMON_SIZE))
  one_hot[np.arange(len(indexed_text)), indexed_text] = 1
  return one_hot

def naive_next_word(word2vec_dict, input_file):
  next_word_list = []
  last_words = get_last_words(input_file)
  for word in last_words:
    vec_of_word = word2vec_dict[word]
    min_dis = LARGE_NUM
    next_word = ""
    for w in word2vec_dict:
      new_dis = get_dist(vec_of_word, word2vec_dict[w])
      if new_dis < min_dis and new_dis != 0:
        min_dis = new_dis
        next_word = w
    next_word_list.append(next_word)
  return next_word_list

def write_to_output(target_file, subject):
  with io.open(target_file, 'w', encoding='utf-8') as tf:
    subject = subject.decode('utf-8')
    tf.write(subject)

def get_dist(vec1, vec2):
  return distance.euclidean(vec1, vec2)

def get_encoders(glove_dir, glove_corpus, glove_vec_size):
  """
  Returns a dictionary that can tokenize words 
  """
  glove_path = os.path.join(glove_dir, "glove.{}.{}d.txt".format(glove_corpus, glove_vec_size))
  sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
  total = sizes[glove_corpus]
  word2vec_dict = {}
  vec2word_dict = {}
  #all_words = ""
  with io.open(glove_path, 'r', encoding='utf-8') as fh:
    for line in tqdm(fh, total=total):
      array = line.split(" ")
      word = array[0].lower() # lower case all inputs
      #all_words += " {0}".format(word)
      # cache as numpy array for Keras
      vector = np.asarray(list(map(float, array[1:])))
      word2vec_dict[word] = vector
      vec2word_dict[tuple(vector)] = word # cache numpy array s.t. it is hashable
  #token1hot = text.one_hot(all_words, total)
  print("Vectorized {0} words from GloVe.{1}.{2}d".format(len(word2vec_dict), glove_corpus, glove_vec_size))
  return word2vec_dict, vec2word_dict

def works_by_author(text_dir, author):
  text_file_dir = os.path.join(text_dir, author)
  texts = [] # list of words in text
  for name in sorted(os.listdir(text_file_dir)):
    fpath = os.path.join(text_file_dir, name)
    with open(fpath) as t:
      # txt = t.read().lower().split()
      print("Cleaing up input text...")
      txt = text.text_to_word_sequence(t.read())
      texts.extend(txt)
  print('Found %s texts.' % len(texts))
  return texts

def add_noise(tokenized_word):
  noise_vector = NU * np.random.randn(EMBEDDING_SIZE)
  return np.add(noise_vector, tokenized_word)

def filter_most_common(word_dict):
  common_dict = {}
  filter_size = MOST_COMMON_SIZE - 1
  filtered_dict = dict(sorted(word_dict.iteritems(), key=operator.itemgetter(1), reverse=True)[:filter_size])
  for idx, word in enumerate(filtered_dict):
    common_dict[word] = idx
  common_dict["<UNK>"] = MOST_COMMON_SIZE
  return common_dict # size of MOST_COMMON_SIZE - 1 (leave one index for unknown tokens)

def tokenize_text(listified_txt, word2vec_dict, vec_size):
  occurences = {}
  existing = 0
  tokenized_text = []
  embedded_text = []
  total = len(listified_txt)
  #word_ahead = [0] * vec_size
  for word in tqdm(listified_txt, total=total):
    if word in occurences:
      occurences[word] += 1
    else:
      occurences[word] = 1
      #existing += 1
      #tokenized_word = np.asarray(word2vec_dict[word])
      #tokenized_text.append(tokenized_word)
      #existing += 1
      #word_ahead = tokenized_word
  common_dict = filter_most_common(occurences)
  for word in tqdm(listified_txt, total=total):
    try:
      # ensure both have to exist for encoding to happen
      cw, vw = common_dict[word], word2vec_dict[word]
      tokenized_text.append(cw)
      embedded_text.append(vw)
      existing += 1
    # if the word either does not have a glove embedding
    # or is not a common word then yield unknown token 
    except KeyError:
      embedded_text.append([LARGE_NUM] * vec_size) # the <unknown> embedding
      tokenized_text.append(MOST_COMMON_SIZE - 1) # the <unknown> token

    # except KeyError: # word is not in embedding dict
    #  word_estimate = add_noise(word_ahead)
    #  tokenized_text.append(word_estimate)
  print("{0}/{1} words successfully embedded".format(existing, total))
  # cache as nested numpy array
  return np.asarray(tokenized_text), np.asarray(embedded_text), common_dict

def nearest_word(vec2word_dict, token):
  old_dist = float('inf')
  closest_word = ""
  for vec in vec2word_dict:
    new_dist = get_dist(np.asarray(vec), token) # unwrap tuple-cached vec (done for hashability) 
    if new_dist < old_dist:
      old_dist = new_dist
      closest_word = vec2word_dict[vec]
  return closest_word
