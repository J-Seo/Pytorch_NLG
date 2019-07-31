# -*- coding: utf-8 -*-
"""tokenize_preprocess.ipynb의 사본

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10xGlIo5KWV5RtCNfeUn7KBTCNJDeL8bU
"""

!pip install https://github.com/rsennrich/subword-nmt/archive/master.zip

from __future__ import absolute_import, division, print_function, unicode_literals

!pip install -q tensorflow-gpu==2.0.0-beta0
import tensorflow as tf

import numpy as np
import os

subword-nmt get-vocab --train_file {train_file} --vocab_file {vocab_file}
subword-nmt segment-char-ngrams --vocab {vocab_file} -n {order} --shortlist {size} < {test_file} > {out_file}

url_list = ['https://drive.google.com/open?id=1I4kynBxgLvy6ukPxt2Nm0e7r67U4xTls',
'https://drive.google.com/open?id=1DKmfllIV5_Y178ubYHVnXITVrbi8IDcZ', 
'https://drive.google.com/open?id=1Ht4ZI12wNSkm5I6g3-KChka9CIyyk36c',
'https://drive.google.com/open?id=1G0ENNV49lYcNpxZE28xcdSOfJgJRoc1G',
'https://drive.google.com/open?id=1PDDfrhdJHYCPTvhP_gF4ArmgOmJLp36O',
'https://drive.google.com/open?id=1HIGcwke5FSDboO9e1YrFjYVIduvix34M']

  
file_names = ['text1.txt', 'text2.txt', 'text3.txt', 'text4.txt', 'text5.txt', 'text6.txt']
i = 0

for name in file_names:
  text_dir = tf.keras.utils.get_file(name, origin = url_list[i])
  i += 1
  
parent_dir = os.path.dirname(text_dir)
parent_dir

os.system("./learn_bpe.py -s 30000 < parent_dir > parent.dir.bpe")

os.system("parent.dir.bpe")

def labeler(example, index):
  return example, tf.cast(index, tf.int64)  

labeled_data_sets = []

for i, file_name in enumerate(file_names):
  lines_dataset = tf.data.TextLineDataset(os.path.join(parent_dir, file_name))
  labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
  labeled_data_sets.append(labeled_dataset)

BUFFER_SIZE = 50000
BATCH_SIZE = 64
TAKE_SIZE = 5000

all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
  all_labeled_data = all_labeled_data.concatenate(labeled_dataset)
  
all_labeled_data = all_labeled_data.shuffle(
    BUFFER_SIZE, reshuffle_each_iteration=False)

for ex in all_labeled_data.take(5):
  print(ex)

## 토큰화하기
tokenizer = tfds.features.text.Tokenizer()

vocabulary_set = set()
for text_tensor, _ in all_labeled_data:
  some_tokens = tokenizer.tokenize(text_tensor.numpy())
  vocabulary_set.update(some_tokens)

vocab_size = len(vocabulary_set)
vocab_size

encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

example_text = next(iter(all_labeled_data))[0].numpy()
print(example_text)

encoded_example = encoder.encode(example_text)
print(encoded_example)

def encode(text_tensor, label):
  encoded_text = encoder.encode(text_tensor.numpy())
  return encoded_text, label

def encode_map_fn(text, label):
  return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))

all_encoded_data = all_labeled_data.map(encode_map_fn)

train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([-1],[]))

test_data = all_encoded_data.take(TAKE_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([-1],[]))

sample_text, sample_labels = next(iter(test_data))

sample_text[0], sample_labels[0]

os.system("subword-nmt learn-bpe -s 30000 < train_data > train_data.bpe")

train_data.bpe

os.system("subword-nmt apply-bpe -c merge_text.en.bpe < train_data.en > train_data_final.en")

os.system("subword-nmt learn-bpe -s 30000 < test_data.en > test_data.en.bpe")





