#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import os
from nltk import word_tokenize


# In[2]:


def load_data(data):
    with open(data, 'r') as f:
        return f.read().splitlines()


# In[3]:


def revert_BPE(sentence):
    sentence = sentence.replace("<EOS>", "")
    command = 'echo "{}" | sed -E "s/(@@ )|(@@ ?$)//g"'.format(sentence)
    give_command = Popen(args=command, stdout=PIPE, shell=True).communicate()
    reverted_sentence = give_command[0]
    return reverted_sentence.decode('utf-8', 'ignore')


# In[4]:


import sentencepiece as spm
spm.SentencePieceTrainer.Train('--input=dataset/Donghwan_vg --model_prefix=m --vocab_size=1000')


# In[5]:


sp = spm.SentencePieceProcessor()
sp.Load("m.model")


# In[9]:





# In[17]:





# In[18]:





# In[ ]:




