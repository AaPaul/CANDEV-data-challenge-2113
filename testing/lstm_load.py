import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, SimpleRNN, Embedding
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras import models, layers
import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
import os

data = pd.read_csv('g.csv', error_bad_lines=False, encoding = "ISO-8859-1")

stopword_list = ["the", "and", "of", "a", "an", "to", "is",
                 "are", "was", "were", "this", "that",
                 "be", "s", "for", "with", "it", "say", 
                 "i", "must", "some"]

data_text = data
documents = data_text

documents['title'] = documents['title'].fillna('999')
documents = documents.drop(documents[(documents.title=='999')].index.tolist())
documents = documents.drop(documents[(documents.title=='Cours de EFPC / CSPS courses')].index.tolist())
documents = documents.drop(documents[(documents.title=='Cours provenant d\'une autre ministère / Other Government Department courses')].index.tolist())
documents = documents.drop(documents[(documents.title=='Externals Ressources Links')].index.tolist())
documents = documents.drop(documents[(documents.title=='Cours de Skillsoft courses')].index.tolist())
documents = documents.drop(documents[(documents.title=='Delegation of Authority - video')].index.tolist())
documents = documents.drop(documents[(documents.title=='Blogs')].index.tolist())
documents = documents.drop(documents[(documents.title=='Cases Studies')].index.tolist())
documents = documents.drop(documents[(documents.title=='hq')].index.tolist())
documents = documents.drop(documents[(documents.title=='Video List')].index.tolist())

for i in documents.index:
    if(' '.join(str(documents['simple description'][i]).split()[:10])==' '.join(str(documents['description for Gccampus'][i]).split()[:10])):
        documents['simple description'][i]= ""

import csv
with open('title.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(documents['title'])

documents['combine'] = documents[['title','simple description','description for Gccampus']].apply(lambda x : '{} {} {}'.format(x[0],x[1],x[2]), axis=1)
from keras.models import load_model
model = load_model("lstm.h5")

train_texts = documents['combine']

max_words = 10000
#import embedding vector with dimension 300
embedding_dimension = 300
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_texts)
sequences = tokenizer.texts_to_sequences(train_texts)
word_index = tokenizer.word_index
# print(f'Found {len(word_index)} unique tokens.')
# #use all kinds of vocabulary
# max_words = len(word_index)

stopword_sequence = []
for sw in stopword_list:
    stopword_sequence.append(word_index[sw])
for s in sequences:
    for i in stopword_sequence:
        while i in s:
            s.remove(i)

glove_dir = os.path.join('../..','glove.6B.300d.txt')
embedding_index = {}
f = open(glove_dir)
for line in f:
  values = line.split()
  word = values[0]
  correlation_vector = np.array(values[1:],dtype='float32')
  embedding_index[word] = correlation_vector
f.close()

embedding_matrix = np.zeros((max_words,embedding_dimension))
for word, i in word_index.items():
  if i < max_words:
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector


max_length = 350
train_data = pad_sequences(sequences,maxlen=max_length)

output = model.predict(train_data, batch_size=32)

np.save('output', output) 

with open('output.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(output)


