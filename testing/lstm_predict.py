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

course_title = "financial_management"
course_description = {'col':["This is an introductory course in financial management, with an emphasis on the major decisions made by the financial executive of an organization. The student studies topics in the financial management of profit-seeking organizations. A major objective is the development of analytical and decision-making skills in finance through the use of theory questions and practical problems."]}

stopword_list = ["the", "and", "of", "a", "an", "is", "to",
                 "are", "was", "were", "this", "that",
                 "be", "s", "for", "with", "it", "say", 
                 "i", "must", "some"]

from keras.models import load_model
model = load_model("lstm.h5")

train_texts = pd.DataFrame(course_description)

train_texts = train_texts['col']

print(train_texts)

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

output = model.predict(course_description)

for i in range(size(output)):
    if(output[i]>=0.5):
        output[i] = 1
    else:
        output[i] = 0

print(output)

#there is a bug here
