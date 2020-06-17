#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[1]:


# Importing all libraries
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import pickle


# In[2]:


# Uploading the dataset
data = pd.read_csv('revdata.txt',sep='\t',quoting=3)
y = data.iloc[:,1].values
data.shape


# In[3]:


data.head(5)


# # Data Preprocessing

# In[4]:

nltk.download('stopwords')
wordnet = WordNetLemmatizer()
# Cleaning the reviews
corpus = []
for i in range(len(y)):

  # Cleaning special character from the reviews
    review = re.sub(pattern='[^a-zA-Z]',repl=' ', string=data['Review'][i])

  # Converting the entire review into lower case
    review = review.lower()

  # Tokenizing the review by words
    review_words = review.split()

  # Removing the stop words
    review_words = [word for word in review_words if not word in set(stopwords.words('english'))]

  # Stemming the words
    review = [wordnet.lemmatize(word) for word in review_words]

  # Joining the stemmed words
    review = ' '.join(review)

  # Creating a corpus
    corpus.append(review)
corpus[0:5]


# # Bag of words model

# In[5]:


cv = CountVectorizer()
x = cv.fit_transform(corpus).toarray()
x


# # Splitting data for training and testing and fitting the model

# In[14]:


x_train,x_test,y_train,y_test = train_test_split(x,y)
classifier = MultinomialNB(alpha=0.2)
classifier.fit(x_train,y_train)


# In[7]:


# Saving model to disk
pickle.dump(classifier, open('model.pkl','wb'))


# In[15]:


#Scores
y_pred = classifier.predict(x_test)
score1 = accuracy_score(y_test,y_pred)
score2 = precision_score(y_test,y_pred)
score3= recall_score(y_test,y_pred)
print("---- Scores ----")
print("Accuracy score is: {}%".format(round(score1*100,2)))
print("Precision score is: {}".format(round(score2,2)))
print("Recall score is: {}".format(round(score3,2)))


# In[16]:


# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm


# # Predictions 

# In[10]:


def prediction(sample_review):
  sample_review = re.sub(pattern='[^a-zA-Z]',repl=' ', string = sample_review)
  sample_review = sample_review.lower()
  sample_review_words = sample_review.split()
  sample_review_words = [word for word in sample_review_words if not word in set(stopwords.words('english'))]
  wordnet = WordNetLemmatizer()
  final_review = [wordnet.lemmatize(word) for word in sample_review_words]
  final_review = ' '.join(final_review)

  temp = cv.transform([final_review]).toarray()
  return temp


# In[11]:


# Predicting values
sample_review = 'I love all of your quotes.'
a = prediction(sample_review)
b = classifier.predict(a)
if b:
  print('This is a POSITIVE review.')
else:
  print('This is a NEGATIVE review!')


# In[12]:


# Predicting values
sample_review = 'Your writeups are mostly sad, i dont like it.'
a = prediction(sample_review)
b = classifier.predict(a)
if b:
  print('This is a POSITIVE review.')
else:
  print('This is a NEGATIVE review!')

