# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 23:59:31 2020

@author: Hamza
"""

import pandas as pd
import re
import string as st
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
from sklearn.linear_model import LogisticRegression as lr
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score as acc,precision_score as prec, recall_score as rec
import pickle #for saving and loading the file


train_data=pd.read_csv(r'D:\Projects\Project Resources\IMDB Dataset.csv')#reading the data
train_data.shape#shows the shape
train_data.head(10)#shows what the top 10 data looks like


def cleaning(text):
    text=text.lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub('[%s]'%re.escape(st.punctuation),'',text)
    text = re.sub('\w*\d\w*','',text)
    text = re.sub('[‘’“”…]','',text)
    text = re.sub('\n','',text)
    return text
#removing all the new line and special characters and converting everything into lower case
    
train_data['Description']=pd.DataFrame(train_data.review.apply(cleaning))#adding the cleaned  data in the training data
train_data.head()

#preparation for training
independent_var=train_data.review
dependent_var=train_data.sentiment
IV_train,IV_test,DV_train,DV_test=tts(independent_var,dependent_var,test_size=0.1,random_state=225)

print('IV_train: ', len(IV_train))
print('IV_test: ', len(IV_test))
print('DV_train: ', len(DV_train))
print('DV_test: ', len(DV_test))

#the data s split into training and test sets. Training part builds the model and test part tests for the accuracy.here, size of test set in 10% of training set

'''random state?
When the Random_state is not defined in the code for every run train data will change and accuracy might change 
for every run. When the Random_state = " constant integer" is defined then train data will be constant For every run
so that it will make easy to debug'''

#training

vectorizer=tfidf()
classifier=lr(solver = "lbfgs")#this solver is used for smaller datasets.
model=Pipeline([('vectorizer',vectorizer),('classifier',classifier)])
model.fit(IV_train, DV_train)
predictions=model.predict(IV_test)

'''
Scikit-learn is a free software machine learning library for the Python programming language. 
It features various classification, regression and clustering algorithms including support vector machines.

With Tfidfvectorizer you compute the word counts, idf and tf-idf values all at once. It’s really simple.

Logistic regression is a classification algorithm used to assign observations to a discrete set of classes. 
Some of the examples of classification problems are Email spam or not spam, Online transactions Fraud or not Fraud, Tumor Malignant or Benign. 
Logistic regression transforms its output using the logistic sigmoid function to return a probability value.
review positive or negative in this case.

it is a predictive analysis algorithm and based on the concept of probability.

pipelines are used to make the code look cleaner

'''

#checking the performance

print("Accuracy: ",acc(predictions,DV_test))
print("Precision: ",prec(predictions,DV_test,average='weighted'))
print("Recall: ",rec(predictions,DV_test,average='weighted'))

'''
The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. 
The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.

weighted: Calculate metrics for each label, and find their average weighted by support 

The recall is intuitively the ability of the classifier to find all the positive samples

'''

with open('sentimental analyze on movie reviews','wb') as f:
    pickle.dump(model,f)
with open('sentimental analyze on movie reviews','rb') as f:
    sentimental_analyze=pickle.load(f)

#lets test it..
test_review=["it was great"]
result=sentimental_analyze.predict(test_review) 
if result[0]=='happy':
    print("positive")
else:
    print("negative")





