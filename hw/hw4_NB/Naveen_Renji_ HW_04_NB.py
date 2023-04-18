#!/usr/bin/env python
# coding: utf-8

# # Name : Naveen Mathews Renji | CWID : 20016323 

# ## Import Libraries

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# ## Load the dataset

# In[2]:


df = pd.read_csv("breast-cancer-wisconsin.csv")


# In[3]:


df.head()


# In[4]:


num_rows = df.shape[0]
print(num_rows)


# ## Taking care of missing values

# In[5]:


df['F6']=df['F6'].replace('?',np.nan)
df = df.dropna()
del df['Sample']


# In[6]:


df.describe()


# ## Creating NB Model and classifying the training data

# In[7]:


y=df['Class']
y=np.where(y==2,0,1)
X=df.drop('Class',axis=1).values


# In[8]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30)


# ## Guassian Naive Bayes

# In[9]:


nbg = GaussianNB()
nbg.fit(X_train, y_train)
nbg.score(X_test, y_test)
Y_pred=nbg.predict(X_test)
print("The Confusion Matrix is -")
print(confusion_matrix(y_test,Y_pred))
print("The Classification Report Matrix is -")
print(classification_report(y_test,Y_pred))


# ## Bernoulli Naive Bayes

# In[10]:


nbb = BernoulliNB()
nbb.fit(X_train, y_train)
nbb.score(X_test, y_test)
Y_pred=nbb.predict(X_test)
print("The Confusion Matrix is -")
print(confusion_matrix(y_test,Y_pred))
print("The Classification Report Matrix is -")
print(classification_report(y_test,Y_pred))


# ## Multinomial Naive Bayes

# In[11]:


nbm = MultinomialNB()
nbm.fit(X_train, y_train)
nbm.score(X_test, y_test)
Y_pred=nbm.predict(X_test)
print("The Confusion Matrix is -")
print(confusion_matrix(y_test,Y_pred))
print("The Classification Report Matrix is -")
print(classification_report(y_test,Y_pred))


# In[ ]:




