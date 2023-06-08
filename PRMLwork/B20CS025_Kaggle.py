#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from warnings import filterwarnings
filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
np.set_printoptions(precision=4)
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score , confusion_matrix
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')


# In[2]:


train = pd.read_csv(r'C:\Users\Kartik\Desktop\PRML Kaggle\train\train')
test = pd.read_csv(r'C:\Users\Kartik\Desktop\PRML Kaggle\test\test')


# In[3]:


X_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]
X_test = test.iloc[:, :]


# In[74]:


X_train_df = StandardScaler().fit_transform(X_train)
X_test_df = StandardScaler().fit_transform(X_test)


# In[4]:


xgb_cl = xgb.XGBClassifier()


# In[5]:


print(type(xgb_cl))


# In[6]:


y_train_processed = y_train.values.reshape(-1, 1)
X_train_df_processed = X_train


# In[7]:


X_train_sample, X_test_sample, y_train_sample, y_test_sample = train_test_split(X_train_df_processed, y_train_processed, test_size = 0.2, random_state = 1234)


# In[8]:


xgb_cl.fit(X_train_sample, y_train_sample)
preds = xgb_cl.predict(X_test_sample)


# In[9]:


from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


# In[10]:


classifiers = [[CatBoostClassifier(verbose=0),'CatBoost Classifier'],[XGBClassifier(),'XGB Classifier'], [RandomForestClassifier(),'Random Forest'], 
    [KNeighborsClassifier(), 'K-Nearest Neighbours'], [SGDClassifier(),'SGD Classifier'], [SVC(),'SVC'],[LGBMClassifier(),'LGBM Classifier'],
              [GaussianNB(),'GaussianNB'],[DecisionTreeClassifier(),'Decision Tree Classifier'],[LogisticRegression(),'Logistic Regression']]


# In[11]:


#for cls in classifiers:
model = classifiers[0][0]
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)


# In[12]:


y_pred = pd.DataFrame(y_pred[:,1], columns=['predicted']).to_csv('new.csv')


# In[13]:


y_pred

