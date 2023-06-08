#!/usr/bin/env python
# coding: utf-8

# # Question 1

# In[1]:


import librosa
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import pandas as pd
import wave
get_ipython().system('pip install pydub')
from pydub import AudioSegment


# ## 1. Reading the Signals

# In[2]:


x1, sr1 = librosa.load(r"C:\Users\Kartik\Desktop\Lab 8\Dataset(Lab8)\signal_1.wav")
x2, sr2 = librosa.load(r"C:\Users\Kartik\Desktop\Lab 8\Dataset(Lab8)\signal_2.wav")
x3, sr3 = librosa.load(r"C:\Users\Kartik\Desktop\Lab 8\Dataset(Lab8)\signal_3.wav")


# In[3]:


print(x1.shape)
print(sr1)


# ## 1. Visualising the signals

# In[4]:


plt.figure(figsize=(14, 5))
plt.title("Signal 1")
librosa.display.waveshow(x1, sr=sr1)


# In[5]:


plt.figure(figsize=(14, 5))
plt.title("Signal 2")
librosa.display.waveshow(x2, sr=sr2)


# In[6]:


plt.figure(figsize=(14, 5))
plt.title("Signal 3")
librosa.display.waveshow(x3, sr=sr3)


# ## 1. Listening to the audio

# In[7]:


import IPython.display as ipd
ipd.Audio(r"C:\Users\Kartik\Desktop\Lab 8\Dataset(Lab8)\signal_1.wav")


# In[8]:


ipd.Audio(r"C:\Users\Kartik\Desktop\Lab 8\Dataset(Lab8)\signal_2.wav")


# In[9]:


ipd.Audio(r"C:\Users\Kartik\Desktop\Lab 8\Dataset(Lab8)\signal_3.wav")


# ## 2. Creating the Dataset

# In[11]:


X = list(zip(x1, x2, x3))
X = np.array(X)


# ## 3. Implementing ICA from scratch

# In[12]:


cols = ['family',
 'product-type',
 'steel',
 'carbon',
 'hardness',
 'temper_rolling',
 'condition',
 'formability',
 'strength',
 'non-ageing',
 'surface-finish',
 'surface-quality',
 'enamelability',
 'bc',
 'bf',
 'bt',
 'bw/me',
 'bl',
 'm',
 'chrom',
 'phos',
 'cbond',
 'marvi',
 'exptl',
 'ferro',
 'corr',
 'blue/bright/varn/clean',
 'lustre',
 'jurofm',
 's',
 'p',
 'shape',
 'thick',
 'width',
 'len',
 'oil',
 'bore',
 'packing',
 'class']


# In[6]:


df1.columns = cols


# In[7]:


df2.columns = cols


# In[8]:


df = pd.concat([df1, df2])


# In[9]:


df.reset_index(inplace = True, drop=True)
df


# In[10]:


print("Missing Values are:")
droppingColumns = []
for ci, column in enumerate(cols):
    count = 0
    for j in range(len(df)):
        if(df.iloc[j, ci]=='?'):
            count+=1
    if(count>=240):
        droppingColumns.append(column)
    print(column, "------>", count)


# In[11]:


df.drop(droppingColumns, axis=1, inplace = True)


# In[12]:


df.head()


# In[13]:


print('Missing Values: ')
droppingColumns = []
for ci,column in enumerate(df.columns):
    count = 0
    for j in range(len(df)):
        if df.iloc[j , ci] == '?':
            count += 1
    if count > 1:
        droppingColumns.append(column)
    print(column, '===>', count)


# In[14]:


for ji, j in enumerate(df['steel']):
    if j == '?':
        df.drop(ji,inplace = True)


# In[15]:


df.reset_index(inplace = True , drop = True)


# In[16]:


print('Unique Values')
for i in df.columns:
    print(i ,'==>' ,df[i].nunique() )


# In[17]:


df.drop(['product-type'] , inplace = True, axis = 1)
df


# In[18]:


def encode(df, colums):
    for col in colums:
        dic = {}
        count = 0
        for ji,j in enumerate(df[col]):
            if j not in dic:
                dic[j] = count
                df.loc[ji, col] = dic[j]
                count += 1
            else:
                df.loc[ji, col] = dic[j]
    return df


# In[19]:


df = encode(df, ['shape','bore','class','steel'])


# In[20]:


df['shape'] = df['shape'].astype('int64')


# In[21]:


df['class'] = df['class'].astype('int64')


# In[22]:


df['steel'] = df['steel'].astype('int64')


# In[23]:


df.info()


# In[102]:


sns.pairplot(df,hue="class")


# In[24]:


df


# In[25]:


df.describe()


# ## splitting the dataset

# In[26]:


X = np.array(df.iloc[:, 0:-1])
y = np.array(df.iloc[:, -1])


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.35, random_state = 12)


# #  3. Training on Three different models :
# ## Model1 = Decision Tree Classifier

# In[28]:


m1 = DecisionTreeClassifier()


# In[29]:


m1_score  =  cross_val_score(m1, X, y, cv=5)


# In[30]:


fold = [1, 2, 3, 4, 5]


# In[31]:


plt.title("5 Fold Cross Validation for Decision Tree Classifier") 
plt.xlabel("Folds") 
plt.ylabel("Score") 
plt.plot(fold, m1_score) 


# ## Model2 = Gaussian Classifier

# In[32]:


m2 = GaussianNB()


# In[33]:


m2_score  =  cross_val_score(m2, X, y, cv=5)


# In[34]:


plt.title("5 Fold Cross Validation for Gaussian Classifier") 
plt.xlabel("Folds") 
plt.ylabel("Score") 
plt.plot(fold, m2_score) 


# ## Model3 = Multinomial Classifier

# In[35]:


m3 = MultinomialNB()


# In[36]:


m3_score = cross_val_score(m3, X, y, cv=5)


# # Question 2

# ## 1. Preprocessing the data

# In[31]:


data = pd.read_csv(r"C:\Users\Kartik\Desktop\Lab 8\Arlline\train.csv")


# In[32]:


data


# In[33]:


cols  = data.columns
(list(cols))


# In[34]:


data.head()


# In[35]:


data.info()


# In[36]:


data.describe()


# In[37]:


data.drop(['Unnamed: 0'], axis = 1, inplace=True)


# In[38]:


data


# In[39]:


from sklearn.preprocessing import LabelEncoder


# In[40]:


encoder = LabelEncoder()


# In[41]:


for i in range(len(data.columns)):
    data.iloc[:, i] = encoder.fit_transform(data.iloc[:, i])


# In[42]:


data.head()


# In[43]:


#satisfaction is the label here and the rest are the features 


# ## Separating features and labels as X and y

# In[44]:


X = data.iloc[:, :-1]


# In[45]:


y = data.iloc[:, -1:]


# In[46]:


corr_matrix = X.corr()


# In[47]:


import seaborn as sns


# In[48]:


#creating the correlation heatmap
plt.figure(figsize=(30, 30))
plt.title("Correlation HeatMap")
a = sns.heatmap(corr_matrix, square=True, annot = True, fmt='.2f', linecolor='black')
a.set_xticklabels(a.get_xticklabels(), rotation=30)
a.set_yticklabels(a.get_yticklabels(), rotation=30)  
plt.show()


# In[49]:


from sklearn.tree import DecisionTreeClassifier


# In[50]:


DTC = DecisionTreeClassifier(random_state=0)


# In[51]:


from mlxtend.feature_selection import SequentialFeatureSelector as SFS


# ## 2. Creating an object of SFS by embedding decision tree classifier object , providing 10 features , forward = true, floating = false , scoring = accuracy

# In[52]:


sfs = SFS(DTC, k_features=10, forward=True, floating = False, scoring='accuracy')
sfs = sfs.fit(X, y)


# In[53]:


sfs.subsets_[10]


# In[55]:


sfs.subsets_


# ## 3. On training SFS, the accuracy for all the 10 features after cv=5 and the names of selected features are

# In[56]:


(sfs.subsets_[10])['avg_score']*100


# In[57]:


(sfs.subsets_[10])['feature_names']


# ## 4. Using the forward and Floating parameter toggle between SFS(forward True, floating False), SBS (forward False, floating False), SFFS (forward True, floating True), SBFS (forward False, floating True), and choose cross validation = 4 for each configuration. Also report cv scores for each configuration.

# In[58]:


sbs = SFS(DTC, k_features=10, forward=False, floating=False, scoring='accuracy', cv=4)
sbs = sbs.fit(X, y)
print('\nSequential Backward Selection:')
print(sbs.k_score_)


# In[59]:


sffs = SFS(DTC, k_features=10, forward=True, floating = True, scoring='accuracy', cv=4)
sffs = sffs.fit(X, y)
print('\nSequential Forward Floating Selection:')
print(sffs.k_score_)


# In[60]:


sbfs = SFS(DTC, k_features=10, forward=False, floating = True, scoring='accuracy', cv=4)
sbfs = sbfs.fit(X, y)
print('\nSequential Backward Floating Selection:')
print(sbfs.k_score_)


# In[61]:


sfs = SFS(DTC, k_features=10, forward=True, floating=False, scoring='accuracy', cv=4)
sfs = sfs.fit(X, y)
print('\nSequential Forward Selection:')
print(sfs.k_score_)


# In[66]:


left = [1, 2, 3, 4]
height = [0.9499826763165999, 0.9514744379427165, 0.9482695565137048, 0.9499826763165999]
 
tick_label = ['SBS', 'SFFS', 'SBFS', 'SFS']
 
plt.bar(left, height, tick_label = tick_label,
        width = 0.8, color = ['red', 'green', 'blue', 'yellow'])
 
plt.xlabel('Permutations')
plt.ylabel('CV Score')
plt.title('SFS Variants')
plt.show()


# ## 5. Visualize the output from the feature selection in a pandas DataFrame format using the get_metric_dict for all four configurations.

# In[67]:


pd.DataFrame.from_dict(sfs.get_metric_dict()).T


# In[68]:


pd.DataFrame.from_dict(sbs.get_metric_dict()).T


# In[69]:


pd.DataFrame.from_dict(sffs.get_metric_dict()).T


# In[70]:


pd.DataFrame.from_dict(sbfs.get_metric_dict()).T


# In[71]:


from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs


# In[72]:


import matplotlib.pyplot as plt


# ## 6. Plotting the results

# In[73]:


sfs_plot = plot_sfs(sfs.get_metric_dict(), kind='std_dev')
plt.ylim([0.6, 1])
plt.title('Sequential Forward Selection')
plt.grid()
plt.show()


# In[74]:


sbs_plot = plot_sfs(sbs.get_metric_dict(), kind='std_dev')
plt.ylim([0.6, 1])
plt.title('Sequential Backward Selection')
plt.grid()
plt.show()


# In[75]:


sffs_plot = plot_sfs(sffs.get_metric_dict(), kind='std_dev')
plt.ylim([0.6, 1])
plt.title('Sequential Forward Floating Selection')
plt.grid()
plt.show()


# In[76]:


sbfs_plot = plot_sfs(sbfs.get_metric_dict(), kind='std_dev')
plt.ylim([0.6, 1])
plt.title('Sequential Backward Floating Selection')
plt.grid()
plt.show()


# ## Twitching the number of features in sfs
# ## By increasing the features 

# In[58]:


sfs = SFS(DTC, k_features=15, forward=True, floating = False, scoring='accuracy')
sfs = sfs.fit(X, y)


# In[59]:


sfs_plot = plot_sfs(sfs.get_metric_dict(), kind='std_dev')
plt.ylim([0.6, 1])
plt.title('Sequential Forward Selection')
plt.grid()
plt.show()


# ## By decreasing the features  

# In[60]:


sfs = SFS(DTC, k_features=8, forward=True, floating = False, scoring='accuracy')
sfs = sfs.fit(X, y)
sfs_plot = plot_sfs(sfs.get_metric_dict(), kind='std_dev')
plt.ylim([0.6, 1])
plt.title('Sequential Forward Selection')
plt.grid()
plt.show()


# In[ ]:
# In[10]:


print("Missing Values are:")
droppingColumns = []
for ci, column in enumerate(cols):
    count = 0
    for j in range(len(df)):
        if(df.iloc[j, ci]=='?'):
            count+=1
    if(count>=240):
        droppingColumns.append(column)
    print(column, "------>", count)


# In[11]:


df.drop(droppingColumns, axis=1, inplace = True)


# In[12]:


df.head()


# In[13]:


print('Missing Values: ')
droppingColumns = []
for ci,column in enumerate(df.columns):
    count = 0
    for j in range(len(df)):
        if df.iloc[j , ci] == '?':
            count += 1
    if count > 1:
        droppingColumns.append(column)
    print(column, '===>', count)


# In[14]:


for ji, j in enumerate(df['steel']):
    if j == '?':
        df.drop(ji,inplace = True)


# In[15]:


df.reset_index(inplace = True , drop = True)


# In[16]:


print('Unique Values')
for i in df.columns:
    print(i ,'==>' ,df[i].nunique() )


# In[17]:


df.drop(['product-type'] , inplace = True, axis = 1)
df


# In[18]:


def encode(df, colums):
    for col in colums:
        dic = {}
        count = 0
        for ji,j in enumerate(df[col]):
            if j not in dic:
                dic[j] = count
                df.loc[ji, col] = dic[j]
                count += 1
            else:
                df.loc[ji, col] = dic[j]
    return df


# In[19]:


df = encode(df, ['shape','bore','class','steel'])


# In[20]:


df['shape'] = df['shape'].astype('int64')


# In[21]:


df['class'] = df['class'].astype('int64')


# In[22]:


df['steel'] = df['steel'].astype('int64')


# In[23]:


df.info()


# In[102]:


sns.pairplot(df,hue="class")


# In[24]:


df


# In[25]:


df.describe()


# ## splitting the dataset

# In[26]:


X = np.array(df.iloc[:, 0:-1])
y = np.array(df.iloc[:, -1])


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.35, random_state = 12)


# #  3. Training on Three different models :
# ## Model1 = Decision Tree Classifier

# In[28]:


m1 = DecisionTreeClassifier()


# In[29]:


m1_score  =  cross_val_score(m1, X, y, cv=5)


# In[30]:


fold = [1, 2, 3, 4, 5]


# In[31]:


plt.title("5 Fold Cross Validation for Decision Tree Classifier") 
plt.xlabel("Folds") 
plt.ylabel("Score") 
plt.plot(fold, m1_score) 


# ## Model2 = Gaussian Classifier

# In[32]:


m2 = GaussianNB()


# In[33]:


m2_score  =  cross_val_score(m2, X, y, cv=5)



