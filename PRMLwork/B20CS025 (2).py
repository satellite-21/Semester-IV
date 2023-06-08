#!/usr/bin/env python
# coding: utf-8

# ## Question 1: Principal Component Analysis

# In[75]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB , MultinomialNB
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1. Converting the data into readable format

# In[2]:


df1 = pd.read_csv('anneal.data')
df2 = pd.read_csv('anneal.test')
f = open('anneal.names','r')


# In[3]:


df2


# In[4]:


(df1.columns)


# ## 2. Preprocessing the data

# In[5]:


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


# In[37]:


plt.title("5 Fold Cross Validation for Multinomial Classifier") 
plt.xlabel("Folds") 
plt.ylabel("Score") 
plt.plot(fold, m3_score) 


# ## Five Fold Cross-Validation of the above models

# In[38]:


plt.plot(fold, m1_score, color='r', label='Decision Tree Classifier')
plt.plot(fold, m2_score, color='g', label='Gaussian Classifier')
plt.plot(fold, m3_score, color='b', label='Multinomial Classifier')
plt.xlabel("Folds")
plt.ylabel("Score")
plt.title("Cross Validation Score on Different Models")
plt.legend()
plt.show()


# # 4. Implementing the Principal Component Analysis

# In[39]:


df.corr()


# In[40]:


correlation = df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='viridis')

plt.title('Correlation between different fearures')


# ## Data Standardisation

# In[41]:


#steel
#carbon
#hardness
#strength
#shape
#thick
#width
#len
#bore
X


# In[42]:


from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)


# ## Computing the Covariance Matrix 

# In[43]:


mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std  - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)


# In[44]:


print('Covariance matrix \n%s' %cov_mat)


# In[45]:


plt.figure(figsize=(8, 8))
sns.heatmap(cov_mat, vmax=1, square=True, annot=True, cmap='viridis')
plt.title("Correlation between different features")


# ## Eigen decomposition of the covariance matrix
# 

# In[46]:


def simultaneous_orthogonalisation(A, tol=0.0001):
    Q, R = np.linalg.qr(A)
    previous = np.empty(shape=Q.shape)
    for i in range(100):
        previous[:] = Q
        X = A@Q
        Q, R = np.linalg.qr(X)
        if np.allclose(Q, previous, atol=tol):
            break
    return Q
eigen_vectors = simultaneous_orthogonalisation(cov_mat)


# In[47]:


eigen_values, _ = np.linalg.eigh(cov_mat)
sorted_index = np.argsort(eigen_values)[::-1]
sorted_eigenvalue = eigen_values[sorted_index]
sorted_eigenvectors = eigen_vectors[:,sorted_index]

n_components = 2
eigenvector_subset = sorted_eigenvectors[:,0:n_components]

X_reduced = np.dot(eigenvector_subset.transpose(),mean_vec.transpose()).transpose()
X_reduced


# #### With linalg:

# In[48]:


eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


# ## Selecting Principal Components

# In[49]:


#making a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

eig_pairs.sort(key=lambda x:x[0], reverse=True)

print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])


# In[50]:


s = 0
for i in eig_pairs:
    s+=i[0]


# In[51]:


for i in eig_pairs:
    print(f'{i[0]}-------> {i[0]*100/s}%')


# In[52]:


tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]


# In[53]:


with plt.style.context('dark_background'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(9), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()


# ## Dropping the last two, since the other features contibutes to a good 90+% 

# ## Creating a feature matrix

# In[54]:


matrix_w = np.hstack((eig_pairs[0][1].reshape(9,1), 
                      eig_pairs[1][1].reshape(9,1),
                      eig_pairs[2][1].reshape(9,1),
                      eig_pairs[3][1].reshape(9,1),
                      eig_pairs[4][1].reshape(9,1),
                      eig_pairs[5][1].reshape(9,1),
                      eig_pairs[6][1].reshape(9,1),
                      
                    ))
matrix_w


# In[55]:


Y = X_std.dot(matrix_w)
Y


# In[56]:


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # covariance, function needs samples as columns
        cov = np.cov(X.T)

        # eigenvalues, eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # -> eigenvector v = [:,i] column vector, transpose for easier calculations
        # sort eigenvectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # store first n eigenvectors
        self.components = eigenvectors[0 : self.n_components]

    def transform(self, X):
        # project data
        X = X - self.mean
        return np.dot(X, self.components.T)


# In[57]:


pca = PCA(6)
pca.fit(X)
X_projected = pca.transform(X)


# In[58]:


print("Shape of X:", X.shape)
print("Shape of transformed X:", X_projected.shape)


# In[67]:


target_data = df.loc[:, 'class']
df_data = pd.DataFrame(X_projected)


# In[68]:


df_data


# In[69]:


target_data


# In[70]:


sns.pairplot(df_data)


# In[71]:


X_train,X_test,y_train,y_test = train_test_split(df_data,target_data,train_size = 0.65,random_state=0)


# ## Choosing the first model on the new projected dataset

# ## Decision Tree Classifier

# In[78]:


m1NewScore = cross_val_score(m1, X_train, y_train, cv=5)
print(m1NewScore)


# In[79]:


y_pred = m1.fit(X_train, y_train).predict(X_test)
print(accuracy_score( y_test, y_pred) * 100 , "%")


# In[80]:


print(f1_score(y_test, y_pred, average=None))


# In[81]:


plt.title("5 Fold Cross Validation for Decision Tree Classifier") 
plt.xlabel("Folds") 
plt.ylabel("Score") 
plt.plot(fold, m1NewScore) 


# ## Gaussian Classifier

# In[82]:


m2NewScore = cross_val_score(m2, X_train, y_train, cv=5)
print(m2NewScore)


# In[83]:


y_pred = m2.fit(X_train, y_train).predict(X_test)
print(accuracy_score( y_test, y_pred) * 100 , "%")


# In[84]:


print(f1_score(y_test, y_pred, average=None))


# In[85]:


plt.title("5 Fold Cross Validation for Gaussian Classifier") 
plt.xlabel("Folds") 
plt.ylabel("Score") 
plt.plot(fold, m2NewScore) 


# In[88]:


plt.plot(fold, m1NewScore, color='r', label='Decision Tree Classifier')
plt.plot(fold, m2NewScore, color='g', label='Gaussian Classifier')
plt.xlabel("Folds")
plt.ylabel("Score")
plt.title("Cross Validation Score on Different Models")
plt.legend()
plt.show()


# # Question 2: Linear Discriminant Analysis

# In[89]:


class ScratchLDA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)
        mean_overall = np.mean(X, axis=0)
        SW = np.zeros((n_features, n_features))
        SB = np.zeros((n_features, n_features))
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            SW += (X_c - mean_c).T.dot((X_c - mean_c))

            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            SB += n_c * (mean_diff).dot(mean_diff.T)

        A = np.linalg.inv(SW).dot(SB)
        eigenvectors = simultaneous_orthogonalization(A)
        eigenvalues, _ = np.linalg.eig(A)
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        self.linear_discriminants = eigenvectors[0:self.n_components]

    def transform(self, X):
        return np.dot(X, self.linear_discriminants.T)


# In[93]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[94]:


new_X = df.iloc[:, 0:9].to_numpy()
new_y = df.loc[:, 'class'].to_numpy()

X_train_, X_test_, y_train_, y_test_ = train_test_split(new_X, new_y, test_size=0.35, shuffle=True)
from sklearn.decomposition import PCA
dt = DecisionTreeClassifier()
gnb = GaussianNB()
lda = LinearDiscriminantAnalysis(n_components=2)
pca = PCA(2)

models = [dt, gnb]
tmp = []
result = []
for idx, model in enumerate(models):
    tmp_X = pca.fit_transform(X_train_)
    model.fit(tmp_X, y_train_)
    preds = model.predict(pca.transform(X_test_))
    acc = accuracy_score(y_test_, preds)
    tmp.append(acc)

    tmp_X = lda.fit_transform(X_train_, y_train_)
    model.fit(tmp_X, y_train_)
    preds = model.predict(lda.transform(X_test_))
    acc = accuracy_score(y_test_, preds)
    tmp.append(acc)

    result.append(tmp)
    tmp = []

result = np.array(result)
results = result.T
print(results)


# In[95]:


final_df = pd.DataFrame()
final_df['Decision Tree'] = results[:, 0] * 100 
final_df['Naive Bayes'] = results[:, 1] * 100

final_df.index = ['PCA', 'LDA']

final_df


# In[97]:


model = LinearDiscriminantAnalysis()
csv_scores = cross_val_score(model, X_train, y_train, cv=5)
plt.plot(range(1, 6), csv_scores)
plt.xlabel("Folds")
plt.ylabel("Cross Val Score of LDA")
plt.show()


# In[100]:


from sklearn.metrics import plot_roc_curve 
X = df_data.iloc[:,:].to_numpy()
y = df.loc[:, 'class'].to_numpy()
print(np.unique(y))
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size = 0.65,random_state=0)
model = LinearDiscriminantAnalysis(n_components = 2)
model = model.fit(X_train, y_train)
plot_roc_curve(model,X_train,y_train)


# In[ ]:




