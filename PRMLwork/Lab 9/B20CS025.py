#!/usr/bin/env python
# coding: utf-8

# # Question 1: K-Means Clustering

# In[18]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from numpy.linalg import norm
from tqdm import tqdm
from collections import Counter
from sklearn import *
from scipy.spatial.distance import cdist
import cv2 as cv
import os 
import shutil
from os import listdir
from os.path import isfile, join
sns.set_style("darkgrid")


# In[2]:


names = []
for i in range(14):
    names.append(i)
names[0] = "Class"


# In[3]:


data = pd.read_csv(r'C:\Users\Kartik\Desktop\Lab 9\wine.data', names=names)
data           


# In[4]:


data.describe()


# In[5]:


X = data.iloc[:, 1:14]
y = data.loc[:, 'Class']


# In[6]:


scaling = StandardScaler()
scaling.fit(X)
X = scaling.transform(X)


# ## a) Using the PCA technique for the dimension Reduction

# In[7]:


pca = PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)


# In[8]:


X = pd.DataFrame(X)
y = pd.DataFrame(y)
data = pd.concat([X, y],axis=1)
data


# In[9]:


sns.pairplot(data, hue="Class")


# ## As we can see from the visualisation that the classes can be divided into three, hence K=3

# ## b) building KMeans Clustering Algorithm

# In[10]:


kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
labels = kmeans.fit_predict(X)


# In[11]:


print("Prediced Class Labels:- ")
print(labels)


# In[12]:


X = X.values


# In[13]:


y = y.values


# ## plotting the centroids

# In[14]:


plt.figure(figsize=(10, 5))
plt.title("Part B plot")
centres = kmeans.cluster_centers_
plt.scatter(X[labels == 0, 0], X[labels == 0, 1], s = 100, c = 'red', label = ' 1')
plt.scatter(X[labels == 1, 0], X[labels == 1, 1], s = 100, c = 'orange', label = ' 2')
plt.scatter(X[labels == 2, 0], X[labels == 2, 1], s = 100, c = 'green', label = ' 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', s=200, alpha=0.5,label = 'Centroids');
plt.legend()
plt.show()


# ## c) Using different values of K and finding the silhouette score and finding the optimal value of k

# In[15]:


score = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init = 'k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    score_temp = silhouette_score(X, kmeans.labels_, metric='euclidean')
    score.append(score_temp)


# In[16]:


plt.plot(range(2, 11), score)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()


# ## d) Using the elbow method for finding the optimal K

# In[17]:


import sklearn.cluster as cluster 


# In[18]:


K = range(1, 11)
WSS = []
for k in K:
    kmeans = cluster.KMeans(n_clusters=k, init=
                           'k-means++')
    kmeans = kmeans.fit(X)
    wss_iter = kmeans.inertia_
    WSS.append(wss_iter)


# In[19]:


mycenters = pd.DataFrame({'clusters' : K, 'WSS' : WSS})
mycenters


# ## Hence from the below elbow method it can be seen that the elbow is formed when k = 3

# In[20]:


sns.scatterplot(x = 'clusters', y='WSS', data = mycenters, marker="*")


# # Question 2: 

# In[3]:


df = pd.read_csv(r'C:\Users\Kartik\Desktop\Lab 9\archive\fashion-mnist_train.csv')
df


# In[4]:


df.describe()


# In[5]:


df.info()


# ## a and b part : Writing the KMeans Clustering Class

# In[24]:


class kMeans:
    def __init__(self,dis='euclidean',clusters = 2, tolerance = 1e-7, Initialise = False,centroids = None): ## 'centroids' parameter enables user to take initial cluster center points as its initialization. (iii)
        self.clusters = clusters ## Taking value from user for k (by default k = 2) (ii)
        self.dis = dis
        self.centroids = centroids ## Stroing Centroids (i)4
        self.Initialise = Initialise
        self.vals = None
        self.tolerance = tolerance
        self.Data = None
    def distance(self,a,b):
        if self.dis == 'euclidean':
            return np.sum((a-b)**2)**0.5
        if self.dis == 'absolute':
            return np.sum(abs(a-b))
    def adjustCentroids(self,centroids, X,y): ## Function to adjust the centroids after every iteration
        centers = centroids.shape[0]
        for i in range(centers):
            centroids[i] = np.mean(X[y==i].copy(), axis=0)
        return centroids
    def fit(self,X, iterations = 100):
        self.vals = np.zeros(X.shape[0])
        self.Data = X
        if self.Initialise == False:
            self.centroids = np.zeros((self.clusters,X.shape[1]))
            for i in range(self.clusters):
                self.centroids[i] = X[np.random.randint(0,X.shape[0])]
        for it in tqdm(range(iterations)): ## Stops iterating after iterations specified by the user
            for ix,row in enumerate(X):
                center = -1
                dist = float("inf")
                for cx,centr in enumerate(self.centroids):
                    if self.distance(row,centr) < dist:
                        dist = self.distance(row,centr)
                        self.vals[ix] = cx
            self.centroids = self.adjustCentroids(self.centroids,X,self.vals)
    def SSE_cluster(self,cluster):
        Xc = self.Data[self.vals == cluster].copy()
        centroid = self.centroids[cluster].copy()
        Xc = Xc - centroid
        Xc = Xc ** 2
        Xc = np.sum(Xc, axis = 1)
        return float(np.sum(Xc))
    def SSE(self):
        sum_ = 0
        for i in self.clusters:
            sum_ += self.SSE_cluster(i)
        return sum_
        
        


# ## c) training K-Means model on f-MNIST data with k=10 and 10 random 784 dimensional points 

# In[20]:


km = kMeans('euclidean', 10)


# In[21]:


X = df.iloc[:, 1:].values.astype('float64') / 255
y = df['label']


# In[22]:


km.fit(X)


# ## Reporting the number of points in each cluster.

# In[26]:


for i in Counter(km.vals):
    print('Cluster:',i,'==>',Counter(km.vals)[i])


# ## d) Visualizing the cluster centers of each cluster as 2-d images of all clusters.

# In[27]:


# Reducing the dimensionality of the data to plot its 2D graph
pc = decomposition.PCA(2)
pc.fit(X)
XPCA = pc.transform(X)
centroids = pc.transform(km.centroids)


# In[28]:


'''
Storing data in dataframe
'''
dicClasses = {}
dicClasses[0] = '0'
dicClasses[1] = '1'
dicClasses[2] = '2'
dicClasses[3] = '3'
dicClasses[4] = '4'
dicClasses[5] = '5'
dicClasses[6] = '6'
dicClasses[7] = '7'
dicClasses[8] = '8'
dicClasses[9] = '9'
yPlot = km.vals.copy()
y = list(km.vals.copy())
for ix,i in enumerate(km.vals):
    y[ix] = dicClasses[i]
'''
Plotting a KDE plot 
'''
dfPlot = pd.DataFrame()
dfPlot['Dim1'] = XPCA[:,0]
dfPlot['Dim2'] = XPCA[:,1]
dfPlot['Class'] = y
fig = plt.figure(figsize=(15,15))
sns.kdeplot(x = 'Dim1', y = 'Dim2',hue = 'Class', data = dfPlot, fill = True)
plt.scatter(x = centroids[:,0], y = centroids[:,1], c = 'black', s = 200, marker = '*')


# In[29]:


'''
Plotting a Scatter plot 
'''
fig = plt.figure(figsize=(15,15))
sns.scatterplot(x = 'Dim1', y = 'Dim2',hue = 'Class', data = dfPlot)
plt.scatter(x = centroids[:,0], y = centroids[:,1], c = 'black', s = 200, marker = '*')


# ## e) Visualising 10 images corresponding to each cluster

# In[30]:


'''
Plotting 10 images corresponding to each cluster.
'''
fig, axes = plt.subplots(2,5,figsize=(30,15))
for r in range(2):
    for c in range(5):
        axes[r][c].imshow(X[yPlot == 5*r + c][2000].reshape(28,28,1), cmap = 'gray')


# ## f) Training another k-means model with 10 images from each class as initializations , and reporting the number of points in each cluster and visualizing the cluster centers.

# In[31]:


'''
Initialising Centroids for the Kmeans model
'''
InitialiseCentroids = np.array([X[df['label'].values==i][3000] for i in range(10)])


# In[32]:


kMeansIni = kMeans(dis = 'euclidean',clusters = 10, centroids = InitialiseCentroids, Initialise = True)


# In[33]:


kMeansIni.fit(X)


# In[35]:


'''
Reporting points in each cluster
'''
for i in Counter(kMeansIni.vals):
    print('Cluster:',i,'==>',Counter(kMeansIni.vals)[i])


# In[36]:


'''
Visualising Centroids
'''
'''
Reducing dimensionality of the Data to plot its 2D graph
'''
pc = decomposition.PCA(2)
pc.fit(X)
XPCA = pc.transform(X)
centroids = pc.transform(kMeansIni.centroids)


# In[37]:


'''
Storing data in dataframe
'''
yPlot = kMeansIni.vals.copy()
y = list(kMeansIni.vals.copy())
for ix,i in enumerate(kMeansIni.vals):
    y[ix] = dicClasses[i]
'''
Plotting a KDE plot 
'''
dfPlot = pd.DataFrame()
dfPlot['Dim1'] = XPCA[:,0]
dfPlot['Dim2'] = XPCA[:,1]
dfPlot['Class'] = y
fig = plt.figure(figsize=(15,15))
sns.kdeplot(x = 'Dim1', y = 'Dim2',hue = 'Class', data = dfPlot, fill = True)
plt.scatter(x = centroids[:,0], y = centroids[:,1], c = 'black', s = 200, marker = '*')


# In[38]:


'''
Plotting a Scatter Plot
'''
fig = plt.figure(figsize=(15,15))
sns.scatterplot(x = 'Dim1', y = 'Dim2',hue = 'Class', data = dfPlot)
plt.scatter(x = centroids[:,0], y = centroids[:,1], c = 'black', s = 200, marker = '*')


# ## g) Visualize 10 images corresponding to each cluster.

# In[39]:


'''
Plotting 10 images corresponding to each cluster.
'''
fig, axes = plt.subplots(2,5,figsize=(30,15))
for r in range(2):
    for c in range(5):
        axes[r][c].imshow(X[yPlot == 5*r + c][1243].reshape(28,28,1), cmap = 'gray')


# ## Part H

# In[40]:


'''
Calculating mse for each cluster for both the models
'''
sseRandom = []
sseIni = []
for i in range(0,10):
    sseRandom.append(km.SSE_cluster(i))
    sseIni.append(kMeansIni.SSE_cluster(i))


# In[41]:


'''
Plotting the SSE's reported
'''
dfPlot = pd.DataFrame()
dfPlot['Clusters'] = [i for i in range(10)]
dfPlot['SSE_Random'] = sseRandom
dfPlot['SSE_Initialised'] = sseIni
fig, axes = plt.subplots(1,2, figsize=(8,4))
sns.pointplot(x = 'Clusters', y = 'SSE_Random', data = dfPlot, ax = axes[0])
sns.pointplot(x = 'Clusters', y = 'SSE_Initialised', data = dfPlot, ax = axes[1])


# In[42]:


'''
Calculating sum of SSE's
'''
SSE_sum_random = np.sum(np.array(sseRandom))
SSE_sum_initialised = np.sum(np.array(sseIni))
print('SSE --> Random -->',SSE_sum_random)
print('SSE --> Initialised -->',SSE_sum_initialised)

#Results for both the models are almost similar with Initialised centroids approach peforming slightly better


# # Question 3: Hierarchical Clustering

# In[44]:


'''
Collecting images of the yes category
'''
yes_no = []
mypath = r'C:\Users\Kartik\Desktop\Lab 9\Q3\yes'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for ix,file in enumerate(onlyfiles):
    if ix == 1500:
        break
    img = cv.imread(mypath+'/'+file,0)
    img = cv.resize(img,(100,100))
    yes_no.append(img)


# In[45]:


'''
Collecting images of the no category
'''
mypath = r'C:\Users\Kartik\Desktop\Lab 9\Q3\no'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for ix,file in enumerate(onlyfiles):
    if ix == 1500:
        break
    img = cv.imread(mypath+'/'+file,0)
    img = cv.resize(img,(100,100))
    yes_no.append(img)
yes_no = np.array(yes_no)


# In[46]:


'''
Reshaping Data collected
'''
yes_no = yes_no.reshape(3000,10000)


# In[47]:


'''
Storing it in a dataframe
'''
dfBrain = pd.DataFrame(yes_no)
dfBrain.columns =['Pixel'+str(i+1) for i in range(10000)]
dfBrain['Class'] = [1 if i < 1500 else 0 for i in range(3000)]


# In[48]:


'''
Scaling Data
'''
dfBrain_ = dfBrain.copy()
dfBrain.iloc[:,:-1] = preprocessing.StandardScaler().fit_transform(dfBrain.iloc[:,:-1])


# ## Part B

# In[49]:


pca = decomposition.PCA(2)
XTemp3 = pca.fit_transform(dfBrain.iloc[:,:-1])
dfPCA3 = pd.DataFrame(XTemp3)
dfPCA3.columns = ['PC1','PC2']
XTemp3_ = pca.fit_transform(dfBrain_.iloc[:,:-1])
dfPCA3_ = pd.DataFrame(XTemp3_)
dfPCA3_.columns = ['PC1','PC2']  


# In[50]:


'''
Removing some outliers
'''
rows2Delete = list(np.where(dfPCA3['PC2'] > 150)[0])
rows2Delete.extend(list(np.where(dfPCA3['PC1'] > 150)[0]))


# In[51]:


'''
Removing Selected Rows
'''
for row in rows2Delete:
    dfPCA3.drop(row, inplace=True)
    dfBrain.drop(row, inplace=True)
    dfPCA3_.drop(row, inplace=True)
dfPCA3.reset_index(drop = True, inplace=True)
dfBrain.reset_index(drop = True, inplace=True)
dfPCA3_.reset_index(drop = True, inplace=True)


# In[52]:


'''
Visualising without scaling
'''
sns.jointplot(x = 'PC1', y = 'PC2', kind = 'kde',data  = dfPCA3_, cmap = 'magma',fill = True)


# In[53]:


'''
Visualising with scaling
'''
sns.jointplot(x = 'PC1', y = 'PC2', kind = 'kde',data  = dfPCA3, cmap = 'magma',fill = True)


# ## Part C

# In[54]:


'''
Visualising communites from Part A through Scatter Plot without scaling
'''
sns.scatterplot(x = 'PC1', y = 'PC2',data  = dfPCA3_, cmap = 'magma')


# In[55]:


'''
Visualising communites from Part A through Scatter Plot with scaling
'''
sns.scatterplot(x = 'PC1', y = 'PC2',data  = dfPCA3, cmap = 'magma')


# In[56]:


'''
Plotting image from Cluster 1
'''
plt.imshow(yes_no[300].reshape(100,100), cmap = 'gray')


# In[57]:


'''
Plotting image from Cluster 2
'''
plt.imshow(yes_no[2700].reshape(100,100), cmap = 'gray')


# ## Part D

# In[64]:


'''
Training Agglomerative hierarchical clustering model
'''
agglClusterModel = cluster.AgglomerativeClustering(2)
agglClusterModel.fit(dfBrain.iloc[:,:-1].values)


# In[65]:


'''
Training Kmeans model
'''
kmeansModel = cluster.KMeans(2)
kmeansModel.fit(dfBrain.iloc[:,:-1].values)


# In[66]:


'''
Storing labels from both the Models
'''
yAgg = agglClusterModel.labels_
yKmeans = kmeansModel.labels_


# In[68]:


yKmeans


# In[69]:


dfPlot = pd.DataFrame()


# In[72]:


temp = pca.fit_transform(dfBrain.iloc[:,:-1])


# In[80]:


temp = pd.DataFrame(temp)


# In[82]:


dfPlot = temp


# In[89]:


dfPlot.columns = ['Dim1', 'Dim2']


# In[90]:





# In[91]:


dfPlot['yAgg'] = yAgg
dfPlot['yKmeans'] = yKmeans
dfPlot['Original'] = dfBrain['Class']
fig, axes = plt.subplots(1,3 , figsize=(15,5))
sns.scatterplot(x = 'Dim1', y = 'Dim2',hue = 'yAgg',data  = dfPlot, ax = axes[0])
axes[0].set_title('Agglomerative hierarchical clustering model')
axes[1].set_title('Kmeans clustering model')
axes[2].set_title('Original Class')
sns.scatterplot(x = 'Dim1', y = 'Dim2',hue = 'yKmeans',  data  = dfPlot, ax = axes[1])
sns.scatterplot(x = 'Dim1', y = 'Dim2',hue ='Original', data = dfPlot, ax = axes[2])


# In[ ]:




