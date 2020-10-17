#!/usr/bin/env python
# coding: Spyder IDE

# Import the required modules and libraries ##########

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing as ppr
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as shc
get_ipython().run_line_magic('matplotlib', 'inline')

#  Import the data and pre-process 

bank = pd.read_csv("D:/BACP/Machine Learning/ML Project/bank_marketing_part1_Data.csv",sep=',')

bank.dtypes
bank.head()
bank.columns

bank['spending'] = bank['spending'] * 1000
bank['advance_payments'] = bank['advance_payments'] * 100
bank['current_balance'] = bank['current_balance'] * 1000
bank['credit_limit'] = bank['credit_limit'] * 10000
bank['min_payment_amt'] = bank['min_payment_amt'] * 100
bank['max_spent_in_single_shopping'] = bank['max_spent_in_single_shopping'] * 1000

# Exploratory Data Analysis

sns.pairplot(bank,kind='reg',plot_kws={'line_kws':{'color':'red'}})

sns.boxplot(data=bank.loc[:,bank.columns!='credit_limit'],orient="h")

sns.heatmap(bank.corr(),cmap=sns.diverging_palette(256,0,sep=80,n=7,as_cmap=True),annot=True)

# Scale the Data

scaler = ppr.StandardScaler()

bankscaled = scaler.fit_transform(bank)
bankscaled = pd.DataFrame(bankscaled)
bankscaled.head()

## pd.Dataframe equivalent to as.dataframe in R ##

sns.boxplot(data=bankscaled)

# -----------First Part : Hierarcical Clustering --------------#########

hc = AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage="ward",compute_full_tree='auto')

agglmodel = hc.fit(bankscaled)

hclabels = agglmodel.labels_

plt.figure(figsize =(6, 6)) 
plt.title('Hierarchical Clustering')
Dendo = shc.dendrogram(shc.linkage(bankscaled,method='ward',metric='euclidean'))
plt.xlabel('Customers')

pcacolms = PCA(n_components=2).fit_transform(bankscaled)

plt.scatter(x=pcacolms[:,0],y=pcacolms[:,1] , c= hclabels, alpha=0.5)

silhouette_avg = silhouette_score(bankscaled, hclabels)

each_silhouette_score = silhouette_samples(bankscaled, hclabels,metric="euclidean")
fig =plt.figure()
ax = fig.add_subplot(1,1,1)
y_lower =10

for i in range(0,3):
    ith_cluster_silhouette_values = each_silhouette_score[hclabels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    
    ax.fill_betweenx(np.arange(y_lower,y_upper),0,ith_cluster_silhouette_values,alpha=0.3)
    
    #label the silhouse plots with their cluster numbers at the middle
    ax.text(-0.05,y_lower + 0.5 * size_cluster_i,str(i))
    
    #compute the new y_lower for next plot
    y_lower = y_upper +10 
    
ax.set_title("Silhuoette plot")
ax.set_xlabel("Silhouette score")
ax.set_ylabel("Cluster label")
    
#the vertical line for average silhouette score of all the values
ax.axvline(x=silhouette_avg,color="red",linestyle="--")
    
ax.set_yticks([])
ax.set_xticks([-0.2,0,0.2,0.4,0.6,0.8,1])


# --------------------- Second Part : K Means-----------------------------------##############

# WSS method
SSE =[]
for i in range(1,20):
    km = KMeans(n_clusters = i)
    km = km.fit(bankscaled)
    SSE.append(km.inertia_)

plt.plot(range(1, 20), SSE,'bx-')
plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('Error')
plt.show()

# Silhoutte score method
silht = []

for k in range(2, 20):
  km1 = KMeans(n_clusters = k).fit(bankscaled)  
  preds = km1.fit_predict(bankscaled)
  silht.append(silhouette_score(bankscaled, preds, metric = 'euclidean'))

plt.plot(range(2, 20), silht,'bx-')
plt.title('Silhouette Method')
plt.xlabel('Number of clusters')
plt.ylabel('Sil')
plt.show()

## ------------Final K means with k = 3 as optimal value---------------#
finalkmean = KMeans(n_clusters=3,init="k-means++",n_init=10,max_iter=300).fit(bankscaled)
centroids = finalkmean.cluster_centers_
print(centroids)
klabel = finalkmean.labels_

pcacolmns = PCA(n_components=2).fit_transform(bankscaled)


plt.scatter(x=pcacolmns[:,0],y=pcacolmns[:,1] , c= finalkmean.labels_, alpha=0.5)
plt.scatter(centroids, c='red')

bank_kmeans = bank
bank_kmeans['Cluster_Mapping'] = finalkmean.labels_
bank_kmeans.head(n=10)

#----------------Calculate the average of silhouette scores

silhouette_avg = silhouette_score(bankscaled, klabel)

each_silhouette_score = silhouette_samples(bankscaled, klabel,metric="euclidean")
fig =plt.figure()
ax = fig.add_subplot(1,1,1)
y_lower =10

for i in range(0,3):
    ith_cluster_silhouette_values = each_silhouette_score[klabel == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    
    ax.fill_betweenx(np.arange(y_lower,y_upper),0,ith_cluster_silhouette_values,alpha=0.3)
    
    #label the silhouse plots with their cluster numbers at the middle
    ax.text(-0.05,y_lower + 0.5 * size_cluster_i,str(i))
    
    #compute the new y_lower for next plot
    y_lower = y_upper +10 
    
ax.set_title("Silhuoette plot")
ax.set_xlabel("Silhouette score")
ax.set_ylabel("Cluster label")
    
#the vertical line for average silhouette score of all the values
ax.axvline(x=silhouette_avg,color="red",linestyle="--")
    
ax.set_yticks([])
ax.set_xticks([-0.2,0,0.2,0.4,0.6,0.8,1])

bank_kmeans['Cluster_Mapping'] = bank_kmeans['Cluster_Mapping'].astype(str) 
bank_kmeans['Cluster_Mapping'] = bank_kmeans['Cluster_Mapping'].replace(['0'],'Cluster1')
bank_kmeans['Cluster_Mapping'] = bank_kmeans['Cluster_Mapping'].replace(['1'],'Cluster2')
bank_kmeans['Cluster_Mapping'] = bank_kmeans['Cluster_Mapping'].replace(['2'],'Cluster3')
bank_kmeans['Cluster_Mapping'] = bank_kmeans['Cluster_Mapping'].astype('category')

sns.pairplot(data=bank_kmeans,hue="Cluster_Mapping")

