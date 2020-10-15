#!/usr/bin/env python
# coding: utf-8

# <font color=Magenta, size = "4"> Import the required modules and libraries </font>

# In[51]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
import statsmodels.api as sm
from scipy.stats import shapiro
from sklearn import preprocessing as ppr
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
from sklearn import metrics


# Import the data and preprocess

# In[3]:


Telecom = pd.read_excel("D:/BACP/Predictive Modelling/Final Project/Cellphone.xlsx")


# In[4]:


Telecom.shape
Telecom.dtypes
Telecom.head()


# In[5]:


Telecom['ContractRenewal'] = Telecom['ContractRenewal'].astype('category')
Telecom['DataPlan'] = Telecom['DataPlan'].astype('category')
#Telecom['CustServCalls'] = Telecom['CustServCalls'].astype('category')


# <font color=Magenta, size = "3"> Exploratory Data Analysis (EDA) </font>

# In[84]:


sns.heatmap(Telecom.isnull(),cbar=False)


# In[85]:


def plot_cor(df,size=11):
    correl = df.corr()
    fig , ax = plt.subplots(figsize = (size,size))
    ax.matshow(correl)
    plt.xticks(range(len(correl.columns)), correl.columns)
    plt.yticks(range(len(correl.columns)), correl.columns)


# In[11]:


plot_cor(Telecom)


# In[9]:


col = list(Telecom)
Telecom[col].hist(stacked=False,figsize=(12,30), layout=(14,2))


# In[25]:


sns.set(style="ticks", color_codes=True)
sns.pairplot(data=Telecom[['AccountWeeks','DataUsage','DayMins','DayCalls','MonthlyCharge','OverageFee','RoamMins']],kind="reg", plot_kws={'line_kws':{'color':'red'}})


# In[87]:


#sns.boxplot(data=Telecom[['AccountWeeks','DayMins','DayCalls','MonthlyCharge']])
sns.boxplot(data=Telecom[['DataUsage','CustServCalls','OverageFee','RoamMins']])


# <font color=Magenta, size = "3"> Outlier Treatment </font>

# In[6]:


Telecom['AccountWeeks'].quantile([0.01,0.99])
Telecom['AccountWeeks'] = np.where(Telecom['AccountWeeks'] > 195, 195, Telecom['AccountWeeks'])
sns.boxplot(data=Telecom['AccountWeeks'])


# In[7]:


Telecom['DataUsage'].quantile([0.01,0.99])
Telecom['DataUsage'] = np.where(Telecom['DataUsage'] > 4.1, 4.1, Telecom['DataUsage'])
sns.boxplot(data=Telecom['DataUsage'])


# In[8]:


Telecom['DayCalls'].quantile([0.01,0.99])
Telecom['DayCalls'] = np.where(Telecom['DayCalls'] > 152, 152, Telecom['DayCalls'])
Telecom['DayCalls'] = np.where(Telecom['DayCalls'] < 47.66, 47.66, Telecom['DayCalls'])
sns.boxplot(data=Telecom['DayCalls'])


# In[9]:


Telecom['DayMins'].quantile([0.01,0.99])
Telecom['DayMins'] = np.where(Telecom['DayMins'] > 340, 340, Telecom['DayMins'])
Telecom['DayMins'] = np.where(Telecom['DayMins'] < 25, 25, Telecom['DayMins'])
sns.boxplot(data=Telecom['DayMins'])


# In[10]:


Telecom['MonthlyCharge'].quantile([0.01,0.99])
Telecom['MonthlyCharge'] = np.where(Telecom['MonthlyCharge'] > 105, 105, Telecom['MonthlyCharge'])
Telecom['MonthlyCharge'] = np.where(Telecom['MonthlyCharge'] < 15, 15, Telecom['MonthlyCharge'])
sns.boxplot(data=Telecom['MonthlyCharge'])


# In[12]:


Telecom['OverageFee'] = np.where(Telecom['OverageFee'] > 16, 16 ,Telecom['OverageFee'])
Telecom['OverageFee'] = np.where(Telecom['OverageFee'] < 3, 3 ,Telecom['OverageFee'])
Telecom['RoamMins'] = np.where(Telecom['RoamMins'] > 18.5, 18.5 ,Telecom['RoamMins'])


# Multicollinearity Check

# In[38]:


y, X = dmatrices("Churn ~ AccountWeeks + ContractRenewal + DataPlan + DataUsage + CustServCalls + DayMins + DayCalls + MonthlyCharge + OverageFee + RoamMins ", data = Telecom, return_type='dataframe')


# In[42]:


vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif


# In[43]:


y, X = dmatrices("Churn ~ AccountWeeks + ContractRenewal + DataPlan + CustServCalls + DayMins + DayCalls + OverageFee + RoamMins ", data = Telecom, return_type='dataframe')


# In[44]:


# Performing again after removing monthly Charge and Data Usage
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif


# Split the data to training and testing set 

# In[69]:


X = Telecom.drop(['Churn','MonthlyCharge','DataUsage'],axis=1)
Y = Telecom['Churn']


# In[70]:


X_Train , x_test , Y_Train , y_test = train_test_split(X,Y,test_size = 0.3,random_state = 42)


# In[71]:


print("{0:0.2f}% data is in training set" .format((len(X_Train)/len(Telecom.index))*100))
print("{0:0.2f}% data is in test set" .format((len(x_test)/len(Telecom.index))*100))


# <font color=Magenta, size = "3"> Logistic Regression </font>

# In[20]:


Xindept = Telecom[['AccountWeeks','ContractRenewal','DataPlan','CustServCalls','DayMins','DayCalls','OverageFee','RoamMins']]
Ydepnt = Telecom[['Churn']]
logitmodel = sm.Logit(Ydepnt,Xindept.astype(float)).fit()


# In[57]:


logitmodel.summary2()


# Logistic Regression plots

# In[88]:


sns.regplot(x=Telecom['CustServCalls'],y=Telecom['Churn'],y_jitter=0.1,data=Telecom,logistic=True,ci=None)
plt.axhline(y=0.5,ls='--',c="red")


# In[89]:


sns.regplot(x=Telecom['DayMins'],y=Telecom['Churn'],y_jitter=0.03,data=Telecom,logistic=True,ci=None)
plt.axhline(y=0.5,ls='--',c="red")


# In[90]:


sns.regplot(x=Telecom['DayCalls'],y=Telecom['Churn'],y_jitter=0.03,data=Telecom,logistic=True,ci=None)
plt.axhline(y=0.5,ls='--',c="red")


# In[91]:


sns.regplot(x=Telecom['MonthlyCharge'],y=Telecom['Churn'],y_jitter=0.03,data=Telecom,logistic=True,ci=None)
plt.axhline(y=0.5,ls='--',c="red")


# In[94]:


sns.regplot(x=Telecom['OverageFee'],y=Telecom['Churn'],y_jitter=0.03,data=Telecom,logistic=True,ci=None)
plt.axhline(y=0.5,ls='--',c="red")


# In[52]:


sns.regplot(x=Telecom['ContractRenewal'].astype(int),y=Telecom['Churn'],y_jitter=0.1,data=Telecom,logistic=True,ci=None)
plt.axhline(y=0.5,ls='--',c="red")


# In[51]:


sns.regplot(x=Telecom['DataPlan'].astype(int),y=Telecom['Churn'],y_jitter=0.1,data=Telecom,logistic=True,ci=None)
plt.axhline(y=0.5,ls='--',c="red")


# In[72]:


logiclass = LogisticRegression(solver="liblinear",random_state=42)


# In[73]:


logimodel = logiclass.fit(X_Train,Y_Train)


# In[74]:


lypredtest = logimodel.predict(x_test)
lyprobtest = logimodel.predict_proba(x_test)[:,1]


# In[75]:


print(metrics.confusion_matrix(y_test,lypredtest))
print(metrics.classification_report(y_test,lypredtest))
metrics.accuracy_score(y_test,lypredtest)


# In[76]:


metrics.plot_precision_recall_curve(logimodel,x_test,y_test)
metrics.plot_roc_curve(logimodel,x_test,y_test)
plt.plot([0,1],[0,1],'k--')


# In[80]:


lypredtrain = logimodel.predict(X_Train)
lyprobtrain = logimodel.predict_proba(X_Train)[:,1]


# In[81]:


print(metrics.confusion_matrix(Y_Train,lypredtrain))
print(metrics.classification_report(Y_Train,lypredtrain))
print(metrics.accuracy_score(Y_Train,lypredtrain))


# In[82]:


fpr, tpr, thresholds = metrics.roc_curve(Y_Train, lyprobtrain)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Logistic Regression ROC curve')
plt.show()


# In[83]:


metrics.plot_precision_recall_curve(logimodel,X_Train,Y_Train)
metrics.plot_roc_curve(logimodel,X_Train,Y_Train)
plt.plot([0,1],[0,1],'k--')


# In[ ]:





# <font color=Magenta, size = "3"> K - Nearest Neighbour </font>

# Note : Knn requires min max or other normalization and scaling

# In[13]:


mscaler = MinMaxScaler()
Telecom_scaled = Telecom


# In[14]:


Telecom_scaled[['AccountWeeks','DataUsage','DayMins','DayCalls','MonthlyCharge','OverageFee','RoamMins','CustServCalls']] = mscaler.fit_transform(Telecom_scaled[['AccountWeeks','DataUsage','DayMins','DayCalls','MonthlyCharge','OverageFee','RoamMins','CustServCalls']])
Telecom_scaled


# Split to training and testing set

# In[15]:


X = Telecom_scaled.drop('Churn',axis=1)
Y = Telecom_scaled['Churn']
kX_Train , kx_test , kY_Train , ky_test = train_test_split(X,Y,test_size = 0.3,random_state = 42)


# Determining Optimal K value for knn

# In[16]:


error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(kX_Train, kY_Train)
    pred_i = knn.predict(kx_test)
    error.append(np.mean(pred_i != ky_test))


# In[17]:


plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')


# Got optimal K value to be 7

# In[30]:


kmidel = KNeighborsClassifier(n_neighbors=7,metric='euclidean')


# In[31]:


finalknn = kmidel.fit(kX_Train,kY_Train)


# In[62]:


ky_predtest = kmidel.predict(kx_test)
ky_probtest = kmidel.predict_proba(kx_test)[:,1]


# In[33]:


print(metrics.confusion_matrix(ky_test,ky_predtest))
print(metrics.classification_report(ky_test,ky_predtest))


# In[64]:


fpr, tpr, thresholds = metrics.roc_curve(ky_test, ky_probtest)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Knn(n_neighbors=7) ROC curve')
plt.show()


# In[65]:


metrics.plot_precision_recall_curve(kmidel,kx_test,ky_test)


# In[73]:


ky_predtrain = kmidel.predict(kX_Train)
ky_probtrain = kmidel.predict_proba(kX_Train)[:,1]


# In[75]:


print(metrics.confusion_matrix(kY_Train,ky_predtrain))
print(metrics.classification_report(kY_Train,ky_predtrain))
metrics.accuracy_score(kY_Train,ky_predtrain)


# In[76]:


fpr, tpr, thresholds = metrics.roc_curve(kY_Train, ky_probtrain)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Knn(n_neighbors=7) ROC curve')
plt.show()


# <font color=Magenta, size = "3"> K - Nearest Neighbour with Cross Validation </font>

# In[66]:


k_list = list(range(1,40))
# creating list of cv scores
cv_scores = []

# perform 10-fold cross validation
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, kX_Train, kY_Train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())


# In[43]:


MSE = [1 - x for x in cv_scores]

plt.figure()
plt.figure(figsize=(12,6))
plt.title('The optimal number of neighbors', fontsize=20, fontweight='bold')
plt.xlabel('Number of Neighbors K', fontsize=15)
plt.ylabel('Misclassification Error', fontsize=15)
sns.set_style("whitegrid")
plt.plot(k_list, MSE)
plt.show()


# In[44]:


# finding best k
best_k = k_list[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d." % best_k)


# <font color=Magenta, size = "3"> Naive Bayes </font>

# In[20]:


stat, p = shapiro(Telecom)
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')


# In[27]:


NBclob = GaussianNB()


# In[28]:


NBModel = NBclob.fit(X_Train,Y_Train)


# In[50]:


importnce = permutation_importance(NBModel, x_test, y_test)
print(importnce.importances_mean)


# In[35]:


NBy_predtest = NBModel.predict(x_test)
NBy_probtest = NBModel.predict_proba(x_test)[:,1]


# In[37]:


print(metrics.confusion_matrix(y_test,NBy_predtest))
print(metrics.classification_report(y_test,NBy_predtest))
print(metrics.accuracy_score(y_test,NBy_predtest))


# In[38]:


fpr, tpr, thresholds = metrics.roc_curve(y_test, NBy_probtest)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Naive Bayes ROC curve')
plt.show()


# In[42]:


metrics.plot_precision_recall_curve(NBModel,x_test,y_test)
metrics.plot_roc_curve(NBModel,x_test,y_test)
plt.plot([0,1],[0,1],'k--')


# In[43]:


NBy_predtrain = NBModel.predict(X_Train)
NBy_probtrain = NBModel.predict_proba(X_Train)[:,1]


# In[44]:


print(metrics.confusion_matrix(Y_Train,NBy_predtrain))
print(metrics.classification_report(Y_Train,NBy_predtrain))
print(metrics.accuracy_score(Y_Train,NBy_predtrain))


# In[45]:


fpr, tpr, thresholds = metrics.roc_curve(Y_Train, NBy_probtrain)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Naive Bayes ROC curve')
plt.show()


# In[46]:


metrics.plot_precision_recall_curve(NBModel,X_Train,Y_Train)
metrics.plot_roc_curve(NBModel,X_Train,Y_Train)
plt.plot([0,1],[0,1],'k--')


# <font color=Magenta, size = "3"> Naive Bayes with Cross Validation </font>

# In[53]:


k_fold = KFold(len(Y_Train), n_splits=10, shuffle=True, random_state=42)


# In[59]:


NBscores = cross_val_score(NBModel,X_Train,Y_Train, cv=k_fold, scoring='accuracy')
NBscores.mean()


# In[60]:


k_fold1 = KFold(len(y_test), n_splits=10, shuffle=True, random_state=42)
NBscores1 = cross_val_score(NBModel,x_test,y_test, cv=k_fold1, scoring='accuracy')
NBscores1.mean()

