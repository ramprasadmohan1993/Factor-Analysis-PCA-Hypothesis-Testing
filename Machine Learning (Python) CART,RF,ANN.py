#!/usr/bin/env python
# coding: utf-8 Spyder

# ------- Import the required modules and libraries ------------###########

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing as ppr
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV
from collections import OrderedDict
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


######-------------------------------  Import the data and preprocess --------------------------#########


Insurance = pd.read_csv("D:/BACP/Machine Learning/ML Project/insurance_part2_data.csv",sep=',')

Insurance.dtypes
#Insurance.head()
#Insurance.shape

Insurance['Agency_Code'] = Insurance['Agency_Code'].astype('category')
Insurance['Type'] = Insurance['Type'].astype('category')
Insurance['Claimed'] = Insurance['Claimed'].astype('category')
Insurance['Channel'] = Insurance['Channel'].astype('category')
Insurance['Product Name'] = Insurance['Product Name'].astype('category')
Insurance['Destination'] = Insurance['Destination'].astype('category')


### ----- Exploratory Data Analysis --------------------------------------####

plt.figure(figsize =(10, 6)) 
plt.title('Agency_Code vs Claims')
sns.countplot(Insurance['Agency_Code'],hue=Insurance['Claimed'],orient="h")

plt.figure(figsize =(10, 6)) 
plt.title('Product Name vs Claims')
sns.countplot(Insurance['Product Name'],hue=Insurance['Claimed'],orient="h")

plt.figure(figsize =(10, 6)) 
plt.title('Product Name vs Claims')
plt.hist(Insurance['Sales'],bins=100)

plt.figure(figsize =(10, 6)) 
plt.title('Type of tour insurance firms vs Claims')
sns.countplot(Insurance['Type'],hue=Insurance['Claimed'],orient="h")

plt.figure(figsize =(10, 6)) 
plt.title('Destination vs Claims')
sns.countplot(Insurance['Destination'],hue=Insurance['Claimed'],orient="h")

#----------- Outlier Treatment ------------------------########

sns.boxplot(data=Insurance.loc[:,Insurance.columns!='Duration'],orient="h")
sns.boxplot(Insurance['Duration'])

Insurance['Duration'].max()
Insurance['Duration'].min()
Insurance = Insurance.drop(Insurance['Duration'].idxmax())
Insurance = Insurance.drop(Insurance['Duration'].idxmin())
Insurance = Insurance.drop(Insurance['Commision'].idxmax())

Insurance = Insurance.drop('Channel',1)

Insurance.head()
Insurance.shape

sns.pairplot(Insurance,kind='reg',plot_kws={'line_kws':{'color':'red'}})

####--------------- Splitting to training and Testing data ------------------#####

X = Insurance.drop('Claimed',axis=1)
Y = Insurance.Claimed

X_Train , x_test , y_Train , y_test = train_test_split(X,Y,test_size = 0.3,random_state = 42)

X_Train.head()

lenc = LabelEncoder()
X_Train = X_Train.apply(lenc.fit_transform)
x_test = x_test.apply(lenc.fit_transform)

RFX_Train = pd.get_dummies(X_Train)
RFX_Train.head()
RFx_test = pd.get_dummies(x_test)

#######--########--- First Part : CART (Classification Decision TREE) ---------------#####

CART = DecisionTreeClassifier(min_samples_split=10,min_samples_leaf=3)

CARTModel = CART.fit(X=X_Train,y=y_Train)

path = CARTModel.cost_complexity_pruning_path(X_Train,y_Train)
ccp_alphas, impurities = path.ccp_alphas , path.impurities

fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")

feat_imp = pd.Series(CARTModel.feature_importances_,index=X_Train.columns)
feat_imp.plot(kind='barh')

CART = DecisionTreeClassifier(min_samples_split=10,min_samples_leaf=3,ccp_alpha=0.004)
CARTModel = CART.fit(X=X_Train,y=y_Train)

plot_tree(CARTModel)

## ------- Predict ----------####

y_predtest = CARTModel.predict(x_test)
y_probtest = CARTModel.predict_proba(x_test)

## -------- Model Validation ------------###

cf = metrics.confusion_matrix(y_test,y_predtest)
metrics.ConfusionMatrixDisplay(cf)

metrics.accuracy_score(y_test,y_predtest)

y_predTrain = CARTModel.predict(X_Train)
y_probTrain = CARTModel.predict_proba(X_Train)
metrics.accuracy_score(y_Train,y_predTrain)

metrics.plot_precision_recall_curve(CARTModel,X_Train,y_Train)
metrics.plot_precision_recall_curve(CARTModel,x_test,y_test)

print(metrics.classification_report(y_test,y_predtest))
metrics.plot_roc_curve(CARTModel,x_test,y_test)
metrics.plot_roc_curve(CARTModel,X_Train,y_Train)

######-----########--------#### Second part : Random Forest (Classification Decision TREE) --------#######-----------#########

RF = RandomForestClassifier(n_estimators=501,oob_score=True,criterion="gini",min_samples_leaf=10,random_state=42,max_features=4)

RFModel = RF.fit(X=RFX_Train,y=y_Train)

RFModel.get_params()

feat_imp = pd.Series(RFModel.feature_importances_,index=RFX_Train.columns)
feat_imp = feat_imp.sort_values()
feat_imp.plot(kind='barh')

ensemble_clfs = [ ("RandomForestClassifier, max_features=None",RandomForestClassifier(warm_start=True, max_features=None, oob_score=True,criterion="gini",min_samples_leaf=10,random_state=42))]
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)
min_estimators = 10
max_estimators = 500

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1):
        clf.set_params(n_estimators=i)
        clf.fit(X=RFX_Train,y=y_Train)
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))


for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)
    
plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show()

#-------------- Predict--------------####

y_predtest = RFModel.predict(RFx_test)
y_probtest = RFModel.predict_proba(RFx_test)

cf = metrics.confusion_matrix(y_test,y_predtest)
cf

metrics.accuracy_score(y_test,y_predtest)

y_predTrain = RFModel.predict(RFX_Train)
y_probTrain = RFModel.predict_proba(RFX_Train)
metrics.accuracy_score(y_Train,y_predTrain)

#-----------Model Validation ---------------####

metrics.plot_precision_recall_curve(RFModel,RFX_Train,y_Train)
metrics.plot_precision_recall_curve(RFModel,RFx_test,y_test)

print(metrics.classification_report(y_test,y_predtest))
metrics.plot_roc_curve(RFModel,RFX_Train,y_Train)
metrics.plot_roc_curve(RFModel,RFx_test,y_test)

####-----######------------- THIRD PART : Artificial Nueral Network ANN ----------------#######

## One Hot encoding and scaling

ANNY_Train = pd.get_dummies(y_Train)
ANNY_Train = pd.DataFrame(ANNY_Train)

ANNy_test = pd.get_dummies(y_test)
ANNy_test = pd.DataFrame(ANNy_test)

# Scale the data

scaler = StandardScaler()

ANNX_Train = scaler.fit_transform(RFX_Train)
ANNX_Train = pd.DataFrame(ANNX_Train,columns=RFX_Train.columns)

ANNx_test = scaler.fit_transform(RFx_test)
ANNx_test = pd.DataFrame(ANNx_test,columns=RFx_test.columns)

#annclf = MLPClassifier(,max_iter=150000,random_state=1,verbose=10,alpha=0.0001,tol=0.00001)
annclf = MLPClassifier(max_iter=150000,random_state=1)

#####  Using gridcv to find best hyperparameters for the nueral network   ####

parameter_space ={'hidden_layer_sizes':[(9,4),(9,3),(9,2),(8,4),(8,3),(8,2),(7,3),(7,2),(6,3),(6,2),(5,2),(4,2)],
                 'activation':['relu','logistic','tanh'],
                 'solver': ['sgd' , 'adam'],
                 'alpha' : [0.0001 , 0.05],
                 'learning_rate' : ['constant','adaptive']}

clf = GridSearchCV(annclf,parameter_space,n_jobs=-1,cv=10)
clf.fit(ANNX_Train,ANNY_Train['Yes'])

print('Best parameters found for ANN : \n', clf.best_params_)


# Running the nueral network with activation : Relu , alpha " 0.0001 , hidden layers (9,4), solver : adam and learning_rate constant

annclfinal = MLPClassifier(hidden_layer_sizes= (9,4), activation= 'relu',solver='adam',
                           alpha = 0.0001,learning_rate= 'constant', max_iter=150000,random_state=1,verbose=10,shuffle=True,tol=0.000001)

annmodel = annclfinal.fit(ANNX_Train,ANNY_Train['Yes'])

plt.ylabel('loss')
plt.xlabel('iterations')
plt.title("Learning rate =" + str(0.001))
plt.plot(annmodel.loss_curve_)
plt.show()

#####-- Predict ----------###

y_predtestann = annmodel.predict(ANNx_test)
y_probtestann = annmodel.predict_proba(ANNx_test)

cf = metrics.confusion_matrix(ANNy_test['Yes'],y_predtestann)

metrics.accuracy_score(ANNy_test['Yes'],y_predtestann)

y_predTrainann = annmodel.predict(ANNX_Train)
y_probTrainann = annmodel.predict_proba(ANNX_Train)
metrics.accuracy_score(ANNY_Train['Yes'],y_predTrainann)

## Model Valdiation --####

metrics.plot_precision_recall_curve(annmodel,ANNX_Train,ANNY_Train['Yes'])
metrics.plot_precision_recall_curve(annmodel,ANNx_test,ANNy_test['Yes'])

print(metrics.classification_report(ANNy_test['Yes'],y_predtestann))
metrics.plot_roc_curve(annmodel,ANNX_Train,ANNY_Train['Yes'])
metrics.plot_roc_curve(annmodel,ANNx_test,ANNy_test['Yes'])

