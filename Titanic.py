#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[2]:


os.chdir("C:\\Users\\91721\\Desktop\\Ims files")


# In[5]:


test_data=pd.read_csv("titanic_test.csv")
train_data=pd.read_csv("titanic_train.csv")

test_data=pd.read_csv("titanic_test.csv")
train_data=pd.read_csv("titanic_train.csv")
# In[7]:


full_data=train_data.append(test_data)


# In[10]:


full_data.isnull().sum()


# In[13]:


drop_columns=["Name","Age","SibSp","Ticket","Cabin","Parch","Embarked"]


# In[18]:


full_data.drop(labels=drop_columns,axis=1,inplace=True)


# In[20]:


full_data=pd.get_dummies(full_data,columns=["Sex"])


# In[22]:


full_data.fillna(value=0.0,inplace=True)


# In[23]:


#splitting into train and test

X=full_data.drop("Survived",axis=1)
y=full_data["Survived"]


# In[24]:


state=12
test_size=0.30
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=state)


# In[25]:


#

scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)


# In[26]:


from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)
X_pred=rfc.predict(X_train)


# In[27]:


print("Confusion matrix",confusion_matrix(y_train,X_pred))
print("Accuracy",accuracy_score(y_train,X_pred))
print("Confusion matrix from testing set",confusion_matrix(y_test,y_pred))
print("Accuracy of testing",accuracy_score(y_test,y_pred))


# In[28]:


'''
n_estimators = number of trees in the foreset
max_features = max number of features considered for splitting a node
max_depth = max number of levels in each decision tree
'''


# In[30]:


# hyperParameter tuning with GridSearchCV

from sklearn.model_selection import GridSearchCV

param_grid = { 
    'n_estimators': [25,50,75,100,150,200],
    'max_depth' : [4,7,9],
    'criterion' : ['gini', 'entropy']
}

gs = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
gs.fit(X_train, y_train)
print (gs.best_params_)
print(gs.best_score_)


# In[32]:


gs


# In[33]:


#hyperParameter tuning with RandomSearchCV

from sklearn.model_selection import RandomizedSearchCV

param_distributions = { 
    'n_estimators': [25,50,75,100,150,200],
    'max_depth' : [4,7,9],
    'criterion' : ['gini', 'entropy']
}

rs = RandomizedSearchCV(estimator=rfc, param_distributions=param_distributions, cv= 5)
rs.fit(X_train, y_train)
print (rs.best_params_)
print(rs.best_score_)


# In[35]:


rs


# In[36]:


##################  SVM with different kernels  ###################

from sklearn.svm import SVC

svc = SVC()

svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
x_pred = svc.predict(X_train)


print("Confusion Matrix of SVM from training set : \n " , confusion_matrix(y_train,x_pred))
print("\n Accuracy of SVM from training set : \n" , accuracy_score(y_train,x_pred))

print(" \n\n Confusion Matrix of SVM from testing set: \n " , confusion_matrix(y_test,y_pred))
print("\n Accuracy of SVM from testing set : \n" , accuracy_score(y_test,y_pred))


# In[39]:


## HyperParameter tuning with grid SearchCV

from sklearn.model_selection import GridSearchCV

param_grid = { 'C' : [1,10,100,1000],
              'kernel' : ['rbf','poly','linear'],
              'gamma' : [0.1,0.2,0.3,0.4]
             }

gs = GridSearchCV(estimator=svc, param_grid=param_grid, cv= 5)
gs.fit(X_train,y_train)

print(gs.best_params_)
print(gs.best_score_)


# In[40]:


## HyperParameter tuning with Random SearchCV


from sklearn.model_selection import RandomizedSearchCV

param_distributions = { 'C' : [1,10,100,1000],
              'kernel' : ['rbf','poly','linear'],
              'gamma' : [0.1,0.2,0.3,0.4,0.5,1,2,3]
             }

rs = RandomizedSearchCV(estimator=svc, param_distributions=param_distributions, cv= 5)
rs.fit(X_train,y_train)

print(rs.best_params_)
print(rs.best_score_)


# In[41]:


# making a new svc model with the best hyper parameters

svc = SVC(C=1, kernel='poly', gamma=3)

svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
x_pred = svc.predict(X_train)


print("Confusion Matrix of SVM from training set : \n " , confusion_matrix(y_train,x_pred))
print("\n Accuracy of SVM from training set : \n" , accuracy_score(y_train,x_pred))

print(" \n\n Confusion Matrix of SVM from testing set: \n " , confusion_matrix(y_test,y_pred))
print("\n Accuracy of SVM from testing set : \n" , accuracy_score(y_test,y_pred))


# In[42]:


##################  Boosting ###########################
# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)
# Train Adaboost Classifer
model = abc.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = model.predict(X_test)


# Model Accuracy, how often is the classifier correct?
print("Accuracy:", accuracy_score(y_test, y_pred))


# In[43]:


from sklearn.neighbors import KNeighborsClassifier


# In[44]:


knn=KNeighborsClassifier()


# In[ ]:


knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
x_pred = knn.predict(X_train)


print("Confusion Matrix of KNN from training set : \n " , confusion_matrix(y_train,x_pred))
print("\n Accuracy of KNN from training set : \n" , accuracy_score(y_train,x_pred))

print(" \n\n Confusion Matrix of KNN from testing set: \n " , confusion_matrix(y_test,y_pred))
print("\n Accuracy of KNN from testing set : \n" , accuracy_score(y_test,y_pred))

