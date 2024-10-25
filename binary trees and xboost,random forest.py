#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


# In[22]:


# taking a dataset of CardioVescular deases
dataset = r"D:\machine learning files\archive (2)\heart.csv"
data_set=pd.read_csv(dataset)
data_set.head()


# In[21]:


cat_gorical = ['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']


# In[30]:


# doing one hot encoded in the dataset for making more features
one_hot_encoded = pd.get_dummies(data =data_set,prefix=cat_gorical,columns=cat_gorical,dtype=int)


# In[31]:


one_hot_encoded.head()


# In[39]:


data.columns


# In[44]:


features = [x for x in one_hot_encoded.columns if x not in 'HeartDisease']
print(features)


# In[71]:


X_train,X_test,Y_train,Y_test =train_test_split(one_hot_encoded[features],one_hot_encoded['HeartDisease'],
                                                test_size=0.2,random_state=55) 


# In[76]:


min_samples_split_list = [2,10, 30, 50, 100, 200, 300, 700] ## If the number is an integer, then it is the actual quantity of samples,


# In[73]:


training_accuracy = []
testing_accuracy = []
for min_split in min_samples_split_list:
    model = DecisionTreeClassifier(min_samples_split=min_split,
                                  random_state=55)
    model.fit(X_train,Y_train)
    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)
    train_acc = accuracy_score(train_prediction,Y_train)
    testing_acc   = accuracy_score(test_prediction,Y_test)
    training_accuracy.append(train_acc)
    testing_accuracy.append(testing_acc)
plt.title('Train x Validation metrics')
plt.xlabel('min_samples_split')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(min_samples_split_list )),labels=min_samples_split_list)
plt.plot(training_accuracy)
plt.plot(testing_accuracy)


# In[54]:


model = DecisionTreeClassifier(min_samples_split =3,
                                  random_state = 55)


# In[59]:


testing_accuracy


# In[60]:


training_accuracy


# In[91]:


max_depth_list =[1,2, 3, 4, 8, 16, 32, 64,None]
max_depth_list


# In[99]:


# max dept list
training_accuracy = []
testing_accuracy = []
for min_split in max_depth_list:
    model = DecisionTreeClassifier(max_depth=min_split,
                                  random_state=55)
    model.fit(X_train,Y_train)
    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)
    train_acc = accuracy_score(train_prediction,Y_train)
    testing_acc   = accuracy_score(test_prediction,Y_test)
    training_accuracy.append(train_acc)
    testing_accuracy.append(testing_acc)
plt.title('Train x Validation metrics')
plt.xlabel('min_samples_split')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(min_samples_split_list )),labels=min_samples_split_list)
plt.plot(training_accuracy)
plt.plot(testing_accuracy)


# In[100]:


training_accuracy


# In[101]:


testing_accuracy


# In[103]:


min_samples_split_list = [2,10, 30, 50, 100, 200, 300, 700] 
max_depth_list = [2, 4, 8, 16, 32, 64, None]
n_estimators_list = [10,50,100,500]


# In[106]:


# max dept list
training_accuracy = []
testing_accuracy = []
for min_split in min_samples_split_list:
    model = RandomForestClassifier(min_samples_split=min_split,
                                  random_state=55)
    model.fit(X_train,Y_train)
    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)
    train_acc = accuracy_score(train_prediction,Y_train)
    testing_acc   = accuracy_score(test_prediction,Y_test)
    training_accuracy.append(train_acc)
    testing_accuracy.append(testing_acc)
plt.title('Train x Validation metrics')
plt.xlabel('min_samples_split')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(min_samples_split_list )),labels=min_samples_split_list)
plt.plot(training_accuracy)
plt.plot(testing_accuracy)


# In[107]:


training_accuracy


# In[108]:


testing_accuracy


# In[110]:


max_depth_list = [2, 4, 8, 16, 32, 64, None]
n_estimators_list = [10,50,100,500]


# In[111]:


# max dept list
training_accuracy = []
testing_accuracy = []
for min_split in max_depth_list:
    model = RandomForestClassifier(max_depth=min_split,
                                  random_state=55)
    model.fit(X_train,Y_train)
    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)
    train_acc = accuracy_score(train_prediction,Y_train)
    testing_acc   = accuracy_score(test_prediction,Y_test)
    training_accuracy.append(train_acc)
    testing_accuracy.append(testing_acc)
plt.title('Train x Validation metrics')
plt.xlabel('min_samples_split')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(min_samples_split_list )),labels=min_samples_split_list)
plt.plot(training_accuracy)
plt.plot(testing_accuracy)


# In[112]:


training_accuracy


# In[113]:


testing_accuracy


# In[114]:


n_estimators_list = [10,50,100,500]


# In[116]:


# max dept list
training_accuracy = []
testing_accuracy = []
for min_split in n_estimators_list:
    model = RandomForestClassifier(n_estimators=min_split,
                                  random_state=55)
    model.fit(X_train,Y_train)
    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)
    train_acc = accuracy_score(train_prediction,Y_train)
    testing_acc   = accuracy_score(test_prediction,Y_test)
    training_accuracy.append(train_acc)
    testing_accuracy.append(testing_acc)
plt.title('Train x Validation metrics')
plt.xlabel('min_samples_split')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(min_samples_split_list )),labels=min_samples_split_list)
plt.plot(training_accuracy)
plt.plot(testing_accuracy)


# In[117]:


training_accuracy


# In[118]:


testing_accuracy


# In[ ]:





# In[ ]:





# In[ ]:




