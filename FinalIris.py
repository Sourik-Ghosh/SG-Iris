#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

iris = pd.read_csv("Iris.csv")


# In[2]:


iris.head()


# In[3]:


iris = iris.drop(columns = ['Id'])
iris.head()


# In[4]:


iris.describe()


# In[5]:


iris.info()


# In[6]:


iris['Species'].value_counts()


# In[7]:


iris.isnull().sum()


# In[8]:


iris['SepalLengthCm'].hist()


# In[9]:


iris['SepalWidthCm'].hist()


# In[10]:


iris['PetalLengthCm'].hist()


# In[11]:


iris['PetalWidthCm'].hist()


# In[12]:


colors = ['red','orange','blue']
species = ['Iris-virginica','Iris-versicolor','Iris-setosa']


# In[13]:


for i in range(3):
    x=iris[iris['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c = colors[i], label=species[i])
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.legend()


# In[14]:


for i in range(3):
    x=iris[iris['Species'] == species[i]]
    plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    plt.legend()


# In[15]:


for i in range(3):
    x=iris[iris['Species'] == species[i]]
    plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
    plt.xlabel("Sepal Width")
    plt.ylabel("Petal Width")
    plt.legend()


# In[16]:


iris.corr()


# In[17]:


corr = iris.corr()
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot=True, ax=ax, cmap = 'coolwarm')


# In[18]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[19]:


iris['Species'] = le.fit_transform(iris['Species'])
iris.head()


# In[20]:


from sklearn.model_selection import train_test_split
X = iris.drop(columns=['Species'])
Y = iris['Species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)


# In[21]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[22]:


model.fit(x_train, y_train)


# In[23]:


print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[24]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


# In[25]:


model.fit(x_train, y_train)


# In[31]:


print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[27]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[28]:


model.fit(x_train, y_train)


# In[29]:


print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[ ]:




