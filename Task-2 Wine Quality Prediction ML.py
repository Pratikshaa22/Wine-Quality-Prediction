#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("winequality.csv")


# In[4]:


df.head(10)


# In[5]:


df.shape


# In[6]:


df.isnull().sum()


# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


df.groupby('quality').mean()


# In[10]:


corr = df.corr()


# In[11]:


plt.figure(figsize=(12,7))
sns.heatmap(corr,cmap="coolwarm",annot=True)
plt.show()


# In[25]:


sns.countplot(df['quality'])
plt.show()


# In[13]:


df.hist(figsize=(15,13),bins=50)
plt.show()


# In[14]:


sns.pairplot(df)


# In[17]:


# Create Classification version of target variable
df['goodquality'] = [1 if x >= 7 else 0 for x in df['quality']]
X = df.drop(['quality','goodquality'], axis = 1)
y = df['goodquality']


# We have created another column named "goodquality" which has value 1 if the wine quality >= 7 otherwise 0. i.e if the value is 1 then the wine is goof quality otherwise it is decent or bad quality.

# In[18]:


df.head(17)


# ### Spliting the dataset

# In[19]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[20]:


X_train


# In[21]:


y_test


# ### Logistic Regression Model

# In[22]:


from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression()

model1.fit(X_train,y_train)

prediction1 = model1.predict(X_test)

prediction1


# In[23]:


from sklearn.metrics import accuracy_score,confusion_matrix

print(accuracy_score(prediction1,y_test))


# In[ ]:




