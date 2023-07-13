#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[3]:


titanic = sns.load_dataset('titanic')
titanic.shape
titanic.head()


# In[4]:


titanic = titanic[['survived','pclass','sex','age']]
titanic.dropna(axis=0,inplace=True)
titanic['sex'].replace(['male','female'],[0,1],inplace=True)
titanic.head()


# In[5]:


from sklearn.neighbors import KNeighborsClassifier


# In[6]:


model = KNeighborsClassifier()


# In[7]:


y = titanic['survived']
x = titanic.drop(['survived'],axis=1)


# In[8]:


model.fit(x,y)
model.score(x,y)


# In[9]:


def survie(model , pclass=1,sex=0,age=19):
    x=np.array([pclass,sex,age]).reshape(1,3)
    print(model.predict(x))
    print(model.predict_proba(x))


# In[10]:


survie(model)


# In[11]:


a=survie(model)


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns

# Prédire les classes des données d'entraînement
predicted = model.predict(x)

# Créer un graphique de dispersion
plt.figure(figsize=(6, 10))
sns.scatterplot(x='age', y='pclass', hue='survived', data=titanic, palette='Set1')
plt.title("Survival Prediction")
plt.xlabel("Age")
plt.ylabel("Passenger Class")

# Ajouter les prédictions du modèle
sns.scatterplot(x=titanic['age'], y=titanic['pclass'], hue=predicted, marker='x', s=100, palette='Set1', legend=False)

plt.show()

