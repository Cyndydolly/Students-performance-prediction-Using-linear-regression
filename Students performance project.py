#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[2]:


data = pd.read_csv(r"C:\Users\lenovo\OneDrive\Desktop\My Projects\StudentsPerformance.csv")
data.head()


# In[3]:


print(data.shape)
print(data.info())
print(data.describe())


# In[4]:


sns.histplot(data['math score'], bins=10, kde=True)
plt.title("Math Score Distribution")
plt.show()


# In[5]:


data = pd.get_dummies(data, drop_first=True)


# In[6]:


data['average'] = data[['math score', 'reading score', 'writing score']].mean(axis=1)
X = data.drop(['math score', 'reading score', 'writing score', 'average'], axis=1)
y = data['average']


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[9]:


y_pred = model.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


# In[10]:


plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Student Performance")
plt.show()


# In[11]:


#predicting pass or fail
data['result'] = np.where(data['average'] >= 50, 'Pass', 'Fail')


# In[12]:


from sklearn.ensemble import RandomForestClassifier


# In[13]:


#Accuracy
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)


# In[16]:


# Convert numeric target to Pass/Fail To be able to get the accuracy of classification 
data['result'] = np.where(data['average'] >= 50, 'Pass', 'Fail')

X = data.drop(['math score', 'reading score', 'writing score', 'average', 'result'], axis=1)
y = data['result']

from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:




