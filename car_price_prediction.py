#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


a=pd.read_csv(r'C:\Users\DEEKSHITH\Desktop\car_price.csv')


# In[3]:


a.head()


# In[4]:


a.isnull().sum()#checking for null values

removing unnecessary columns
# In[5]:


a.describe()


# In[6]:


a=a.drop(['name'],axis=1)
a.head()

LabelEncoding
# In[7]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
a['fuel_type']=le.fit_transform(a.fuel)
a['transmission_type']=le.fit_transform(a.transmission)
a['sellertype']=le.fit_transform(a.seller_type)
a['owner_type']=le.fit_transform(a.owner)
a=a.drop(['fuel','seller_type','transmission','owner'],axis=1)#converting to categorical data
a


# In[36]:


import matplotlib.pyplot as plt
plt.scatter(a[['fuel_type']],a[['selling_price']],color='red',marker='o',label='fuel_type')
plt.scatter(a[['transmission_type']],a[['selling_price']],color='magenta',marker='1',label='transmission_type')
plt.scatter(a[['sellertype']],a[['selling_price']],color='blue',marker='2',label='seller_type')
plt.scatter(a[['owner_type']],a[['selling_price']],color='purple',marker='3',label='owner_type')
plt.ylabel('selling_price')
plt.legend()
plt.show()

splitting into dependent and independent variable
# In[39]:


x=a.drop(['selling_price','year'],axis=1)#independent variables
y=a['selling_price']#dependent variable
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
#selecting the best suited machine learning model for prediction
from sklearn.linear_model import LinearRegression
b=LinearRegression()
#training the dataset
c=b.fit(x_train,y_train)
#testing the score of the model
b.score(x_test,y_test)
#predicting the model with independent dataset

predicting the machine learning model
# In[42]:


data_set=pd.DataFrame({
    'km_driven':120238,
    'fuel_type':0,
    'transmission_type':1,
    'sellertype':0,
    'owner_type':2
},index=[0])
b.predict(data_set)


# In[ ]:




