#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as n


# In[2]:


df=pd.read_csv('crime_incident_data2017.csv')
df.head()


# In[3]:


df.tail()


# In[4]:


print('Before dropping nulls:',df.shape)


# In[5]:


df1=df.dropna()


# In[6]:


print('After dropping nulls:',df1.shape)


# In[7]:


df.info()


# In[8]:


from sklearn.impute import SimpleImputer


# In[9]:


print(df['Open Data Lat'].skew())
print(df['Open Data Lon'].skew())
print(df['Open Data X'].skew())
print(df['Open Data Y'].skew())


# In[10]:


meanImp=SimpleImputer(missing_values=n.nan,strategy='mean')
medImp=SimpleImputer(missing_values=n.nan,strategy='median')
freqImp=SimpleImputer(missing_values=n.nan,strategy='most_frequent')
addImp=freqImp.fit(df[['Address']])
df['Address']=addImp.transform(df[['Address']].values)
#I used 'most frequent' for the missing addresses as it is an object data type.
neighborImp=freqImp.fit(df[['Neighborhood']])
df['Neighborhood']=neighborImp.transform(df[['Neighborhood']].values)
#I used 'most frequent' for the missing neighvorhoods as it is an object data type.
latImp=meanImp.fit(df[['Open Data Lat']])
df['Open Data Lat']=latImp.transform(df[['Open Data Lat']].values)
#Since the data for latitude is not heavily skewed, I used 'mean' for the missing latitude data.
lonImp=meanImp.fit(df[['Open Data Lon']])
df['Open Data Lon']=lonImp.transform(df[['Open Data Lon']].values)
#Since the data for longitude is not heavily skewed, I used 'mean' for the missing longitude data.
XImp=meanImp.fit(df[['Open Data X']])
df['Open Data X']=XImp.transform(df[['Open Data X']].values)
#Since the data for X values are not heavily skewed, I used 'mean' for the missing X data.
YImp=meanImp.fit(df[['Open Data Y']])
df['Open Data Y']=YImp.transform(df[['Open Data Y']].values)
#Since the data for Y values are not heavily skewed, I used 'mean' for the missing Y data.
df.head()


# In[11]:


print('The earliest report date is:',pd.to_datetime(df['Report Date']).min())
print('The latest report date is:',pd.to_datetime(df['Report Date']).max())


# In[12]:


print(df['Report Date'].value_counts())
print('May 31 was the worst day (with 229 reports), January 11 was the best day (with 81 reports)')


# In[18]:


df_4July=df[df['Report Date']=='7/4/17'][['Offense Type', 'Offense Count']]


# In[19]:


df_4July.head()


# In[20]:


df_4July.groupby(['Offense Type']).sum()


# In[29]:


print(df['Offense Category'].unique())
print(df['Offense Category'].value_counts())


# In[30]:


dffinal=df.drop('Offense Type',axis=1)


# In[31]:


dffinal.head()


# In[ ]:




