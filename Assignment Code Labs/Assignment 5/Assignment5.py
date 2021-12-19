#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from matplotlib import style
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv('DataSetCln.csv')


# In[3]:


df.head(1)


# In[4]:


plt.plot(df['Age'],df['Salary'],'r.')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('Age vs Salary')
plt.show()


# In[5]:


plt.scatter(df['Age'],df['Salary'], color='r')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('Age vs Salary')
plt.show()


# In[6]:


plt.subplot(1,2,1)
plt.plot(df['Age'],df['Salary'], 'r.')

plt.subplot(1,2,2)
plt.plot(df['Age'],df['Purchased'], 'b*')
plt.show()


# In[7]:


plt.subplot(2,1,1)
plt.plot(df['Age'],df['Salary'], 'r.')

plt.subplot(2,1,2)
plt.plot(df['Age'],df['Purchased'], 'b*')
plt.show()


# In[8]:


fig=plt.figure()

axes=fig.add_axes([0.1,0.1,0.8,0.8])
axes.plot(df.Age,df.Salary,'b*')
axes.set_xlabel('Age')
axes.set_ylabel('Salary')
axes.set_title('Age vs Salary')
plt.show()


# In[9]:


fig,axes=plt.subplots()
axes.plot(df.Age,df.Salary, 'r*')
axes.set_xlabel('Age')
axes.set_ylabel('Salary')
axes.set_title('Age vs Salary')
plt.show()


# In[10]:


fig,axes=plt.subplots(nrows=2,ncols=1)

for ax in axes:
    ax.plot(df.Age,df.Salary, 'g*')
    ax.set_xlabel('Age')
    ax.set_ylabel('Salary')
    ax.set_title('Age vs Salary')
fig
plt.tight_layout()


# In[11]:


fig=plt.figure(figsize=(8,4),dpi=100)

fig,axes=plt.subplots(figsize=(5,4))
axes.plot(df.Age,df.Salary, 'b*')
plt.show()


# In[12]:


fig.savefig('test.png',dpi=200)


# In[13]:


fig=plt.figure()
ax=fig.add_axes([0,0,1,1])

ax.plot(df.Age,df.Salary, 'r.', label='Age')
ax.plot(df.Salary,df.Age, 'b.',label='Sal')
ax.legend()
plt.show()


# In[14]:


plt.scatter(df['Age'],df['Purchased'])
plt.xlabel('Age')
plt.ylabel('Purchased')
plt.title('Age vs Salary')
plt.show()


# In[16]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('DataSetCln.csv')


# In[17]:


sns.barplot(x='Age',y='Purchased',data=df)
plt.show()


# In[18]:


sns.barplot(x='Age',y='Salary',data=df)
plt.show()


# In[19]:


sns.barplot(x='Age',y='Salary',data=df, hue='Purchased')
plt.show()


# In[20]:


sns.scatterplot(x='Age',y='Salary',data=df)
sns.kdeplot(df.Age,df.Salary)
plt.show()


# In[21]:


df.head(1)


# In[22]:


sns.regplot(x='Age',y='Salary', ci=None, data=df)
plt.show()


# In[23]:


sns.regplot(x='Age',y='Salary',color='b',data=df)
plt.show()


# In[24]:


sns.regplot(x='Age',y='Salary', ci=None, data=df)
sns.kdeplot(df.Age,df.Salary)
plt.show()


# In[25]:


sns.lmplot(x='Age',y='Salary',hue='Purchased',data=df)
plt.show()


# In[26]:


sns.lmplot(x='Age',y='Salary',hue='Purchased',palette='Set1',data=df)


# In[27]:


sns.countplot(x='Age',data=df)
plt.show()


# In[28]:


fig_d=(10,10)
fig,ax=plt.subplots(figsize=fig_d)
sns.countplot(x='Age',ax=ax,data=df)
plt.show()


# In[32]:


sns.boxplot(x='Age',y='Country',data=df,palette='rainbow')
plt.show()


# In[33]:


sns.boxplot(x='Age',y='Salary',hue='Purchased',data=df,palette='coolwarm')
plt.show()


# In[34]:


sns.violinplot(x='Age',y='Country',data=df.iloc[0:15,0:15],palette='rainbow')
plt.show()


# In[35]:


sns.set_style('ticks')
fig,ax=plt.subplots()
fig.set_size_inches(11.7,8.27)
sns.swarmplot(x='Country',y='Salary',data=df,ax=ax)


# In[36]:


sns.set_style('ticks')
fig,ax=plt.subplots()
fig.set_size_inches(11.7,8.27)
sns.violinplot(x='Country',y='Salary',data=df,ax=ax)


# In[37]:


sns.set_style('ticks')
fig,ax=plt.subplots()
fig.set_size_inches(11.7,8.27)
sns.boxplot(x='Age',y='Salary',data=df,palette='rainbow',ax=ax)
plt.show()


# In[38]:


sns.set_style('ticks')
fig,ax=plt.subplots()
fig.set_size_inches(11.7,8.27)
sns.boxplot(x='Age',y='Salary',hue='Purchased',data=df,palette='coolwarm',ax=ax)


# In[ ]:




