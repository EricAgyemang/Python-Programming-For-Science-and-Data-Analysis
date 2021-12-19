#!/usr/bin/env python
# coding: utf-8

# ### Lab 9: Pandas and Numpy
# 

# #### Problem 1: Create Table

# In[25]:


import pandas as pd
import numpy as np

data={'Name':['Alice','Bob','Caro','David'],'Age':[19,21,20,22],'Midterm_1':[100,85,90,77],'Midterm_2':[80,99,85,75]}

df=pd.DataFrame(data,index=[1,2,3,4],columns=['Name','Age','Midterm_1','Midterm_2'])


df['Average']=(df['Midterm_1']+df['Midterm_2'])/2


df['Major']=['IT',np.nan,'Mathematics','IT']
df.index=df.Name
del df['Name']
df


# #### Problem 2: Create Data frame with random numbers

# In[69]:


import pandas as pd
import numpy as np

data=np.random.randn(5,5)

df=pd.DataFrame(np.random.randn(5,5))
print(df,'\n\n')

print('The sum of all the positive values:',df[df>0].fillna(0).values.sum())
print('The sum of all the negative values:',df[df<0].fillna(0).values.sum())
print('Rows that meet the requirement:')
print(df[(np.abs(df)>1.8).any(1)])
print('After dropping the 3rd row')
print(df.drop(2,))
print('After dropping the 2nd column:')
print(df.drop(1,axis=1))


# In[ ]:




