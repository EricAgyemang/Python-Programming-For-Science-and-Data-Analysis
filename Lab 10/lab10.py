#!/usr/bin/env python
# coding: utf-8

# ### Lab 10


# #### Problem 1

# In[10]:


import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

x=np.arange(-8,8.1,0.1)
#A[Start,Stop]
y1=np.sin(x)
y2=np.cos(x)
plt.plot(x,y1,'g--',label='sin(x)')
plt.plot(x,y2,'o',color='r',label='cos(x)')
plt.legend(loc='best')
plt.xlim([-10,10])
plt.ylim([-2,2])
plt.show()


# #### Problem 2

# In[7]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

df=pd.read_csv('lab10_movie_scores.csv')
print('The first 10 lines of the data are')
df.head(10)


# In[9]:


print('All the columns of the data are')
df.columns[0:]


# In[38]:


#Only care about FILM and IMDB

df1=df[['FILM','IMDB','IMDB_norm','IMDB_norm_round','IMDB_user_vote_count']]
df1


# In[40]:


#List only movies whose ratings are over 7 (out of 10) in IMDB
df1[df1['IMDB']>=7]


# In[48]:


#Find lesser-known movies to watch, with fewer than 20K votes in IMDB
df2=df1[df1['IMDB_user_vote_count']<20000]
df2


# In[49]:


#Finally, export this file to an Excel spreadsheet (another csv file) -- without the DataFrame index.
df2.to_csv('data_barnes_lab10.csv')


# In[ ]:




