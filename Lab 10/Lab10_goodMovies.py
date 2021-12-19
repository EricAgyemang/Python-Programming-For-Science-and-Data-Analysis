#!/usr/bin/env python
# coding: utf-8

# In[2]:


from matplotlib import pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
x=np.arange(-8,8.1,0.1)
y1=np.sin(x)
y2=np.cos(x)
plt.plot(x,y1,'b--',label='sin(x)')
plt.plot(x,y2,'g:',label='cos(x)')
plt.legend(loc='best')
plt.xlim([-10,10])
plt.ylim([-2,2])


# In[3]:


# Import pandas library
import pandas as pd


# In[4]:


# Open the csv file, movie_scores, using pandas
# 'movie_scores.csv'
dat=pd.read_csv('lab10_movie_scores.csv')


# In[5]:


# Display the first 10 rows 
dat.head(10)


# In[6]:


# List all the columns the table provides
dat.columns


# In[7]:


# only care about Imdb, so create a new table that takes the Film and all the columns relating to IMDB
dat2=dat[['FILM','IMDB','IMDB_norm','IMDB_norm_round','IMDB_user_vote_count']]


# In[8]:


# List only movies whose ratings are over 7 (out of 10) in IMDB
datgood=dat['IMDB'].astype(int)>7
dat3=dat2[datgood]
dat3


# In[9]:


# Find lesser-known movies to watch, with fewer than 20K votes in IMDB
datless=dat['IMDB_user_vote_count']<20000
lessknown=dat2[datless]
lessknown


# In[10]:


# Finally, export this file to an Excel spreadsheet (another csv file) -- without the DataFrame index.
lessknown.to_csv('lab10Edit.csv',index=False)


# In[ ]:




