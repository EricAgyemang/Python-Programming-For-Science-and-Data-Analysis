#!/usr/bin/env python
# coding: utf-8

# ### Lab 11: Matplotlib

# #### Problem 1: Plot sin(x) and cos(x)

# In[37]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Use matplotlib to plot sin(x) and cos(x), where x is in [-8,8]
x=np.arange(-8,8,0.1)
y1=np.sin(x)
y2=np.cos(x)
plt.plot(x,y1,'g--',label='sin(x)')
plt.plot(x,y2,'o',color='r',label='cos(x)')
plt.legend(loc='best')
plt.xlim([-10,10])
plt.ylim([-2,2])
plt.show()


# #### Problem 2: Four subplots using 'tips.csv'

# In[38]:


#Use pandas to read the 'tips.csv' file into a DataFrame. Create a figure that has four subplots, arranging the plots using a 
#2 by 2 mesh

df=pd.read_csv('Lab11_tips.csv')

fig=plt.figure()

#plot1, splitting this code up (the one previous, and the one after)
ax1=fig.add_subplot(2,2,1)
ax1.hist(df['tip'],bins=10,alpha=0.3,color='g')

#plot2
ax2=fig.add_subplot(2,2,2)
ax2=grouped=df['day'].groupby(df['day'])
ax2=grouped.keys.value_counts().plot(kind='bar',rot=360)

#plot3
ax3=fig.add_subplot(2,2,3)
ax3=df['tip'].plot.box()

#plot4
ax4=fig.add_subplot(2,2,4)
ax4=df['size'].hist()


# In[ ]:




