#!/usr/bin/env python
# coding: utf-8

# ### Lab 8: Numpy


# #### Problem 1: Max, min, sum, mean

# In[21]:


# Launch Numpy
import numpy as np
size=50

#Create random one-dimensional array
A=np.random.randn(size)
B=np.random.randn(size)
C=A+B

#Call for the .max() function
print("The maximum is:%.2f"%  C.max())
print("The minimum is:%.2f"%  C.min())
print("The sum is:%.2f"% C.sum())
print("The mean is:%.2f"% C.mean())

#For negative count, use .where() function
print("The number of negative values is:%.2f"% np.where(C<0,1,0).sum())


# #### Problem 2: Max, min, sub-matrix

# In[38]:


import numpy as np

#Create random list of 5x4 array
x=np.random.randn(5,4)
#use \n" to make the matrix cleaner
print("The original Matrix is:\n",x)

#Find max value for each ROW
print("The maximum value for each row is:", x.max(axis=1))

#Find max values for each COLUMN
print("The maximum value for each column is:", x.max(axis=0))

#Find min values for each ROW
print("The minimum value for eah row is:", x.min(axis=1))

#Find minimum values for each COLUMN
print("The minium value for each column is:", x.min(axis=0))

#Find a sub matrix of x and display it
#Recall, doesn't count the first index
print("The sub-matrix is:\n", x[2:,1:])


# #### Problem 3: Sum of all neg and pos values in array

# In[42]:


import numpy as np

y=np.random.randn(10,10)
print("The sum of all positive values is:%.2f"% (y*(y>0)).sum())
print("The sum of all negative values is:%.2f"% (y*(y<0)).sum())

