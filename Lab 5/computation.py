#!/usr/bin/env python
# coding: utf-8

# ### Lab 5
# ### Computation.py

# In[1]:


#Create a module for Problem 3 in lab5
def exp(x,n):
    den=1;result=1
    for i in range(1,n+1):
    #den=den*i
        den*=i
    #result=result+x**i/den
        result+=x**i/den
    return result


# In[ ]:




