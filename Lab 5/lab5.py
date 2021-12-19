#!/usr/bin/env python
# coding: utf-8

# ### Lab 5

# #### Problem 1

# In[23]:


# Empty a list of randomly generated numbers
import random
my_list=[random.randint(0,1000) for i in range(500)]
print('The length of the list is',len(my_list))
print('First 20 numbers in the list:',my_list[:20])

def empty_list(list):
    while len(list)>0:
        list.pop()
empty_list(my_list)
print('The list after the function call:',my_list)


# #### Problem 2

# In[47]:


#Dictionary
import random
def empty_list(list):
    while len(list)>0:
        list.pop()
def dict_gen():
    return{x:[random.randint(0,1000) for i in range(5)] for x in range(1,11)}
d=dict_gen()
print('The generated dictionary:',d)
for key in list(d.keys()):
         empty_list(d[key])
print('The dictionary after the cleaning up:',d)
        


# #### Problem 3

# In[7]:


#Learn to call a module
from computation import exp
import math as m
x=15
print('The ground truth for e**15 is:',x,exp(x,n))
n=10
print('Result based on',n,'expansions:',x,exp(x,n))
n=20
print('Result based on',n,'expansions:',x,exp(x,n))
n=100
print('Result based on',n,'expansions:',x,exp(x,n))

