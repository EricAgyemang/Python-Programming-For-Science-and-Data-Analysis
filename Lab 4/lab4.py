#!/usr/bin/env python
# coding: utf-8

# ## Lab 4
# 
# #### Problem 1

# In[22]:


#Mean (Mu) and standard deviation (sd)

import random, math as m
a_random_num=[random.randint(1,500) for xi in range(50)]
length_a_random_num=len(a_random_num)
print('The length of the list is:',length_a_random_num)
print('The list is:',sorted(a_random_num))

#Mean (Mu)
n=0
Mu=0
for xi in a_random_num:
    Mu=Mu+xi
    n=n+1
x_bar=Mu/n
print('The Mean is:',x_bar)

#Standard Deviation (sd)
sq_diff=0
sigma=0
for xi in a_random_num:
    sq_diff=(xi-x_bar)**2
    sigma=sigma+sq_diff
variance=sigma/(n-1)
sd=m.sqrt(variance)
print('The standard deviation is:',round(sd,2))


# #### Problem 2

# In[35]:


#The Computation of e^x
import math as m
x=10
print('The expontential is:',x)
n=10
print('There are',n,'steps of expansions.')
den=1;result=1
for i in range(1,n+1):
    den=den*i
    result=result+x**i/den
print('The ground truth for e**10 is',m.e**10)
print('Our result is',result)


# #### Problem 3

# In[1]:


#Write a program that solves the equation

import random
result=0
while result !=66:
    x=[random.randint(1,9) for _ in range(9)]
    result=x[0]+13*x[1]/x[2]+x[3]+12*x[4]-x[5]-11+x[6]*x[7]/x[8]-10
print("One solution is",x)
print("The result is:",result)


# In[ ]:




