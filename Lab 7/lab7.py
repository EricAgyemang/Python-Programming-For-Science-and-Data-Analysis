#!/usr/bin/env python
# coding: utf-8

# ### Lab 7

# #### Problem 1: Python Loop

# In[6]:


# Use a python loop to compute: 1+2-3+4-5+...-99+100

sum=0
for i in range(1,101):
    if i==1:
        sum+=i
    elif i%2==0:
        sum+=i
    else:
        sum-=i
print(sum)
    


# #### Problem 2: Define a function

# In[27]:


#Define a python function which takes a list of numbers as the input
import random
list=[random.randint(0,100) for i in range(5)]
list=sorted(list)
print('The list is:',list)
def difference(list):
    num=[]
    for i in range(1,len(list)):
        num.append(list[i]-list[i-1])
    return num
print('The difference between two numbers in the list is:', difference(list))


# #### Problem 3: Volume of a Sphere

# In[16]:


#allow user to input radius, calculate the volume of the sphere
import math as m
#radius=input('What is the radius of the sphere?')
def myfunction(radius):
    volume=(4/3)*m.pi*(radius**3)
    return volume
print('The Volume of a Sphere with radius 1 is:',myfunction(1))
print('The Volume of a Sphere with radius 2 is:',myfunction(2))
print('The Volume of a Sphere with radius 3 is:',myfunction(3))


# #### Problem 4: Dictionary, names and ages

# In[31]:


#Given a list of names and ages, find whose the oldest/youngest at the same time.
infor = {'Alice':20,'Bob':25,'Caro':40,'David':32,'Evan':60,'Frank':38,'Greg':45,'Hason':12,'Izzy':33,'Jason':17,'Kyle':23}
#Create two different lists to add the keys to one, and add the values to another
values=list(infor.values())
keys=list(infor.keys())
biggervalue=values[0]
key1=0
smallervalue=values[-1]
key2=0
for i in range(len(values)):
    if values[i-1] > biggervalue:
        biggervalue=values[i-1]
        key1=i-1
    if values[i-1]<smallervalue:
        smallervalue=values[i-1]
        key2=i-1
print('The oldest is:'+str(keys[key1])+','+ 'The youngest is:'+str(keys[key2]))


# In[ ]:




