#!/usr/bin/env python
# coding: utf-8

# # problem 1 use a python code to compute: 1+2-3+4-5+....-99+100

# In[1]:


sum=0
for i in range(1,101):
    if i ==1:
        sum=sum+i
    elif i %2==0:
        sum+=i
    else: 
        sum-=i
print(sum)
        


# #problem 2

# In[12]:


list=[1,2,3,4]
def difference(list):
    num=[]
    for i in range (1,len(list)):
        num.append(list[i]-list[i-1])
    return num
print(difference(list))


# # problem 3

# In[18]:


import math as m
def myFunction(radius):
    Volume = (4/3)*m.pi*(radius**3)
    return Volume
print("The volume of a sphere with a radius of 1 is: "+ str(myFunction(1)))


# # problem 4

# In[2]:


myDictionary={"joe":70, "Eric":2, "krystina": 20, "ryan": 18, "jess": 90, "anna": 76}
values=list(myDictionary.values())
keys=list(myDictionary.keys())
biggerValue=values[0]
key1=0
smallerValue=values[0]
key2=0

for i in range(len(values)):
    if values[i-1]>biggerValue:
        biggerValue=values[i-1]
        key1=i-1
    if values[i-1]<smallerValue:
        smallerValue=values[i-1]
        key2=i-1
print("The oldest is: "+str(keys[key1]))
print("The youngest is: "+str(keys[key2]))


# In[ ]:




