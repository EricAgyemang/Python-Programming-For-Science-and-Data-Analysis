#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Problem 1: (6 points)
#Create a list with 999 randomly generated
#integers, each of which is generated in between -1000 and 1000.
#And write Python statements to find out the
#mean, median, minimum, and maximum value inside the list.
#Hint: You will need to import the random module and use the randint() method
#for generating the integers.

import random
import statistics
import math

#Your code goes below
l = []
for _ in range(999):
    l.append(random.randint(-1000,1000))
print("The maximum of the list is "+str(max(l)))
print("The mean of the list is "+str(sum(l)/len(l)))
print("The minimum of the list is "+str(min(l)))
print("The median of the list is "+str(statistics.median(l)))


# In[2]:


#Problem 2: (10 points)
#Your code goes below
n = 3
s=0
e=0
x = [random.randint(0,100) for _ in range(n)]
y = [random.randint(0,100) for _ in range(n)]
for _ in range(n):
    s += abs(x[_-1]-y[_-1])
    e += (x[_-1]-y[_-1])**2
e = math.sqrt(e)
print("Manhattan(x,y) = " + str(s))
print("Euclidean(x,y) = "+str(e))


# In[3]:


#Problem 3: (6 points)
#Implement the insertion-sort algorithm
#Pseudo code:
#Input: A Python list, A, with unsorted numbers
#Output: A sorted list, A, where the numbers are sorted ascendingly
#for i from 1 to n (n is the length of A)
#   assign i-1 to j
#   assign A[i] to key
#   while j is greater than or equal to 0 and A[j] is greater than key
#       assign A[j] to A[j+1]
#       decreament j by 1
#   assign key to A[j+1]

A = [random.randint(0,1000) for i in range(50)]
#Your code goes below
N = []
temp = 0
for i in range(0,len(A)):
    N.append(A[i])
for i in range(len(A)):
    for j in range(i+1, len(A)):
        if(N[i]>N[j]):
            temp = N[i]
            N[i] = N[j]
            N[j] = temp
A = N
print(A)


# In[4]:


#Problem 4: (8 points)
#Write Python statements that create a dictionary of word counts.
#Specifically, keys of the dictionary are words; values of the dictionary
#are the number of occurances of the words
#For example, given s = 'go go hurry up', the dictionary, D, should be like
# {'go':2, 'hurry':1, 'up':1}

s = '''
Deep learning also known as deep structured learning hierarchical learning or deep machine learning is a branch of machine learning
based on a set of algorithms that attempt to model high level abstractions in data In a simple case there might be two sets of neurons
ones that receive an input signal and ones that send an output signal When the input layer receives an input it passes on a modified
version of the input to the next layer In a deep network there are many layers between the input and output and the layers are not made
of neurons but it can help to think of it that way allowing the algorithm to use multiple processing layers composed of multiple linear
and non-linear transformations
'''

#Your code goes below
s_new = s.lower().split()
counts={}
for _ in s_new:
    if _ not in counts.keys():
        counts[_] = 0
    counts[_] += 1
print(counts)


# In[ ]:




