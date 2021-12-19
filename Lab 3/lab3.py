#!/usr/bin/env python
# coding: utf-8

# # Lab 3

# #### Problem: 1 ULID Analysis

# In[32]:


ulid=input('Enter a ULID: ')


# In[33]:


checklist=['0','1','2','3','4','5','6','7','8','9']
#If the ULID has only English letters, display it.
if ulid.isalpha():
    print(ulid,'consists of only alphabetical letters.')
#If the ULID ends w/ nums, display the reversed ULID.
elif ulid[-1] in checklist:
    print(ulid,'ends with a number.')
    print('The reversed ULID is:', ulid[::-1])
#If ULID doesn't belong to either category, display ULID is not valid
else:
    print(ulid, 'is not a valid ULID.')


# #### Problem 2: Numbers

# In[58]:


#Floating-point number or integer.
number= input('Enter a number: ')


# In[59]:


if '.' in number:
    print('It is a floating-point number and the value is',number)
elif int(number)%2==0:
    print('It is an even number and the value is', number)
else:
    print('It is an odd number and the value is', number)
    


# #### Problem 3: Numbers (Advanced)

# In[156]:


# Modify Problem 2, add if its a positive or negative number.
number=input('Enter a number:')


# In[157]:


if '.' in number:
    if float(number)<0:
        print(number,' is a negative floating-point number.')
    else:
        print(number, 'is a positive floating-point number.')
elif int(number)%2==0 and int(number)>0:
        print(number, 'is a positive even number.')
elif int(number)%2==0 and int(number)<0:
        print(number, 'is a negative even number.')
elif int(number)>0:
        print(number, 'is a positive odd number')
else:
        print(number,'is a negative odd number.')


# In[ ]:




