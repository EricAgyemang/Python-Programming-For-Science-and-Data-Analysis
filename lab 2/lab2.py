#!/usr/bin/env python
# coding: utf-8

# In[35]:


#3.5x^2+20x = 10
#ax^2+bx+c = 0

import math
a = 3.5
b = 20
c = -10

root1 = (- b + (math.sqrt((b**2)-(4*a*c))))/(2*a)
root2 = (- b - (math.sqrt((b**2)-(4*a*c))))/(2*a)

print("< ", round(root1,2), " > and <", round(root2,2), " >")


# In[17]:


price = input("Enter the price of an item in cents: ")
qt = 25
dim = 10
nic = 5
price_int = int(price)
changes_dict = {"quart": 0, "dime": 0, "nick": 0}

change = 100 - price_int

change_q = change//25
rem_q = change % 25
change_d = rem_q // 10
rem_d = rem_q % 10
change_n = rem_d // 5

changes_dict["quart"] = change_q
changes_dict["dime"] = change_d
changes_dict["nick"] = change_n

print("You bought an item for ", price_int, " cents and gave me a dollar, so your change is ", change_q, "- quarter, ", change_d, "-dime, ", change_n, "-nickel. The dictionary is: ", changes_dict)


# In[15]:


information_list = ["Ben", "Doiron", 26, "IT 166", "MAT 340", "PHY 240", "PHY 220"]
 
print("Name: ", information_list[0], information_list[1])
print("Age: ",information_list[2])
print("Courses attending this semester: ", information_list[3],", ", information_list[4],", ",information_list[5], ", and ", information_list[6])


# In[ ]:




