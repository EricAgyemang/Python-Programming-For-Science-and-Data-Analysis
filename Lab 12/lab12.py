#!/usr/bin/env python
# coding: utf-8

# ### Lab 12: Times Series in Pandas

# #### Problem 1: Apple Stock Plot

# In[47]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

#index_col sets index for whatever column
#parse_dates=True decreases parsing time
df=pd.read_csv('AAPL.csv',index_col='Date',parse_dates=True)

#Check if index iS what you want it to be
df.index

#select information for 2016
df2=df['2016-01-01':'2016-12-31']

#plot open and high prices using lines
df2['Open'].plot(legend=True)
df2['High'].plot(legend=True)

#mean for may and june of 2016
m1=df2['Close'].mean()

#difference of mean between the two months
print("The difference for the Close Prices in May and June is %.2f"% (df2.loc['2016-05'].Close.mean()
                                                                     -df2.loc['2016-06'].Close.mean()))


# #### Problem 1: Date Range function

# In[87]:


#for times series and dates
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.holiday import AbstractHolidayCalendar,nearest_workday,Holiday
from pandas.tseries.offsets import CustomBusinessDay

#create an ISU calendar
class ISUFALL2018Calendar(AbstractHolidayCalendar):
    rules=[Holiday('Thanksgiving',month=11,day=19),Holiday('Thanksgiving',month=11,day=20),
           Holiday('Thanksgiving',month=11,day=21),Holiday('Thanksgiving',month=11,day=22),
           Holiday('Thanksgiving',month=11,day=23)]
ISU=CustomBusinessDay(calendar=ISUFALL2018Calendar())

#print the date range for students
calendar=pd.date_range(start='8/20/2018',end='12/14/18',freq=ISU)
print(calendar)

#count number of days in the calendar using %d
print("There are %d weekdays in Fall 2018."%(calendar.size))

#it166 classes
weekmask='Mon Wed'
it166=CustomBusinessDay(weekmask='Mon Wed',calendar=ISUFALL2018Calendar())
it166Calendar=pd.date_range(start='8/20/2018',end='12/14/18',freq=it166)

#removes the last week of finals from this calendar
removes=[]
for each in it166Calendar:
    if each.month==12 and each.day in range(10,15):
        removes.append(each)
it166Calendar=it166Calendar.drop(removes)
print(it166Calendar)
print("There are %d lectures of IT166 in Fall 2018."%(it166Calendar.size))


# In[ ]:




