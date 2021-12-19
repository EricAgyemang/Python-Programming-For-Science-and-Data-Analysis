# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 20:07:32 2019

@author: xfang13
"""

import pandas as pd
ozone = pd.read_csv("ozone.csv")
ozone = ozone.drop(["Unnamed: 0"],axis=1)

solar = pd.read_csv("solar.csv")
solar = solar.drop(["Unnamed: 0"],axis=1)

data = ozone.join(solar)

data = data.dropna()

import numpy as np

x = data['airquality.Solar.R'].values
y = data['x'].values

z = np.polyfit(x,y,3)
p = np.poly1d(z)

import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.scatter(x,p(x))
plt.show()