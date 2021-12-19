#!/usr/bin/env python
# coding: utf-8

# ### Lab 14: Machine Learning Hands-on Lab
# #### 5. Import Libraries

# In[1]:


#this python 3 environment comes with many helpful analytics libraries installed
#import libraries

import numpy as np
import pandas as pd

#Data visualization
import matplotlib.pyplot as plt

#statistical data visualization
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

#input data files are available in the "../it 166/" directory
#the input directory

import os
os.getcwd()


# In[2]:


import warnings

warnings.filterwarnings('ignore')


# #### 6.Import datasheet

# In[3]:


df=pd.read_csv('weather-aus.csv')
df.head()


# #### 7.Exploratory data analysis
# 
# Now I will explore the data to gain insights about the data

# In[4]:


#view dimensions of dataset
df.shape


# We can see that there are 142193 instances and 24 vriables in the data set

# In[5]:


col_names=df.columns

col_names


# #### Drop RISK_MM variable
# 
# It is given in the dataset description, that we should drop the RISK_MM feature variable from the dataset descriprtion. So, we should drop it as follows-

# In[6]:


#df.drop(['column_name'],axis=1,inplace=True)


# In[7]:


#view summary of dataset
df.info()


# In[8]:


df=df.drop_duplicates()


# In[9]:


df=df.dropna()


# In[10]:


df.shape


# #### Types of variables
# 
# In this section, I segregate the dataset into categoriacal and numerical values. There are a mixture of categorical and numerical variables in the dataset. Categorical variables have sata type object. Numerical varibel have data type float64.
# 
# First of all, I will find categorical variables.

# In[11]:


# find categorical variables

categorical=[var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)


# In[12]:


#view the categorical variables
df[categorical].head()


# #### Summary of categorical variables
# 
# There is a date variable. It is denoted by Date column
# 
# There are 6 categorical variables. These are given by Location, WindGustDir, WindDir9am, WindDir3pm,RainToday and RainTomorrow.
# 
# There are two binary categorical variables- RainToday and RainTomorrow.
# 
# RainTomorrow is the target variable.

# #### Explore problems within categorical variables
# First, I will explore the categorical variables.

# #### Missing values in categorical variables

# In[13]:


#check missing values in categorical variables

df[categorical].isnull().sum()


# In[14]:


#print ctegorical variables containing the missing values

cat1=[var for var in categorical if df[var].isnull().sum()!=0]

print(df[cat1].isnull().sum())


# We can see that there are only 4 categorical variables in the dataset which contains missing values. There are WindGustDir,WindDir9am and RainToday.

# #### Frequency Count of categorical data
# 
# Now I will check the frequency counts of categorical variables.

# In[15]:


#view frequency of categorical variables

for var in categorical:
    print(df[var].value_counts())


# In[16]:


#view frequency distribution of categorical variables

for var in categorical:
    print(df[var].value_counts()/float(len(df)))


# #### Number of labels: cardinality
# 
# The number of labels within a categorical variable is known as cardinality. A high number of labels within a variable is known as high cardinality. High cardinality may pose some serious problems in the machine learning model. So, I will check for high cardinality.

# In[17]:


# check for cardinality in categorical variables

for var in categorical:
    print(var,'contains',len(df[var].unique()),'labels')


# We can see the there is a Date variable which needs to be preprocessed. I will do preprocessing in the foloowing section.
# 
# All the other variables contain relativley smaller number of variables.

# #### Feature Engineering of Date variable

# In[18]:


df['Date'].dtypes


# We can see that the data type of Date variable is object. I will parse the date currently coded as object into datetime format.

# In[19]:


#parse the dates, currently coded as strings into datetime format
df['Date']=pd.to_datetime(df['Date'])


# In[20]:


#exrtact the year from date

df['Year']=df['Date'].dt.year

df['Year'].head()


# In[21]:


#extract month from date

df['Month']=df['Date'].dt.month

df['Month'].head()


# In[22]:


#extract day from date

df['Day']=df['Date'].dt.day

df['Day'].head()


# In[23]:


#again, view the summmart of dataset

df.info()


# We can see that there are three additional columns created from Date variable. Now, I will drop the original Date variable from the dataset.

# In[24]:


#drop the original date variable

df.drop('Date',axis=1,inplace=True)

#preview the dataset again
df.head()


# Now, we can see that the Date variable has been removed from the dataset.

# #### Explor Categorical Variables
# 
# Now, I will explore the categorical variables one by one.

# In[25]:


#find categorical variables

categorical=[var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :',categorical)


# We can see that there are 6 categorical variables in the dataset. The Date variable has been removed. First, I will check missing values in categorical variables.

# In[26]:


#check for missing values in categorical variables

df[categorical].isnull().sum()


# We can see that WindGustDir,WindDir9am,WindDir3pm,RainToday variables contain missing values. I will explore these variables one by one.

# #### Explore Location variable

# In[27]:


#print number of labels in Location variable

print('Location contains',len(df.Location.unique()),'lables')


# In[28]:


#check labels in location variable

df.Location.unique()


# In[29]:


#check frequency distribution of value in location variable
df.Location.value_counts()


# In[30]:


#let's do One Hot Encoding of location variable get k-1 dummy variables after
#One Hot Encoding preview the dataset with head() method

pd.get_dummies(df.Location,drop_first=True).head()


# #### Explore WindGustDir variable

# In[31]:


#print number of labels in WindGustDir variable

print('WindGustDir contains',len(df['WindGustDir'].unique()),'labels')


# In[32]:


#check labels in WindGustDir variable

df['WindGustDir'].unique()


# In[33]:


#Check frequency distribution of values in WindDustDir variable

df.WindGustDir.value_counts()


# In[34]:


#let's do One Hot Encoding of WindGustDir variable get k-1 dummy variables after
#One Hot Encoding also add an additional dummy variable to indicate there was missing
#data preview the dataset with head() methof
pd.get_dummies(df.WindGustDir,drop_first=True,dummy_na=True).head()


# In[35]:


#sum the number of 1s per boolean variable over the rows of the dataset
#it will tell us how many observations we have for each category

pd.get_dummies(df.WindGustDir,drop_first=True,dummy_na=True).sum(axis=0)


# We can see that there are 9330 missing values in WindGustDir variable.

# #### Explore WindDir9am variable

# In[36]:


#print number of labels in WindGustDir variable

print('WindDir9am contains',len(df['WindDir9am'].unique()),'labels')


# In[37]:


#check labels in WindDir9am variable

df['WindDir9am'].unique()


# In[38]:


#check frequency distribution of values in WindDir9am variable

df['WindDir9am'].value_counts()


# In[39]:


#Let's do One Hot Encoding of WindDir9am variable. get k-1 dummy variables
#after One Hot Encoding also add an additional dummy variable to indicate there
#was a missing data preview the dataset with head() method

pd.get_dummies(df.WindDir9am,drop_first=True,dummy_na=True).head()


# In[40]:


#sum the number of 1s per boolean variable over the rows of the dataset
#it will tell us how many observations we have for each category

pd.get_dummies(df.WindDir9am,drop_first=True,dummy_na=True).sum(axis=0)


# we can see that there are 10013 missing values in the WindDir9am variable.

# #### Explore WindDir3pm variable

# In[41]:


#print number of labels in WindDir3pm variable

print('WindDir3pm contains',len(df['WindDir3pm'].unique()),'labels')


# In[42]:


#check labels in WindDir3pm variable
df['WindDir3pm'].unique()


# In[43]:


#check frequency distribution of values in WIndDir3pm variable
df['WindDir3pm'].value_counts()


# In[44]:


#lets do One HOt Encoding of WindDir3pm variable get k-1 dummer variables after
#One Hot Encoding also add an additional dummy variable to indicate there was 
#missing data preview the dataset with head() method

pd.get_dummies(df.WindDir3pm,drop_first=True,dummy_na=True).head()


# In[45]:


#sum the number of 1s per boolean variable over the rows of the dataset it will
#tell us how many obeservations we have for each category

pd.get_dummies(df.WindDir3pm,drop_first=True,dummy_na=True).sum(axis=0)


# There are 3778 missing values in the WindDir3pm variable.

# ### Explore RainToday variable

# In[46]:


#print number of labels in RainToday variable

print('RainToday contains',len(df['RainToday'].unique()),'Labels')


# In[47]:


#Check labels in WindGustDir variable

df['RainToday'].unique()


# In[48]:


#check frequency distributions of values in WindGustDir variable

df.RainToday.value_counts()


# In[49]:


#lets do One HOt Encoding of RainToday variable get k-1 dummer variables after
#One Hot Encoding also add an additional dummy variable to indicate there was 
#missing data preview the dataset with head() method

pd.get_dummies(df.RainToday,drop_first=True,dummy_na=True).head()


# In[50]:


#sum the number of 1s per boolean variable over the rows of the dataset it will
#tell us how many obeservations we have for each category

pd.get_dummies(df.RainToday,drop_first=True,dummy_na=True).sum(axis=0)


# There are 1406 missing values in the RainToday variable.

# #### Explore Numerical variables

# In[51]:


#find numerical variables

numerical=[var for var in df.columns if df[var].dtype!='O']

print('There are {} numberical variables\n'.format(len(numerical)))

print('The numerical variables are :',numerical)


# In[52]:


#view the numreical variable

df[numerical].head()


# ### Explore problems within numerical variables

# In[53]:


#check mmissing values in numerical ariables
df[numerical].isnull().sum()


# We can see that all the 16 numerical variables conatins missing values.

# #### Outliers in numerical variables

# In[54]:


#view summary statistics in numerical variables
print(round(df[numerical].describe()),2)


# On closer inspection, we can see that the Rainfall, Evaporation, WindSpeed9am and WindSpeed3pm columns may contain outliers
# 
# I will draw a boxplot to visualize outliers in the above variables.

# In[55]:


#draw boxplots to visualize outliers

plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
fig=df.boxplot(column='Rainfall')
fig.set_title('')
fig.set_ylabel('Rainfall')

plt.subplot(2,2,2)
fig=df.boxplot(column='Evaporation')
fig.set_title('')
fig.set_ylabel('Evaporation')

plt.subplot(2,2,3)
fig=df.boxplot(column='WindSpeed9am')
fig.set_title('')
fig.set_ylabel('WindSpeed9am')


plt.subplot(2,2,4)
fig=df.boxplot(column='WindSpeed3pm')
fig.set_title('')
fig.set_ylabel('WindSpeed3pm')


# ### Check the Distribution of variables

# In[56]:


#plot histogram to check distributions

plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
fig=df.Rainfall.hist(bins=10)
fig.set_xlabel('Rainfall')
fig.set_ylabel('RainTomorrow')

plt.subplot(2,2,2)
fig=df.Rainfall.hist(bins=10)
fig.set_xlabel('Evaporation')
fig.set_ylabel('RainTomorrow')

plt.subplot(2,2,3)
fig=df.Rainfall.hist(bins=10)
fig.set_xlabel('WindSpeed9am')
fig.set_ylabel('RainTomorrow')

plt.subplot(2,2,4)
fig=df.Rainfall.hist(bins=10)
fig.set_xlabel('WindSpeed3pm')
fig.set_ylabel('RainTomorrow')


# In[57]:


#find outliers for Rainfall variable

IQR= df.Rainfall.quantile(0.75)-df.Rainfall.quantile(0.25)
Lower_fence=df.Rainfall.quantile(0.25)-(IQR*3)
Upper_fence=df.Rainfall.quantile(0.75)+(IQR*3)
print('Rainfall outliers are values< {lowerboundary} or > {upperboundary}\n'
     .format(lowerboundary=Lower_fence,upperboundary=Upper_fence))


# For Rainfall, the minimum and maximum values are 0.0 and 371.0 So, the outliers are values>3.2

# In[58]:


#find outliers for Evaporation variable

IQR= df.Evaporation.quantile(0.75)-df.Evaporation.quantile(0.25)
Lower_fence=df.Evaporation.quantile(0.25)-(IQR*3)
Upper_fence=df.Evaporation.quantile(0.75)+(IQR*3)
print('Evaporation outliers are values< {lowerboundary} or > {upperboundary}\n'
     .format(lowerboundary=Lower_fence,upperboundary=Upper_fence))


# For Evaporation, the minimum and maximum vaues are 0.0 and 145.0 So, the outliers are values>21.8

# In[59]:


#find outliers for WindSpeed9am variable


IQR= df.WindSpeed9am.quantile(0.75)-df.WindSpeed9am.quantile(0.25)
Lower_fence=df.WindSpeed9am.quantile(0.25)-(IQR*3)
Upper_fence=df.WindSpeed9am.quantile(0.75)+(IQR*3)
print('WindSpeed9am outliers are values< {lowerboundary} or > {upperboundary}\n'
     .format(lowerboundary=Lower_fence,upperboundary=Upper_fence))


# for WindSpede9am, the minimum and maximum values are 0.0 and 87.0, So, the outliers are values>57.0

# #### Delcare Feature vector and target variable

# In[60]:


X=df.drop(['RainTomorrow'],axis=1)

y=df['RainTomorrow']


# #### Split data into separate training and test set

# In[61]:


#split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=0)


# In[62]:


#check the shape of X_train and X_test

X_train.shape,X_test.shape


# #### Feature Engineering

# In[63]:


#check data types in X_train

X_train.dtypes


# In[64]:


#display categorical variables

categorical=[col for col in X_train.columns if X_train[col].dtypes=='O']
categorical


# In[65]:


#display numerical variables

numerical=[col for col in X_train.columns if X_train[col].dtypes!='O']
numerical


# #### Engineering missing values in numerical variables

# In[66]:


#check missing values in numerical variables in X_train

X_train[numerical].isnull().sum()


# In[67]:


#check missibg values in numerical variables in X_test

X_test[numerical].isnull().sum()


# In[68]:


#print pecentage of missing values in the numerical variables in training set

for col in numerical:
    if X_train[col].isnull().mean()>0:
        print(col,round(X_train[col].isnull().mean(),4))


# ### Assumption
# 
# I assume that the data are missing completely at random (MCAR).There are two methods which can be used to impute missing values. One is mean or median imputation and other one is random sample imputation. When there are outliers in the dataset, we should use median imputation. So, I will use median imputation because median imputation is robust to outliers.
# 
# 
# I will impute missing values with the appropriate statistical measures of the data, in this case median. Imputation should be done over the training set, and then propagated to the test set. It means that the statistical measures to be used to fill missing values both in train and test set, should be extracted from the train set only.This is to avoid overfitting.

# In[69]:


#impute missing values in X_train and X_test with respective column median in 
#X_train

for df1 in [X_train,X_test]:
    for col in numerical:
        col_median=X_train[col].median()
        df1[col].fillna(col_median,inplace=True)


# In[70]:


#check again missing values in numerical variables in X_train

X_train[numerical].isnull().sum()


# In[71]:


#check mising values in numerical variables in X_test
X_test[numerical].isnull().sum()


# Now we can see that there are no missing values in the numerical columns of training and test set.

# #### Engineering missing values in categorical variables

# In[72]:


#print pecentage of missing vlaues in the categorical variables in training set

X_train[categorical].isnull().mean()


# In[73]:


#print categorical variables with missing data

for col in categorical:
    if X_train[col].isnull().mean()>0:
        print(col,(X_train[col].isnull().mean()))


# In[74]:


#impute missing categorical variables with most frequent values

for df2 in [X_train,X_test]:
    df2['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0],inplace=True)
    df2['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0],inplace=True)
    df2['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0],inplace=True)
    df2['RainToday'].fillna(X_train['RainToday'].mode()[0],inplace=True)


# In[75]:


#check missing values in categorical variables in X_train

X_train[categorical].isnull().sum()


# In[76]:


#Check missing values in categorical variables in X_test

X_test[categorical].isnull().sum()


# As a final check, I will check for missing values in X_train and X_test

# In[77]:


#check missing values in X_train

X_train.isnull().sum()


# In[78]:


#check missing values in X_test

X_test.isnull().sum()


# We can see that there are no missing values in X_train and X_test

# #### Engineering outliers in numerical variables
# 
# We have seen that the Rainfall, Evaporation, WindSpeed9am and WindSpeed3pm columns contain outliers. I will use top-coding approach to cap maximum values and remove outliers from the above variables.

# In[79]:


def max_value(df3,variable,top):
    return np.where(df3[variable]>top,top,df3[variable])

for df3 in [X_train,X_test]:
    df3['Rainfall']=max_value(df3, 'Rainfall',3.2)
    df3['Evaporation']=max_value(df3, 'Evaporation',21.8)
    df3['WindSpeed9am']=max_value(df3, 'WindSpeed9am',55)
    df3['WindSpeed3pm']=max_value(df3, 'WindSpeed3pm',57)


# In[80]:


X_train.Rainfall.max(), X_test.Rainfall.max()


# In[81]:


X_train.Evaporation.max(), X_test.Evaporation.max()


# In[82]:


X_train.WindSpeed9am.max(), X_test.WindSpeed9am.max()


# In[83]:


X_train.WindSpeed3pm.max(), X_test.WindSpeed3pm.max()


# In[84]:


X_train[numerical].describe()


# #### Encode categorical variables

# In[85]:


categorical


# In[86]:


X_train[categorical].head()


# In[87]:


#encode RainToday variable

import category_encoders as ce

encoder=ce.BinaryEncoder(cols=["RainToday"])

X_train=encoder.fit_transform(X_train)

X_test=encoder.transform(X_test)


# In[88]:


X_train.head()


# We can see that two additional variables RainToday_0 and RainToday_1 are created from RainToday variable.
# 
# Now, I will create the X_train training set.

# In[89]:


X_train=pd.concat([X_train[numerical],X_train[['RainToday_0','RainToday_1']],
                   
                   pd.get_dummies(X_train.Location),
                   pd.get_dummies(X_train.WindGustDir),
                   pd.get_dummies(X_train.WindDir9am),
                   pd.get_dummies(X_train.WindDir3pm)],axis=1)


# In[90]:


X_train.head()


# similarly, I will create the X_test testing set.

# In[91]:


X_test=pd.concat([X_test[numerical],X_test[['RainToday_0','RainToday_1']],
                   
                   pd.get_dummies(X_test.Location),
                   pd.get_dummies(X_test.WindGustDir),
                   pd.get_dummies(X_test.WindDir9am),
                   pd.get_dummies(X_test.WindDir3pm)],axis=1)


# In[92]:


X_test.head()


# We now have training and testing set ready for model building. Before that, we should map all the feature variables onto the same scale. It is called feature scaling. I will do it as follows.

# #### 11. Feature Scaling

# In[93]:


X_train.describe()


# In[94]:


cols=X_train.columns


# In[95]:


from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

X_train=scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)


# In[96]:


X_train=pd.DataFrame(X_train,columns=[cols])


# In[97]:


X_test=pd.DataFrame(X_test,columns=[cols])


# In[98]:


X_train.describe()


# We now have X_train dataset ready to be fed into the Logistic Regression classifier. I will do it as follows.

# #### 12. Model Training

# In[99]:


#train a logistic regression model on the traing set
from sklearn.linear_model import LogisticRegression

#instantiate the model
logreg=LogisticRegression(solver='liblinear',random_state=0)

#fit the model
logreg.fit(X_train,y_train)


# #### 13. Predict results

# In[100]:


y_pred_test=logreg.predict(X_test)

y_pred_test


# #### predict_proba method
# #### predict_proba method
# gives the probabilities for the target variable (0 and 1) in this case, in array form.
# 0 is for probability of no rain and 1 is for prabability of rain.

# In[101]:


#probability of getting output as 0-no rain
logreg.predict_proba(X_test)[:,0]


# In[102]:


#probability of getting output as 1-rain

logreg.predict_proba(X_test)[:,1]


# #### 14. Check accuracy score

# In[103]:


from sklearn.metrics import accuracy_score
print('Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred_test)))


# #### Check for overfitting and underfitting

# In[104]:


#print the scores on training and test set
print('Training set score: {:.4f}'.format(logreg.score(X_train,y_train)))

print('Test set score: {:.4}'.format(logreg.score(X_test,y_test)))


# The training-set accuracy score is 0.8476 while the test-set accuracy to be 0.8501. These two values are quite comparable. So, there is no question of overfitting.
# 
# In Logistic Regression, we use default value of C = 1. It provides good performance with approximately 85% accuracy on both the training and the test set. But the model performance on both the training and test set are very comparable. It is likely the case of underfitting.
# 
# I will increase C and fit a more flexible model.

# In[105]:


#fit the Logistics Regression model with C=100

#instantiate the model
logreg100=LogisticRegression(C=100,solver='liblinear',random_state=0)

#fit the model
logreg100.fit(X_train,y_train)


# In[106]:


#print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg100.score(X_train,y_train)))

print('Test set score: {:.4f}'.format(logreg100.score(X_test,y_test)))


# We can see that, C=100 results in higher test set accuracy and also a slightly increased training set accuracy. So, we can conclude that a more complex model should perform better.
# 
# Now, I will investigate, what happens if we use more regularized model than the default value of C=1, by setting C=0.01.

# In[107]:


#fir the logistic regression mmoel with C=001

#instantiate the model
logreg001=LogisticRegression(C=0.01,solver='liblinear',random_state=0)

#fit the model
logreg001.fit(X_train,y_train)


# In[108]:


#print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg001.score(X_train,y_train)))

print('Test set score: {:.4f}'.format(logreg001.score(X_test,y_test)))


# So, if we use more regularized model by setting C=0.01, then both the training and test set accuracy decrease relatiev to the default parameters.

# #### Compare model accuracy with null accuracy
# 
# So, the model accuracy is 0.8501. But, we cannot say that our model is very good based on the above accuracy. We must compare it with the null accuracy. Null accuracy is the accuracy that could be achieved by always predicting the most frequent class.
# 
# So, we should first check the class distribution in the test set.

# In[109]:


#check class distribution in test set

y_test.value_counts()


# We can see that the occurences of most frequent class is 22067. So, we can calculate null accuracy by dividing 22067 by total number of occurences.
# 

# In[110]:


#check null accuracy score

null_accuracy=(22067/(22067+6372))

print('Null accuracy score: {0:0.4f}'.format(null_accuracy))


# We can see that our model accuracy score is 0.8501 but null accuracy score is 0.7759. So, we can conclude that our Logistic Regression model is doing a very good job in predicting the class labels.
# 
# Now, based on the above analysis we can conclude that our classification model accuracy is very good. Our model is doing a very good job in terms of predicting the class labels.
# 
# But, it does not give the underlying distribution of values. Also, it does not tell anything about the type of errors our classifer is making.
# 
# We have another tool called Confusion matrix that comes to our rescue

# #### 15. Confusion matrix
# 
# A confusion matrix is a tool for summarizing the performance of a classification algorithm. A confusion matrix will give us a clear picture of classification model performance and the types of errors produced by the model. It gives us a summary of correct and incorrect predictions broken down by each category. The summary is represented in a tabular form.
# 
# Four types of outcomes are possible while evaluating a classification model performance. These four outcomes are described below:-
# 
# True Positives (TP) – True Positives occur when we predict an observation belongs to a certain class and the observation actually belongs to that class.
# 
# True Negatives (TN) – True Negatives occur when we predict an observation does not belong to a certain class and the observation actually does not belong to that class.
# 
# False Positives (FP) – False Positives occur when we predict an observation belongs to a certain class but the observation actually does not belong to that class. This type of error is called Type I error.
# 
# False Negatives (FN) – False Negatives occur when we predict an observation does not belong to a certain class but the observation actually belongs to that class. This is a very serious error and it is called Type II error.
# 
# These four outcomes are summarized in a confusion matrix given below

# In[111]:


#print the confusion matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred_test)

print('Confusion matrix\n\n',cm)

print('\nTrue Positives(TP)=',cm[0,0])

print('\nTrue Negatives(TN)=',cm[1,1])

print('\nFalse Positives(FP)=',cm[0,1])

print('\nFalse Negatives(FN)=',cm[1,0])


# The confusion matrix shows 20892 + 3285 = 24177 correct predictions and 3087 + 1175 = 4262 incorrect predictions.
# 
# In this case, we have
# 
# True Positives (Actual Positive:1 and Predict Positive:1) - 20892
# True Negatives (Actual Negative:0 and Predict Negative:0) - 3285
# False Positives (Actual Negative:0 but Predict Positive:1) - 1175 (Type I error)
# False Negatives (Actual Positive:1 but Predict Negative:0) - 3087 (Type II error)

# In[112]:


#visualize confusion matrix with seaborn heatmap

cm_matrix=pd.DataFrame(data=cm,columns=['Actual Positive:1','Actual Negative:0'],
                      index=['Predict Positive:1','Predict Negative:0'])
sns.heatmap(cm_matrix,annot=True,fmt='d',cmap='YlGnBu')


# #### 16. Classification metrics

# Classification Report
# Classification report is another way to evaluate the classification model performance. It displays the precision, recall, f1 and support scores for the model. I have described these terms in later.
# 
# We can print a classification report as follows:-

# In[113]:


from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred_test))


# #### classification accuracy

# In[114]:


TP=cm[0,0]
TN=cm[1,1]
FP=cm[0,1]
FN=cm[1,0]


# In[115]:


#print classification accuracy

classification_accuracy=(TP+TN)/float(TP+TN+FP+FN)

print('Classification accuracy: {0:0.4f}'.format(classification_accuracy))


# #### Classification error

# In[116]:


#print classification error

classification_error=(FP+FN)/float(TP+TN+FP+FN)

print('Classification error: {0:0.4f}'.format(classification_error))


# #### Precision
# 
# Precision can be defined as the percentage of correctly predicted positive outcomes out of all the predicted positive outcomes. It can be given as the ratio of true positives (TP) to the sum of true and false positives (TP + FP).
# 
# So, Precision identifies the proportion of correctly predicted positive outcome. It is more concerned with the positive class than the negative class.
# 
# Mathematically, precision can be defined as the ratio of TP to (TP + FP).

# In[117]:


#print precision score

precision=TP/float(TP+FP)

print('Precision: {0:0.4}f'.format(precision))


# #### Recall
# Recall can be defined as the percentage of correctly predicted positive outcomes out of all the actual positive outcomes. It can be given as the ratio of true positives (TP) to the sum of true positives and false negatives (TP + FN). Recall is also called Sensitivity.
# 
# Recall identifies the proportion of correctly predicted actual positives.
# 
# Mathematically, recall can be given as the ratio of TP to (TP + FN).

# In[118]:


recall=TP/float(TP+FN)

print('Recall or Sensitivity:{0:0.4f}'.format(recall))


# #### True Positive Rate
# 
# synonymous with Recall

# In[119]:


true_positive_rate=TP/float(TP+FN)

print('True Positive Rate; {0:0.4f}'.format(true_positive_rate))


# #### Specificity

# In[120]:


specificity=TN/(TN+FP)

print("Specificity:{0:0.4f}".format(specificity))


# #### F1-score
# 
# f1-score is the weighted harmonic mean of precision and recall. The best possible f1-score would be 1.0 and the worst would be 0.0. f1-score is the harmonic mean of precision and recall. So, f1-score is always lower than accuracy measures as they embed precision and recall into their computation. The weighted average of f1-score should be used to compare classifier models, not global accuracy.

# #### Support
# 
# Support is the actual number of occurrences of the class in our dataset.

# ### 17. Adjusting the threshold level

# In[121]:


#print the first 10 predicted probabilities of two classes 
#s-0 and 1
y_pred_prob=logreg.predict_proba(X_test)[0:10]

y_pred_prob


# Observations
# In each row, the numbers sum to 1.
# There are 2 columns which correspond to 2 classes - 0 and 1.
# 
# Class 0 - predicted probability that there is no rain tomorrow.
# 
# Class 1 - predicted probability that there is rain tomorrow.
# 
# Importance of predicted probabilities
# 
# We can rank the observations by probability of rain or no rain.
# predict_proba process
# 
# Predicts the probabilities
# 
# Choose the class with the highest probability
# 
# Classification threshold level
# 
# There is a classification threshold level of 0.5.
# 
# Class 1 - probability of rain is predicted if probability > 0.5.
# 
# Class 0 - probability of no rain is predicted if probability < 0.5.

# In[122]:


#store the probabilities in dataframe

y_pred_prob_df=pd.DataFrame(data=y_pred_prob,columns=['Prob\n'
'of-No rain tomorrow (0)','Prob of= Rain tomorrow (1)'])

y_pred_prob_df


# In[123]:


#print the first 10 predicted probabilities for class 1
#probability of rain

logreg.predict_proba(X_test)[0:10,1]


# In[124]:


#store the predicated probabilities for class 1-
#probability of rain

y_pred1=logreg.predict_proba(X_test)[:,1]


# In[125]:


#plot histogram of predicted probabilities


#adjust the font size 
plt.rcParams['font.size'] = 12


#plot histogram with 10 bins
plt.hist(y_pred1, bins = 10)


#set the title of predicted probabilities
plt.title('Histogram of predicted probabilities of rain')


#set the x-axis limit
plt.xlim(0,1)


# set the title
plt.xlabel('Predicted probabilities of rain')
plt.ylabel('Frequency')


# Observations
# We can see that the above histogram is highly positive skewed.
# The first column tell us that there are approximately 15000 observations with probability between 0.0 and 0.1.
# There are small number of observations with probability > 0.5.
# So, these small number of observations predict that there will be rain tomorrow.
# Majority of observations predict that there will be no rain tomorrow.

# #### Lower the threshold

# In[126]:


from sklearn.preprocessing import binarize

for i in range(1,5):
    cm1=0
    y_pred1 = logreg.predict_proba(X_test)[:,1]
    y_pred1 = y_pred1.reshape(-1,1)
    y_pred2 = binarize(y_pred1, i/10)
    y_pred2 = np.where(y_pred2 == 1, 'Yes', 'No')
    cm1 = confusion_matrix(y_test, y_pred2)
    print ('With',i/10,'threshold the Confusion Matrix is ','\n\n',cm1,'\n\n',
           
            'with',cm1[0,0]+cm1[1,1],'correct predictions, ', '\n\n', 
           
            cm1[0,1],'Type I errors( False Positives), ','\n\n',
           
            cm1[1,0],'Type II errors( False Negatives), ','\n\n',
           
           'Accuracy score: ', (accuracy_score(y_test, y_pred2)), '\n\n',
           
           'Sensitivity: ',cm1[1,1]/(float(cm1[1,1]+cm1[1,0])), '\n\n',
           
           'Specificity: ',cm1[0,0]/(float(cm1[0,0]+cm1[0,1])),'\n\n',
          
            '====================================================', '\n\n')


# Comments
# In binary problems, the threshold of 0.5 is used by default to convert predicted probabilities into class predictions.
# Threshold can be adjusted to increase sensitivity or specificity.
# Sensitivity and specificity have an inverse relationship. Increasing one would always decrease the other and vice versa.
# We can see that increasing the threshold level results in increased accuracy.
# Adjusting the threshold level should be one of the last step you do in the model-building process.

# ##### 18. ROC - AUC

# ROC Curve
# Another tool to measure the classification model performance visually is ROC Curve. ROC Curve stands for Receiver Operating Characteristic Curve. An ROC Curve is a plot which shows the performance of a classification model at various classification threshold levels.
# 
# The ROC Curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold levels.
# 
# True Positive Rate (TPR) is also called Recall. It is defined as the ratio of TP to (TP + FN).
# 
# False Positive Rate (FPR) is defined as the ratio of FP to (FP + TN).
# 
# In the ROC Curve, we will focus on the TPR (True Positive Rate) and FPR (False Positive Rate) of a single point. This will give us the general performance of the ROC curve which consists of the TPR and FPR at various threshold levels. So, an ROC Curve plots TPR vs FPR at different classification threshold levels. If we lower the threshold levels, it may result in more items being classified as positve. It will increase both True Positives (TP) and False Positives (FP).

# In[127]:


#plot ROC Curve

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred1, pos_label = 'Yes')
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0,1], [0,1], 'k--' )
plt.rcParams['font.size'] = 12
plt.title('ROC curve for RainTomorrow classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.show()


# ROC curve help us to choose a threshold level that balances sensitivity and specificity for a particular context.
# 
# ROC-AUC
# ROC AUC stands for Receiver Operating Characteristic - Area Under Curve. It is a technique to compare classifier performance. In this technique, we measure the area under the curve (AUC). A perfect classifier will have a ROC AUC equal to 1, whereas a purely random classifier will have a ROC AUC equal to 0.5.
# 
# So, ROC AUC is the percentage of the ROC plot that is underneath the curve.

# In[128]:


#compute ROC AUC

from sklearn.metrics import roc_auc_score
ROC_AUC = roc_auc_score(y_test, y_pred1)
print('ROC AUC : {:.4f}'.format(ROC_AUC))


# Comments
# ROC AUC is a single number summary of classifier performance. The higher the value, the better the classifier.
# 
# ROC AUC of our model approaches towards 1. So, we can conclude that our classifier does a good job in predicting whether it will rain tomorrow or not.

# In[129]:


#calculate cross validated ROC AUC 

from sklearn.model_selection import cross_val_score
Cross_validated_ROC_AUC = cross_val_score(logreg, X_train, y_train, cv=5, scoring='roc_auc').mean()
print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))


# #### 19. k fold Cross Validation

# In[130]:


#applying 5 fold cross validation

from sklearn.model_selection import cross_val_score
scores = cross_val_score(logreg, X_train, y_train, cv = 5, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))


# We can summarize the cross-validation accuracy by calculating its mean

# #### 20. Hyperparameter Optimization using GridSearch CV

# In[ ]:


from sklearn.model_selection import GridSearchCV

parameters=[{'penalty':['l1','l2']},{'C':[1,10,100,1000]}]
grid_search=GridSearchCV(estimator=logreg,param_grid=parameters,scoring='accuracy',cv=5, verbose=0)
grid_search.fit(X_train,y_train)


# In[ ]:


# examine the best model

# best score achieved during the GridSearchCV
print('GridSearch CV best score : {:.4f}\n\n'.format(grid_search.best_score_))

# print parameters that give the best results
print('Parameters that give the best results :','\n\n', (grid_search.best_params_))

# print estimator that was chosen by the GridSearch
print('\n\nEstimator that was chosen by the search :','\n\n', (grid_search.best_estimator_))


# In[ ]:


# calculate GridSearch CV score on test set

print('GridSearch CV score on test set: {0:0.4f}'.format(grid_search.score(X_test, y_test)))


# Comments
# Our original model test accuracy is 0.8501 while GridSearch CV accuracy is 0.8507.
# We can see that GridSearch CV improve the performance for this particular model.

# #### 21. Results and conclusion
# 
# The logistic regression model accuracy score is 0.8501. So, the model does a very good job in predicting whether or not it will rain tomorrow in Australia.
# 
# Small number of observations predict that there will be rain tomorrow. Majority of observations predict that there will be no rain tomorrow.
# 
# The model shows no signs of overfitting.
# 
# Increasing the value of C results in higher test set accuracy and also a slightly increased training set accuracy. So, we can conclude that a more complex model should perform better.
# 
# Increasing the threshold level results in increased accuracy.
# 
# ROC AUC of our model approaches towards 1. So, we can conclude that our classifier does a good job in predicting whether it will rain tomorrow or not.
# 
# Our original model accuracy score is 0.8501 whereas accuracy score after RFECV is 0.8500. So, we can obtain approximately similar accuracy but with reduced set of features.
# 
# In the original model, we have FP = 1175 whereas FP1 = 1174. So, we get approximately same number of false positives. Also, FN = 3087 whereas FN1 = 3091. So, we get slighly higher false negatives.
# 
# Our, original model score is found to be 0.8476. The average cross-validation score is 0.8474. So, we can conclude that cross-validation does not result in performance improvement.
# 
# Our original model test accuracy is 0.8501 while GridSearch CV accuracy is 0.8507. We can see that GridSearch CV improve the performance for this particular model.

# #### 22. References
# 
# The work done in this project is inspired from following books and websites:-
# 
# Hands on Machine Learning with Scikit-Learn and Tensorflow by Aurélién Géron
# 
# Introduction to Machine Learning with Python by Andreas C. Müller and Sarah Guido
# 
# Udemy course – Machine Learning – A Z by Kirill Eremenko and Hadelin de Ponteves
# 
# Udemy course – Feature Engineering for Machine Learning by Soledad Galli
# 
# Udemy course – Feature Selection for Machine Learning by Soledad Galli
# 
# https://en.wikipedia.org/wiki/Logistic_regression
# 
# https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html
# 
# https://en.wikipedia.org/wiki/Sigmoid_function
# 
# https://www.statisticssolutions.com/assumptions-of-logistic-regression/
# 
# https://www.kaggle.com/mnassrib/titanic-logistic-regression-with-python
# 
# https://www.kaggle.com/neisha/heart-disease-prediction-using-logistic-regression
# 
# https://www.ritchieng.com/machine-learning-evaluate-classification-model/
