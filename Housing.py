#!/usr/bin/env python
# coding: utf-8

# Surprise Housing - Advanced Regression

# Let's first import numpy and pandas

# In[ ]:


#Importing basic packages
import numpy as np
import pandas as pd
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',100)
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#Importing packages for regression
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split,GridSearchCV,KFold,cross_val_score
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score


# In[ ]:


#Supressing warnings
import warnings
warnings.filterwarnings('ignore')


# Let's define all custom methods here for code conciseness.

# In[ ]:


# Let's define a method for checking missing value count per column.
def missingValueCount(data_frame, threshold):
    missing_value = data_frame.isnull().sum()
    return missing_value.loc[missing_value > threshold]


# In[45]:


# Let's define a method for checking missing value percentage per column.
def missingValuePercentage(data_frame, threshold):
    missing_value = round(df.isnull().sum()/len(df.index),2).sort_values(ascending=False)
    return missing_value.loc[missing_value > threshold]


# In[58]:


#Defining method to calculate age from year
def yearToAge(data_frame,col):
    new_cal = col + '_Old'
    data_frame[new_cal] = data_frame[col].max()-df[col]


# In[15]:


#Defining method for imputing missing values 
def imputingMissingValue(data_frame,col,value):
    data_frame[col].fillna(value, inplace = True)


# In[ ]:





# Custom methods for Plotting and Visualization

# In[20]:


# Defining method for plotting graphs based on numerical/continuous variables
def numericColAnalysis(data_frame, index,independent_variable ,target_variable):
    plt.figure(figsize = (20, 26))
    plt.figure(index)
    sns.jointplot(x = independent_variable, y = target_variable, data = data_frame[[independent_variable,target_variable]])


# Step 1

# In[ ]:


# reading data set
df = pd.read_csv(r"C:\Users\Nirali\Downloads\train.csv")
df.head()


# In[31]:


#Firstly, let's have a look at the dimension of data
df.shape


# In[33]:


#Let's have look on the metadata of the dataset
df.info()


# In[35]:


#Let's have a look at the first few rows of the Data
df.head()


# In[38]:


#Let's have a look at all the column names
df.columns


# In[40]:


#Let's have a look at statistic part of data
df.describe([0.25,0.50,0.75,0.99])


# In[41]:


#Let's check number of missing value per column.
missingValueCount(df, 0)


# In[47]:


#Let's check percentage of missing value per column.
missingValuePercentage(df, 0)


# In[49]:


missing_data = missingValuePercentage(df, 0.10)
missing_data


# In[ ]:





# In[53]:


#Checking the columns where the missing values less than or equal to 10%
missingValuePercentage(df, 0)


# In[55]:


#Let's have a look on first few data after droping missing values
df.head()


# Before proceeding ahaed, we will try to convert the Year columns into the age where we are going to fill these columns with number, And max year for all these columns come out to be 2010. For example, suppose the YrSold=2000 , Then YrSold_Old = 2010-2000 = 10

# In[61]:


#Converting the year column into age
yearToAge(df,'YearBuilt')
yearToAge(df,'YearRemodAdd')
yearToAge(df,'GarageYrBlt')
yearToAge(df,'YrSold')


# In[63]:


#Let's have a look on data after converting into age
df[['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold','YearBuilt_Old','YearRemodAdd_Old',
             'GarageYrBlt_Old','YrSold_Old']].head()


# In[70]:


imputingMissingValue(df,'MasVnrType', 'None')
imputingMissingValue(df,'MasVnrArea' ,df.MasVnrArea.mean())
imputingMissingValue(df,'BsmtQual', 'TA')
imputingMissingValue(df,'BsmtCond', 'TA')
imputingMissingValue(df,'BsmtExposure', 'No')
imputingMissingValue(df,'BsmtFinType1', 'Unf')
imputingMissingValue(df,'BsmtFinType2', 'Unf')
imputingMissingValue(df,'GarageType', 'Attchd')
imputingMissingValue(df,'GarageYrBlt_Old', -1)
imputingMissingValue(df,'GarageFinish', 'Unf')
imputingMissingValue(df,'GarageQual', 'TA')
imputingMissingValue(df,'GarageCond', 'TA')


# In[71]:


df['Utilities'].value_counts()


# In[73]:


df['Street'].value_counts()


# In[82]:


#Checking the columns where the missing values
missingValuePercentage(df, 0)


# Lets plot some graph for the EDA purpose

# In[84]:


#Get all numerical Columns
numerical_columns = df.select_dtypes(include = np.number).columns.tolist()
numerical_columns.remove('SalePrice')


# In[105]:


#Get all categorical Columns
categorical_columns = df.select_dtypes(include = np.object).columns.tolist()
for col in categorical_columns:
    print(col,' : ',df[col].head().unique())


# In[106]:


#Ploting the graph for all numerical variables
for index, col in enumerate(numerical_columns):
    numericColAnalysis(df, index, col, 'SalePrice')


# In[107]:


plt.figure(figsize=(16,16))
sns.heatmap(df[numerical_columns].corr(), annot = True)
plt.show()


# Step 2: Preparing tha Data for Modelling

# In[108]:


#Let's chcking data_frame shape
df.shape


# In[109]:


numerical_columns = ['LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','1stFlrSF','GrLivArea','OpenPorchSF',
           'EnclosedPorch','3SsnPorch',
           'ScreenPorch' ,'PoolArea','MiscVal','SalePrice']
house_price = dropOutliers(df, numerical_columns)


# In[111]:


#Let's have a look on first few columns
df[categorical_columns].head()


# In[113]:


ordinal_columns = ['LandSlope','ExterQual','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
            'HeatingQC','CentralAir',  'KitchenQual','GarageFinish','GarageQual','GarageCond',
             'ExterCond','LotShape']
df[ordinal_columns].head()


# In[115]:


# Let's convert categorical variable to numeric. 
#we convert categorical variables into factors i.e number because to make things easy.
df[ordinal_columns[0]] = df[ordinal_columns[0]].map({'Gtl':0,'Mod':1,'Sev':2})
df[ordinal_columns[1]] = df[ordinal_columns[1]].map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})
df[ordinal_columns[2]] = df[ordinal_columns[2]].map({'NA':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
df[ordinal_columns[3]] = df[ordinal_columns[3]].map({'NA':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
df[ordinal_columns[4]] = df[ordinal_columns[4]].map({'NA':0,'No':1,'Mn':2,'Av':3,'Gd':4})
df[ordinal_columns[5]] = df[ordinal_columns[5]].map({'NA':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6})
df[ordinal_columns[6]] = df[ordinal_columns[6]].map({'NA':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6})
df[ordinal_columns[7]] = df[ordinal_columns[7]].map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})
df[ordinal_columns[8]] = df[ordinal_columns[8]].map({'N':0,'Y':1})
df[ordinal_columns[9]] = df[ordinal_columns[9]].map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})
df[ordinal_columns[10]] = df[ordinal_columns[10]].map({'NA':0,'Unf':1,'RFn':2,'Fin':3})
df[ordinal_columns[11]] = df[ordinal_columns[11]].map({'NA':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
df[ordinal_columns[12]] = df[ordinal_columns[12]].map({'NA':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
df[ordinal_columns[13]] = df[ordinal_columns[13]].map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})
df[ordinal_columns[14]] = df[ordinal_columns[14]].map({'IR1':0,'IR2':1,'IR3':2,'Reg':3})


# In[117]:


df[ordinal_columns].head()


# In[132]:


dummy_col = pd.get_dummies(df[['MSZoning','LandContour','LotConfig','Neighborhood','Condition1','Condition2','BldgType',
             'HouseStyle','RoofStyle','RoofMatl','Exterior1st',  'Exterior2nd','MasVnrType','Foundation',
             'Heating','Electrical','Functional','GarageType','PavedDrive','SaleType','SaleCondition']],
                           drop_first=True)

house_price = pd.concat([df,dummy_col],axis='columns')

house_price = house_price.drop(['MSZoning','LandContour','LotConfig','Neighborhood','Condition1','Condition2','BldgType',
             'HouseStyle','RoofStyle','RoofMatl','Exterior1st',  'Exterior2nd','MasVnrType','Foundation',
             'Heating','Electrical','Functional','GarageType','PavedDrive','SaleType','SaleCondition'],axis='columns')


# In[133]:


# Plotting the graph
plt.figure(figsize=(16,6))
sns.distplot(df.SalePrice)
plt.show()


# In[134]:


df_train,df_test = train_test_split(house_price,train_size=0.7,test_size=0.3,random_state=50)


# In[135]:


numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
scaler = StandardScaler()
df_train[numerical_columns] = scaler.fit_transform(df_train[numerical_columns])
df_test[numerical_columns] = scaler.transform(df_test[numerical_columns])


# In[136]:


plt.figure(figsize=(16,6))
plt.subplot(121)
sns.distplot(df_train.SalePrice)
plt.subplot(122)
sns.distplot(df_test.SalePrice)


# Step- 3: Training the Model

# In[141]:


y_train = df.pop('SalePrice')
X_train = df


# Trying the model of RFE and LASSO,
# However they are not runing currently.
# 

# Conclusion
# These variables are significant in predicting the price of a house :
# 
# BsmtFinSF2 : Type 2 finished square feet.
# LotShape : General shape of property.
# ExterCond : Evaluates the present condition of the material on the exterior.
# GarageCars : Size of garage in car capacity.
# Neighborhood_Gilbert : Physical locations within Ames city limits(Gilbert).
# BsmtFinSF1 : Type 1 finished square feet.
# OverallQual : Rates the overall material and finish of the house.
# BsmtExposure : Refers to walkout or garden level walls.
# CentralAir : Central air conditioning.
# OverallCond : Rates the overall condition of the house.

# The optimal value of lambda for ridge and lasso regression are :
# 
# Best alpha value for Lasso : 0.001
# 
# Best alpha value for Ridge : 20.0
