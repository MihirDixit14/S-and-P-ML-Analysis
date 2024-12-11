# %% [markdown]
# <!-- #Here is my new ML Project based on Stock price of S and P.
# #Different ML models will be used for data analysis and visualization.
# #Programming Language: Python
# #IDE: VSCode
# #Dataset: Sand P500.csv (Kaggle.com)
#  -->
# 

# %% [markdown]
# <!-- 1. Understanding the data set and cleaning the dataset -->

# %%
#Import all the important libraries
import pandas as pd
import numpy as np
from sklearn import linear_model
from scipy import stats
#Check the rows, columns, datatypes of the dataset

df=pd.read_csv(".\sp500_companies.csv")

#Here we can get the general view of the dataset
print(df.head())
print(df.info())
print(df.columns)
#check the column name data types. Here is an example of Current price
print(df.Currentprice.dtype)

print(df.Currentprice.duplicated())


# %% [markdown]
# Basic Operations performed: 
# Calculate the important values of Mean, median, Mode
# 1. Mean- The average Value of the current price, 
# 2. Median- The midpoint value of the current price, 
# 3. Mode: The most common value of the current price  
# 4. Standard Deviation: A number that describes how spread out the values of the current price.

# %% [markdown]
# <!-- # Calculate the important values of Mean, median, Mode
# 1. Mean- The average Value, 
# 2. Median- The midpoint value, 
# 3. Mode: The most common value  
# 4. Standard Deviation: A number that describes how spread out the values.-->

# %%
#Mean of the current price in the S and P dataset

Mean=df['Currentprice'].mean()
#The average of the S and P 500 share price
print('Mean=',Mean)

# The median of the S and P 500 share price
Median=df['Currentprice'].median()
print('Median=',Median)

# The Mode of the S and P 500 share price
Mode=df['Currentprice'].mode()
print('Mode=',Mode)

#The standard deviation of the Current price of the S and P dataset
Std=df['Currentprice'].std()
print('Standard Deviation=',Std)

# %% [markdown]
# Machine Learning Models
# Linear regression. Lets check if there is a positive relation between Current price and Marketcap.

# %% [markdown]
# <!-- Linear regression of the  S and P dataset 
# In order to test out linear regression lets check slope, 
# -->

# %%
current_price=df['Currentprice']
MarketCap=df['Marketcap']
slope,intercept, r, p, std_err=stats.linregress(current_price,MarketCap)
print('Slope=',slope)
print('intercept=',intercept)
print('r=',r)






