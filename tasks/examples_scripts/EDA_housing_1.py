#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Amad Diouf, amaddioufb13@gmail.com
# Created Date: 05/05/2022
# version ='1.0'
# ---------------------------------------------------------------------------
""" script 1 to understand how EDA is done using the housig dataset"""
# ---------------------------------------------------------------------------
# Imports
import pandas as pd
import matplotlib # change the backend used by maplotlib from the default 'QtAgg' to one that enables interactive plots 'Qt5Agg'
matplotlib.use('Qt5Agg')
matplotlib.get_backend()
import matplotlib.pyplot as plt
import seaborn as sns
# for sns plots, add plt.show() to make the plots appear (replace it with the "%matplotlib inline" in the case of a notebook to let the backend solve that issue while displaying the plots inside the notebook)
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
# ---------------------------------------------------------------------------
# IMPORTING OUR DATA (end of EDA's step 1 - data collection)
df_train = pd.read_csv('/home/amad/PALADIN_2/3CEREBRO/garage/projects/EDA_ledger/Simplilearn_1h19m42s_V1/datasets/train.csv')
df_train
# we have a [1460 rows x 81 columns] dataframe
# some cols are mostly missing values as NAN so we might as well delete them later
# ---------------------------------------------------------------------------
# DATA CLEANING (EDA's step 2 - data cleaning)
# ---> missing data :
# from the df, we make df of wether true (1) or false(0) that the value in the cell is missing, then we get a table "col 1 : colname, col 2: num missing values in that col", then we sort by descending values the table
total = df_train.isnull().sum().sort_values(ascending=False)
# to get % of missing values by cells in a col, we divide the table "col 1 : colname, col 2: num missing values in that col" by the table table "col 1 : colname, col 2: num values in that col wether missing value or not", then we sort in desc
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
# we concatenate the 2 previous table into a table "col 1 : colname, col 2: num missing values in that col, col 3: % missing values in that col"
missing_data = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
# lets visualize the first 20 lines of this table
missing_data.head(20)
# until the col FireplaceQu, all above cols have more than half of their values as missing data and the FireplaceQu has even almost half of its data as missing data
# ---------------------------------------------------------------------------
# ---> DEALING WITH MISSSING DATA :
# there are 2 things we can do : drop some values or fill in some values
# to decide : we will consider that when more than 15% of the values are missing, we will delete the variable and pretend it never existed and we will not try to do anything to fill in the missing data in this case
# NB1 : As a result in our example here, all cols until LotFrontage included at 17% missing values, will e deleted.
# A view from a point of view of understanding the problem would suggest that those deleted cols variables are strong candidates
# for outliers as not all houses will have a pool, alley, fence etc so it can disproportionately push up the price of a house
# NB2 : Some groupe of variables, that are recognizable as related by having the same prefix (e.g. garage, basement),
# have the same % of missing values hence must be about the same sphere of variables and so one variable can be chosen to be kept to express that variable
# (e.g. GarageQual for all things garage,BsmntCond for all things basement ) ad delete the others related variables of that sphere
# NB3 : some sphere of variables can come in 2 only variables with similar low number of variables (e.g. : MasVnrArea, MasVnrType). we can consider them as non essential (precise what it does mean in practice later) ##?

df_train_cleaned = df_train.drop((missing_data[missing_data['Total'] > 1]).index, 1) # drop all the cols with more than one missing value
df_train_cleaned = df_train_cleaned.drop(df_train_cleaned.loc[df_train_cleaned['Electrical'].isnull()].index) # drop the row where Electrical has a missing value
df_train_cleaned.isnull().sum().max() # to check that there is no mssing data left, we get the table of num mssing data for each col, we find the max value in it (should be 0 if no missing data left)
# lets take a look at our final dataset
df_train_cleaned # 20 cols and 1 row have been deleted
# ---------------------------------------------------------------------------
# ---> DESCRIPTIVE STATISTICS SUMMARY :
# for the sale price (dependent variable to predict here)
df_train_cleaned['SalePrice'].describe()
# """
# count      1459.000000
# mean     180930.394791
# std       79468.964025
# min       34900.000000
# 25%      129950.000000
# 50%      163000.000000
# 75%      214000.000000
# max      755000.000000
# Name: SalePrice, dtype: float64
# """
# Now we know how the price is distributed approximately and how it varies globally
# ---------------------------------------------------------------------------
# UNIVARIATE ANALYSIS
# we choose to loot at the outcome variable to predict (Sale Price)
# ---> histogram
sns.displot(df_train_cleaned['SalePrice'])
plt.show()
# - from this plots, we can tell tell a variety of things :
# we can tell its a normal distribution. however it has a slight deviation from the normal distribution, it has positive skewness (ie skewed towards the left side),
# it also has a peak so it shows peakness
# ---> skewness and kurtosis
print("Skewess: %f" % df_train_cleaned['SalePrice'].skew())
print("Skewess: %f" % df_train_cleaned['SalePrice'].kurt())
# Remarque 1 : the shape and high degree of skewness shows that it has a bunch of outliers present in it
# Need : we have to establish a threshold that defines an observation as an outlier.
# To do this, we'll standardize the data and this means transforming the data to have a mean of 0 and a std dev of 1.
# ---> standardizing data
saleprice_scaled = StandardScaler().fit_transform(df_train_cleaned['SalePrice'][:,np.newaxis]) # np.newaxis change what was a 1D table into a matrix of cols (all lines kept in 1 col)
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10] # argsort() does a sorting and gives an array of the indices of the sorted array
high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print("outer range (low) of the distribution:")
print(low_range)
print("\nouter range (high) of the distribution:")
print(high_range)
# Remarque 1 : we can see low range values are not very from 0 but high range values are very far from zero
# (eg all of the 7 values are really out of range)
# This is just what looking at 1 variable told us. Now we can look at 2 variables and compare them.
# ---------------------------------------------------------------------------
# BIVARIATE ANALYSIS
# we use two variables and them. and that way, we can see how one feature affects the an other.
# it is done with :
# - scatter plots (plots indvidual data points)
# - or correlation matrices (plots the correlation in terms of hues) (also the higher the correlation value, the higher the correlation between 2 variables)
# - box-plots can also be used
# lets look at different methods to perform bivariate analysis...
# ---> bivariate analysis saleprice/grlivarea :
# - lets look at how the saleprice and the greater living area crrespond to each other
var = 'GrLivArea'
data = pd.concat([df_train_cleaned['SalePrice'], df_train_cleaned[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
# Remark 1 : there is a tendancy of saleprice going higher with the greater living area being larger...
# ...except for the 2 highest values that remains at lower prices : one thing that could explain it is there are not really houses but something like an agricultural area.
# In any case, thes two values seem like outliers thus we can delete them
# Remark 2 : The 2 following values must be the two 7... values seen prviously. They seems far from the cloud of points but they also seems to follow the trend so w will keep them
# Globally : after one bivariate analysis, we have found a couple of outliers and we can now delete them
# ---> deleting points
df_train_cleaned.sort_values(by='GrLivArea', ascending=False)[:2]
df_train_cleaned2 = df_train_cleaned
df_train_cleaned2 = df_train_cleaned2.drop(df_train_cleaned2[df_train_cleaned2["Id"] ==1299].index)
df_train_cleaned2 = df_train_cleaned2.drop(df_train_cleaned2[df_train_cleaned2["Id"] ==524].index)
# - lets look at how the total bsmnt area correspond to the saleprice
var = 'TotalBsmtSF'
data = pd.concat([df_train_cleaned2['SalePrice'], df_train_cleaned2[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
# Remark 1 : the general trend of higher the basement surface the higher the price is respected throughout the cloud of points
# 3 values on the far right seems to stagnate price wise but we will keep them as they are not far from the trend

# Remark 2 : Until now, we compared two variables that when both numerical. What to do if we want to compare categorical values with numerical ?
# Solution : we can plots this in the form of boxplots to see how they correspond to each other
# ---> box plots overallqual / saleprice :
var = 'OverallQual'
data = pd.concat([df_train_cleaned2['SalePrice'], df_train_cleaned2[var]], axis=1)
f, ax = plt.subplots(figsize=(8,6))
fig = sns.boxplot(x=var, y='SalePrice', data=data)
fig.axis(ymin=0, ymax= 800000)
# Remark 1 : A trend of "the higher the overall qality the higher the saleprice" seems to be respected.
# Rmark 2 : Some outliers seems to exist, most of them being the last box, the 4th box, a bit on the higher end boxes

# Globall : this is what bivariate analysis look like, basically comparing 2 variables with each other with the help of graphical methods
# now lets take a look at multivariate analysis...
# ---------------------------------------------------------------------------
# MULTIVARIATE ANALYSIS
# when analysing x amount of variables at a time, univariate was for one, bivariate was for 2 and now multivariate is for multiple variables at a time
# - it can be done wih the help of a correlation matrix to see how well different variables are correlated to each other
# - A correlation matrix will plot the correlation between different variables
# -  variable which are more correlated to each other are ones that affects each other to a greater degree
# - (plotted in a more redish color, in a blue-to-red color gradient, and variable which are less correlated to each other in a darker color)
# - these correlation matrices are also called heatmaps
# ---> correlation matrix
corrmat = df_train_cleaned2.corr()
f, ax  = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
# Remark 1 : At first sight, we look for the more colored in red squares : there are particurarly 2 of them
# - one is referring to "TotalBsmtSF" correlated with "1sFlrSF"
# - the second one is referring to "GrLivArea" correlated with "TotRmsAbvGrd"
# These 2 cases shows a significant correlation between two variables and that is so strong that it shows a situation
# of multicollinearity. Thinking deeper about these variable, we can conclude that they give almost the same information
# Heatmaps are a great way to detect these situations and these variables are so correlated with each other
# that we might drop one of more of them

# ---> saleprice correlation matrice
# now lets get the saleprice correlation price to get the variables that are more correlated with saleprice
k = 10 # number of variables for heatmap
cols = corrmat.nlargest(k,'SalePrice')['SalePrice'].index # nlargest(n, columns,...) return the first n rows with the largest values in columns, in descending order, as a df
# restrict our dataset to only the k highest correlated variables to Saleprice, keep the values only as an array,...
# ...transpose it (cols are now rows), then compute correlation matrix (default option is rows are named as the variables hence the previous transposition)
cm = np.corrcoef(df_train_cleaned2[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt=' .2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
# Remark 1 : On this correlation matrix, we have the highest correlated to SalePrice k variables.
# - 3 var ('OverallQual', 'GrLivArea', 'TotalBsmtSF') are strongly correlated with SalePrice (corr >= 0.65)
# - 3 var ('GarageCars', '1stFlrSF','GarageArea') are all also among some of the most strongly correlated to SalePrice (0.5 < corr < 0.65)
# Remark 2 : 'GarageCars' is a consequence of 'GarageArea'. So they are like twin variables
# in the same idea, 'TotalBsmtSF' and '1stFlrSF' are also twin variables as the total area covered by the 1st floor will be the same ara as the basement
# Hence we can keep only one var for each set of these (i would keep 'GarageArea' and 'TotalBsmtSF') for being more informative to the human need
# Remark 3 : 'YearBuilt' is slightly correlated with 'SalePrice' (corr slighlty > 0.5 ie 0.52)
# ---> scatterplot
# now, lets create a pairplot consisting of the k variables most correlated with SalePrice
# this is great way to display a lot of information in a small space
# here we create multiple different scatterplots between many variables
# NB : the variables chosen here were at the arbitrary but i would only remove the correlated with other var and the ones not informative in real life situations
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train_cleaned2[cols], size= 2.5)
plt.show()
# Remark 1 : the scatterplot between "TatalBsmtSf" and "GrLivArea" shows a trend of dots rarely going past the x = y line towards the y axis
# showing that its not commonly expected to have a basement area larger than the above ground living area
# Remark 2 : the plot between 'SalePrice' and 'YearBuilt' : an exponential trend can be observed with the cloud of points itself and its upper limit
# Also regarding the last year, the dots tends to stay above a certain limit : this means prices are incrasing way faster now than they were in the earlier years

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------



