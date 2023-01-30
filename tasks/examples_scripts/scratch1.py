#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Amad Diouf, amaddioufb13@gmail.com
# Created Date: 27/05/2022
# version ='1.0.0'
# ---------------------------------------------------------------------------
""" script 1 used a workbench to tests small chunks of code and leave them there in case for when needed later"""
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
# IMPORTING OUR DATA
# ----------- some quick lines for tsting
# label_directory = "/home/amad/PALADIN_2/3CEREBRO/garage/projects/Owkin_Test1_Diouf_Amad/data/input_data/training_setA_test"
# label_directory = "/home/amad/PALADIN_2/3CEREBRO/garage/projects/Owkin_Test1_Diouf_Amad/data/input_data/training_setA"

# input_directory = "/home/amad/PALADIN_2/3CEREBRO/garage/projects/Owkin_Test1_Diouf_Amad/data/input_data/training_setA_test"
# input_directory = "/home/amad/PALADIN_2/3CEREBRO/garage/projects/Owkin_Test1_Diouf_Amad/data/input_data/training_setA"

# output_directory = "/home/amad/PALADIN_2/3CEREBRO/garage/projects/Owkin_Test1_Diouf_Amad/data/output_data/output_test"

#---------------
for k in range(5): # check if range starts by 0
    print(k)
#---------------------
for i, f in enumerate(files): # check if enumerate starts by 0 at i
    print(i)
#------------
size_longest_column_names_space = max(len(l) for l in existing_column_names)
considered_column_names = list(range(size_longest_column_names_space))
# or using
full_table.shape[1]
# -----------------
# where a break in missing data values happens () are :
# the break happens below 40% missing data : a group of cols have less than 16% missing data with 1 slight above 15%
missing_data_lowest_grp = missing_data[(missing_data.Percent < 40)] # 10
lowest_grp_list = missing_data_lowest_grp.index.values.tolist()
# its ['SBP', 'O2Sat', 'MAP', 'Resp', 'HR', 'HospAdmTime', 'Gender', 'Age', 'ICULOS', 'SepsisLabel'];
# ['SBP', 'O2Sat', 'MAP', 'Resp', 'HR'] in 15.21% to 7.74%; 'HospAdmTime' is 0.00101% with only 8 missing; ['Gender', 'Age', 'ICULOS', 'SepsisLabel'] have 0 missing
#--------------------
# - step 1 : drop these columns : ['EtCO2', 'Unit1', 'Unit2', 'DBP', 'SBP'] (df1)
# - step 2-a : make a copy of df1 and we drop from it ['Temp'] and the 26 lab values variables, found in largely_grp_list or see below (df1_MinVarSpace)

# - step 2-b : make a copy of df1 and we drop from it the missing rows of ['Temp'] and we drop the 26 lab values variables, found in largely_grp_list (df1_MinVarSpace_wTemp)

# - step 3 : make a copy of df1 where we remove also the 26 lab values variables, found in largely_grp_list or see below (df1_

# step 3 :
#--------------------------
# action : drop Unit1 and Unit2 as even tho they are from the demographics variables, they are not variables that we would like the phenomenon to be predicted with.
#-----------------------------------
# 3) df1_wLabVals_wTemp = df1, ['Temp'] rows with missing data dropped, rows with missing data in 26 lab values variables dropped # idea : use the most variables possible even if we risk low obs remaining
# NB : the 26 lab values variables are the content of largely_grp_list or see below:
# largely_grp_list = ['TroponinI', 'Bilirubin_direct', 'Fibrinogen', 'Bilirubin_total', 'Alkalinephos', 'AST', 'Lactate', 'PTT', 'SaO2', 'Calcium', 'Phosphate', 'Platelets', 'Creatinine', 'WBC', 'Magnesium', 'HCO3', 'BUN', 'Chloride', 'PaCO2', 'Hgb', 'BaseExcess', 'Potassium', 'pH', 'Hct', 'Glucose', 'FiO2']


#----------------------------------
x = data[-1, 0:34]
c = data[-1, 34:40]
x = df0[-1, 0:34]
c = df0[-1, 34:40]
#-------------------------------

# we get the rows to keep using the "true or false, the value is an outlier" table and the ~ that serves to invert a boolean table/col to get what is not an outlier as True
dfi = dfi.loc[~((dfi[outliers_rm_var] < var_fence_low) | (dfi[outliers_rm_var] > var_fence_high))] # also called dfi_no_outliers_for_outliers_rm_var

# - getting the list of variables on which to apply outliers removal
list_vars_df4_for_outliers_rm = list(df4.columns) # for df4
list_vars_df4_for_outliers_rm.remove(dep_var)
list_vars_df4_for_outliers_rm.remove("Gender")
list_vars_df4_wTemp_for_outliers_rm = list(df4_wTemp.columns) # for df4_wTemp
list_vars_df4_wTemp_for_outliers_rm.remove(dep_var)
list_vars_df4_wTemp_for_outliers_rm.remove("Gender")
# - removing the outliers
# for df4
df4_Q1 = df4[list_vars_df4_for_outliers_rm].quantile(0.25)
df4_Q3 = df4[list_vars_df4_for_outliers_rm].quantile(0.75)
df4_IQR = df4_Q3 - df4_Q1
df4_fence_low = df4_Q1 - 1.5 * df4_IQR
df4_fence_high = df4_Q3 + 1.5 * df4_IQR
# these following one liners throw a ValueError: Cannot index with multidimensional key. So we use a loop and do it col by col
# df4_no_outliers = df4.loc[(df4 >= df4_fence_low) & (df4 <= df4_fence_high)]
# df4_outliers_only = df4.loc[(df4 < df4_fence_low) | (df4 > df4_fence_high)]
df4_as_truefalse_table_for_outliers = ((df4 < df4_fence_low) | (df4 > df4_fence_high))
for outliers_rm_var in list_vars_df4_for_outliers_rm : # to test, outliers_rm_var = "HR" or outliers_rm_var = "O2Sat"
    num_obs_in_df4 = df4.shape[0]
    print("- Removing outliers in df4 for the var :",outliers_rm_var,"and this is var",(list_vars_df4_for_outliers_rm.index(outliers_rm_var)+1),"out of",len(list_vars_df4_for_outliers_rm))
    num_obs_to_keep = df4_as_truefalse_table_for_outliers[outliers_rm_var].value_counts()[0]
    num_obs_to_rm = df4_as_truefalse_table_for_outliers[outliers_rm_var].value_counts()[1]
    print("    num total obs was ",num_obs_in_df4,"and num obs being removed is",num_obs_to_rm,"so num obs remaining is",num_obs_to_keep)
    df4_no_outliers_for_outliers_rm_var = df4.loc[~(((df4 < df4_fence_low) | (df4 > df4_fence_high))[outliers_rm_var])] # the ~ serves to inverts a true or false table/col to get what is not an outlier as True
    df4 = df4_no_outliers_for_outliers_rm_var


# for df4_wTemp





# -
df3_Q1 = df3.quantile(0.25)
df3_Q3 = df3.quantile(0.75)
df3_IQR = df3_Q3 - df3_Q1
df3_fence_low = df3_Q1 - 1.5 * df3_IQR
df3_fence_high = df3_Q3 + 1.5 * df3_IQR
df3_out_nooutliers = df3.loc[(df3 > df3_fence_low) & (df3 < df3_fence_high)]
df3_out_ouliers = df3.loc[(df3 < df3_fence_low) | (df3 > df3_fence_high)]


df3_outliers_only = df3[~((df3 < (df3_Q1 - 1.5 * df3_IQR))) | ((df3 > (df3_Q3 + 1.5 * df3_IQR)))]
df3_not_outliers_only = df3[~((df3 >= (df3_Q1 - 1.5 * df3_IQR))) | ((df3 <= (df3_Q3 + 1.5 * df3_IQR)))]

#---------------------

# ---------> DESCRIPTIVE STATISTICS SUMMARY FOR OUR DEPENDENT VARIABLE (IMPORTANT TO KNOW HOW TO KNOW ITS VALUE AND HOW THEY ARE INITIALLY DISTRIBUTED)
# + lets check the classes in our dependent variable (we should find only 2 classes noted 0 and 1)
# - for df3 :
df3.SepsisLabel.value_counts()
# # """
# 0.0    646112
# 1.0     14905
# Name: SepsisLabel, dtype: int64
# # """
df3.SepsisLabel.value_counts(normalize=True)
# # """
# 0.0    0.977451
# 1.0    0.022549
# Name: SepsisLabel, dtype: float64
# # """
perc_class0_df3 = round(df3.SepsisLabel.value_counts(normalize=True)[0]*100,2) # 97.75
perc_class1_df3 = round(df3.SepsisLabel.value_counts(normalize=True)[1]*100,2) # 2.25
ratio_sup_inf_classes_df3 = round(max([perc_class0_df3,perc_class1_df3])/min([perc_class0_df3,perc_class1_df3]),0) # 43
# - for df3_wTemp :
df3_wTemp.SepsisLabel.value_counts()
# # """
# 0.0    244983
# 1.0      5121
# Name: SepsisLabel, dtype: int64
# # """
df3_wTemp.SepsisLabel.value_counts(normalize=True)
# # """
# 0.0    0.979525
# 1.0    0.020475
# Name: SepsisLabel, dtype: float64
# # """
perc_class0_df3_wTemp = round(df3_wTemp.SepsisLabel.value_counts(normalize=True)[0]*100,2) # 97.95
perc_class1_df3_wTemp = round(df3_wTemp.SepsisLabel.value_counts(normalize=True)[1]*100,2) # 2.05
ratio_sup_inf_classes_df3_wTemp = round(max([perc_class0_df3_wTemp,perc_class1_df3_wTemp])/min([perc_class0_df3_wTemp,perc_class1_df3_wTemp]),0) # 48

#------------------------------------
#--------------
le = LabelEncoder()
A3_df3 = df3.copy()
gender_encoded_df3 = le.fit_transform(A3_df3['Gender']) # for Gender variable
df3['Gender'] = gender_encoded_df3
#--------------------------------
a = df3.copy()
a_scaled = fts_scaling(a,dep_var)
print("Gender has", len(a_scaled["Gender"].unique()), "levels and they are", list(a_scaled["Gender"].unique()))
print(dep_var,"has", len(a_scaled["SepsisLabel"].unique()), "levels and they are", list(a_scaled["SepsisLabel"].unique()))

b = df3["Gender"]
c = scaler.fit_transform(b)
dfTest[['A', 'B']] = scaler.fit_transform(dfTest[['A', 'B']])

k = df3.copy()
listcols = list(k.columns)
listcols = listcols.remove(dep_var)
d=scaler.fit_transform(k[listcols])
k[["HR","Gender"]] = scaler.fit_transform(k[["HR","Gender"]].to_numpy())
#--------------------------------------
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,LabelBinarizer

X = df3.copy()
numeric_columns=list(X.select_dtypes('float64').columns)
categorical_columns=list(X.select_dtypes('int64').columns)

pipeline=ColumnTransformer([
    ('num',StandardScaler(),numeric_columns),
    ('cat',LabelBinarizer(),categorical_columns),
])

new_X=pipeline.fit_transform(X)

X1 = df3.copy()
X2 = X1["Gender"]
X2 = np.array(X2).reshape((len(X2), 1))

scaler = StandardScaler()
print(scaler.fit(X2))
# StandardScaler()
print(scaler.mean_)
# [0.5 0.5]
print(scaler.transform(X2))  ##!
#---------------------------------------
data = df3.copy()
col_names = list(data.columns)
# col_names = col_names.remove(dep_var)
del col_names[-1]

features = data[col_names]

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

ct = ColumnTransformer([
        ('somename', StandardScaler(), col_names)
    ], remainder='drop')

feats = ct.fit_transform(features)

# - for df3
df3_scaled = df3.copy()
list_fts_df3_scaled = list(df3_scaled.columns)
list_fts_df3_scaled.remove("SepsisLabel")
fts_frame_2_change = df3_scaled.loc[:, list_fts_df3_scaled]
fts_frame_scaled = scaler.fit_transform(fts_frame_2_change)
df3_scaled.loc[:, list_fts_df3_scaled] = fts_frame_scaled
# - for df3_wTemp
df3_wTemp_scaled = df3_wTemp.copy()
list_fts_df3_wTemp_scaled = list(df3_wTemp_scaled.columns)
list_fts_df3_wTemp_scaled.remove("SepsisLabel")
fts_frame_2_change = df3_wTemp_scaled.loc[:, list_fts_df3_wTemp_scaled]
fts_frame_scaled = scaler.fit_transform(fts_frame_2_change)
df3_wTemp_scaled.loc[:, list_fts_df3_wTemp_scaled] = fts_frame_scaled


#----------------------
df3["Gender"].describe()
A1_df3 = df3.copy()
A2_df3 = fts_scaling(A1_df3,dep_var)
A2_df3["Gender"].describe()
A2_df3["Gender"].std()
#-------------------------------
# lets make versions of our datasets that has understandable values for the categorical cols
df4_viz = df4
df4_viz['Gender'] = df4_viz['Gender'].map({sorted(df4_viz["Gender"].unique())[0]: "Female", sorted(df4_viz["Gender"].unique())[0]: "Male"})
#----------------------
# version 1
# df4_unscaled.Gender.value_counts().nlargest(2).plot(kind='bar', figsize=(10,4))
# plt.title("Number of patients by Gender")
# plt.ylabel('Number of patients')
# plt.xlabel("Gender")
# version 2 (nicer)
#----------------------
var1 = "HR"
var2 = "O2Sat"
data = pd.concat([df4_unscaled[var1], df4_unscaled[var2]], axis=1)
graph = data.plot.scatter(x=var1, y=var2)
graph = data.plot.scatter(x=var1, y=var2, ylim=(0,1.5))
#----------------------
var = 'HR'
data = pd.concat([df4[dep_var], df4[var]], axis=1)
graph = data.plot.scatter(x=dep_var, y=var)
# because the dependent var is a categorical var,
#----------------------
var = 'HR'
data = pd.concat([df4[dep_var], df4[var]], axis=1)
f, ax = plt.subplots(figsize=(8,6))
fig = sns.boxplot(x=dep_var, y=var, data=data) # putting
fig.axis(ymin=0, ymax= 1.5)
#----------------------
sns.heatmap(corrmat, annot=True, square=True)
sns.heatmap(corrmat, cmap="seismic", annot=True,square=True)
sns.heatmap(corrmat, cmap="RdBu", annot=True,square=True)
sns.heatmap(corrmat, cmap="BrBG", annot=True,square=True)
sns.heatmap(corrmat, vmax=.8, square=True)
#----------------------
sns.set()
sns.set(font_scale=1.25)
#----------------------
graph = sns.heatmap(cm, mask=mask_used ,cmap="bwr", annot=True, square=True,
                 fmt=' .2f', annot_kws={'size': 10},
                 yticklabels=cols.values, xticklabels=cols.values)
#----------------------
# > with only the vars most correlated to the dep var
k = 10
corrmat = df4_wTemp.corr()
corrmat_all_abs = corrmat.abs()
k_cols_in_descorder_of_corr_with_dep_var = corrmat_all_abs.nlargest(k,dep_var)[dep_var].index
cm = df4_wTemp[k_cols_in_descorder_of_corr_with_dep_var].corr()
mask_used = np.triu(np.ones_like(cm, dtype=bool))
f, ax  = plt.subplots(figsize=(12, 9))
graph = sns.heatmap(cm, mask=mask_used, cmap="bwr", annot=True, square=True)
plt.title("Heatmap focused on the features most correlated to the dependent variable")
plt.show()
# Remark 1 :
#----------------------
k = 6
corrmat = df4.corr()
cols = corrmat.nlargest(k,dep_var)[dep_var].index
cm = np.corrcoef(df4[cols].values.T)
mask_used = np.triu(np.ones_like(cm, dtype=bool))
f, ax  = plt.subplots(figsize=(12, 9))
graph = sns.heatmap(cm, mask=mask_used ,cmap="bwr", annot=True, square=True,
                 fmt=' .2f', annot_kws={'size': 10},
                 yticklabels=cols.values, xticklabels=cols.values)
plt.title("Heatmap focused on the features most correlated to the dependent variable")
plt.show()
#----------------------
k = 10
corrmat = df4_wTemp.corr()
corrmat_all_abs = corrmat.abs()
k_cols_in_descorder_of_corr_with_dep_var = corrmat_all_abs.nlargest(k,dep_var)[dep_var].index
cm = df4_wTemp[k_cols_in_descorder_of_corr_with_dep_var].corr()
mask_used = np.triu(np.ones_like(cm, dtype=bool))
f, ax  = plt.subplots(figsize=(12, 9))
graph = sns.heatmap(cm, mask=mask_used, cmap="bwr", annot=True, square=True)
plt.title("Heatmap focused on the features most correlated to the dependent variable")
plt.show()


k = 10
initial_corrmat = df4.corr()
corrmat_all_abs = initial_corrmat.abs()
k_cols_in_descorder_of_corr_with_dep_var = corrmat_all_abs.nlargest(k,dep_var)[dep_var].index
final_corrmat = df4[k_cols_in_descorder_of_corr_with_dep_var].corr()
mask_used = np.triu(np.ones_like(final_corrmat, dtype=bool))
f, ax  = plt.subplots(figsize=(12, 9))
graph = sns.heatmap(final_corrmat, mask=mask_used, cmap="bwr", annot=True, square=True)
plt.title("Heatmap showing the correlation between all remaining variables")
plt.show()
#----------------------
# + heatmap using correlation matrix (variables that are the most correlated with the dep var)
# now lets get the dep var correlation matrice to get the variables that are more correlated with the dep var
# - k is number of variables for heatmap (number of vars most correlated + 1 for the dep var)
# - nlargest(n, columns,...) return the first n rows with the largest values in columns, in descending order, as a df
# then we restrict our df to der var, keep the values only as an array,
# then we transpose it (cols are now rows),
# then compute correlation matrix (default option is rows are named as the variables hence the previous transposition)

# Remark : from highest to lowest, the vars most correlated with the dep are ranked like this :
# ICULOS, HR, Resp, HospAdmTime,Gender
#----------------------
# the source "temp management and sepsis" reports that temp management is detrimental to a good progress of the care during
# source "temp management and sepsis" : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5702415/#:~:text=Fever%20in%20sepsis%20could%20lead,energy%20requirements%20of%20the%20tissues

#----------------------
# But for a more time-efficient process, we chose these 2 steps :
# step 2 : use a multivariate view dedicated to show bivariate situations on all our variables
# step 3 : use a multivariate view dedicated to show correlation between all our variables



# + a multivariate view dedicated to show bivariate situations on all our variables

# + a multivariate view dedicated to show correlation between all our variables
#----------------------
var1 = "HR"
var2 = "O2Sat"
data = pd.concat([df4[var1], df4[var2]], axis=1)
graph = data.plot.scatter(x=var1, y=var2)
plt.title("Scatter plot of the variables HR and O2Sat")
plt.ylabel('O2Sat', fontsize=12)
plt.xlabel('HR', fontsize=12)
plt.show()

#----------------------
for f in os.listdir(input_directory):
    if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('csv'):
        files.append(f)

#----------------------
# ---------------------------------------------------------------------------
""" script taking as entry a path to a folder with multiple files (one for each patient of a hospital X),
 and output a full table with all patients of the hospital X"""
# (for training_setA)

# ---------------------------------------------------------------------------
# Imports
import numpy as np
import os, os.path, sys, warnings
import pandas as pd
import matplotlib # change the backend used by maplotlib from the default 'QtAgg' to one that enables interactive plots 'Qt5Agg'
matplotlib.use('Qt5Agg')
matplotlib.get_backend()
import matplotlib.pyplot as plt
import seaborn as sns
# for sns plots, add plt.show() to make the plots appear (replace it with the "%matplotlib inline" in the case of a notebook to let the backend solve that issue while displaying the plots inside the notebook)
# import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from engines.data_mgmt_engine import Prefix_in_absolute_path
#----------------------
# Imports
import os, os.path, sys, warnings
import pandas as pd
import numpy as np
import matplotlib # change the backend used by maplotlib from the default 'QtAgg' to one that enables interactive plots 'Qt5Agg'
matplotlib.use('Qt5Agg')
matplotlib.get_backend()
import matplotlib.pyplot as plt
import seaborn as sns
# for sns plots, add plt.show() to make the plots appear (replace it with the "%matplotlib inline" in the case of a notebook
# to let the backend solve that issue while displaying the plots inside the notebook)
from scipy.stats import norm
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from engines.data_mgmt_engine import Prefix_in_absolute_path # a defined prefix of all absolute paths used in order to keep them universal for all users


# ----------------------------------------------------------------------------------------------------------------------
# >>>>>> Objective : clean our dataset, understand it better, and prepare it for the basics of learning tasks.
# ----------------------------------------------------------------------------------------------------------------------

#----------------------

# for df4 # (using df4_unscaled to display adequate names for categories)
sns.set()
var1 = 'HR'
var2 = dep_var
order=["0","1"]
data = pd.concat([df4_unscaled[var2], df4_unscaled[var1]], axis=1)
f, ax = plt.subplots(figsize=(8,6))
fig = sns.boxplot(x=var2, y=var1, data=data, order=order)
pairs_to_compare=[("0","1")]
our_annotations_to_add = add_stat_annotation(ax, data=data, x=var2, y=var1, order=order,
                                             box_pairs=pairs_to_compare,
                                             test='Mann-Whitney', text_format='star',
                                             loc='inside', verbose=2)
our_annotations_to_add
plt.title("Boxplot of the variables HR and SepsisLabel (dtsetA, var Temp dropped)")
plt.ylabel('HR', fontsize=12)
plt.xlabel('SepsisLabel', fontsize=12)
fullname_file_of_plot = output_directory+"/"+"dtsetA_STEP2_EDA_BA_Case2_1ContVar1CatVar_HR_theDepVar.png"
plt.savefig(fullname_file_of_plot, bbox_inches='tight')
plt.show()
plt.close()
# """
# p-value annotation legend:
# ns: 5.00e-02 < p <= 1.00e+00
# *: 1.00e-02 < p <= 5.00e-02
# **: 1.00e-03 < p <= 1.00e-02
# ***: 1.00e-04 < p <= 1.00e-03
# ****: p <= 1.00e-04
# 0 v.s. 1: Mann-Whitney-Wilcoxon test two-sided with Bonferroni correction, P_val=2.683e-257 U_stat=4.026e+09
# """
# - Remark 1 :  some outliers seem to exist (e.g. outside of both whiskers of both boxes)
# This can also be observed in a pairplot during multivariate analysis that implicates these vars.
# - Remark 2 : the 2 two groups of the dep var are differenciated by the HR var (P_val=2.683e-257 ie <= 1.00e-04 ie **** significance)

#----------------------
# for df4 # (using df4_unscaled to display adequate names for categories)
sns.set()
var1 = 'HR'
var2 = dep_var
var3 = "Gender"
order=["0","1"]
data = pd.concat([df4_unscaled[var2], df4_unscaled[var1], df4_unscaled[var3]], axis=1)
f, ax = plt.subplots(figsize=(8,6))
fig = sns.boxplot(x=var2, y=var1, hue=var3,data=data, order=order)
pairs_to_compare=[(("0","Male"),("1","Male")),(("0","Female"),("1","Female")),
                  (("0","Male"),("0","Female")),(("1","Male"),("1","Female"))]
our_annotations_to_add = add_stat_annotation(ax, data=data, x=var2, y=var1,  hue=var3,order=order,
                                             box_pairs=pairs_to_compare,
                                             test='Mann-Whitney', text_format='star',
                                             loc='inside', verbose=2)
our_annotations_to_add
plt.title("Boxplot of the variables HR and SepsisLabel, for each Gender (dtsetA, var Temp dropped)")
plt.ylabel('HR (beats per minute)', fontsize=12)
plt.xlabel('SepsisLabel', fontsize=12)
fullname_file_of_plot = output_directory+"/"+"dtsetA_STEP2_EDA_BA_Case3_2CatVars1ContVar_Gender_theDepVar_HR.png"
plt.savefig(fullname_file_of_plot, bbox_inches='tight')
plt.show()
plt.close()
"""
p-value annotation legend:
ns: 5.00e-02 < p <= 1.00e+00
*: 1.00e-02 < p <= 5.00e-02
**: 1.00e-03 < p <= 1.00e-02
***: 1.00e-04 < p <= 1.00e-03
****: p <= 1.00e-04
1_Male v.s. 1_Female: Mann-Whitney-Wilcoxon test two-sided with Bonferroni correction, P_val=5.898e-03 U_stat=2.570e+07
0_Male v.s. 0_Female: Mann-Whitney-Wilcoxon test two-sided with Bonferroni correction, P_val=2.549e-113 U_stat=4.920e+10
0_Male v.s. 1_Male: Mann-Whitney-Wilcoxon test two-sided with Bonferroni correction, P_val=3.050e-155 U_stat=1.414e+09
0_Female v.s. 1_Female: Mann-Whitney-Wilcoxon test two-sided with Bonferroni correction, P_val=3.990e-105 U_stat=6.657e+08
"""
# - Remark 1 :  some outliers seem to exist (e.g. outside of both whiskers of all boxes)
# - Remark 2 : a) the HR var differentiates significantly the two genders in each SepsisLabel.
# b) the HR differentiates significantly the SepsisLabel in each gender.
#----------------------

#----------------------
y_pred_100 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
y_pred_90 = [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
y_pred_90 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
y_pred_100 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred_80 = [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
y_pred_70 = [1, 0, 0, 0, 0, 1, 1, 1, 0, 0]
f1_score(y_true, y_pred_100, average='binary')
Traceback (most recent call last):
  File "/home/amad/anaconda3/envs/Owkin_Test1_Diouf_Amad/lib/python3.10/code.py", line 90, in runcode
    exec(code, self.locals)
  File "<input>", line 1, in <module>
NameError: name 'f1_score' is not defined
from sklearn.metrics import f1_score
f1_score(y_true, y_pred_100, average='binary')
1.0
f1_score(y_true, y_pred_90, average='binary')
0.888888888888889
f1_score(y_true, y_pred_80, average='binary')
0.7499999999999999
f1_score(y_true, y_pred_70, average='binary')
0.6666666666666665
f1_score(y_true, y_pred_100, average='weighted')
1.0
f1_score(y_true, y_pred_90, average='weighted')
0.898989898989899
f1_score(y_true, y_pred_80, average='weighted')
0.7916666666666666
f1_score(y_true, y_pred_70, average='weighted')
0.6969696969696969
y_true = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
y_pred_100 = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
y_pred_90 = [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
y_pred_80 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
y_pred_70 = [1, 0, 0, 0, 0, 0, 0, 1, 0, 0]
f1_score(y_true, y_pred_100, average='binary')
1.0
f1_score(y_true, y_pred_100, average='weighted')
1.0
f1_score(y_true, y_pred_90, average='binary')
0.8
f1_score(y_true, y_pred_90, average='weighted')
0.8933333333333333
f1_score(y_true, y_pred_80, average='binary')
0.5
f1_score(y_true, y_pred_80, average='weighted')
0.7625000000000001
f1_score(y_true, y_pred_70, average='binary')
0.4
f1_score(y_true, y_pred_70, average='weighted')
0.6799999999999999
#----------------------

# ------------- reserve of ideas to add

# # - the params_list that we should print, copy the print and paste it in the code part were we want all these values
# params_list = {'numThreads': -1, 'verbose': True, 'it0': 10, 'max_it': 200, 'L0': 0.1, 'intercept': False, 'pos': False}
# params_list['compute_gram'] = True
# params_list['loss'] = 'square'
# params_list['regul'] = 'l1'
# params_list['tol'] = 0.01
# params_list["lambda1"] = 0.05
# # params_list["lambda2"] = 0.05
# # params_list["lambda3"] = 0.05
# print(params_list)

# - the param_grid is the gallery of values to be given to the gridsearchcv function for the search
# to explore one of the lambdas, uncomment of these
# param_grid = {"lambda1": space_of_values}
# param_grid = {"lambda2": space_of_values}
# param_grid = {"lambda3": space_of_values}

#----------------------
# > setting it up...

# param_grid_gallery = param_grid_lambdas123_space_maker1("geom",0.0001,100.0,0,"yes",None,"no","geom",0.0001,100.0,200,"yes",None,"no") # 200 values of lambda2 only
# param_grid_gallery = param_grid_lambdas123_space_maker1("geom",0.0001,100.0,200,"yes",None,"no","geom",0.0001,100.0,200,"yes",None,"no") # 200 values of lambda1 and 200 values of lambda2
# > ... calling it
param_grid = param_grid_gallery[0]
param_grid_sizes = param_grid_gallery[1]
#----------------------checking on dataset if for a patient that has sepsis all entries after the early entry 1 are marked 1
f_pos = [] # 1790 patients
for fi_1 in files_tables1:
    fi_2 = fi_1[:,-1]
    if 1 in fi_2:
        f_pos.append(fi_1)

f_pos_lastas0 = [] # 0 patients
f_pos_lastas1 = [] # 1790 patients
for fi_3 in f_pos:
    if fi_3[-1, -1]==0.0:
        f_pos_lastas0.append(fi_3)
    elif fi_3[-1, -1]==1.0:
        f_pos_lastas1.append(fi_3)



#----------------------
# df4 = df4_list_of_versions_after_outliers_rm[0] # [444258 rows x 9 columns]
# df4_unscaled = df4_list_of_versions_after_outliers_rm[1] # [444258 rows x 9 columns]
# #----------------------
# df4_wTemp = df4_wTemp_list_of_versions_after_outliers_rm[0] # [179436 rows x 10 columns]
# df4_wTemp_unscaled = df4_wTemp_list_of_versions_after_outliers_rm[1] # [179436 rows x 10 columns]
#----------------------
# - Remark 4 :
# between fts, the highest pairwise corrs distinguished are, in descending order :
#----------------------
# (same group and order in comparison to hospital A)
#----------------------hosp a then hosp b
# ICULOS, HR, Resp, MAP, HospAdmTime, Gender, Age, and O2Sat (a)
# HR, MAP, ICULOS, HospAdmTime, Resp, O2Sat, Gender, and Age (b)
#----------------------
# ICULOS, HR, Resp, MAP. (a)
# HR, MAP, ICULOS, HospAdmTime.(b)
#----------------------hosp a then hosp b with Temp
# Resp, Temp, ICULOS, HR, HospAdmTime, MAP, O2Sat, Age, Gender (a)
# HR, ICULOS, Temp, MAP, HospAdmTime, Resp, Age, Gender, O2Sat (b)
#----------------------
# Resp, Temp, ICULOS, HR, HospAdmTime, MAP (a)
# HR, ICULOS, Temp, MAP, HospAdmTime (b)
#----------------------
# (different order in comparison to hospital A)
# (3/4 ie 75% similar in comparison to hospital A)
#----------------------
# - Remark 2 :
# this is for HospAdmTime being left skewed while having a negative correlation to the dep var. Same remark as in (dtsetB, var Temp dropped),
# but here, a longer HospAdmTime tends to be in case of a non sepsis patient.
# And with the dataset largely imbalanced with much more non sepsis patients,
# therefore its is understandable to have more data points on the right (ie less on the left) of the distribution of HospAdmTime.
#----------------------df0 for a then b
# [790215 rows x 41 columns]
# [761995 rows x 41 columns]
#----------------------
#
#
#----------------------df4 for b (df4 then wtemp version)
# [444258 rows x 9 columns]
# [179436 rows x 10 columns]
#----------------------
##! last stopped here
##! stopped here
#----------------------
param_grid = dict_lambda1_spaceofvalues
param_grid_size = dict_lambda1_sizespaceofvalues["lambda1"]
#----------------------
a_seed = 0
modelselector_by_GSCV_T0 = modelselector_by_GSCV
a_seed = 1
modelselector_by_GSCV_T1 = modelselector_by_GSCV
a_seed = 2
modelselector_by_GSCV_T2 = modelselector_by_GSCV
#----------------------
# # ~~~~
    # if X_test_probs_all_classes.ndim == 1: # case where predict_proba gives only 1 col that is for "X_test_probs_of_class1_when_classes_are_bin"
    #     X_test_probs_of_class1_when_classes_are_bin = X_test_probs_all_classes
    # elif X_test_probs_all_classes.ndim > 1: # case where predict_proba gives a table with "col 1 to col n" for "class 0 to class last_class"
    #
    #     X_test_probs_of_class1_when_classes_are_bin = X_test_probs_all_classes[:, 1]
    # else : # (if X_test_probs_all_classes.ndim > 1 :)
    #     index_last_col_table_of_probs = X_test_probs_all_classes.shape[1] - 1 # get index last col because it where the positive class is always (wether both classes probs are computed or not)
    #     X_test_probs_of_class1_when_classes_are_bin = X_test_probs_all_classes[:,index_last_col_table_of_probs]
    # # ~~~~

#----------------------
# ...for each metric that we had in our list of metrics to compute, we get the test score of the best model
    for a_metric_of_testscore_to_compute in list_tags_metrics_computed_for_testscore:
        if a_metric_of_testscore_to_compute == "MM4":
            model_retained_asbest_testscore = MM4_bespoke_score(y_test, X_test_preds)  # MM4
        elif a_metric_of_testscore_to_compute == "MCC":
            model_retained_asbest_testscore = matthews_corrcoef(y_test, X_test_preds)  # MCC
        elif a_metric_of_testscore_to_compute == "CK":
            model_retained_asbest_testscore = cohen_kappa_score(y_test, X_test_preds)  # cohen_kappa
        elif a_metric_of_testscore_to_compute == "F1":
            model_retained_asbest_testscore = f1bin_score(y_test, X_test_preds)  # f1_score
        elif a_metric_of_testscore_to_compute == "BalAcc":
            model_retained_asbest_testscore = balanced_accuracy_score(y_test, X_test_preds)  # bal acc
        elif a_metric_of_testscore_to_compute == "Acc":
            model_retained_asbest_testscore = acc_norm_score(y_test, X_test_preds)  # acc
        elif a_metric_of_testscore_to_compute == "Prec":
            model_retained_asbest_testscore = precbin_score(y_test, X_test_preds) # precision
        elif a_metric_of_testscore_to_compute == "Rec":
            model_retained_asbest_testscore = recbin_score(y_test, X_test_preds)  # recall
        elif a_metric_of_testscore_to_compute == "AUC":
            model_retained_asbest_testscore = roc_auc_score(y_test, X_test_probs_of_class1_when_classes_are_bin)  # auc
        else : # tag of a metric for which a computation has not been implemented
            model_retained_asbest_testscore = 0.0
            print("Warning! the tag",a_metric_of_testscore_to_compute,"does not correspond to an implemented metric computation. Test score of 0.0 is given to it by default")
        # - Part 2 : stash the test score computed for each metric computed (stashed in a dict "metric as key and list of testscores as value")
        dict_val_updater_valueaslist_V1(DictCol_test_scores, a_metric_of_testscore_to_compute, model_retained_asbest_testscore)
#----------------------
# + f1 for binary classification but not weighted (ie not specially geared towards accounting for imbalance)
# NB1 : this function is defined to return a version of the f1_score dedicated to binary problem with the average argument set to average='binary'
# NB2 : As per the source, this only report results for the class specified by pos_label (class 1 by default). This is applicable only if targets (y_{true,pred}) are binary.

def f1score_as_binary(actual,prediction):
    f1score_as_binary_value = f1_score(actual, prediction, average='binary')
    return f1score_as_binary_value

# + f1 for each label and weighted (ie geared towards accounting for imbalance)
# NB1 : this function is defined to return a version of the f1_score dedicated to account for label imbalance with the average argument set to average='weighted'
# NB2 : As per the source, this calculates metrics for each label, and find their average weighted by support
# (the number of true instances for each label). This accounts for label imbalance.
# NB3 : this can result in an F-score that is not between precision and recall.

def f1score_as_weighted(actual,prediction):
    f1score_as_weighted_value = f1_score(actual, prediction, average='weighted')
    return f1score_as_weighted_value
#----------------------
"f1score_as_binary"
"f1score_as_weighted"
"MCC_V1"
"CK_non_weighted"
"CK_weighted_as_linear"
"acc_V1"
"bal_acc_V1"
"prec_as_binary"
"prec_as_weighted"
"rec_as_binary"
"rec_as_weighted"
"roc_auc_score_as_macro"
"roc_auc_score_as_weighted"
"MM4_bespoke_score"

#----------------------
# all metrics that are ready to be computed, in order of preference, and if the metric has a version that manages imbalance better,
# the version "with better imbalance management" placed just before the version "unfit for imbalance management" (14 metrics)
list_a= ["f1score_as_weighted", "f1score_as_binary", "MCC_V1", "CK_weighted_as_linear", "CK_non_weighted", "bal_acc_V1", "acc_V1",
    "MM4_bespoke_score",
    "prec_as_weighted", "prec_as_binary", "rec_as_weighted", "rec_as_binary", "roc_auc_score_as_weighted", "roc_auc_score_as_macro"]
# only the metrics that manages imbalance better, in order of preference (8 metrics)
list_b= ["f1score_as_weighted", "MCC_V1", "CK_weighted_as_linear", "bal_acc_V1",
    "MM4_bespoke_score",
    "prec_as_weighted", "rec_as_weighted", "roc_auc_score_as_weighted"]
# among the metrics that manages imbalance better, in order of preference,
# only those who would make it to publication if we decided to keep the list as short as possible (6 metrics)
list_c= ["f1score_as_weighted", "MCC_V1", "bal_acc_V1",
    "prec_as_weighted", "rec_as_weighted", "roc_auc_score_as_weighted"]
#----------------------
roc_curve_updater_after_one_iteration_of_the_mdl2(y_test,
                                                      X_test_probs_of_class1_when_classes_are_bin,
                                                      RespClassesList,
                                                      mean_fpr_by_seed_one_alg,
                                                  fprs_col_by_seed_one_alg,
                                                      tprs_col_by_seed_one_alg,
                                                      aucs_col_by_seed_one_alg,
                                                      a_seed,
                                                      fig_subplot_as_axi)
#----------------------
# that can be an interval with points not adequately spaced for a figure,
	# we can interpolate these points to obtain another version of tpr_mdl_one_iter with values adequately spaced for a figure and
	# we can even choose the number of values n that will be outputed as corresponding to n values on the x axis
#----------------------
roc_curve_updater_after_one_iteration_of_the_mdl2(y_test,
													  X_test_probs_of_class1_when_classes_are_bin,
													  ClassesList,
													  mean_fpr_by_seed_one_alg,
													  tprs_col_by_seed_one_alg,
													  aucs_col_by_seed_one_alg,
													  a_seed,
													  fig_subplot_as_axi)
#----------------------
# NB : this is the succession of the arguments in the older model that was used when a 3rd axi in the plot is dedicated to showing the average roc curves only
# the part to add the average roc curve to that 3rd plot is uncommented so the 3rd axi is removed from the arguments (it was the 3rd arg)
# old succesio of args was : fig,fig_subplot_as_axi,mdls_comp_fig_subplot_as_axi,tprs_col_of_mdl,mean_fpr_of_mdl,aucs_col_of_mdl,basedir,task_type,scheme_used,the_model_compared,tag_ctype,tag_drugname,tag_profilename,tag_num_trial

#----------------------
collector_metrics_type_OneValMadeFromTheConfMatrix_supplier_V1
collector_metrics_type_ValuesUsedToPlotRocCurves_supplier_V1
#----------------------
task_type
scheme_used
the_model_compared
tag_ctype
tag_drugname
tag_profilename
tag_num_trial
#in
subplot_ax1outof1.set_title("\n".join(wrap('ROC curve of %(Task)s using %(Alg)s-%(Model)s model, on case %(Ctype)s-%(Drug)s, %(Profile)s profile,  %(Trial)s' %
    {"Task": task_type, "Alg": scheme_used, "Model": the_model_compared, "Ctype": tag_ctype, "Drug": tag_drugname,
     "Profile": tag_profilename, "Trial": tag_num_trial})))
task_type
scheme_used
the_model_compared
tag_cond
tag_pop
tag_dataprofile
tag_num_trial
# with
task_type = "Regr"
scheme_used = "L1LogReg"
the_model_compared = "SingleTaskwAllFts"
tag_cond = "SepsisOnset"
tag_pop = tag_cohort.split("_")[0] # using dtsetA or dtsetB as pop
tag_dataprofile = tag_cohort.split("_")[1] # use woTemp or wTemp as profiles
tag_num_trial = "Trial" + "1"

#----------------------
print("- finished model selection step for working with seed", a_seed)
print("- started collecting results step for working with seed", a_seed) ##!
print("- finished collecting results step for working with seed", a_seed)
#----------------------
# (uncomment one between these "#~~~start #~~~end" to use it)
#----------------------
# ---> Grouping all our dict "key as HP name" and "values as list_of_values_to_explore_for_HP"
# in one dict param_grid that will be explored during model selection
#----------------------
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#----------------------

#----------------------

#----------------------

#----------------------

#----------------------

#----------------------

#----------------------

#----------------------

#----------------------

#----------------------

#----------------------

#----------------------

#----------------------

#----------------------

#----------------------

#----------------------

#----------------------

#----------------------