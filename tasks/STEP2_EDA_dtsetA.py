#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Created By  : Amad Diouf, amaddioufb13@gmail.com
# Created Date: 27/05/2022
# version ='1.0.0'
# ------------------------------------------------------------------------------
""" script taking as input file a full table with all patients of a hospital,
then we realise here the EDA on it,
to obtain as output the curated version(s)
"""
# (for training_setA)

# ------------------------------------------------------------------------------
# + NOTE ON SCRIPT LAUNCH :
# this script is initially designated to be followed line by line in an IDE where the linked environment supplied is opened.
# This is in order to explore all the steps done, choices made and remarks given for more clarity on the task carried out here.
# + HOW TO LAUNCH THIS SCRIPT :
# with the supplied environment opened, you can either :
# - follow this script line by line in an IDE (all necessary inputs are given here, all outputs will be produced in designated locations)
# - or call it in a terminal (no arguments needed, all necessary inputs are given here, all outputs will be produced in designated locations)
# + ARGUMENTS HANDLING :
# we decided to include in this script all the inputs needed for the task as well as all the info needed to give an output.
# We made this choice to focus as best as we can in the developments oriented towards the task realized by this script,
# knowing how vast arguments handling (throughout all the workings of a script) can be if we dont want errors coming from inadequate user given arguments.
# NB : To answer inquiries about how we would handle arguments (if we gave the choice to the user to
# use them in order to supply input and output related information), we would mainly use sys.argv and/or the argparse library.

# ------------------------------------------------------------------------------
# Imports
import os, os.path, sys, warnings, argparse # standard files manipulation set of imports
import pandas as pd
import numpy as np
import matplotlib # change the backend used by matplotlib from the default 'QtAgg' to one that enables interactive plots 'Qt5Agg'
matplotlib.use('Qt5Agg')
matplotlib.get_backend()
import matplotlib.pyplot as plt
import seaborn as sns
# for sns plots, add plt.show() to make the plots appear (replace it with the "%matplotlib inline" in the case of a notebook
# to let the backend solve that issue while displaying the plots inside the notebook)
from sklearn.preprocessing import LabelEncoder # to encode values of categorical variables
from engines.data_mgmt_engine import Prefix_in_absolute_path # a defined prefix of all absolute paths used in order to keep them universal for all users
from engines.data_mgmt_engine import fts_scaling # A function carrying out the scaling of the features in a table
from engines.data_mgmt_engine import first_removal_outliers # A function carrying out the first removal of outliers
from engines.data_mgmt_engine import imbalance_summary # A function carrying the estimate of the imbalance level
from statannot import add_stat_annotation # to add statistical annotations on figures (e.g. statistical difference on boxplots)
# ----------------------------------------------------------------------------------------------------------------------
print("# >>>>>> Objective : clean our dataset, understand it better, and prepare it for the basics of learning tasks.")
# ----------------------------------------------------------------------------------------------------------------------

print("# >>>>>> Part 1 : Input and output related information.")

print("# ---> input related information")
# + File location of the full table of our dataset
# - to test with small file
# input_file = Prefix_in_absolute_path + "/Owkin_Test1_Diouf_Amad/data/output_data/after_step1_data_gathering/dtsetA_forEDA_test_V1.csv"
# - to use the full file
input_file = Prefix_in_absolute_path + "/Owkin_Test1_Diouf_Amad/data/output_data/after_step1_data_gathering/dtsetA_forEDA_full_V1.csv"

print("# ---> output related information")
# + the output folder location
output_directory = Prefix_in_absolute_path + "/Owkin_Test1_Diouf_Amad/data/output_data/after_step2_EDA"
# + checks
# - if output folder is existing (if not we create it)
if not os.path.isdir(output_directory):
    os.mkdir(output_directory)

print("# ---> End of part : all input and output related info accounted for !")
# ----------------------------------------------------------------------------------------------------------------------

print("# >>>>>> Part 2 : Loading our file info as a dataframe and getting our first observations of its content.")

print("# ---> importing our file to get one full table of the population")
df0 = pd.read_csv(input_file)

print("# ---> displaying a preview of the data")
# uncomment an option to use it

# # + option 1 : displaying a head (some of the first lines of the table)
# df0.head(10) # the first 10 lines

# # + option 2 : displaying a tail (some of the last lines of the table)
# df0.tail(10) # the last 10 lines

# + option 3 : displaying fully the data
# NB1 : the display will depend on IDE displaying capabilities :
# either the whole table will be displayed if the table is not large enough
# or if the table is large, some of the first lines and some of the last lines will be displayed
# NB2 : by using this option, the dimensions of the table are already given below
# the table before the next step to see how large is the table
df0
# + Remark 1 :
# there are missing values

print("# ---> description of the data")
# NB : the previous command can already give us the dimensions of the table but we prefer the df.info() command that is more informative
df0.info()
# """
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 790215 entries, 0 to 790214
# Data columns (total 41 columns):
#  #   Column            Non-Null Count   Dtype
# ---  ------            --------------   -----
# """
# + Remark 1 :
# the table dimensions are [790215 rows x 41 columns]
# + Remark 2 :
# - rows are the samples/observations/data points (790215).
# - columns are the variables (40 independent variables/features + 1 dependant variable).
# + Remark 3 :
# in the rest of the work, we use some abbreviations to refer to some entities :
# we will use 'row' or 'obs' to refer to any of the samples/observations/data points
# we will use 'col' or 'var' to refer to any of the 41 variables
# we will use 'fts' to refer to any of the 40 independent variables/features
# we will use 'dep var' to refer to the dependant variable


print("# ---> Records to facilitate later navigation of the vars in the dataset")
# + keeping the dep var in a coding variable to facilitate future lines of code that uses it
dep_var = 'SepsisLabel'
# + keeping a list of the initial features :
list_fts0 = list(df0.columns)
list_fts0.remove(dep_var)

# ---> checking the type of data in each col
# df0.dtypes
# Remark 1 :
# this is not useful yet at this point because we are potentially looking at cols we wont keep.
# It will be done after the table is cleaned and only the rows and cols to keep are remaining.

print("# ---> End of part !")
# ----------------------------------------------------------------------------------------------------------------------

print("# >>>>>> Part 3 : Data cleaning 1/3 : initial sanitation of columns and rows.")

print("# ---> initial sanitation of columns (1/2) : dropping irrelevant columns")

# NB : "the columns" are all the vars (all the features space + the dep var) but
# the sanitation of columns is focusing on the features space. The dep var, remains untouched yet at this point, because
# it is absolutely necessary for the learning task that will follow the EDA
# hence sanitation actions on columns like dropping the column or renaming it are not done on the dep var to preserve it.
# Actions that go towards curating the dep var column will follow in future steps and they will be done with a precisely explained objective in mind.

# + how we approach the overview of the features space :
# we have a displayable number of features here so we can display the list and make our remarks on it.
# - Remark 1 : In a case where the features are in a non displayable number (e.g. thousands of fts), it is better to use lines of codes
# to "automatically" look for particularities among them (e.g. vars that are duplicates but named with slightly different names, etc.)

# + displaying the features space
list_fts0
# ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
# 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose',
# 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets',
# 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']

# - Remark 1 : no apparent redundancy observed in the features space.

# + Topology of the features space :
# > a group of 8 features called "Vital Signs" ('HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2')
# > a group of 26 features called "Laboratory values" ('BaseExcess' to 'Platelets')
# > a group of 6 features called "Demographics" ('Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS')
# - Remark 1 :
# At a first glance, all vars seem interesting to inquire about for the prediction of a medical condition...
# ... except the Unit1 and Unit2 as they are about identifiers in the patient management.
# We would like to predict the patient state without it depending on an ID but more so on
# physiological, clinical and any other medically observed value related to the patient body functions.
# - action : So we drop Unit1 and Unit2.
# NB : a deeper discussion can be had about whether or not the existence of these 2 IDs for the patient can help his management in the healthcare system
# hence the evolution of his state could depend on them. But for a first go at the task of early prediction of sepsis onset,
# we prefer to not go into such aspects and focus solely on body functions related values.

# + dropping columns
df1 = df0.copy() # at each major change in the df we are analysing, we make a copy of the df and keep going through with the changes on the copy (useful to see the changes done at anytime)
df1 = df1.drop(['Unit1', 'Unit2'], axis=1) # [790215 rows x 39 columns]

# + Remark 1 :
# With more and more understanding of the data as we go deeper in the EDA, cols will be dropped later on if they are deemed
# non informative, bias prone, or they are identified as not interesting to explore while answering our scientific question.

# + Remark 2 : Feature engineering
# A usual step found in EDA and that focuses also on columns that are features is the feature engineering step.
# It consists of analysing the features space and execute "transformations that could make it better" for learning tasks later.
# These "transformations that could make it better" can be :
# - reducing the features space size (to reduce high dimensionality) by combining a group of features into less features (e.g. by using PCA)
# - splitting a feature into more features to add more clarity in the information processed by learning tasks later
# (e.g. a feature "HairColor_EyeColor" can be split into the features "HairColor" and "EyeColor")
# This step of feature engineering wont be applied here as this is our first go at the dataset and we dont have yet
# a deep understanding of the features and how they interact in a prediction, to be able to seek combinations of some of them.
# Also the features are presently conveying a single information for each one of them so no need for splitting a feature for clarity.

print("# ---> initial sanitation of columns (2/2) : renaming columns")
# + Remark 1 : cols names are satisfying so no need to rename any of them.

print("# ---> initial sanitation of rows (1/2) : dropping duplicate rows")
# A dataset containing a high number of rows could potentially have duplicate rows.
# + counting the number of duplicate rows
df1_duplicate_rows = df1[df1.duplicated(keep='first')] # gets all duplicates ie existing copies of a first row (gets only the copies, the first occurrence is ignored)
df1_duplicate_rows.shape # (457, 39) ie 457 duplicates rows found
# + dropping the duplicate rows
df2 = df1.copy()
df2 = df2.drop_duplicates() # [789758 rows x 39 columns]

print("# ---> initial sanitation of rows (2/2) : renaming rows")
# + Remark 1 :
# we wont rename the rows here as the index they use is satisfying.
# But we will only reset the index because after removing duplicate rows, the existing index might be misleading
# + resetting the index of the rows (to have again a sequential index starting from 0 to 'index last row = num_rows-1')
df2 = df2.reset_index(drop=True)

print("# ---> displaying the dimensions of the df obtained after all columns and rows drops")
df2.shape # (789758, 39)

print("# ---> End of part !")
# ----------------------------------------------------------------------------------------------------------------------

print("# >>>>>> Part 4 : Data cleaning 2/3 : dealing with missing data.")

# ---> Mindset of cleaning the missing data :
# + There are 3 things we can do about missing data in order :
# - drop the column : case of a var containing too much missing data (e.g. >= 15%) hence useless or prone to outliers and/or unrealistic values if filled in
# - drop the row : case of a var to keep (e.g. to implicate it in the learning tasks later), and we have enough obs (rows) for the learning tasks hence dropping a row is not that taxing
# - fill in with some statistic values (e.g. mean, mode, or a value following a specific distribution) : case where the var has to be kept, and also the obs has to be kept

# + Course of action followed to better decide which of the previous actions to do :
# - we estimate the % of missing data in each var and rank the vars by this %
# - we group the vars by percentage of missing data exceeded :
# a) we choose 80%, 50% and 15% missing data as important thresholds and we try to find the group of vars that goes past each threshold
# b) a different grouping can also appear following certain breaks we observe while ranking the vars by % of missing data
# - we consider this rule of thumb to decide which actions to do to handle missing data (in general, not specific to this data)  :
# a) 15% is the threshold that when missing data is equals to or past it, we prefer to drop the var unless it is to be kept for a desired inquiry in later learning tasks
# b) if the var has to be kept despite having too much missing data, the rows with missing data for the var are dropped.
# c) in the case where the var has to be kept as well as its rows with missing data, we can fill in the missing data with mode
# (as it is the most frequently found value hence might be the value missing or the closest value to it)

# + The considerations that are specific to this dataset :
# We will still drop in priority the vars with too much missing data (>= 15%).
# But when we will want to keep a var with missing data, we will prefer to drop rows rather than fill in the missing data.
# This preference is due to the fact that we have a dataset of 100s of 1000s of rows so dropping rows is not that taxing
# compared to the risk of bias and potentially unrealistic values that filling in missing data can bring to the critical
# next step of training models for early prediction of sepsis onset.


print("# ---> estimate the % of missing data in each var :")
# + number of missing values by var
# from the df, we make df of wether true (1) or false(0) that the value in the cell is missing,
# then we get a table "index : colname, col 1/1: num missing values in colname",
# then we sort the table by descending values
total_mv_incol = df2.isnull().sum().sort_values(ascending=False)
# + percentage of missing values by var
# to get it,  we divide each value in the table "index : colname, col 1/1: num missing values in colname" by the number of rows in colname,
# then we sort the table by descending values
percent_mv_incol = (df2.isnull().sum()*100/df2.isnull().count()).sort_values(ascending=False)
# we concatenate the 2 previous tables into a table "index : colname, col 1/2: num missing values in that col, col 2/2: % missing values in that col"
missing_data = pd.concat([total_mv_incol, percent_mv_incol], axis=1, keys=["Total", "Percent"])
# lets visualize the first 20 lines of this table
missing_data.head(20)
# + Remark 1 :
# a majority of cols have more than 15% missing data.
# lets see how many vars go past each of the 80%, 50% and 15% thresholds...

print("# ---> group the vars by percentage of missing data exceeded :")
# + number of cols with at least 80 % missing data :
missing_data_al80 = missing_data[(missing_data.Percent >= 80)] # 27
missing_data_al80_share = (missing_data_al80.shape[0]) / (missing_data.shape[0])
missing_data_al80_perc = round((missing_data_al80_share*100), 2) # 69.23
print("Number of vars where % missing data >= 80 :", missing_data_al80.shape[0], "and this is approx", missing_data_al80_perc, "% of total vars")
# + number of cols with at least 50 % missing data :
missing_data_al50 = missing_data[(missing_data.Percent >= 50)] # 28
missing_data_al50_share = (missing_data_al50.shape[0]) / (missing_data.shape[0])
missing_data_al50_perc = round((missing_data_al50_share*100), 2) # 71.79
print("Number of vars where % missing data >= 50 :", missing_data_al50.shape[0], "and this is approx", missing_data_al50_perc, "% of total vars")
# + number of cols with at least 15 % missing data :
missing_data_al15 = missing_data[(missing_data.Percent >= 15)] # 30
missing_data_al15_share = (missing_data_al15.shape[0]) / (missing_data.shape[0])
missing_data_al15_perc = round((missing_data_al15_share*100), 2) # 76.92
print("Number of vars where % missing data >= 15 :", missing_data_al15.shape[0], "and this is approx", missing_data_al15_perc, "% of total vars")
# + Remark 1 :
# A large majority of the vars can be dropped if we respect the threshold of >= 15% missing data.
# But we remember that our features space has these 3 groups :
# - a group of 8 vars called "Vital Signs" ('HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2')
# - a group of 26 vars called "Laboratory values" ('BaseExcess' to 'Platelets')
# - a group of 4 (out of the initial 6) vars called "Demographics" ('Age', 'Gender', 'HospAdmTime', 'ICULOS') # 'Unit1', 'Unit2' have been dropped because they are deemed irrelevant to our inquiry

# So we can go through the vars, from the highest % of missing data to the lowest, and look at remarkable groups of vars,
# give the choice of missing data management (dropping those vars or not) and say the impacted group of vars in the feature space (on Vital Signs, Laboratory values or Demographics)
# This way we can know what remains of each group of vars in the feature space (Vital Signs, Laboratory values or Demographics)
# and what vars we would also like to keep, despite the amount of missing data in them, because they are interesting in regards to the scientific question

print("# ---> group the var by remarkable groups, give choices of missing data management and its impact on the features space :")
# + a group of vars missing all data :
missing_data_missall_grp = missing_data[(missing_data.Percent ==  100)] # 1
missall_grp_list = missing_data_missall_grp.index.values.tolist()
print("List of vars with % missing data == 100 has length", len(missall_grp_list), "and is", missall_grp_list)
# the list is ['EtCO2'] : the EtCO2 has 100% missing value so its useless as a var anyway, even if it is from group of vars "Vital signs")
# - action : drop the var 'EtCO2'
# - situation report :
# the group of vars "Vital Signs" will have 1 out of 8 vars dropped ('EtCO2'), and 7 out of 8 vars to remain
# the group of vars "Lab values" will have 26 out of 26 vars to remain
# the group of vars "Demographics" have 2 out 6 vars already dropped ('Unit1', 'Unit2') so 4 out of 4 vars to remain
# > vars with missing data to keep : None for now

# + a group largely missing data (85% to 100%) :
missing_data_largely_grp = missing_data[(missing_data.Percent > 85) & (missing_data.Percent < 100)] # 26
largely_grp_list = missing_data_largely_grp.index.values.tolist()
print("List of vars with % missing data > 85 and < 100 has length", len(largely_grp_list), "and is", largely_grp_list)
# the list is all the 26 lab values vars :
# - action : drop them all (ie not involving later, in the learning tasks, the lab values vars but having much more observations)
# NB : another option is to keep them (ie involving later, in the learning tasks, the lab values vars but remove the missing rows) but
# this has the risk of producing a dataset with a too low number of obs remaining
# (see the NB made on this aspect after this entire part of which vars to drop...)
# - situation report :
# the group of vars "Vital Signs" will have 1 out of 8 vars dropped ('EtCO2'), and 7 out of 8 vars to remain
# the group of vars "Lab values" will have 26 out of 26 vars dropped
# the group of vars "Demographics" have 2 out 6 vars already dropped ('Unit1', 'Unit2') so 4 out of 4 vars to remain
# > vars with missing data to keep : None for now

# + a group that equals or exceed the 15% of missing data threshold, up to a certain point (15% to 67%) :
missing_data_lessthanlargely_grp = missing_data[(missing_data.Percent >= 15) & (missing_data.Percent < 67)] # 3
lessthanlargely_grp_list = missing_data_lessthanlargely_grp.index.values.tolist()
print("List of vars with % missing data >= 15 and < 67 has length", len(lessthanlargely_grp_list), "and is", lessthanlargely_grp_list)
# the list is ['Temp', 'DBP', 'SBP'] :
# For Temp (66.% missing data), 2 choices exist :
# - choice 1 : drop it (ie not involving later, in the learning tasks, the Temp that is part of the group of vars "Vital Signs" and is widely known as reflecting in the physiology of the patient)
# - choice 2 : keep it (ie involving later, in the learning tasks, the Temp (and remove the missing rows in this var hence work with less obs))
# For DBP and SBP (resp. 48.09% and 15.16% missing data) :
# they seem related as they are both in the vital signs vars group and seem to be about blood pressure.
# A further inquiry in source A and source B (see below) shows that MAP = 1/3(SBP)+2/3(DBP) and that MAP is used more by
# health care staff because it is a better indicator of blood flow to organs than SBP and DBP.
# source A (article) : https://www.ahajournals.org/doi/10.1161/01.hyp.36.5.801?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%20%200pubmed
# source B (website) :  https://www.machinedesign.com/community/article/21837907/whats-the-difference-between-systolic-and-diastolic-blood-pressures
# > This means the information conveyed by SBP and DBP vars is found in MAP. Dropping vars that has their information
# # transmitted already through another var X and only keep the var X, can prevent multicollinearity during models building.
# Hence, we prefer to keep MAP and drop both SBP and DBP.
# > Also, an additional argument in favor of keeping MAP only is that, being preferred by health care staff, it will tend to be
# more available in populations data so it is more interesting to test it for predictions models rather than DBP or SBP.
# NB : if availability in populations can be judged by % of missing data, we see here that MAP has less missing data than DBP and SBP,
# all the more reasons to favor it
# - action : drop DBP and SBP
# - situation report :
# the group of vars "Vital Signs" will have 3 out of 8 vars dropped ('EtCO2','DBP','SBP'), the var Temp that can be kept or not, and 4 out of 8 vars to remain
# the group of vars "Lab values" will have 26 out of 26 vars dropped
# the group of vars "Demographics" have 2 out 6 vars already dropped ('Unit1', 'Unit2') so 4 out of 4 vars to remain
# > vars with missing data to keep : Temp if it is kept,

# + a group with lowest missing data (at least 1 missing data and < 15%) :
missing_data_lowest_grp = missing_data[(missing_data.Total >= 1) & (missing_data.Percent < 15)] # 5
lowest_grp_list = missing_data_lowest_grp.index.values.tolist()
print("List of vars with num missing data >= 1 and % missing data < 15 has length", len(lowest_grp_list), "and is", lowest_grp_list)
# the list is ['O2Sat', 'MAP', 'Resp', 'HR', 'HospAdmTime'] :
# its the remaining 4 vars left from the group of vars "Vital Signs" + 1 var (HospAdmTime) from the group of vars "Demographics"
# - action 1 : keep the 4 vars as they are vars that we would like to know how they participate in the prediction.
# - action 2 : 'HospAdmTime' (with 'ICULOS') might be essential to computing later on the utility metric so its extremely important to keep them both of them.
# For these 2 vars that are related to the length of stay of the patient, the attitude is to always keep them and drop the rows/obs with missing data.

# - situation report :
# the group of vars "Vital Signs" will have 3 out of 8 vars dropped ('EtCO2','DBP','SBP'), the var Temp that can be kept or not, and 4 out of 8 vars to remain
# the group of vars "Lab values" will have 26 out of 26 vars dropped
# the group of vars "Demographics" have 2 out 6 vars already dropped ('Unit1', 'Unit2') so 4 out of 4 vars to remain
# > vars with missing data to keep : Temp if it is kept, 'O2Sat', 'MAP', 'Resp', 'HR', 'HospAdmTime',

# - Remark 1 :
# the impact of keeping some vars that have missing data (without imputing data but instead choosing to remove rows/obs of missing data),
# is that we reduce the number of observations for the future learning tasks. Our attitude towards this choice is that, with the large
# amount of total samples we have, combined with, the low percent of missing values the vars to keep have, removing the observations (rows)
# with missing data on the vars to keep should leave us still with a number of obs to learn from.

# + a group with no missing data :
missing_data_nomiss_grp = missing_data[(missing_data.Percent == 0)] # 4
nomiss_grp_list = missing_data_nomiss_grp.index.values.tolist()
print("List of vars with no missing data has length", len(nomiss_grp_list), " and is ", nomiss_grp_list)
# the list is ['Gender', 'Age', 'ICULOS', 'SepsisLabel'] : its the other 3 vars from the group of vars "Demographics" + the dep var
# (in fact, these are the only vars among "Demographics" that we would like to see if they participate or not in the prediction, as the other 2 vars are about Ids)

# - situation report :
# the group of vars "Vital Signs" will have 3 out of 8 vars dropped ('EtCO2','DBP','SBP'), the var Temp that can be kept or not, and 4 out of 8 vars to remain
# the group of vars "Lab values" will have 26 out of 26 vars dropped
# the group of vars "Demographics" have 2 out 6 vars already dropped ('Unit1', 'Unit2') so 4 out of 4 vars to remain
# > vars with missing data to keep : Temp if it is kept, 'O2Sat', 'MAP', 'Resp', 'HR', 'HospAdmTime',

# + Remark about the dep var :
# the dep var is not missing values so its good news. if it had missing values, the action to take would be to remove those observations (rows)
# as we cant remove the dep var because we need it for the learning tasks and we would not fill it with imputed values that would create unwanted bias.

# + Summary of result of dealing with missing data :
# - we have the choice to keep or not the var Temp. This means our data cleaning can result in 2 different version of the dataset :
# a) a version df3 : this is our main version of the dataset to study
# b) a version df3_wTemp : this is the same dataset as df3, but with the var Temp in order to see its impact on the prediction. this version of the dataset has also less obs than df3.
# This version of the dataset is not "officially" an output of a strict EDA process but its a "curiosity based version of the official dataset".
# It is less representative of the larger population than the main version dataset df3 but will serve to initiate an answer to
# the question "If we could get consistently the temperature for all patients, would it help in the early prediction of sepsis onset ?"

# NB : an additional option is to make a version of the dataset where, in addition to keep the var Temp,
# we keep also the 26 vars in the group "Laboratory values" to see their importance in the prediction.
# But this has 2 issues :
# issue 1 : by working with vars with extreme levels of missing data while choosing to not impute data (hence we will have to remove a lot of rows),
# we are working with a sample of the larger population that may be too reduced to allow all the splitting needed for the learning tasks,
# but also it is far from being representative of the population.
# issue 2 : Also, in terms of time management for this project, adding another version of the dataset can be detrimental, knowing all the learnings tasks will also be ran for this version.
# This is why we choose to expose the idea of this possible 3rd version of the dataset, what it could serve to research, what could be its issues and why we cant do it in time.

print("# ---> dealing with missing data :")

# + Summary of actions to deal with missing data :
# - making df2_aft_initial_cleaning (obtained after all the cleaning steps that apply to any version of the dataset that will be output)
# df2_aft_initial_cleaning = df2 with ['EtCO2', 'DBP', 'SBP'] cols dropped, ['O2Sat', 'MAP', 'Resp', 'HR', 'HospAdmTime'] rows with missing data dropped
# - making df3
# df3 = df2_aft_initial_cleaning, 26 lab values vars dropped, ['Temp'] col dropped
# - making df3_wTemp
# df3_wTemp = df2_aft_initial_cleaning, 26 lab values vars dropped, ['Temp'] rows with missing data dropped

# + making df2_aft_initial_cleaning
df2_aft_initial_cleaning = df2.copy() # [789758 rows x 39 columns]
df2_aft_initial_cleaning = df2_aft_initial_cleaning.drop(['EtCO2', 'DBP', 'SBP'], axis=1) # [789758 rows x 36 columns]
df2_aft_initial_cleaning = df2_aft_initial_cleaning.dropna(subset=['O2Sat', 'MAP', 'Resp', 'HR', 'HospAdmTime']) # [661017 rows x 36 columns] ie (128741 rows dropped)
# + making the list of 26 lab values vars to drop
# > largely_grp_list contained all the vars with % missing data > 85 and < 100 and corresponded exactly to the list of the 26 lab values vars
list_labvalues_vars_todrop = largely_grp_list
# + making df3
df3 = df2_aft_initial_cleaning.copy() # [661017 rows x 36 columns]
df3 = df3.drop(list_labvalues_vars_todrop, axis=1) # [661017 rows x 10 columns]
df3 = df3.drop(['Temp'], axis=1) # [661017 rows x 9 columns]
df3.isnull().sum().max() # check to see if no missing data left (output is 0 if no missing data left)
# + making df3_wTemp
df3_wTemp = df2_aft_initial_cleaning.copy() # [661017 rows x 36 columns]
df3_wTemp = df3_wTemp.drop(list_labvalues_vars_todrop, axis=1) # [661017 rows x 10 columns]
df3_wTemp = df3_wTemp.dropna(subset=['Temp']) # [250104 rows x 10 columns] ie (410913 rows dropped)
df3_wTemp.isnull().sum().max()

# # + just out of curiosity, we try to see how many samples would remain on the version of the dataset with Temp as well as the 26 vars of the group Lab values...
# df_wLabVals_wTemp = df2_aft_initial_cleaning.copy() # [661017 rows x 36 columns]
# df_wLabVals_wTemp = df2_aft_initial_cleaning.dropna(subset=['Temp']) # [250104 rows x 36 columns]
# df_wLabVals_wTemp = df_wLabVals_wTemp.dropna(subset=list_labvalues_vars_todrop) # [1 rows x 36 columns]
# # NB : df_wLabVals_wTemp only has 1 observations so even if pursued, this dataset could not serve to experiment with a learning task on it...

# + resetting the index of the rows for the versions of the dataset that we will go on with
df3 = df3.reset_index(drop=True)
df3_wTemp = df3_wTemp.reset_index(drop=True)

print("# ---> End of part !")
# ----------------------------------------------------------------------------------------------------------------------

print("# >>>>>> Part 5 : Data cleaning 3/3 : handling outliers.")

# ---> Mindset of handling outliers :

# + Definition of outliers :
# Outliers are data points that are different from others data points by the fact that they are not following the general trend of the data.
# They are observed as much higher or lower values in the group of data points.
# They have to be detected and removed because their presence in the data, while learning tasks are creating a model from the data,
# results in a model that is using information that is less likely to happen in the population
# This can be summarized as "outliers create unwanted bias in prediction models and results in less accurate models"

# + The chosen approach to detect and remove outliers :
# Outliers can be detected on visualizations like boxplots and for the case of boxplots, they are shown as data points outside of the box and beyond the whiskers.
# But we want a more automatic/strict mean of detecting outliers and not rely on human observation that can be faulty.
# The visual observation with boxplots (points beyond the whiskers) is formalized by the following definition of an outlier :
# "an outlier Ot is such as Ot < Q1 - 1.5*IQR or Ot > Q3 + 1.5*IQR, where Q1 and Q3 are the 25th and 75th percentiles, and IQR = Interquartile range = Q3 - Q1"
# - action : we use this mathematical rule to find, for each feature, all the rows that correspond to this definition and we drop them
# - Remark : this is a first removal of outliers. To better "polish" the data, it could be ran more times on certain fts (all or some) until each feature has no value looking like an outlier.
# But we choose to only run the first removal because the "true" outliers are the ones detected and removed by this first removal.
# The others "maybe outliers" values (removed with additional runs of the rule) can be important information for the learning models so for the moment we keep them.

# + Steps preceding outliers removal :
# Outliers, whether visually or mathematically detected, are based on values. So the values in a dataset have to be set up
# in all the ways they are considered proper before outliers removal. Doing this will consist of preliminary steps that mainly consist of :
# - addressing the need for expansion or restriction of the dataset (make sure the dataset containing all the useful classes for the prediction and is only limited to those)
# - handling the units of the values across the vars inside the dataset (more coherent units means more coherent values across the board)
# - formatting the type of values in each column (encoding the categorical vars and scaling the fts)

print("# ---> Handling outliers 1/2 : Formatting the data before handling outliers")
print("# + addressing the need for expansion and/or restriction of the dataset")

# The "big picture" objective of this work is to develop models for early prediction of sepsis onset, effectively 6 hours before for each sample,
# a sample being an entry of patients measurables/info at a precise hour, one patient having multiple samples.

# - In order to do that, the models building in this work will contain 2 parts :
# > a 1st initial part of proper models building where we develop "discriminative models" (ie models that dont just focus on predicting effectively
# one of the classes but that focus on differentiating all the classes involved and equally penalizing false predictions as well as equally rewarding true predictions, on each class)
# > a 2nd part of seeking what is the performance of the best models developed in the 1st part,
# when we use the original patients data and we estimate the performance using the utility metric of the data challenge

# - For the "discriminative models" of the 1st part, an important requirement is that the data after being curated should :

# > contain all the clusters needed for the early prediction of sepsis onset :
# if all the clusters needed are not present, its important to point out a need to expand the dataset in order to get the occurrences of the lacking clusters,
# ie to get the different cases of what a positive and a negative observation "could" look like
# so that the learning tasks can effectively heighten their ability to differentiate both cases.

# > be restricted to clusters needed for the early prediction of sepsis onset :
# if others clusters than the ones needed are present, its important to remove them of the dataset so that the learning tasks
# can only be limited to learning operations differenciating the cases of positive and negative observation.

# - What should the needed clusters look like in our dataset ?

# > "Cluster A" is consisting of patients entries that has been medically observed as "has developed sepsis onset in the timeframe satisfyingly considered "early".
# This cluster is built by focusing on the patients that have had, at some point during their stay, a sepsis onset and
# pronouncing their entries as "positive" 6 hours before (ie for each entry that is medically observed as positive for sepsis,
# all entries since the 6th entry before are marked as 1).

# > this result in also having "Cluster B" consisting of entries marked 0 because, even if they are for a patient that at some point will have sepsis,
# they are before the 6 hours window before the first entry where sepsis has been medically observed.
# NB1 : Clusters A is giving to the learning tasks a sense of "what could an observation that is positive to sepsis development and also positive to the early aspect of the prediction, look like"
# and Cluster B is giving to the learning tasks a sense of "what could an observation that is positive to sepsis development but negative to the early aspect of the prediction, look like".

# > Cluster C is consisting of patients entries that has been medically observed as "never have developed sepsis during the stay".
# This cluster is focusing on patients that dont have a record of being positive to sepsis during their stay (they are marked 0)
# NB2 : Cluster C is giving to the learning tasks a sense of "what could an observation that is negative to sepsis development look like".

# > NB3 : logically, it should exist a Cluster D that would consist of "patients entries where sepsis has been observed but late" and would be marked 0...
# This cluster D is not obtained because the early criteria of the prediction (marking the entry as 1 only if positive to the early observation of sepsis) is done for all entries.
# The result is that, each patient that at some point have sepsis, will be marked either 0 for too early entries or 1 if the entry is in the satisfying time window of "earlyness".
# Hence no room for setting some entries as "sepsis has been medically observed but it was done late".

# NB4 : Another way here, could be to get a Cluster D by arbitrarily choosing a timeframe to mark as late some entries of sepsis patients (e.g. 3 hours after sepsis is medically observed).
# But the entries that will be marked as late (SepsisLabel is 0 instead of 1) will be less useful to the prediction than in their present state.
# This is because, the early prediction of sepsis onset require 2 information : information on prediction of sepsis onset (early or not) and information on being early observed sepsis;
# and the information on prediction of sepsis onset (early or not) is reduced when we decide to mark as 0 some of the last entries for a patients that has sepsis,
# ie we are missing out of useful information for what is a part of the predictive problem.

# - Looking closely at our datasets, what clusters do we have ? Do we have to think about expanding the dataset or applying a restriction on it ?
# > when we examine the dep var SepsisLabel, we have the class 1 as our Cluster A and the class 0 is the union of our Clusters B and C.
# (see the definition of SepsisLabel that is "For sepsis patients, SepsisLabel is 1 if t≥tsepsis−6 and 0 if t<tsepsis−6. For non-sepsis patients, SepsisLabel is 0.".
# source : https://physionet.org/content/challenge-2019/1.0.0/
# the dataset is also limited to only those clusters.
# > Conclusions on need for expansion and/or restriction of the dataset :
# 1- the dataset has all the needed clusters for the task of the early prediction of sepsis onset so no need to expand our pool of data to make the learning tasks possible.
# 2- also, the dataset dooes not include any others clusters that is not related to being positive or negative to the early prediction of sepsis so no need for restriction of the dataset.

print("# + handling the units of the vars")
# Among features, some vars can describe a certain space (e.g. here, the vars that are physiological metrics describe the space that is the patient body functions).
# For a specific space, it is important to push as best as we can for using one unit to refer to a physical, chemical or conceptual trait (e.g. time, fraction, pressure, etc.)
# An example is : for the patient body functions space, do our best to put all time units in minutes
# This way, the nuances between the values of the vars in the space are more appearing and the values are more in line even if they are estimates of different traits.

# - Remark 1 : in the features space at its present state, we recognize 2 spaces described by a group of vars each :  the group of vars for "Vital signs" and the group of vars for "Demographics".
# For the space "Vital signs", all vars use the same unit for each physical, chemical or conceptual trait (minutes for time, % for fractions, mmHg for pressure, etc.)
# For the space "Demographics", this is also already done (e.g. hours for time)
# NB : it can be argued that time units across the 2 spaces are different (minutes for the space "Vital signs" and hours for the space "Demographics").
# But this is acceptable as it is 2 different spaces described.

print("# + formatting the type of values in each column 1/3 : checking the present state of the vars")
# Among our cols, we can find cols with categorical values and cols with continuous values.
# - Present state of the vars
# NB : knowing that df3 and df3_wTemp are only different by df3_wTemp having Temp as an additional var, we only display the vars types of df3_wTemp
# df3.dtypes # not needed
df3_wTemp.dtypes
# """
# HR             float64
# O2Sat          float64
# Temp           float64
# MAP            float64
# Resp           float64
# Age            float64
# Gender         float64
# HospAdmTime    float64
# ICULOS         float64
# SepsisLabel    float64
# dtype: object
# """
# + Remark 1 :
# - All the vars are in the type float. So no changes for the ones that are continuous vars.
# - But, Gender and the dep var have categorical values while they are in the type float. So we have to encode them to
# have int type (they will still be encoded as 0 and 1)

# + lets check if our categorical vars have the only levels announced
# for df3 (we check only for df3 as df3_wTemp is a subset of df3 when it comes to our categorical vars)
print("Gender has", len(df3["Gender"].unique()), "levels and they are", list(df3["Gender"].unique()))
print(dep_var,"has", len(df3[dep_var].unique()), "levels and they are", list(df3[dep_var].unique()))
# only 2 levels (0 and 1) have been observed for Gender and the dep var, as announced !

print("# + formatting the type of values in each column 2/3 : categorical values encoding")
# NB : as a good practice, LabelEncoder() should only be used for the dep var (and OneHotEncoder() for the categorical feature).
# But, OneHotEncoder() expands the number of columns and we dont want that, so we use LabelEncoder() also for Gender. The objective here is to just get categories values as Ints.
le = LabelEncoder()
# - for the dep var, in df3 and df3_wTemp
y_encoded_df3 = le.fit_transform(df3[dep_var]) # for df3
df3[dep_var] = y_encoded_df3
y_encoded_df3_wTemp = le.fit_transform(df3_wTemp[dep_var]) # df3_wTemp
df3_wTemp[dep_var] = y_encoded_df3_wTemp
# - for the categorical var, in df3 and df3_wTemp
gender_encoded_df3 = le.fit_transform(df3['Gender']) # for df3
df3['Gender'] = gender_encoded_df3
gender_encoded_df3_wTemp = le.fit_transform(df3_wTemp['Gender']) # df3_wTemp
df3_wTemp['Gender'] = gender_encoded_df3_wTemp

# + lets check again the types of the values in each column
# for df3
df3.dtypes
# """
# HR             float64
# O2Sat          float64
# MAP            float64
# Resp           float64
# Age            float64
# Gender           int64
# HospAdmTime    float64
# ICULOS         float64
# SepsisLabel      int64
# dtype: object
# """
# for df3_wTemp
df3_wTemp.dtypes
# """
# HR             float64
# O2Sat          float64
# Temp           float64
# MAP            float64
# Resp           float64
# Age            float64
# Gender           int64
# HospAdmTime    float64
# ICULOS         float64
# SepsisLabel      int64
# dtype: object
# """

# + Considerations of good practice before outliers removal :
# - Remark 1 :
# when handling outliers, we consider that a var with only 2 categories does not have outliers in its data.
# This is the case for our categorical vars Gender and the dep var.
# So we apply the outliers removal to all vars except these two (ie only to the continuous vars).

# - Remark 2 :
# for outliers removal in the continuous vars, we have to establish a threshold that defines an observation as an outlier.
# So, for the values of the data points and of the threshold to be more in context, we need to standardize the data before even computing any threshold.
# Standardizing the data, means here, transforming the data to have a mean of 0 and a std dev of 1.
# NB : outside of proper detection of outliers needs, this is good practice as by doing this, we are also putting the data
# into the proper format for most learning methods that require for the data to be standardized first.

print("# + formatting the type of values in each column 3/3 : standardizing the data (for features)")
# NB : we keep a copy of each dataset that will be kept unscaled for some visualizations later
df3_unscaled = df3.copy() # [661017 rows x 9 columns]
df3_wTemp_unscaled = df3_wTemp.copy() # [250104 rows x 10 columns]
#...also we change the categories names in each of the categorical features into adequate names (for the 2 cat vars)
# > for the Gender
df3_unscaled['Gender'] = df3_unscaled['Gender'].map({0: "Female", 1: "Male"})
df3_wTemp_unscaled['Gender'] = df3_wTemp_unscaled['Gender'].map({0: "Female", 1: "Male"})
# > for the dep var
# NB : the dep var is left in the levels 0 and 1 as this variable is not about having sepsis or not but more so about
# "having been suspected of sepsis early and not too soon enough = 1 or being anything but that = 0 "
# so (0, 1) summarize this idea better, hence they are not replaced with something like ("dont have sepsis", "has sepsis").
# But, for some statistical tests used in adding annotations to our figures to work, the levels need to be strings.
# So here, we keep the levels but we change them into strings.
df3_unscaled['SepsisLabel'] = df3_unscaled['SepsisLabel'].astype('string')
df3_wTemp_unscaled['SepsisLabel'] = df3_wTemp_unscaled['SepsisLabel'].astype('string')


# - use a function to scale the features
# (for the scaling, a function defined in a separate module is used. The scaler used in the function is defined in that module and is StandardScaler() from sklearn)
# NB : fts_scaling(dftoscale, dep_var_to_ignore) is a function that given a df and the dep var to not scale, will output a copy of the same df but with all features scaled
# to test, dftoscale = df3_unscaled.copy() and dep_var_to_ignore = dep_var

# output the scaled datasets
df3_scaled = fts_scaling(df3, dep_var) # [661017 rows x 9 columns]
df3_wTemp_scaled = fts_scaling(df3_wTemp, dep_var) # [250104 rows x 10 columns]


print("# ---> Handling outliers 2/2 : the actual (first) removal of outliers")
# + making copies of our datasets to apply changes on them
# - for unscaled dfs
df4_unscaled = df3_unscaled.copy() # [661017 rows x 9 columns]
df4_wTemp_unscaled = df3_wTemp_unscaled.copy() # [250104 rows x 10 columns]
# - for scaled dfs
df4 = df3_scaled.copy() # [661017 rows x 9 columns]
df4_wTemp = df3_wTemp_scaled.copy() # [250104 rows x 10 columns]

# + detecting outliers
# - a list of the vars to not apply the outliers removal on
list_cols_ignored_in_outliers_rm = ["Gender", dep_var] # this list is common to all our versions of the dataset

# - use a function to make the first removal of outliers remove and return a version of the dataset without the outliers
# NB : first_removal_outliers(dfin, dfin_unsc, list_cols_ignored) is a function for all the workings in order to remove outliers in each concerned col
# to test, dfin = df4 and dfin_unsc = df4_unscaled and list_cols_ignored = list_cols_ignored_in_outliers_rm

# for df4
df4_list_of_versions_after_outliers_rm = first_removal_outliers(df4, df4_unscaled, list_cols_ignored_in_outliers_rm)
df4 = df4_list_of_versions_after_outliers_rm[0] # [512140 rows x 9 columns]
df4_unscaled = df4_list_of_versions_after_outliers_rm[1] # [512140 rows x 9 columns]
# """
# - Detecting outliers for the var HR and this is var 1 out of 7
#     num total obs is 661017 and num obs to remove is 6034 ie num obs to remain is 654983
# - Detecting outliers for the var O2Sat and this is var 2 out of 7
#     num total obs is 661017 and num obs to remove is 10347 ie num obs to remain is 650670
# - Detecting outliers for the var MAP and this is var 3 out of 7
#     num total obs is 661017 and num obs to remove is 11158 ie num obs to remain is 649859
# - Detecting outliers for the var Resp and this is var 4 out of 7
#     num total obs is 661017 and num obs to remove is 11724 ie num obs to remain is 649293
# - Detecting outliers for the var Age and this is var 5 out of 7
#     num total obs is 661017 and num obs to remove is 2380 ie num obs to remain is 658637
# - Detecting outliers for the var HospAdmTime and this is var 6 out of 7
#     num total obs is 661017 and num obs to remove is 92810 ie num obs to remain is 568207
# - Detecting outliers for the var ICULOS and this is var 7 out of 7
#     num total obs is 661017 and num obs to remove is 30491 ie num obs to remain is 630526
# ------Removing outliers------
# - Removed a total of outliers of 148877 out of 661017 obs ie 22.52 %
# - Remaining num total obs is 512140 out of 661017 obs ie 77.48 %
# """

# for df4_wTemp
df4_wTemp_list_of_versions_after_outliers_rm = first_removal_outliers(df4_wTemp, df4_wTemp_unscaled, list_cols_ignored_in_outliers_rm)
df4_wTemp = df4_wTemp_list_of_versions_after_outliers_rm[0] # [192271 rows x 10 columns]
df4_wTemp_unscaled = df4_wTemp_list_of_versions_after_outliers_rm[1] # [192271 rows x 10 columns]
# """
# - Detecting outliers for the var HR and this is var 1 out of 8
#     num total obs is 250104 and num obs to remove is 3285 ie num obs to remain is 246819
# - Detecting outliers for the var O2Sat and this is var 2 out of 8
#     num total obs is 250104 and num obs to remove is 2270 ie num obs to remain is 247834
# - Detecting outliers for the var Temp and this is var 3 out of 8
#     num total obs is 250104 and num obs to remove is 3561 ie num obs to remain is 246543
# - Detecting outliers for the var MAP and this is var 4 out of 8
#     num total obs is 250104 and num obs to remove is 4702 ie num obs to remain is 245402
# - Detecting outliers for the var Resp and this is var 5 out of 8
#     num total obs is 250104 and num obs to remove is 4297 ie num obs to remain is 245807
# - Detecting outliers for the var Age and this is var 6 out of 8
#     num total obs is 250104 and num obs to remove is 2951 ie num obs to remain is 247153
# - Detecting outliers for the var HospAdmTime and this is var 7 out of 8
#     num total obs is 250104 and num obs to remove is 32900 ie num obs to remain is 217204
# - Detecting outliers for the var ICULOS and this is var 8 out of 8
#     num total obs is 250104 and num obs to remove is 11051 ie num obs to remain is 239053
# ------Removing outliers------
# - Removed a total of outliers of 57833 out of 250104 obs ie 23.12 %
# - Remaining num total obs is 192271 out of 250104 obs ie 76.88 %
# """

print("# ---> End of part !")
# ----------------------------------------------------------------------------------------------------------------------

print("# >>>>>> Part 6 : Descriptive statistics.")

# ---> Mindset in this part :
# our "big picture" objective is to make good datasets that will be used for early prediction models of sepsis onset.
# So, we focus on the tasks that goes towards checking that every aspect of the datasets is polished.
# Most of the remaining tasks of discussing features and making hypothesis on how features must probably interact with each other or vary together, will be kept at a minimum.
# Although, for demonstrative purposes, we will show the commands of the some of tasks that we dont go through properly.

# ---> Use of descriptive statistics :
# + For each var, we can get :
# - the count and proportion of each level in the column by respectively using df.colname.value_counts() and df.colname.value_counts(normalize=True)
# (more directed towards categorical features and the dep var)
# - the basic statistics (mean, median, etc.) obtained by using df[colname].describe(). They are mainly used to guess skewness and kurtosis on a var.
# NB1 : skewness
# for an asymmetric distribution, it is a deviation from the normal distribution.
# A deviation with most values on the left and a tail at the right is a positive skewness or right-skewed (Mode > Median > Mean)
# A deviation with most values on the right and a tail at the left is a negative skewness or left-skewed (Mean < Median < Mode)
# (And a normal distribution have Mean = Median = Mode)
# skewness can be computed using df[colname].skew()
# NB2 : kurtosis
# it is the peakness of the distribution. It is a manifestation of outliers presence.
# kurtosis can be estimated using df[colname].kurt()

# ---> Our strategy of using the descriptive statistics :
# + Choice 1 : Imbalance of the datasets
# Only for the dep var, we will display counts and proportions of the 2 classes (0 and 1) and this is to estimate how imbalanced our versions of the dataset are.

# - Use a function to estimate the imbalance level
# NB : imbalance_summary(dftocheck, colnametocheck) is a function to compute all the workings of imbalance and return the estimate of the imbalance level
# # to test, use dftocheck = df4 and colnametocheck = dep_var

print("# ---> Imbalance summary based on the dependent variable")
# - imbalance summary for df4
imbalance_summary(df4,dep_var)
# """
# - Num obs : class 0 has 504343 and class 1 has 7797
# - % obs in class : class 0 accounts for 98.48 and class 1 accounts for 1.52
# - ie a ratio classSup over classInf of :  65.0
# """
# - imbalance summary for df4_wTemp
imbalance_summary(df4_wTemp,dep_var)
# """
# - Num obs : class 0 has 189850 and class 1 has 2483
# - % obs in class : class 0 accounts for 98.71 and class 1 accounts for 1.29
# - ie a ratio classSup over classInf of :  77.0
# """
# - Remark 1 :
# both datasets show a strong imbalance, with the size of the class 0 being at least 65 times the size of the class 1, that we are trying to predict.
# This suggests favoring later, learning methods that manage well imbalance in the data.

# + Choice 2 :
# For skewness, we prefer to observe it on the visualisations later.
# For kurtosis, it is mainly used as an indicator of presence of outliers, so we dont need to examined it again as we already have applied the first removal of outliers
# ie "the acceptable level of outliers to remove for the moment without sacrificing potentially useful information for the prediction".
# Although, if we remark outliers on the visualisations, they could be pointed out but we will not remove them as they passed the first removal.

print("# ---> End of part !")
# ----------------------------------------------------------------------------------------------------------------------

print("# >>>>>> Part 7 : Univariate, bivariate and multivariate analysis.")

# ---> Mindset of univariate, bivariate and multivariate analysis :
# Univariate, bivariate, and multivariate analysis are in genral done step by step in order to :

# 1) for univariate analysis :
# + have a view of the distribution of one var
# - This is done using :
# > histograms (for continuous vars),
# > barplots (for categorical vars).

# 2) for bivariate analysis :
# + know what is the trend when a couple of vars are plotted one against another
# - This serves mainly :
# > to bring out initial insights from pairwise trends between vars,
# > to detect outliers
# - This is done using :
# > scatter plots (for a pair of numerical vars),
# > box-plots (when at least one of the vars is a categorical var).

# 3) for multivariate analysis :
# a) + offer the same insights as bivariate analysis but for multiple pairs of vars at the same time
# - This is done using :
# > pairplots
# (a pairplot is a display of multiple visualisations at the same time, each one an appropriate visualisation for the pair of vars
# ie a scatter plot for a pair of different vars and a histogram/bar plot for a pair of the same vars)
# b) + observe which vars are correlated
# - This is done using :
# > heatmaps exploiting correlation matrices
# (correlation matrices contain pairwise correlations such that "the higher the correlation value in absolute value, the higher the correlation between 2 vars",
# and heatmaps plots the pairwise correlations in terms of hues, varying in intensity with the absolute value of the correlation)

# + NB : Insights can already be obtained from visualisations before doing the learning tasks
# The visualisations of univariate, bivariate and multivariate analysis can give insights, even before doing the learning tasks, such as :
# a) how a var is correlated with the dep var and after learning tasks, we can see if models selected the vars previously seen as correlated to the dep var
# b) how some vars have pairwise trends that shows a relationship between them.
# These can either be reported as insights of a study without learning tasks, or be pointed out again after the learning
# tasks feature selections are obtained to put more attention on certain features.


# ---> Our strategy in univariate, bivariate and multivariate analysis:

# + Choice 1 :
# Additional outliers can be observed in the visualisations of univariate, bivariate and multivariate analysis.
# But, we already made a first removal of outliers and this is a preprocessing of the data for a first go at the learning tasks,
# so we consider any additional outliers detected as a particularity of the data that can add information to the model and not an outlier to remove presently.
# If this was a second go at the data following initial results of learning tasks, we could "polish even more" the data
# by removing additional outliers remaining after the first removal.
# > So our choice for now is that, even if we can look for and detect outliers on the visualisations, we wont do it.

# + Choice 2 :
# Here, our "big picture" objective is to properly format the data in order to make models that are capable of early prediction of sepsis onset.
# Hence, the objective is making correct data for the models. Whatever previous insight obtained can make our study richer but does not replace the production of the models.
# > So, for time management purposes, we will not extensively search the visualisations of univariate, bivariate and multivariate analysis
# for the full range of insights on the vars. Instead, just for demonstration purposes, we will give an example of
# each visualisation commonly done in this part and give also example of insights.

# + A summary of the versions of the dataset produced :
# From the EDA of the dataset, we ended up with 2 versions of the dataset to use for the learning tasks later:
# a) one version called df4, where the Temp var is dropped because it had more than 15% of missing data,
# b) one version called df4_wTemp, where the Temp var has been kept and his rows with missing data dropped
# Thus, there is more data points in the version without Temp (df4) than the version with Temp (df4_wTemp).

# + Choice 3 :
# > For our visualisations that are mostly for demonstrative purposes, the visualisation will be done for both versions of the dataset only if it can implicate the var Temp
# (ie a visualisation, using only vars that are common to both versions of the dataset, will only be done for the version without Temp)
# This is a better time management and another advantage is that we only display visualisations that uses data with the most data points

# + Choice 4 :
# > Also, each version of the dataset has a copy containing unscaled values (df4_unscaled and df4_wTemp_unscaled).
# When a plot needs to show values before scaling (e.g. the Gender categories), this pre-scaling version of the dataset will be used.



print("# ---> Univariate analysis")
# (for the distribution of a continuous var or the size of each category of a categorical var)

# + Case 1 : count of each category of a categorical var (a bar plot is used)
# - for the dep var
# for df4
# NB : sns.set() # to get seaborn default blue background theme with white line to follow more easily the y values levels
sns.set()
var = dep_var
counts_by_cat = df4[var].value_counts()
plt.figure(figsize=(7,5))
graph = sns.barplot(counts_by_cat.index, counts_by_cat.values, alpha=0.8)
plt.title("Number of patients for each category of SepsisLabel (dtsetA, var Temp dropped)")
plt.ylabel('Count', fontsize=12)
plt.xlabel('SepsisLabel', fontsize=12)
for p in graph.patches:
        graph.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()),
                       ha='center', va='center', color='black', xytext = (25, 5), textcoords = 'offset points')
fullname_file_of_plot = output_directory+"/"+"dtsetA_STEP2_EDA_UA_Case1_1CatVar_theDepVar.png"
plt.savefig(fullname_file_of_plot, bbox_inches='tight')
plt.show()
plt.close()

# - for var Gender
# for df4 (using df4_unscaled to display adequate names for categories)
sns.set()
var = "Gender"
counts_by_cat = df4_unscaled[var].value_counts()
plt.figure(figsize=(7,5))
graph = sns.barplot(counts_by_cat.index, counts_by_cat.values, alpha=0.8)
plt.title("Number of patients for each category of Gender (dtsetA, var Temp dropped)")
plt.ylabel('Count', fontsize=12)
plt.xlabel('Gender', fontsize=12)
for p in graph.patches:
        graph.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()),
                       ha='center', va='center', color='black', xytext = (25, 5), textcoords = 'offset points')
fullname_file_of_plot = output_directory+"/"+"dtsetA_STEP2_EDA_UA_Case1_1CatVar_Gender.png"
plt.savefig(fullname_file_of_plot, bbox_inches='tight')
plt.show()
plt.close()

# + Case 2 : the distribution of the counts of a continous var (a histogram is used)
# - for var HR
# for df4
sns.set()
var = "HR"
graph = sns.displot(df4[var], kde=True)
plt.title("Distribution of the HR variable (dtsetA, var Temp dropped)")
plt.ylabel('Count', fontsize=12)
plt.xlabel('HR', fontsize=12)
fullname_file_of_plot = output_directory+"/"+"dtsetA_STEP2_EDA_UA_Case2_1ContVar_HR.png"
plt.savefig(fullname_file_of_plot, bbox_inches='tight')
plt.show()
plt.close()
# - for var Temp (to have a look at the var Temp in particular)
# (done because studying impact of Temp was the reason to make df4_wTemp)
sns.set()
var = "Temp"
graph = sns.displot(df4_wTemp[var], kde=True)
plt.title("Distribution of the Temp variable (dtsetA, var Temp kept)")
plt.ylabel('Count', fontsize=12)
plt.xlabel('Temp', fontsize=12)
fullname_file_of_plot = output_directory+"/"+"dtsetA_STEP2_EDA_UA_Case2_1ContVar_Temp.png"
plt.savefig(fullname_file_of_plot, bbox_inches='tight')
plt.show()
plt.close()


print("# ---> Bivariate analysis")
# (for how 2 vars correspond ie a pairwise trend of the population)

# + Case 1 : for 2 continuous vars (a scatter plot is used)
# for df4
sns.set()
var1 = "O2Sat"
var2 = "Resp"
data = pd.concat([df4[var1], df4[var2]], axis=1)
graph = data.plot.scatter(x=var1, y=var2)
plt.title("Scatter plot of the variables O2Sat and Resp (dtsetA, var Temp dropped)")
plt.ylabel('O2Sat', fontsize=12)
plt.xlabel('Resp', fontsize=12)
fullname_file_of_plot = output_directory+"/"+"dtsetA_STEP2_EDA_BA_Case1_2ContVars_O2Sat_Resp.png"
plt.savefig(fullname_file_of_plot, bbox_inches='tight')
plt.show()
plt.close()
# for df4_unscaled
# (to see if the scaling has made some points to be obvious as outliers)
sns.set()
var1 = "O2Sat"
var2 = "Resp"
data = pd.concat([df4_unscaled[var1], df4_unscaled[var2]], axis=1)
graph = data.plot.scatter(x=var1, y=var2)
plt.title("Scatter plot of the variables O2Sat and Resp (dtsetA, var Temp dropped)")
plt.ylabel('O2Sat (%)', fontsize=12)
plt.xlabel('Resp (breaths per minute)', fontsize=12)
fullname_file_of_plot = output_directory+"/"+"dtsetA_STEP2_EDA_BA_Case1_2ContVars_O2Sat_Resp_unscaled_values.png"
plt.savefig(fullname_file_of_plot, bbox_inches='tight')
plt.show()
plt.close()
# no obvious outliers observed even if we compare with the values before scaling

# + Case 2 : for 1 continuous var and 1 categorical var (a boxplot is used)
# NB1 :
# the cat var is put on the x-axis
# NB2 :
# to test if 2 groups are significantly different, we use the non-parametric Mann-Whitney-Wilcoxon (or rank sum) test two-sided with Bonferroni correction.
# > The Mann-Whitney-Wilcoxon test is to test the null hypothesis that the means of two groups are the same, hence under this null hypothesis,
# the distribution of the variable in question is identical in the two groups.
# Therefore the variable does not differentiate the 2 groups.
# The non-parametric aspect means the test does not need to assume that the 2 distributions tested are normal.
# Instead, the test uses the ranks of the values.
# > The results of the test are read in this way :
# "a p-value and U_stat are outputs;  if the p-value < 0.05, th 2 groups are different (ie null hypothesis rejected)"
# > NB : the Mann-Whitney-Wilcoxon is usually done with a Bonferroni correction.
# > The Bonferroni correction is used when multiple comparisons are being made, and it is an adjustment of the significance level of a statistical test (alpha)
# in order to protect against Type I errors. In fact, when several dependent or independent statistical tests are being performed simultaneously,
# a given alpha value may be appropriate for each individual comparison, but it might not be for the set of all comparisons.
# So, in order to avoid a lot of spurious positives (the Type I errors ie mistaken rejection of an actually true null hypothesis also known as a "false positive"),
# the alpha value needs to be lowered to account for the number of comparisons being performed.
# The simplest and most conservative approach to do this is done by the Bonferroni correction, which sets the alpha value
# for the entire set of n comparisons equal to alpha by taking the alpha value for each comparison equal to alpha/n.
# For example,  if 10 tests need to be performed on a set of scores,
# we can use a Bonferroni-corrected significance level of .05/10 = .005 instead of the conventional .05.
# > source Mann-Whitney-Wilcoxon test 1 : https://thestatsgeek.com/2014/04/12/is-the-wilcoxon-mann-whitney-test-a-good-non-parametric-alternative-to-the-t-test/
# source Mann-Whitney-Wilcoxon test 1 : http://perso.ens-lyon.fr/lise.vaudor/test-de-wilcoxon-mann-whitney/#:~:text=Le%20test%20U%20de%20Mann%2DWhitney%20est%20souvent%20utilis%C3%A9%20comme,les%20donn%C3%A9es%20sont%20peu%20nombreuses
# > source bonferroni correction 1 : https://mathworld.wolfram.com/BonferroniCorrection.html
# source bonferroni correction 2 : https://www.oxfordreference.com/view/10.1093/oi/authority.20110803095517119


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


# for df4_wTemp (to have a look at the var Temp in particular)
# (using df4_wTemp_unscaled to display adequate names for categories)
sns.set()
var1 = 'Temp'
var2 = dep_var
order=["0","1"]
data = pd.concat([df4_wTemp_unscaled[var2], df4_wTemp_unscaled[var1]], axis=1)
f, ax = plt.subplots(figsize=(8,6))
fig = sns.boxplot(x=var2, y=var1, data=data, order=order)
pairs_to_compare=[("0","1")]
our_annotations_to_add = add_stat_annotation(ax, data=data, x=var2, y=var1, order=order,
                                             box_pairs=pairs_to_compare,
                                             test='Mann-Whitney', text_format='star',
                                             loc='inside', verbose=2)
our_annotations_to_add
plt.title("Boxplot of the variables Temp and SepsisLabel (dtsetA, var Temp kept)")
plt.ylabel('Temp', fontsize=12)
plt.xlabel('SepsisLabel', fontsize=12)
fullname_file_of_plot = output_directory+"/"+"dtsetA_STEP2_EDA_BA_Case2_1ContVar1CatVar_Temp_theDepVar.png"
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
# 0 v.s. 1: Mann-Whitney-Wilcoxon test two-sided with Bonferroni correction, P_val=3.011e-99 U_stat=5.192e+08
# """
# - Remark 1 :  some outliers seem to exist (e.g. outside of both whiskers of SepsisLabel 0 boxplot)
# - Remark 2 : the 2 two groups of the dep var are differenciated by the Temp var (P_val=3.011e-99 ie <= 1.00e-04 ie **** significance)

# + Case 3 : for 2 categorical vars
# (a scatter plot for 2 cat vars is not informative.
# It is better to use one of the cat var as a hue on a boxplot of "1 cont var and 1 cat var")
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
# """
# p-value annotation legend:
# ns: 5.00e-02 < p <= 1.00e+00
# *: 1.00e-02 < p <= 5.00e-02
# **: 1.00e-03 < p <= 1.00e-02
# ***: 1.00e-04 < p <= 1.00e-03
# ****: p <= 1.00e-04
# 1_Male v.s. 1_Female: Mann-Whitney-Wilcoxon test two-sided with Bonferroni correction, P_val=5.898e-03 U_stat=2.570e+07
# 0_Male v.s. 0_Female: Mann-Whitney-Wilcoxon test two-sided with Bonferroni correction, P_val=2.549e-113 U_stat=4.920e+10
# 0_Male v.s. 1_Male: Mann-Whitney-Wilcoxon test two-sided with Bonferroni correction, P_val=3.050e-155 U_stat=1.414e+09
# 0_Female v.s. 1_Female: Mann-Whitney-Wilcoxon test two-sided with Bonferroni correction, P_val=3.990e-105 U_stat=6.657e+08
# """
# - Remark 1 :  some outliers seem to exist (e.g. outside of both whiskers of all boxes)
# - Remark 2 : a) the HR var differentiates significantly the two genders in each SepsisLabel.
# b) the HR differentiates significantly the SepsisLabel in each gender.

# for df4_wTemp (to have a look at the var Temp in particular)
# (using df4_wTemp_unscaled to display adequate names for categories)
sns.set()
var1 = 'Temp'
var2 = dep_var
var3 = "Gender"
order=["0","1"]
data = pd.concat([df4_wTemp_unscaled[var2], df4_wTemp_unscaled[var1], df4_wTemp_unscaled[var3]], axis=1)
f, ax = plt.subplots(figsize=(8,6))
fig = sns.boxplot(x=var2, y=var1, hue=var3,data=data, order=order)
pairs_to_compare=[(("0","Male"),("1","Male")),(("0","Female"),("1","Female")),
                  (("0","Male"),("0","Female")),(("1","Male"),("1","Female"))]
our_annotations_to_add = add_stat_annotation(ax, data=data, x=var2, y=var1, hue=var3, order=order,
                                             box_pairs=pairs_to_compare,
                                             test='Mann-Whitney', text_format='star',
                                             loc='inside', verbose=2)
our_annotations_to_add
plt.title("Boxplot of the variables Temp and SepsisLabel, for each Gender (dtsetA, var Temp kept)")
plt.ylabel('Temp (Deg C)', fontsize=12)
plt.xlabel('SepsisLabel', fontsize=12)
fullname_file_of_plot = output_directory+"/"+"dtsetA_STEP2_EDA_BA_Case3_2CatVars1ContVar_Gender_theDepVar_Temp.png"
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
# 1_Male v.s. 1_Female: Mann-Whitney-Wilcoxon test two-sided with Bonferroni correction, P_val=2.772e-03 U_stat=3.276e+06
# 0_Male v.s. 0_Female: Mann-Whitney-Wilcoxon test two-sided with Bonferroni correction, P_val=3.427e-202 U_stat=7.739e+09
# 0_Male v.s. 1_Male: Mann-Whitney-Wilcoxon test two-sided with Bonferroni correction, P_val=5.443e-59 U_stat=1.918e+08
# 0_Female v.s. 1_Female: Mann-Whitney-Wilcoxon test two-sided with Bonferroni correction, P_val=4.454e-40 U_stat=7.993e+07
# """
# - Remark 1 :  some outliers seem to exist (e.g. outside of both whiskers of all boxes)
# - Remark 2 : a) the Temp var differentiates significantly the two genders in each SepsisLabel.
# b) the Temp differentiates significantly the SepsisLabel in each gender.

print("# ---> Multivariate analysis")
# (for a group of vars, the pairwise correlations or the pairwise observations, for multiple vars at a time)

# + Case 1 : the pairwise correlations for multiple vars at a time (we use a heatmap exploiting a correlation matrix)

# NB1 :
# heatmaps are symmetrical along the main diagonal so for clarity and easier reading,
# we only display the lower half of heatmaps. This is done by using the mask argument.
# The mask given to the mask argument is created like this :
# a matrix of Trues with same shape as the correlation matrix used is created,
# then that matrix is passed to np.triu() that will change to Falses the lower half of the matrix of Trues,
# and during creation of the heatmap, the mask answers this question about each cell of the correlation matrix :
# "should this cell be not represented ?" with "True : cell is not represented and False : cell is represented"
# NB2 :
# Also, the vars are ordered in descending order of the correlation with the dep var for easier reading
# NB3 :
# We interpret the strength of correlation coefficients using their absolute value and these 4 intervals :
# highly correlated (>= 0.7), moderately correlated (0.5 included to 0.7 not included),
# low correlation (0.3 included to 0.5 not included), little to no correlation (< 0.3)
# NB4 :
# Even if in regular practice correlations less than 0.5 in abs val are not focused on,
# if a heatmap is produced here with all correlations absolute values < 0.5, we interpret them for demonstrative purposes

# for df4
sns.set() # to have a blue background; use sns.set(style="white") for a white background
k = df4.shape[1]
initial_corrmat = df4.corr()
corrmat_all_abs = initial_corrmat.abs()
k_cols_in_desc_order_of_corr_with_dep_var_in_df4 = corrmat_all_abs.nlargest(k,dep_var)[dep_var].index
final_corrmat = df4[k_cols_in_desc_order_of_corr_with_dep_var_in_df4].corr()
mask_used = np.triu(np.ones_like(final_corrmat, dtype=bool))
f, ax  = plt.subplots(figsize=(12, 9))
graph = sns.heatmap(final_corrmat, mask=mask_used, cmap="bwr", annot=True, square=True)
plt.title("Heatmap showing the correlations between all remaining variables (dtsetA, var Temp dropped)")
fullname_file_of_plot = output_directory+"/"+"dtsetA_STEP2_EDA_MA_Case1_PairwiseCorrelationsHeatmap_VarTempDropped.png"
plt.savefig(fullname_file_of_plot, bbox_inches='tight')
plt.show()
plt.close()
# - Remark 1 :
# no var is showing a very high corr with the dep var (all absolute correlations < 0.5)
# - Remark 2 :
# also, no pairwise corr is quite high ie we dont have a significant corr that would make us suspect a situation
# of multicollinearity (ie 2 or more fts that are so correlated with each other that one of them is enough to predict the dep var)
# In such a situation, we could drop at least one of them for good practice and
# because some learning tasks dont handle multicollinearity very well
# - Remark 3 :
# regarding the abs corr of each feature to the dep var, the vars ordered in desc order are :
# ICULOS, HR, Resp, MAP, HospAdmTime, Gender, Age, and O2Sat
# We decide that those we will study more are vars with abs corr with dep var > 0.01 :
# ICULOS, HR, Resp, MAP.
# - Remark 4 :
# between fts, the highest (>= 0.01) pairwise corrs distinguished are, in descending order :
# the positive corrs :
# > 4 corrs are related to the act of breathing :
# ### the corr between Resp and HR (not surprising as they seem to be both varying in the same direction than the patient breathing cadence)
# ### the corr between HR and MAP (not surprising as the HR is a cause of blood pressure in arteries)
# ### the corr between MAP and O2Sat : (not surprising as a higher blood pressure pushes for more oxygen saturation)
# ### the corr between Age and Resp (not surprising as with more age, comes a need to draw more breath due to effort;
# can also be due to patient having more of those moments were they are having difficulties and compensate with more breathing)
# ### the corr between Resp and MAP (not surprising as Resp favors HR that favors MAP)
# > 4 corrs are related to the length of stay of the patient :
# ### the corr between ICULOS and Resp, the corr between HospAdmTime and MAP, the corr between ICULOS and MAP, the corr between ICULOS and HR,
# This is also not surprising as, if the care is actively working on the patient state,
# a longer care should be concurrent with measures like blood pressure and breathing getting better.
# This also can be seen on a simpler level, as the longer care having a good effect on breathing (that is characterized
# by a higher Resp), and a higher Resp pushes for higher HR, and a higher HR pushes for by a higher MAP.
# > 2 corrs are related to the Gender :
# NB : explanations of correlations to Gender can be either a bit based on not enough information or open to too many explanations...
# hence we will only give the most plausible case of variabilities that could happen under the remarked correlations
# ### the corr between Gender and MAP, the corr between Gender and Resp :
# a variability proposed here is that Resp is higher for male patients and this leads also a higher MAP

# the negative corrs :
# > 5 corrs are related to Age :
# ### the corr between Age and HR, and, the corr between Age and MAP :
# this can be explained by HR and MAP being reduced naturally with age advancing; also we can argue that,
# among people admitted at a hospital, those values are even lower for older patients that are more on distress than younger people
# ### the corr between Age and HospAdmTime (with a pos corr between Age and ICULOS):
# can be explained by the fact that the older the patient, the shorter the time after admission before being transferred to an ICU
# (and this can be supported by the pos corr between age and ICULOS)
# ie the older the patient, less time in hospital and more time in ICU.
# This can be seen as more precaution taken due this patient being generally more at risk than a younger patient)
# ### the corr between Age and Gender :
# this is a corr open to too many explanations and we can only give the 2 observations that it means :
# ... as the age of the patients advances, the more the patients tend to be females
# ... as the patients are going towards young people, the more the patients tend to be males
# ### the corr between Age and O2Sat :
# this can be explained by the fact that with more Age, the oxygen saturation become lower and added to that is the fact
# that older patient are admitted most of the time in a distress state where they are breathing is a struggle hence
# an even lower oxygen saturation

# > 3 corrs are related to the O2Sat :
# ### the corr between O2Sat and Resp :
# can be explained by the moments where oxygen saturation dimishes and the patient Resp increases to account for the loss as mentioned in source below.
# """
# It is important to remember that pulse oximetry measures oxygen saturation while RR measures ventilation. During early
# stages of deterioration, patients’ SpO2 may appear to be in the normal range, but the RR will increase in response to
# inadequate gaseous exchange. Changes in RR is often the first sign of deterioration
# """
# source O2Sat_Resp_1 : https://www.nursingtimes.net/clinical-archive/respiratory-clinical-archive/respiratory-rate-2-anatomy-and-physiology-of-breathing-31-05-2018/
# ### the corr between O2Sat and ICULOS (with a pos corr between O2Sat and HospAdmTime) :
# (can be explained by this : patients going to, during or after surgery are more oriented towards a ICU;
# the care following or around surgery contains anaesthetics, concentrated in opioids that
# act on the central chemoreceptors suppressing the drive to breathe, hence can depress respiration and reduce Resp;
# and a loss in Resp will be manifested in a loss of O2Sat;
# ie longer ICULOS can be concurrent with lower O2Sat (see source source O2Sat_Resp_1 )
# (Also, the pos corr between O2Sat and HospAdmTime, shows that this effect is not observed in hospital care, far from the opioids of ICU care)
# ### the corr between O2Sat and HR :
# HR and O2Sat are most of the time positively correlated but in situations of effort (patient having difficulties),
# the constrained heart rate provokes an oxygen uptake so HR reducing can provokes O2Sat increasing
# source "altering heart rate on oxygen uptake" : https://pubmed.ncbi.nlm.nih.gov/2491802/

# > 3 corrs related to the Gender :
# NB : explanations of correlations to Gender can be either a bit based on not enough information or open to too many explanations...
# hence we will only give the most plausible case of variabilities that could happen under the remarked correlations
# ### the corr between Gender and O2Sat, the corr between Gender and HR :
# For HR and O2Sat, a variability proposed is that these appear to be higher for female patients. An explanation of this is :
# previously, we observed a negative corr between age and gender (can translated into "the older the patient, the more likely it is a female"),
# we can chain it with a tendency of orienting faster to an ICU the older patients (see negative corr between Age and HospAdmTime),
# to come this hypothesis : "older patients, are oriented to ICU for better care due to their age and this leads to
# bettering their measurables hence a higher HR and a higher O2Sat for the females that are a certain portion of these older patients"
# ### the corr between Gender and HospAdmTime :
# it could be that female patients have a longer HospAdmTime due to the fact that among admitted female patients, a certain portion
# is coming in due to a need of care that is more managed under a hospital care that an ICU (e.g. : giving birth, gynecology, etc).

# > 2 corrs related to the length of stay :
# ### the corr between HospAdmTime and HR, the corr between HospAdmTime and Resp :
# can be explained by a shorter HospAdmTime (ie the patient being transferred faster in an ICU), translating into
# a better care hence measurables of the patient being better like a higher HR and a higher Resp

# df4_wTemp (to have a look at the var Temp in particular)
# NB1 : remarks already made with the df4 version of the dataset wont be repeated
# NB2 : the focus will be on how the var Temp behaves with the dep var and other fts
sns.set() # to have a blue background; use sns.set(style="white") for a white background
k = df4_wTemp.shape[1]
initial_corrmat = df4_wTemp.corr()
corrmat_all_abs = initial_corrmat.abs()
k_cols_in_desc_order_of_corr_with_dep_var_in_df4_wTemp = corrmat_all_abs.nlargest(k,dep_var)[dep_var].index
final_corrmat = df4_wTemp[k_cols_in_desc_order_of_corr_with_dep_var_in_df4_wTemp].corr()
mask_used = np.triu(np.ones_like(final_corrmat, dtype=bool))
f, ax  = plt.subplots(figsize=(12, 9))
graph = sns.heatmap(final_corrmat, mask=mask_used, cmap="bwr", annot=True, square=True)
plt.title("Heatmap showing the correlations between all remaining variables (dtsetA, var Temp kept)")
fullname_file_of_plot = output_directory+"/"+"dtsetA_STEP2_EDA_MA_Case1_PairwiseCorrelationsHeatmap_VarTempKept.png"
plt.savefig(fullname_file_of_plot, bbox_inches='tight')
plt.show()
plt.close()
# Remark 1 : also, no high correlations showcased across the board (all abs corr < 0.5)
# Remark 2 : corr with the dep var : Temp and the dep var are slightly positively correlated.
# Remark 3 : regarding corr of each var to the dep var, the vars in desc order are :
# Resp, Temp, ICULOS, HR, HospAdmTime, MAP, O2Sat, Age, Gender
# We decide that those we will study more are vars with abs corr with dep var > 0.01 :
# Resp, Temp, ICULOS, HR, HospAdmTime, MAP
# Remark 3 : the highest pairwise corrs distinguished, between Temp and fts, are in descending order :
# ### the positive corrs :
# with HR, Resp, ICULOS, and Gender
# ### the negative corrs :
# with Age, O2Sat, MAP and HospAdmTime
# NB : the correlations of Temp with other vars is little so we choose to not push here for possible explanations.


# + Case 2 : the pairwise observations for multiple vars at a time (we use a pairplot)

# for df4
# # on vars with abs corr with dep var > 0.01 : ICULOS, HR, Resp, MAP.
sns.set()
# cols_selected = k_cols_in_desc_order_of_corr_with_dep_var_in_df4 # for all vars but in desc order of corr with the dep var
cols_selected = ['ICULOS', 'HR', 'Resp', 'MAP']
graph = sns.pairplot(df4[cols_selected], corner=True)
graph.fig.suptitle("Pairplot of sorted variables having correlation with dep var > 0.01 (dtsetA, var Temp dropped)", ha='center', va='center')
fullname_file_of_plot = output_directory+"/"+"dtsetA_STEP2_EDA_MA_Case2_PairwiseObservationsPairplot_VarTempDropped.png"
plt.savefig(fullname_file_of_plot, bbox_inches='tight')
plt.show()
plt.close()
# - Remark 1 :
# the scatterplots for the pairwise observations of features are too cluttered to be readable for insights due to the large number
# of data points we have. But previous observations in bivariate analysis show that, in most of the pairwise observations of features,
# var 1 has, for most of its values ,all the space of values of var 2.
# This is additionally supported by the low correlations absolute values between pairs of features.
# - Remark 2 :
# the distribution plots of each var are close to a normal distribution (excepted after a first removal of outliers)...
# ... except for ICULOS that is right skewed
# ### For ICULOS right skewness :
# This right skewness of ICULOS distribution is understandable due to the following :
# ICULOS is positively correlated with the dep var
# (a patient having sepsis tends to have a longer ICULOS, ie the right of the distribution of ICULOS tends to be made up of the sepsis patients).
# But, our dataset is largely imbalanced with a lot less sepsis patients than non-sepsis patients.
# Hence, a lot less data points on the right of the distribution of ICULOS.
# - Remark 3 :
# the same trend of "imbalance combined with correlation of a feature with the dep var,
# to cause skewness in distribution" is slightly visible on the others distributions.

# for df4_wTemp (to have a look at the var Temp in particular)
# # on vars with abs corr with dep var > 0.01 : Resp, Temp, ICULOS, HR, HospAdmTime, MAP
sns.set()
# cols_selected = k_cols_in_desc_order_of_corr_with_dep_var_in_df4_wTemp # for all vars but in desc order of corr with the dep var
cols_selected = ['Resp', 'Temp', 'ICULOS', 'HR', 'HospAdmTime', 'MAP']
sns.pairplot(df4_wTemp[cols_selected], corner=True)
graph.fig.suptitle("Pairplot of sorted variables having correlation with dep var > 0.01 (dtsetA, var Temp kept)", ha='center', va='center')
fullname_file_of_plot = output_directory+"/"+"dtsetA_STEP2_EDA_MA_Case2_PairwiseObservationsPairplot_VarTempKept.png"
plt.savefig(fullname_file_of_plot, bbox_inches='tight')
plt.show()
plt.close()
# - Remark 1 :
# same as the remark for the cluttered scatterplots in the pairplot of (dtsetA, var Temp dropped)
# - Remark 2 :
# same remark for ICULOS right skewness (dtsetA, var Temp dropped)
# ### For HospAdmTime left skewness :
# HospAdmTime is positively correlated with the dep var
# (a patient having sepsis tends to have a longer HospAdmTime, ie the right of the distribution of HospAdmTime tends to be made up of the sepsis patients).
# But, our dataset is largely imbalanced with a lot less sepsis patients than non-sepsis patients.
# Hence, there should a lot less data points on the right of the distribution of HospAdmTime.
# But instead it has a lot more patients on the right of its distribution.
# This is due to the fact that the pool of patients with longer HospAdmTime here is a sum of the 2 following pools :
# a) the pool of sepsis patients that have had a long hospital care as well as a long ICU care, and,
# b) the pool of non sepsis patients that as usual stay longer in hospital care
# These 2 pools added up to form a lot more data points in the right of distribution of HospAdmTime.
# - Remark 3 :
# same as the remark about "imbalance combined with correlation manifest in skewness" in the pairplot of (dtsetA, var Temp dropped)

print("# ---> End of part !")
# ----------------------------------------------------------------------------------------------------------------------

print("# >>>>>> Part 8 : Saving the final version(s) of the dataset.")

# ---> Mindset :
# - we started with a dataset not preprocessed
# (df0 had [790215 rows x 41 columns])
# - we produced from it 2 versions that are fit to be used for most learning task (cleaned and scaled)
# but different by one having the var Temp
# (df4 has [512140 rows x 9 columns] and df4_wTemp has [192271 rows x 10 columns])
# - we produced also 2 versions that are not fit for learning tasks (cleaned but not scaled)
# but useful for some visualisations using original values of the continuous fts before scaling
# (df4_unscaled has [512140 rows x 9 columns] and df4_wTemp_unscaled has [192271 rows x 10 columns])

# ---> Saving these 4 versions of the dataset
# NB : we keep the columns names as they are and we dont add another column as a new index (using index=False).
# By doing this, we just keep the existing index. That way, when loading the dataset,
# we dont have that extra not needed column at the beginning

# - version being saved : fit for learning task (cleaned, scaled) without var Temp
# df to save is df4 # [512140 rows x 9 columns]
fullname_file_of_df4 = output_directory+"/"+"dtsetA_forML_woTemp_V1.csv"
df4.to_csv(fullname_file_of_df4, header=True, index=False)
print("File dtsetA_forML_woTemp_V1 saved !")
# - version being saved : fit for learning task (cleaned, scaled) with var Temp
# df to save is df4_wTemp # [192271 rows x 10 columns]
fullname_file_of_df4_wTemp = output_directory+"/"+"dtsetA_forML_wTemp_V1.csv"
df4_wTemp.to_csv(fullname_file_of_df4_wTemp, header=True, index=False)
print("File dtsetA_forML_wTemp_V1 saved !")
# - version being saved : not fit for learning task but useful for some visualisations (cleaned, not scaled) without var Temp
# df to save is df4_unscaled # [512140 rows x 9 columns]
fullname_file_of_df4_unscaled = output_directory+"/"+"dtsetA_forViz_woTemp_V1.csv"
df4_unscaled.to_csv(fullname_file_of_df4_unscaled, header=True, index=False)
print("File dtsetA_forViz_woTemp_V1 saved !")
# - version being saved : not fit for learning task but useful for some visualisations (cleaned, not scaled) with var Temp
# df to save is df4_wTemp_unscaled # [192271 rows x 10 columns]
fullname_file_of_df4_wTemp_unscaled = output_directory+"/"+"dtsetA_forViz_wTemp_V1.csv"
df4_wTemp_unscaled.to_csv(fullname_file_of_df4_wTemp_unscaled, header=True, index=False)
print("File dtsetA_forViz_wTemp_V1 saved !")

print("# ---> End of part : all files to keep have been saved!")
# ----------------------------------------------------------------------------------------------------------------------
print("# >>>>>> All tasks realized !")
# end of file
# ----------------------------------------------------------------------------------------------------------------------
