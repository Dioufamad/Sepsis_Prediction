#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Amad Diouf, amaddioufb13@gmail.com
# Created Date: 27/05/2022
# version ='1.0.0'
# ---------------------------------------------------------------------------
"""  this is the module for all functions related to data management tasks """
# ---------------------------------------------------------------------------
# Imports
import os, os.path, sys, warnings, argparse # standard files manipulation set of imports
import numpy as np
from sklearn.preprocessing import StandardScaler # for scaling

# ----------------------------------------------------------------------------------------------------------------------

# >>>>>> Part 1 : my own created functions.

# ---> The prefix path to the project
# + Mindset :
# Absolute paths throughout the project dont need to be changed from one user to another.
# All absolute paths throughout the project are defined such as "Absolute_path = Prefix_in_absolute_path  + / + Rest_of_the_absolute_path_since_the_project_folder".
# Rest_of_the_absolute_path_since_the_project_folder is "Sepsis_Prediction/..."
# Prefix_in_absolute_path is the path to the folder where the project folder "Sepsis_Prediction" is located.
# e.g. : if Absolute_path = "/home/user/HARD_DRIVE1/Work/projects/Sepsis_Prediction/.....", Prefix_in_absolute_path = "/home/user/HARD_DRIVE1/Work/projects"
# So we define each user, has just to change here below the value of Prefix_in_absolute_path and have access to all files like it was intended to

# + Defining Prefix_in_absolute_path
Prefix_in_absolute_path = "/home/amad/PALADIN_2/3CEREBRO/garage/projects"

# ---> A function that given a df and the dep var to not scale, will output a copy of the same df but with all features scaled
scaler = StandardScaler()
def fts_scaling(dftoscale, dep_var_to_ignore):
    dftoscale_scaled = dftoscale.copy() # make a copy of the dataset so the output df is completely different from the input df
    list_fts_dftoscale_scaled = list(dftoscale_scaled.columns)
    list_fts_dftoscale_scaled.remove(dep_var_to_ignore)
    fts_frame_2_change = dftoscale_scaled.loc[:, list_fts_dftoscale_scaled]
    fts_frame_scaled = scaler.fit_transform(fts_frame_2_change)
    dftoscale_scaled.loc[:, list_fts_dftoscale_scaled] = fts_frame_scaled
    return dftoscale_scaled

# ---> A function for all the workings in order to remove outliers in each concerned col
def first_removal_outliers(dfin,dfin_unsc, list_cols_ignored):
    # - Mindset:
    # dfin is the scaled version of the input dataset
    # dfin_unsc is the unscaled version of the input dataset
    # both versions of the datasets are identical except for the scaling in one
    # (row content, row order, row index, vars space and vars order are all the same between both versions of the dataset)
    # dfin will be searched for rows that are outliers and those rows will be removed in both dfin and dfin_unsc
    # the versions of dfin and dfin_unsc without outliers will be returned

    # - getting the list of vars on which to apply outliers removal
    list_vars_dfin_for_outliers_rm = list(dfin.columns)
    for a_colname_to_ignore in list_cols_ignored:
        list_vars_dfin_for_outliers_rm.remove(a_colname_to_ignore)

    # - removing the outliers (initial method)
    # NB : when trying to remove all outliers in all concerned cols at once using a "true or false, the value is an outlier" table,
    # we use this following line and it throws a "ValueError: Cannot index with multidimensional key". So we use a loop and do it col by col
    # dfin_outliers_only = dfin.loc[~((dfin < dfin_fence_low) | (dfin > dfin_fence_high))]

    # - removing the outliers (better method)
    print("------Detecting outliers for a dataset------")
    # we loop on the vars concerned and collect indexes of the rows to remove later. This way, the outliers are always computed from the initial pool of data)
    full_list_indexes_to_rm = []
    num_obs_in_dfin_before = dfin.shape[0]
    for outliers_rm_var in list_vars_dfin_for_outliers_rm:  # to test, outliers_rm_var = "HR" or outliers_rm_var = "O2Sat"
        print("- Detecting outliers for the var", outliers_rm_var, "and this is var",(list_vars_dfin_for_outliers_rm.index(outliers_rm_var) + 1), "out of", len(list_vars_dfin_for_outliers_rm))
        var_Q1 = dfin[outliers_rm_var].quantile(0.25)
        var_Q3 = dfin[outliers_rm_var].quantile(0.75)
        var_IQR = var_Q3 - var_Q1
        var_fence_low = var_Q1 - 1.5 * var_IQR
        var_fence_high = var_Q3 + 1.5 * var_IQR
        # a table with only the outliers
        dfin_outliers_for_present_var = dfin.loc[(dfin[outliers_rm_var] < var_fence_low) | (dfin[outliers_rm_var] > var_fence_high)]
        # stashing the outliers rows indexes
        list_indexes_outliers = dfin_outliers_for_present_var.index.values.tolist()
        for an_index_to_rm in list_indexes_outliers:
            if an_index_to_rm not in full_list_indexes_to_rm :
                full_list_indexes_to_rm.append(an_index_to_rm)
        # showing a summary of the operation
        num_obs_dfin_to_remove = dfin_outliers_for_present_var.shape[0]
        num_obs_in_dfin_to_remain = num_obs_in_dfin_before - num_obs_dfin_to_remove
        print("    num total obs is", num_obs_in_dfin_before, "and num obs to remove is", num_obs_dfin_to_remove,"ie num obs to remain is", num_obs_in_dfin_to_remain)
    # - the table with all outliers removed
    print("------Removing the outliers detected------")
    # removing the outliers in the scaled version of the dataset
    dfout = dfin.copy()
    dfout = dfout.drop(full_list_indexes_to_rm, axis=0)
    dfout = dfout.reset_index(drop=True)
    # removing the outliers in the unscaled version of the dataset
    dfout_unsc = dfin_unsc.copy()
    dfout_unsc = dfout_unsc.drop(full_list_indexes_to_rm, axis=0)
    dfout_unsc = dfout_unsc.reset_index(drop=True)
    # - a summary of the results
    # (can be done on dfout as well as dfout_unsc, no difference there, but we use dfout as in mindset,
    # it is the proper version of the dataset to continue the analysis with because it is scaled)
    num_total_outliers_removed = len(full_list_indexes_to_rm)
    perc_total_outliers_removed = round((num_total_outliers_removed / num_obs_in_dfin_before)*100,2)
    num_total_obs_remaining = dfout.shape[0]
    perc_total_obs_remaining = round((num_total_obs_remaining / num_obs_in_dfin_before) * 100, 2)
    print("- Removed a total of outliers of", num_total_outliers_removed, "out of", num_obs_in_dfin_before, "obs ie", perc_total_outliers_removed, "%")
    print("- Remaining num total obs is", num_total_obs_remaining, "out of", num_obs_in_dfin_before, "obs ie", perc_total_obs_remaining, "%")
    return dfout, dfout_unsc

# ---> A function to compute all the workings of imbalance and return the estimate of the imbalance level
def imbalance_summary(dftocheck, colnametocheck):
    # count in each class
    num_class0_dftocheck = dftocheck[colnametocheck].value_counts()[0]
    num_class1_dftocheck = dftocheck[colnametocheck].value_counts()[1]
    # percentage of count in each class
    perc_class0_dftocheck = round(dftocheck[colnametocheck].value_counts(normalize=True)[0] * 100, 2)
    perc_class1_dftocheck = round(dftocheck[colnametocheck].value_counts(normalize=True)[1] * 100, 2)
    # a ratio R to use in sentences like "the larger class in R times larger than the smaller class"
    ratio_classSup_classInf_dftocheck = round(max([perc_class0_dftocheck, perc_class1_dftocheck]) / min([perc_class0_dftocheck, perc_class1_dftocheck]),0)
    # display a summary
    print("- Num obs : class 0 has", num_class0_dftocheck, "and class 1 has", num_class1_dftocheck)
    print("- % obs in class : count class 0 accounts for", perc_class0_dftocheck, "and count class 1 accounts for", perc_class1_dftocheck)
    print("- ie a ratio classSup over classInf of : ", ratio_classSup_classInf_dftocheck)
    return

# ---> A function to update, inside a dict "key-value_as_a_list_of_entries", for a certain key, the value_as_a_list_of_entries
# NB : updating value_as_a_list_of_entries means one of these 2 actions :
# a) if the key exists already, its value_as_a_list_of_entries has an entry added to it
# b) if the key does not exists already, the key is created and to its value_as_a_list_of_entries that is an empty list, an entry is added

def dict_val_updater_valueaslist_V1(dict_to_update,key_to_target,entry_to_add_to_value):
    # + option 1 : the key is considered as existing already in the dict keys
	try:
		dict_to_update[key_to_target].append(entry_to_add_to_value)
    # + option 2 : the key is NOT considered as existing already in the dict keys (case of the first append to the key's value that is at this point an empty list)
    # "except" means "in case of this following event, do this..."
    # NB: maybe change this later to an easier to read formulation along the lines of "if key not in list of keys of the dict, do this..."
	except KeyError:
		dict_to_update[key_to_target] = [entry_to_add_to_value]
	return

# ---> A function that, given an array, will output a sorted list of the unique values in that array
# NB : only implemented for arrays of shape (n,1)
def sorted_list_of_uniques_in_array_maker_V1(array_of_interest):
    if array_of_interest.shape[1] == 1: # case of an array of shape (n, 1) like the dep var column of a dataset
        array_of_interest_as_1d_arr = array_of_interest.flatten()
        array_of_interest_as_1d_arr_as_set = set(array_of_interest_as_1d_arr)
        array_of_interest_as_1d_arr_as_set_as_sortedlist = sorted(array_of_interest_as_1d_arr_as_set) # sorted() outputs is a list
    else:
        print("only array with shape1 equals to 1 are managed by this function ! Please supply array of adequate shape.")
    return array_of_interest_as_1d_arr_as_set_as_sortedlist

# ---> A function that given, a lower_limit_value and higher_limit_value as well as number of n values,
# outputs a 1D array with n evely spaced values while including the 2 given limits
# NB : this can be used, during roc curves drawing, to get gallery of values to base interpolation on
def arr_evenly_spaced_val_maker_V1(lower_limit, higher_limit, num_values_to_compose):
    # NB1 : using np.linspace returns a number of evenly spaced values, calculated over an interval [start, stop],
    # here num_values_to_compose values over [lower_limit,higher_limit]
    arr_of_values_composed = np.linspace(lower_limit, higher_limit, num_values_to_compose)
    return arr_of_values_composed

# --->

# --->

# ----------------------------------------------------------------------------------------------------------------------

# >>>>>> Part 2 : functions copied or customized from others authors.

# --->


def load_challenge_data(file):
    with open(file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')

    # Ignore SepsisLabel column if present.
    if column_names[-1] == 'SepsisLabel':
        column_names = column_names[:-1]
        data = data[:, :-1]

    return data

def save_challenge_predictions(file, scores, labels):
    with open(file, 'w') as f:
        f.write('PredictedProbability|PredictedLabel\n')
        for (s, l) in zip(scores, labels):
            f.write('%g|%d\n' % (s, l))



#
# The load_column function loads a column from a table.
#
# Inputs:
#   'filename' is a string containing a filename.
#
#   'header' is a string containing a header.
#
# Outputs:
#   'column' is a vector containing a column from the file with the given
#   header.
#
# Example:
#   Omitted.

def load_column(filename, header, delimiter):
    column = []
    with open(filename, 'r') as f:
        for i, l in enumerate(f):
            arrs = l.strip().split(delimiter)
            if i == 0:
                try:
                    j = arrs.index(header)
                except:
                    raise Exception('{} must contain column with header {} containing numerical entries.'.format(filename, header))
            else:
                if len(arrs[j]):
                    column.append(float(arrs[j]))
    return np.array(column)

# ----------------------------------------------------------------------------------------------------------------------

# end of file
# ----------------------------------------------------------------------------------------------------------------------