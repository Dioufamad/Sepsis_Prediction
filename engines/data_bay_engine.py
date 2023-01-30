#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Amad Diouf, amaddioufb13@gmail.com
# Created Date: 27/05/2022
# version ='1.0.0'
# ---------------------------------------------------------------------------
""" This is the module containing the functions used to keep
the datasets calls, paths, and manipulations at bay...in a bay"""
# ---------------------------------------------------------------------------
# Imports
import os, os.path, sys, warnings, argparse # standard files manipulation set of imports
import numpy as np
from sklearn.preprocessing import StandardScaler # for scaling
import pandas as pd # for dataframes manipulation
from engines.data_mgmt_engine import Prefix_in_absolute_path # a defined prefix of all absolute paths used in order to keep them universal for all users

# ----------------------------------------------------------------------------------------------------------------------

# >>>>>> Part 1 : my own created functions.

# ---> A function used as a library of links for the datasets frequently used in the project to easily call them in learning tasks and in fused datasets makings

# you give it a specific string as the tag of a cohort and it gives you back the link to the dataset you want
def info_pointing_to_dataset_getter(tag_cohort):
	# + the output folder where the datasets have been stored
	source_folder_for_datasets = Prefix_in_absolute_path + "/Sepsis_Prediction/data/output_data/after_step2_EDA"

	# + links for the datasets, each one pointed using a specific tag_cohort
	if tag_cohort == "dtsetA_woTemp" : # for trsetA, version wo Temp var
		filepath_of_ml_dataset = source_folder_for_datasets + "/" + "dtsetA_forML_woTemp_V1.csv"
		dataset_format_for_separator_used_in_file = "csv"
		dep_var_in_dataset = "SepsisLabel"
	elif tag_cohort == "dtsetA_wTemp" : # for trsetA, version w Temp var
		filepath_of_ml_dataset = source_folder_for_datasets + "/" + "dtsetA_forML_wTemp_V1.csv"
		dataset_format_for_separator_used_in_file = "csv"
		dep_var_in_dataset = "SepsisLabel"
	elif tag_cohort == "dtsetB_woTemp" : # for trsetB, version wo Temp var
		filepath_of_ml_dataset = source_folder_for_datasets + "/" + "dtsetB_forML_woTemp_V1.csv"
		dataset_format_for_separator_used_in_file = "csv"
		dep_var_in_dataset = "SepsisLabel"
	elif tag_cohort == "dtsetB_wTemp" : # for trsetB, version w Temp var
		filepath_of_ml_dataset = source_folder_for_datasets + "/" + "dtsetB_forML_wTemp_V1.csv"
		dataset_format_for_separator_used_in_file = "csv"
		dep_var_in_dataset = "SepsisLabel"
	elif tag_cohort == "dtsetAB_woTemp":  # trset A and trset B joined in 1 superdataset for multitask, version wo Temp var
		filepath_of_ml_dataset = source_folder_for_datasets + "/" + "dtsetAB_forML_woTemp_V1.csv"
		dataset_format_for_separator_used_in_file = "csv"
		dep_var_in_dataset = "SepsisLabel"
	elif tag_cohort == "dtsetAB_wTemp": # trset A and trset B joined in 1 superdataset for multitask, version with Temp var
		filepath_of_ml_dataset = source_folder_for_datasets + "/" + "dtsetAB_forML_wTemp_V1.csv"
		dataset_format_for_separator_used_in_file = "csv"
		dep_var_in_dataset = "SepsisLabel"

	# + in the event where what has been supplied as tag_cohort is not recognized among the stored datasets
	else :
		filepath_of_ml_dataset = ""
		dataset_format_for_separator_used_in_file = ""
		dep_var_in_dataset = ""
		print("Warning ! tag_cohort supplied for dataset file selection is not related to an existing file. ")

	return filepath_of_ml_dataset, dataset_format_for_separator_used_in_file, dep_var_in_dataset

# ---> A function to use a given path to a dataset and summon the dataset in the format it will be needed

def dataset_summoner(path_to_dataset, dataset_format_for_separator_used_in_file,
					 add_intercept, dep_var_in_dataset, dep_var_values_type,
					 list_fts_to_restrict_to="All_fts", list_fts_to_drop=None):
	# + load the dataset in a df
	# NB : we precise the option sep = "," so that in the future, when files with different separators have to be used, the function is easily upgraded to also support that)
	# - for a csv
	if dataset_format_for_separator_used_in_file == "csv":
		df = pd.read_csv(path_to_dataset, sep=",")
	# - for another format
	# elif dataset_format_for_separator_used_in_file == "another_format":
	#     "the commands to read file in the format another_format"
	# - as a default option, using pd.read_csv()
	else:
		df = pd.read_csv(path_to_dataset)

	# + restricting the dataset to fts the user wants to work with
	# - step 1 : we restrict the dataset to a list of fts that we want to limit it to
	if isinstance(list_fts_to_restrict_to, list) and list_fts_to_restrict_to != []:
		# make a list that contains fts to restrict to + the dep var
		if dep_var_in_dataset not in list_fts_to_restrict_to:
			list_vars_to_restrict_to = list_fts_to_restrict_to + [dep_var_in_dataset]
		else:
			list_vars_to_restrict_to = list_fts_to_restrict_to
		# make a df with, among all initial fts, only the fts to keep
		df = df[list_vars_to_restrict_to]
	# - step 2 : we drop features that we dont want to work with
	elif isinstance(list_fts_to_drop, list) and list_fts_to_drop != []:
		# make a list that dont contain for sure the dep var
		# NB : any feature should be able to be dropped except the dep var that is needed for the learning tasks)
		if dep_var_in_dataset in list_fts_to_drop :
			list_fts_to_drop.remove(dep_var_in_dataset)
		# we dropped the fts in the list of fts to drop
		df = df.drop(list_fts_to_drop, axis=1)

	# + including or not a column at the first position for the intercept used in regressions tasks
	# NB : this is a column filled with value 1 on all its rows and inserted at first position; also 1 is put as 1.0 to stay in the float type theme of all the fts values
	if add_intercept == "yes":
		df.insert(0, '_SyntheticFeat4Intercept', 1.0)

	# + separating the data into a gallery of features (fts) and the dependent variable (dep var)
	# - a df of the input (fts gallery)...
	df_input = df.drop([dep_var_in_dataset], axis=1)  # same as doing df_input = df.iloc[:, :-1] # recommended to use iloc to produce a slice # :-1 means all except the last one
	# ...and its array version
	X = np.array(df_input)  # same as X = df_input.values but with a possibility to force values to be floats
	# - a df of the output (dep var column)...
	df_output = df[[dep_var_in_dataset]] # df_output = df['SepsisLabel'] is a series that has same content than df_output = df.iloc[:,-1] # -1 means select the last one only
	# NB : we used df_output = df[['SepsisLabel']] to get a real df just like with df_input # for explanation, see (https://stackoverflow.com/questions/11285613/selecting-multiple-columns-in-a-pandas-dataframe)
	# ...and its array version
	if dep_var_values_type == "dep_var_val_type_as_float":
		y = np.array(df_output, dtype='float')  # same as  y = df_ouput.values but with a possibility to force values to be floats
	else: # leave the response values as ints
		y = np.array(df_output)

	# + store some information tied to the dataset
	# - getting a list of the features in the same setup as in the input
	list_all_fts_df_input = list(df_input.columns)
	# - getting the number of features
	num_fts_in_df_input = len(list_all_fts_df_input)
	# - getting the numbers of observations (samples, rows when the fts are in the columns, etc)
	num_obs_in_df_input = df_input.shape[0]


	return df_input, df_output, X, y, list_all_fts_df_input, num_fts_in_df_input, num_obs_in_df_input


# --->

# --->

# ----------------------------------------------------------------------------------------------------------------------

# >>>>>> Part 2 : functions copied or customized from others authors.

# --->

# ----------------------------------------------------------------------------------------------------------------------

# end of file
# ----------------------------------------------------------------------------------------------------------------------