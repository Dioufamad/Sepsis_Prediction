#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Created By  : Amad Diouf, amaddioufb13@gmail.com
# Created Date: 27/05/2022
# version ='1.0.0'
# ------------------------------------------------------------------------------
""" script taking as input a path to a folder with multiple files
(one for each patient of a hospital X),
 and output a full table with all patients of the hospital X"""
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
from engines.data_mgmt_engine import Prefix_in_absolute_path # a defined prefix of all absolute paths used in order to keep them universal for all users

# ----------------------------------------------------------------------------------------------------------------------
print("# >>>>>> Objective : Retrieve all patients files, unify them into one table (the dataset), save it for the following EDA.")
# ----------------------------------------------------------------------------------------------------------------------

print("# >>>>>> Part 1 : Input and output related information.")

print("# ---> input related information")
# + File location of the patients data for one hospital
# - to test with small number of files
# input_directory = Prefix_in_absolute_path + "/Sepsis_Prediction/data/input_data/used_data/training_setA_test"
# - to use all the files
input_directory = Prefix_in_absolute_path + "/Sepsis_Prediction/data/input_data/used_data/training_setA"

print("# ---> output related information")
# + the output folder location
output_directory = Prefix_in_absolute_path + "/Sepsis_Prediction/data/output_data/after_step1_data_gathering"
# + checks
# - if output folder is existing (if not we create it)
if not os.path.isdir(output_directory):
    os.mkdir(output_directory)

print("# ---> End of part : all input and output related info accounted for !")
# ----------------------------------------------------------------------------------------------------------------------

print("# >>>>>> Part 2 : find all files corresponding each to what would be a patient in input folder")

files = []
for f in os.listdir(input_directory):
    if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('psv'):
        files.append(f)

print("# ---> End of part !")
# ----------------------------------------------------------------------------------------------------------------------

print("# >>>>>> Part 3 : making tables from the file (one file, one table) and adding them to a list (to later concatenate them into one full table)")

files_tables = []
existing_column_names = []
num_files = len(files)
for i, f in enumerate(files):
    # a counter to check on evolution of the process
    print('    {}/{}...'.format(i+1, num_files))
    # Load the data in the present file as a table
    input_file = os.path.join(input_directory, f)
    with open(input_file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')
    # stash the table made...
    files_tables.append(data)
    # ...also stash the list of column name if new (to sort out the cases of different cols order if it happens)
    if column_names not in existing_column_names:
        existing_column_names.append(column_names)

print("# ---> End of part !")
# ----------------------------------------------------------------------------------------------------------------------

print("# >>>>>> Part 4 : report on the patients data found.")

print("Number of patients for which data has been found and made into a table : ", len(files_tables))
if (len(files_tables) == 20336):
    print("Good news : amount of data announced and amount of data found are the same !")
else:
    print("Warning : There is a discrepancy between the amount of data announced and the amount of data found !")
print("Number of different columns_names spaces found (difference can be of the length or of order of cols) :", len(existing_column_names))
if (len(existing_column_names) == 1):
    print("Good news : the same columns_space, with a specific length and order of cols, has been found as common to all patients !")
else:
    print("Warning : Different columns_spaces have been found (difference can be of the length or of order of cols) !")

# """
# Number of patients for which data has been found and made into a table :  20336
# Good news : amount of data announced and amount of data found are the same !
# Number of different columns_names spaces found (difference can be of the length or of order of cols) : 1
# Good news : the same columns_space, with a specific length and order of cols, has been found as common to all patients !
# """

print("# ---> End of part !")
# ----------------------------------------------------------------------------------------------------------------------

print("# >>>>>> Part 5 : making one unique full table from the fleet of patient specific tables")

print("# ---> getting the official colum_names space")
considered_column_names = []
if len(existing_column_names) == 1:
    considered_column_names = existing_column_names[0]

print("# ---> making one unique table containing all the patients")
full_table = np.vstack(files_tables)  # full_table.shape is 790215x41 for tr_setA
df_full_table = pd.DataFrame(data=full_table, columns=considered_column_names)

print("# ---> End of part !")
# ----------------------------------------------------------------------------------------------------------------------

print("# >>>>>> Part 6 : Saving the final version(s) of the dataset.")

# ---> saving the df_full_table as it is for the EDA as a next step
# NB : we keep the columns names as they are and we dont add another column as a new index (using index=False).
# By doing this, we just keep the existing index. That way, when loading the dataset,
# we dont have that extra not needed column at the beginning

# + uncomment a version to use it
# # - version being saved : for tests on a small file
# # df to save is df_full_table
# fullname_file_of_df_full_table = output_directory+"/"+"dtsetA_forEDA_test_V1.csv" # for test
# df_full_table.to_csv(fullname_file_of_df_full_table, header=True, index=False)
# print("File dtsetA_forEDA_test_V1 saved !")

# - version being saved : the full table
# df to save is df_full_table
fullname_file_of_df_full_table = output_directory+"/"+"dtsetA_forEDA_full_V1.csv"
df_full_table.to_csv(fullname_file_of_df_full_table, header=True, index=False)
print("File dtsetA_forEDA_full_V1 saved !")

print("# ---> End of part : all files to keep have been saved!")
# ----------------------------------------------------------------------------------------------------------------------
print("# >>>>>> All tasks realized !")
# end of file
# ----------------------------------------------------------------------------------------------------------------------