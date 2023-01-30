#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Created By  : Amad Diouf, amaddioufb13@gmail.com
# Created Date: 27/05/2022
# version ='1.0.0'
# ------------------------------------------------------------------------------
""" script for taking as input file a curated version of the dataset (ie obtained after EDA),
build a predictive model from it using a specific learning algorithm,
output the performance of the best model as metrics values,
save the model.
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
print("# >>>>>> Objective : models building.")
# ----------------------------------------------------------------------------------------------------------------------

print("# >>>>>> Part 1 : Input and output related information.")

print("# ---> input related information")

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################






################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################


print("# ---> End of part : all files to keep have been saved!")
# ----------------------------------------------------------------------------------------------------------------------
print("# >>>>>> All tasks realized !")
# end of file
# ----------------------------------------------------------------------------------------------------------------------