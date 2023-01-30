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

#-----

import pandas as pd


from sklearn.preprocessing import LabelEncoder # to encode values of categorical variables
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
# making an estimator that will use one single task dataset (all fts)

# ---> imports (mandatory)
import os, os.path, sys, warnings, argparse # standard files manipulation set of imports (sys is used to make all stdout display go to a log file)
from engines.watcher_engine import timer_started # for duration of operations
from engines.data_mgmt_engine import Prefix_in_absolute_path # a defined prefix of all absolute paths used in order to keep them universal for all users
from engines.data_bay_engine import info_pointing_to_dataset_getter,dataset_summoner # to get all about the dataset to analyse just from a tag specifying the dataset to use
from engines.data_mgmt_engine import sorted_list_of_uniques_in_array_maker_V1 # to obtain uniques classes in y_test
from engines.metrics_engine import defining_the_scorer_in_gridsearch
import numpy as np # to fixate a random state using a chosen seed
from sklearn.model_selection import train_test_split # to plit data of analysis into a "train+val" set and a "test" set
from sklearn import model_selection # to use gridsearchcv to search for the best model
# ---> imports (optional or depending on the learning task used)
from engines.learning_algs_preps import dict_lambdas123_space_of_values_maker_V1 # a function building a gallery of regularization strengths values
from sklearn import linear_model # for sklearn regressions
from engines.learning_algs_preps import SPAMS_grp_membership_vector_maker_V1 # used in multitask regression using SPAMS to create a group membership vector
from engines.learning_algs_preps import list_seeds_maker # used to make a list of seeds to explore from a number of seeds chosen
from engines.metrics_engine import collector_fts_contrib_type_RegrFtsCoefs_maker_V1, collector_metrics_type_MadeFromTheConfMatrix_maker_V1 # for model selection, creating the collectors to stash qualities of a model selected
from engines.metrics_engine import collector_fts_contrib_type_RegrFtsCoefs_supplier_V1, collector_metrics_type_MadeFromTheConfMatrix_supplier_V1 # for model selection, supplying the collectors to stash qualities of a model selected
from engines.metrics_engine import roc_curve_type_AfterAllIterOfModelSelection_maker_V1 # for plotting ROC curves of the model selection

#-----------------


# # ~~~~ to check
# import matplotlib # change the backend used by matplotlib from the default 'QtAgg' to one that enables interactive plots 'Qt5Agg'
# matplotlib.use('Qt5Agg')
# matplotlib.get_backend()
# import matplotlib.pyplot as plt # used to make plots # for roc curves
# import seaborn as sns
# # for sns plots, add plt.show() to make the plots appear (replace it with the "%matplotlib inline" in the case of a notebook
# # to let the backend solve that issue while displaying the plots inside the notebook)

# from engines.watcher_engine2 import report_maker1,roc_curve_finisher_after_all_iterations_of_the_mdl2
# # ~~~~

# ----------------------------------------------------------------------------------------------------------------------

print("# >>>>>> Part 1 : Analysis starting : time of start noted...")

# ---> start a clock to get the time after all the analysis
globalstart = timer_started()

# ----------------------------------------------------------------------------------------------------------------------
print("# >>>>>> Part 2 : Specifying the precise analysis to run by defining tags")

# NB : for each tag that has multiple options, uncomment one option to use

print("# >>>>>> Part 2-1 : Specifying the precise analysis to run by defining tags : tags that have a functional role...")

# NB : these tags have a functional role throughout the analysis hence
# the choices made here change the research intended and the precise analysis ran

# ---> Defining tags and information related to the input and output...

# + Tag to choose whether we make a log file (to write inside all the steps of the analysis as well as some results)
# tag_decision_make_log = "yes"
tag_decision_make_log = "no"

# + Tag to choose a dataset to analyse
tag_cohort = "dtsetA_woTemp"
# tag_cohort = "dtsetA_wTemp"
# tag_cohort = "dtsetB_woTemp"
# tag_cohort = "dtsetB_wTemp"
# tag_cohort = "dtsetAB_wTemp"
# tag_cohort = "dtsetAB_woTemp"

# + Tag "separator used in fts name of the superdataset" (set if the dataset is a superdataset destined to multitask learning)
# ftname_cohort_sep="_in_"

# + Tag to choose wether or not to add a synthetic feature for intercept (in Regression models, a column of 1s will be added)
# add_intercept = "yes"
add_intercept = "no"

# + Tag to choose the type in which the dependent variable values will be put into
dep_var_values_type = "dep_var_val_type_as_int"
# dep_var_values_type = "dep_var_val_type_as_float"

# + Path to the general output dir where all learning tasks results are put
output_directory = Prefix_in_absolute_path + "/Sepsis_Prediction/data/output_data/after_step3_modelsbuilding"

# ---> Defining tags that are used in the workings of the best model selection...

# + Tag to specify the number of seeds (each seed is for a random state where we have to estimate the model selection then average all of them later)
# num_seeds_chosen = 10 # default for full runs
num_seeds_chosen = 3 # for tests

# + Tags to specify the splitting of "all the samples" into training set, validation set and test set
# - Tag to specify the share of "all the samples" to use as a test set (remaining is used for training+validation sets)
test_set_share = 0.1
# - Tag to specify the cross-validation to use in GridSearchCV (for splitting of "all the samples used as training+validation sets")
cv_on_train_plus_val_data = 10

# + Tag to specify the scorer used in the best model selection (in gridsearchcv)
# - choice 1 : for f1score_as_weighted as the most advised for the gridscorer due to its better management of unbalance of datasets
tag_scorer = "f1score_as_weighted"
# # - choice 2 : for "Mixed Metric of 4", an experimental metric including correlation coefficient based metrics MCC and weighted Cohen's kapp,
# # the f1 weighted and the balanced accuracy
# tag_scorer = "MM4_bespoke_score"

# + Tags to specify the list of tags for the metrics to compute and report for each best model
# - choice 1: all metrics that are ready to be computed, in order of preference, and if the metric has a version that manages imbalance better,
# the version "with better imbalance management" placed just before the version "unfit for imbalance management" (14 metrics)
list_tags_metrics_computed_for_testscore = ["f1score_as_weighted", "f1score_as_binary", "MCC_V1", "CK_weighted_as_linear", "CK_non_weighted", "bal_acc_V1", "acc_V1",
					"MM4_bespoke_score",
					"prec_as_weighted", "prec_as_binary", "rec_as_weighted", "rec_as_binary", "roc_auc_score_as_weighted", "roc_auc_score_as_macro"]
# # - choice 2: only the metrics that manages imbalance better, in order of preference (8 metrics)
# list_tags_metrics_computed_for_testscore = ["f1score_as_weighted", "MCC_V1", "CK_weighted_as_linear", "bal_acc_V1",
# 					"MM4_bespoke_score",
# 					"prec_as_weighted", "rec_as_weighted", "roc_auc_score_as_weighted"]
# # - choice 3: among the metrics that manages imbalance better, in order of preference,
# # only those who would make it to publication if we decided to keep the list as short as possible (6 metrics)
# list_tags_metrics_computed_for_testscore = ["f1score_as_weighted", "MCC_V1", "bal_acc_V1",
# 					"prec_as_weighted", "rec_as_weighted", "roc_auc_score_as_weighted"]

# + Tag to specify the number of values used for interpolation while drawing roc curves
# NB1 : an int, this is approx the number of points on each roc curve drawn and
# obtained by interpolating the real values computed on x and y axis of the roc curves
# NB2 : advised to give value >=100 for visually better curves.
# NB3 : we chose value=2000 due to initial tests showing a roc curve with ~ 1527 points and arbitrarily decided to plot 2000 points
num_values_from_interpolation_for_roc_curves = 2000

# ----------------------------------------------------------------------------------------------------------------------

print("# >>>>>> Part 2-2 : Specifying the precise analysis to run by defining tags : tags that have a descriptive role only...")

# NB : these tags are mostly not functional but are used to better mark our final files
# with the specifications of the precise analysis that we have done

# ---> Defining tags related to the problem studied...

# + Tag specifying the data side of the problem
# - the condition ie an event to predict one or multiple of its outcomes
tag_cond = "SepsisOnset"
# - for the condition, the specific population studied (used for at least for one of learning and predicting)
tag_pop = tag_cohort.split("_")[0] # using dtsetA or dtsetB as pop
# - for the condition, for the specific population studied, the type of dataset used
tag_dataprofile = tag_cohort.split("_")[1] # use woTemp or wTemp as profiles

# + Tag specifying the algorithmic side of the problem (ie about the learning task being ran...)
# - Tag to specify the type of SL task (classification, regression, etc)
# tag_LearningTaskType = "Classif"
tag_LearningTaskType = "Regr"
# - Tag to specify the learning algorithm used
tag_LearningAlgUsed = "L1LogReg"
# - Tag to specify the implementation of the alg used
tag_EstimatorUsed = "SklearnLinearModel"
# - Tag to specify the type of model built (model from all fts, multitask model, Optimal Model Complexity, etc.)
tag_ModelBuilt = "SingleTaskwAllFts"
# - Tag to specify a Trial name or Number to uniquely mark the present analysis
# tag_NumTrial = "Trial" + "3" # for runs
tag_NumTrial = "TrialTest" + "3" # for testing

# ----------------------------------------------------------------------------------------------------------------------

print("# >>>>>> Part 3 : data treatment")

print("# >>>>>> Part 3-1 : data treatment : for output related information...")

# ---> Creating path to the specific output folder for this run
# NB : it is a dir located inside the general output dir and that will contain all the results files for this specific run

# + creating a string specifying this run
main_str_of_tags_specifying_this_analysis = tag_LearningTaskType + "_" + tag_LearningAlgUsed + "_" + tag_ModelBuilt + "_" + tag_cond + "_" + tag_pop + "_" + tag_dataprofile + "_" + tag_NumTrial
# + name of the specific output folder for this run
name_output_folder_specific_to_this_run = 'outputs' + '_' + main_str_of_tags_specifying_this_analysis
# + path to the specific output folder for this run
results_dir_for_this_run = output_directory + '/' + name_output_folder_specific_to_this_run
# + checks :
# - if general output folder is existing (if not we create it)
if not os.path.isdir(output_directory):
    os.mkdir(output_directory)
# - if specific output folder for this run is existing (if not we create it)
if not os.path.isdir(results_dir_for_this_run):
    os.mkdir(results_dir_for_this_run)

# ---> Redirection of stdout to .o file as a log (step 1/2)
# NB : step 2/2 is located at the end of the script and is used, in case a log file is to be created,
# to reinstate original as output space, after all redirections of outputs is over

# + Creating a log file named (e.g. : Output_main_string_of_tags_specifying_this_analysis.o
log_filename = 'Log' + '_' + main_str_of_tags_specifying_this_analysis + ".o"

# + Diverting (or not) the std output towards the log file
if tag_decision_make_log == "yes":
	original_out = sys.stdout
	sys.stdout = open(results_dir_for_this_run + '/' + log_filename, 'w')
	print('This is the log file following the course of the analysis:')
else :
	print("A log file is not to be created. Analysis is to be followed on this standard output :")

# ----------------------------------------------------------------------------------------------------------------------

print("# >>>>>> Part 3-2 : data treatment : for input related information...")

# ---> Loading the data

# + Loading the data 1/2 : getting the pointers needed to call the specific dataset needed
info_pointing_to_dataset = info_pointing_to_dataset_getter(tag_cohort)
path_to_dataset = info_pointing_to_dataset[0]
dataset_format_for_separator_used_in_file = info_pointing_to_dataset[1]
dep_var_in_dataset = info_pointing_to_dataset[2]

# + Loading the data 2/2 : summoning the dataset and splitting it in the specific parts needed by a learning task
# NB : no restriction of initial fts list is done so last 2 args are omitted (default behaviour)
list_data_summoned = dataset_summoner(path_to_dataset, dataset_format_for_separator_used_in_file,
									  add_intercept, dep_var_in_dataset, dep_var_values_type)
df_input = list_data_summoned[0]
df_output = list_data_summoned[1]
X = list_data_summoned[2]
y = list_data_summoned[3]
list_all_fts_df_input = list_data_summoned[4]
num_fts_in_df_input = list_data_summoned[5]
num_obs_in_df_input = list_data_summoned[6]
# - get a sorted list of the unique classes present in y_test
# NB1 : used later for some metrics computations that focus on the positive class or class of interest
# NB2 : make sure your positive class or class of interest is at last position in a sorted list of the uniques classes (e.g. : 1 in [0,1])
classes_list = sorted_list_of_uniques_in_array_maker_V1(y)

# ----------------------------------------------------------------------------------------------------------------------

print("# >>>>>> Part 4 : Preparatives actions for the model selection")
print("# >>>>>> Part 4-1 : Preparatives actions for the model selection : common and specific actions")

# ---> Actions commonly used
# + define our scorer used in the model selection
grid_scorer = defining_the_scorer_in_gridsearch(tag_scorer)
# + choice of our class of interest during prediction on the dep var
# NB1 : the class of interest in a classification-like task, is, among the categories of the dep var,
# the one that we want to predict hence metrics that focus on one class will need that the operator precises which class it is.
# Here, this information is mainly needed for the function roc_curve() that produces the TPR and FPR rates used to plot ROC curves.
# NB2 : Conventionally, in a binary classification (2 classes in the dep var),
# the class of interest is labeled "positive class" (1) (ie the other class is labeled "negative" or 0);
# NB3 : if in a setting of multiclassification (ie more than 2 classes are in the dep var),
# this choice is needed to declare which class is to be considered class of interest;
# NB4 : in all cases (binary classification as well as multiclassification), this line below is valid as long as your
# class of interest is labeled such it is at last position in a sorted list of the uniques classes (e.g. : 1 in [0,1])
class_of_interest = classes_list[-1]

# ---> Actions relative to the specific learning task ran

# # + making the group membership vector (only used for multitask) ##! to remove from singletask scripts
# groups_in_data = SPAMS_grp_membership_vector_maker_V1(list_all_fts_df_input,ftname_cohort_sep)

# ----------------------------------------------------------------------------------------------------------------------

print("# >>>>>> Part 4-2 : Preparatives actions for the model selection : Building the grid of values for hyperparameters")

# NB : During model selection (search of the best model), for some hyperparameters (HP), multiple values are tested to see
# which one is utilized by the best model. Hence, exploring multiple HPs, each with multiple values to test, gives a grid of
# values for hyperparameters (HP) to vary during model selection

# ---> Creating param_grid as one dict that will be explored during model selection
param_grid = {} # a dict of "for each param to test, the list of values tested"
param_grid_size = {} # a dict of "for each param to test, the number of values tested"

# ---> Choosing the hyperparameters to vary during model selection
print("- the Hyperparameters that we choose to vary are:")

# + the lambda 1 values used for a regression task from sklearn (uncomment one to use it)
# - for test with 10 values of lamda1 from 0.0001 to 100.0
print("> the lambda1 (L1) regularization strength value :")
print(">> name hyperparameter to vary in learning task used is : C with C = 1/L1")
print(">> lambda1 values explored :")
print(">> min is 0.0001, max is 100.0, 5 values initially in the space, spaced evenly on a log scale")
print(">> with values forcefully added from list MajorTestingValuesInRegul1, with values forcefully removed from list OnlyTheNullVal")
print(">> all values changed into C with C = 1/L1, all values sorted")
print(">> final number of values explored : 10")
gallery_dict_lambdas1_space_of_values = dict_lambdas123_space_of_values_maker_V1("geom", 0.0001, 100.0, 5, float,
																				 "yes", "MajorTestingValuesInRegul1",
																				 "yes", "OnlyTheNullVal",
																				 "yes", "yes")

# # - for run with 206 values of lambda1 from 0.0001 to 100.0
# print("> the lambda1 (L1) regularization strength value :")
# print(">> name hyperparameter to vary in learning task used is : C with C = 1/L1")
# print(">> lambda1 values explored :")
# print(">> min is 0.0001, max is 100.0, 200 values initially in the space, spaced evenly on a log scale")
# print(">> with values forcefully added from list MajorTestingValuesInRegul1, with values forcefully removed from list OnlyTheNullVal")
# print(">> all values changed into C with C = 1/L1, all values sorted")
# print(">> final number of values explored : 206")
# gallery_dict_lambdas1_space_of_values = dict_lambdas123_space_of_values_maker_V1("geom", 0.0001, 100.0, 200, float,
# 																				 "yes", "MajorTestingValuesInRegul1",
# 																				 "yes", "OnlyTheNullVal",
# 																				 "yes", "yes")


# # + the lambda 1 values used for a regression task from SPAMS (uncomment one to use it)
# # - for test with 10 values of lamda1 from 0.0001 to 100.0
# print("> the lambda1 (L1) regularization strength value :")
# print(">> name hyperparameter to vary in learning task used is : l1 with l1=L1")
# print(">> lambda1 values explored :")
# print(">> min is 0.0001, max is 100.0, 5 values initially in the space, spaced evenly on a log scale")
# print(">> with values forcefully added from list MajorTestingValuesInRegul1, with values forcefully removed from list OnlyTheNullVal")
# print(">> No value changed into C with C = 1/L1, all values sorted")
# print(">> final number of values explored : 10")
# gallery_dict_lambdas1_space_of_values = dict_lambdas123_space_of_values_maker_V1("geom", 0.0001, 100.0, 5, float,
# 																				 "yes", "MajorTestingValuesInRegul1",
# 																				 "yes", "OnlyTheNullVal",
# 																				 "no", "yes")
#
# # # - for run with 206 values of lambda1 from 0.0001 to 100.0
# # print("> the lambda1 (L1) regularization strength value :")
# # print(">> name hyperparameter to vary in learning task used is : l1 with l1=L1")
# # print(">> lambda1 values explored :")
# # print(">> min is 0.0001, max is 100.0, 200 values initially in the space, spaced evenly on a log scale")
# # print(">> with values forcefully added from list MajorTestingValuesInRegul1, with values forcefully removed from list OnlyTheNullVal")
# # print(">> No value changed into C with C = 1/L1, all values sorted")
# # print(">> final number of values explored : 206")
# # gallery_dict_lambdas1_space_of_values = dict_lambdas123_space_of_values_maker_V1("geom", 0.0001, 100.0, 200, float,
# # 																				 "yes", "MajorTestingValuesInRegul1",
# # 																				 "yes", "OnlyTheNullVal",
# # 																				 "no", "yes")


# ---> Updating param_grid with info of each HP to explore

# + for the lambda 1 values used for a regression task from sklearn or SPAMS
# - Putting values of the HP to explore in dict format "key as HP name" and "values as list_of_values_to_explore_for_HP"
dict_lambda1_spaceofvalues = gallery_dict_lambdas1_space_of_values[0]
dict_lambda1_sizespaceofvalues = gallery_dict_lambdas1_space_of_values[1]
# - updating param_grid with info of the HP to explore
param_grid.update(dict_lambda1_spaceofvalues)
param_grid_size.update(dict_lambda1_sizespaceofvalues)

# NB : update() method adds element(s) from a key-value type of entity to the dictionary if the key is not in the dictionary.
# If the key is in the dictionary, it updates the key with the new value.

# ---> getting a list of all the HPs explored
# NB : used to update later a dict key is "HP" and value as "value HP for a model selected as best"
list_names_HPs_explored = list(param_grid.keys())

# ----------------------------------------------------------------------------------------------------------------------

print("# >>>>>> Part 5 : The best model selection, using GridSearchCV, repeated on multiple seeds")

# ---> The idea :
# Go through the seeds and at each seed, do these actions :
# + Action 1 : Define the primary data splitting (into 2 parts) :
# - one part is used as a test set and is dedicated to best model performance estimation
# - another part is for train+validation sets and is dedicated to model selection (search of best model among all models compared)
# > NB : a secondary internal split of the training+validation data is made by the cv of the gridsearchcv

# + Action 2 : at each seed, for the best model obtained from model selection, we keep in the respective collectors these values :
# - the features contribution in the model,
# - the values of HPs explored in the model,
# - the validation score,
# - the test score for each of the metrics to be computed
# > NB : the set of "one-value-metrics" to compute is chosen by the operator; all computations related to ROC curves
# as well as plotting them is mandatory.

# + Action 3 : report on the results obtained :
# - with metrics averaged over the different seeds ran
# - figures (ROC curves are always produced)
# - an optional log file can exist to relate all about the precise analysis ran

# ---> creating the results collectors
# + set of collectors 1/2 : for features contribution in the best model
list_collectors_created_for_fts_contrib = collector_fts_contrib_type_RegrFtsCoefs_maker_V1(list_all_fts_df_input)
PdCol_fts_contrib = list_collectors_created_for_fts_contrib[0]
list_of_cols_fts_contrib = list_collectors_created_for_fts_contrib[1]
# + set of collectors 2/2 : for the metrics values recorded for the best model ("one-value-metrics" and "metrics values used to plot ROC curves")
list_collectors_created_for_metrics = collector_metrics_type_MadeFromTheConfMatrix_maker_V1()
DictCol_bestval_HPsExplored = list_collectors_created_for_metrics[0]
ListCol_val_scores = list_collectors_created_for_metrics[1]
DictCol_test_scores = list_collectors_created_for_metrics[2]
fprs_col_by_seed_one_alg = list_collectors_created_for_metrics[3]
tprs_col_by_seed_one_alg = list_collectors_created_for_metrics[4]
aucs_col_by_seed_one_alg = list_collectors_created_for_metrics[5]


# ---> carrying out the idea
# + given a number of seeds to explore, make a list of seeds, each one for a random state where we have to estimate the model selection
list_seeds = list_seeds_maker(num_seeds_chosen) # a function making a list of seeds to explore; this list is conditionally set (default=listof10 if input is <3)
# + keep the final number of seeds that will be explored (for reports and prints used as checkpoints)
num_seeds_used = len(list_seeds)
print("Model selection operations will be repeated for",num_seeds_used,"random states.")
# + looping on the seeds and carrying out the idea
for a_seed in list_seeds:  # to test, use these 3 values # a_seed = 0 # a_seed = 1 # a_seed = 2
	print("- Starting model selection operations using seed", a_seed)
	rank_present_seed_in_list_seeds = list_seeds.index(a_seed) + 1
	print("- step 1/3 of model selection : primary data splitting and defining gridsearchcv, for seed",a_seed)
	# - fixate the actual random seed (in order to fixate a random state)
	np.random.seed(a_seed)
	# - get the different "train+val" and "test" parts of our data
	X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=test_set_share, random_state=a_seed)
	# - define the estimator ie the learning algorithm for which we are researching its best model
	estimator_used = linear_model.LogisticRegression(penalty='l1', solver='saga', random_state=a_seed, class_weight="balanced",
														   fit_intercept=True, max_iter=10000, tol=1.0)
	# - define the gridsearchcv
	modelselector_by_GSCV = model_selection.GridSearchCV(estimator_used, param_grid, scoring=grid_scorer, cv=cv_on_train_plus_val_data, n_jobs=-1, verbose=3)
	# - carry out the grid search
	print("- step 2/3 of model selection : carrying out model selection by using gridsearchcv, for seed",a_seed)
	modelselector_by_GSCV.fit(X_train, y_train.flatten())
	# NB1 : the labels inside the dep var are in a column format while it is expected in a row.
	# ie we have the values in a numpy array (shape: (n,1) and .flatten() will convert that array shape to (n, )
	# NB2 : whenever we will need to transform an array of shape (n,1) into an array of shape (n,),
	# we prefer to use .flatten() rather than ravel because flatten() returns a copy so no risk of
	# changing the original array of shape (n,1) while working on the obtained array of shape (n,)
	# - collecting metrics values for the model selected from the best model search
	print("- step 3/3 of model selection : collecting qualities of the model selected as best, for seed", a_seed)
	# > supplying the collector of the features contribution in the model
	collector_fts_contrib_type_RegrFtsCoefs_supplier_V1(modelselector_by_GSCV, PdCol_fts_contrib, list_of_cols_fts_contrib, a_seed)
	# > supplying the collector of the metrics values recorded ("one-value-metrics" and "metrics values used to plot ROC curves")
	collector_metrics_type_MadeFromTheConfMatrix_supplier_V1(modelselector_by_GSCV, X_test, y_test, class_of_interest,
															 DictCol_bestval_HPsExplored, list_names_HPs_explored,
															 ListCol_val_scores,
															 DictCol_test_scores, list_tags_metrics_computed_for_testscore,
															 fprs_col_by_seed_one_alg, tprs_col_by_seed_one_alg, aucs_col_by_seed_one_alg)
	# - announce the end of the operations for a seed
	print("- Finished model selection operations using seed", a_seed, "ie", rank_present_seed_in_list_seeds, "out of", num_seeds_used, "seeds done !")

# + Plotting ROC curves of the model selection (1 "by seed roc curve" * number_of_seeds+ 1 "mean roc curve" over all seeds)
roc_curve_type_AfterAllIterOfModelSelection_maker_V1(fprs_col_by_seed_one_alg,
												 tprs_col_by_seed_one_alg,
												 aucs_col_by_seed_one_alg,
												 list_seeds,
												 num_values_from_interpolation_for_roc_curves,
												 tag_LearningTaskType, tag_LearningAlgUsed, tag_ModelBuilt,
												 tag_cond, tag_pop, tag_dataprofile,
												 tag_NumTrial, results_dir_for_this_run)

##!


# + Produce a report of the analysis using variables and collectors updated
report_maker1(tag_cohort,num_seeds_used,dep_var_values_type,add_intercept,tag_scorer,list_tags_metrics_computed_for_testscore,
			  tag_cond,tag_pop,tag_dataprofile,
			  tag_LearningTaskType,tag_LearningAlgUsed,tag_EstimatorUsed,tag_ModelBuilt,tag_NumTrial,
			  list_seeds,
			  PdCol_fts_contrib,ListCol_val_scores,DictCol_test_scores,DictCol_bestval_HPsExplored,
			  num_fts_in_df_input,list_of_cols_fts_contrib,
			  results_dir_for_this_run,
			  globalstart)

# ----------------------------------------------------------------------------------------------------------------------

# >>>>>> Part 1 : Redirection of stdout to .o file as a log (step 2/2)

# ---> stop redirection of output to stdout if it was being done
if tag_decision_make_log == "yes":
	sys.stdout = original_out
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# end of analysis





################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################


print("# ---> End of part : all files to keep have been saved!")
# ----------------------------------------------------------------------------------------------------------------------
print("# >>>>>> All tasks realized !")
# end of file
# ----------------------------------------------------------------------------------------------------------------------