#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Amad Diouf, amaddioufb13@gmail.com
# Created Date: 27/05/2022
# version ='1.0.0'
# ---------------------------------------------------------------------------
"""  this is the module for all functions related to data bay tasks """
# ---------------------------------------------------------------------------
# Imports
from datetime import datetime # for time functions

# ----------------------------------------------------------------------------------------------------------------------

# >>>>>> Part 1 : my own created functions.

# ---> Timers functions
def timer_started():   # start a clock to get the time before a step
	start = datetime.now()
	return start

def duration_from(time_at_start):  # start a clock to get the time after a step and remove it from an initial time to get the duration of the step
	time_at_end = datetime.now()
	elapsed_time = time_at_end - time_at_start
	return elapsed_time

###################################################################################

# --->
import numpy as np # to compute the different types of averages for metrics over the random states of model selection
def report_maker1(tag_cohort, num_seeds_used, dep_var_values_type, add_intercept,tag_scorer,list_tags_metrics_computed_for_testscore,
                  tag_cond,tag_pop,tag_dataprofile,
                  tag_LearningTaskType,tag_LearningAlgUsed, tag_EstimatorUsed, tag_ModelBuilt, tag_NumTrial,
                  list_seeds,
                  PdCol_fts_contrib,ListCol_val_scores, DictCol_test_scores, DictCol_bestval_HPsExplored,
                  num_fts_in_df_input,list_of_cols_coefs,
                  results_dir_for_this_run,
                  globalstart):

    print("# >>>>>> Summary of model selection analysis : ")

    # ----> State the choices made that will influence the run
    print("# ----> These choices have been made and influence the run : ")
    print("+ Dataset analysed is known as : ", tag_cohort)
    print("+ Number of random states (where the model selection is done and they are averaged) explored : ", num_seeds_used)
    print("+ Formatting of dependant variable values : ", dep_var_values_type)
    print("+ Adding a synthetic feature for intercept (in Regression models, a column of 1s will be added) : ", add_intercept)
    print("+ Scorer used in the best model selection (in gridsearchcv) : ", tag_scorer)
    print("+ List of metrics to compute and report for each best model : ", list_tags_metrics_computed_for_testscore, "and ROC curves.")

    # ----> Describe the specicity of the run using some tags entered
    print("# ----> The run is described by the following : ")
    print("+ Illness studied :", tag_cond)
    print("+ Drug studied :", tag_pop)
    print("+ Profile of data used :", tag_dataprofile)
    print("+ Type of Supervised Learning task used :", tag_LearningTaskType)
    print("+ Scheme/Alg used :", tag_LearningAlgUsed)
    print("+ Estimator used :", tag_EstimatorUsed)
    print("+ The type of model produced :", tag_ModelBuilt)
    print("+ trial : ", tag_NumTrial)

    # ------------ report the integrity of the results
    print(" >>>> Checking for integrity of the results collected...")
    # a counter of the verifications passed : if it is in the right number at the end, we mark all is okay
    num_global_verifs_passed = 0
    # check the coefs of best model : one col by seed + 1 col for the first col that has the names of the fts
    if PdCol_fts_contrib.shape[1] == (num_seeds_used + 1):
        num_global_verifs_passed += 1
    else:
        print("+ Warning : the df used as best model coefs collector has wrong numbers of columns !")
    # check the list of val score saved : one val score for each seed
    if len(ListCol_val_scores) == num_seeds_used:
        num_global_verifs_passed += 1
    else:
        print("+ Warning : the list used as best model validation score collector has wrong numbers of elements !")
    # check for each HP, the size of the list of best values : one best value by seed
    num_verifications_passed_in_best_val_HP_dict = 0
    for a_HP_that_has_list_of_best_values in list(DictCol_bestval_HPsExplored.keys()):
        if len(DictCol_bestval_HPsExplored[a_HP_that_has_list_of_best_values]) == num_seeds_used:
            num_verifications_passed_in_best_val_HP_dict += 1
        else:
            print("+ Warning : the HP", a_HP_that_has_list_of_best_values, " has a list of best values that has wrong numbers of elements !")
    if num_verifications_passed_in_best_val_HP_dict == len(list(DictCol_bestval_HPsExplored.keys())):
        num_global_verifs_passed += 1
    # check for each metric of test score, the size of the list of best values : one best value by seed
    num_verifications_passed_in_best_val_testscoremetric_dict = 0
    for a_testscoremetric_that_has_list_of_best_values in list(DictCol_test_scores.keys()):
        if len(DictCol_test_scores[a_testscoremetric_that_has_list_of_best_values]) == num_seeds_used:
            num_verifications_passed_in_best_val_testscoremetric_dict += 1
        else:
            print("+ Warning : the test score metric", a_testscoremetric_that_has_list_of_best_values, " has a list of best values that has wrong numbers of elements !")
    if num_verifications_passed_in_best_val_testscoremetric_dict == len(list(DictCol_test_scores.keys())):
        num_global_verifs_passed += 1

    ##! add the verif part for the roc curves related collectors

    # give an answer to the full verification
    if num_global_verifs_passed == 4:
        print("+ All results are accounted for!")
    else:
        print("+ Please, check the faulty collectors or the process.")

    # ------------ for each seed, report the results obtained :
    print(" >>>> Report, for each seed, the results (best value for HPs explored, validation score, test score) for the best model found : ")
    for a_certain_seed in list_seeds:
        # an index of the seed, used each time to get the best value corresponding to the seed in a collector
        index_present_seed = list_seeds.index(a_certain_seed)
        # get the val_score
        seed_best_model_val_score = ListCol_val_scores[index_present_seed]
        # a dict with each key being an HP and its value being the best value for the seed
        seed_best_model_val_params = {}
        for one_of_the_HP_explored in list(DictCol_bestval_HPsExplored.keys()):
            seed_best_model_val_params[one_of_the_HP_explored] = DictCol_bestval_HPsExplored[one_of_the_HP_explored][index_present_seed]
        if "C" in list(DictCol_bestval_HPsExplored.keys()):
            seed_best_model_val_params_Ctranslated = 1 / (seed_best_model_val_params["C"])  # obtain a lambda value instead of a C value
        else:
            seed_best_model_val_params_Ctranslated = "not_in_this_context"
        # a dict with each key being a test score metric and its value being the value of the metric for the seed
        seed_best_model_test_score = {}
        for one_of_the_testscoremetric_computed in list(DictCol_test_scores.keys()):
            seed_best_model_test_score[one_of_the_testscoremetric_computed] = DictCol_test_scores[one_of_the_testscoremetric_computed][index_present_seed]
        # print the results depending if C is in the HPs explored or not (use % knowing %d is for the int and %s change everything into a string so no need to change dictionnaries of params into str first
        if "C" in list(DictCol_bestval_HPsExplored.keys()):
            print("+ Seed %d best model had : best params as %s (ie lambda from C of %s ) , validation score of %s and test score as %s." % (a_certain_seed, seed_best_model_val_params, seed_best_model_val_params_Ctranslated, seed_best_model_val_score, seed_best_model_test_score))
        else:
            print("+ Seed %d best model had : best params as %s , validation score of %s and test score as %s." % (a_certain_seed, seed_best_model_val_params, seed_best_model_val_score, seed_best_model_test_score))

    # ------------ get the perfs and best HPs values averages across the seeds :
    print(" >>>> Report, the averages across the seeds, for the results (test score, best value for HPs explored) for the best model  : ")
    # + for the perf across seeds
    dict_testscoremetrics_averaged_as_mean = {}  # for the mean values
    dict_testscoremetrics_averaged_as_median = {}  # for the median values
    dict_testscoremetrics_averaged_as_std = {}  # for the std values
    for a_testscoremetric_known in list(DictCol_test_scores.keys()):
        dict_testscoremetrics_averaged_as_mean[a_testscoremetric_known] = np.nanmean(DictCol_test_scores[a_testscoremetric_known])
        dict_testscoremetrics_averaged_as_median[a_testscoremetric_known] = np.nanmedian(DictCol_test_scores[a_testscoremetric_known])
        dict_testscoremetrics_averaged_as_std[a_testscoremetric_known] = np.nanstd(DictCol_test_scores[a_testscoremetric_known])
    print("+ The average test score, across the seeds : Mean %s" % dict_testscoremetrics_averaged_as_mean)
    print("+ The average test score, across the seeds : Median %s " % dict_testscoremetrics_averaged_as_median)
    print("+ The average test score, across the seeds : Std %s " % dict_testscoremetrics_averaged_as_std)

    # + for the hp of interest across seeds
    dict_HPexploredBestValues_averaged_as_mean = {}  # for the mean values
    dict_HPexploredBestValues_averaged_as_median = {}  # for the median values
    dict_HPexploredBestValues_averaged_as_std = {}  # for the std values
    for a_HPexplored_known in list(DictCol_bestval_HPsExplored.keys()):
        dict_HPexploredBestValues_averaged_as_mean[a_HPexplored_known] = np.nanmean(DictCol_bestval_HPsExplored[a_HPexplored_known])
        dict_HPexploredBestValues_averaged_as_median[a_HPexplored_known] = np.nanmedian(DictCol_bestval_HPsExplored[a_HPexplored_known])
        dict_HPexploredBestValues_averaged_as_std[a_HPexplored_known] = np.nanstd(DictCol_bestval_HPsExplored[a_HPexplored_known])
    if "C" in list(DictCol_bestval_HPsExplored.keys()):
        average_as_mean_of_Ctranslated = 1 / (dict_HPexploredBestValues_averaged_as_mean["C"])  # obtain a lambda value instead of a C value
        average_as_median_of_Ctranslated = 1 / (dict_HPexploredBestValues_averaged_as_median["C"])  # obtain a lambda value instead of a C value
        average_as_std_of_Ctranslated = 1 / (dict_HPexploredBestValues_averaged_as_std["C"])  # obtain a lambda value instead of a C value
    else:
        average_as_mean_of_Ctranslated = "not_in_this_context"
        average_as_median_of_Ctranslated = "not_in_this_context"
        average_as_std_of_Ctranslated = "not_in_this_context"
    # print the results depending if C is in the HPs explored or not
    if "C" in list(DictCol_bestval_HPsExplored.keys()):
        print("+ The average best value for the HPs explored, across the seeds : Mean %s (ie lambda from C of %s ) " % (dict_HPexploredBestValues_averaged_as_mean, average_as_mean_of_Ctranslated))
        print("+ The average best value for the HPs explored, across the seeds : Median %s (ie lambda from C of %s ) " % (dict_HPexploredBestValues_averaged_as_median, average_as_median_of_Ctranslated))
        print("+ The average best value for the HPs explored, across the seeds : Std %s (ie lambda from C of %s ) " % (dict_HPexploredBestValues_averaged_as_std, average_as_std_of_Ctranslated))
    else:
        print("+ The average best value for the HPs explored, across the seeds : Mean %s " % dict_HPexploredBestValues_averaged_as_mean)
        print("+ The average best value for the HPs explored, across the seeds : Median %s " % dict_HPexploredBestValues_averaged_as_median)
        print("+ The average best value for the HPs explored, across the seeds : Std %s " % dict_HPexploredBestValues_averaged_as_std)

    # ------------ for each seed, report the counts on the coefs obtained : # collectors are also created here to use them later for the Mean, Median and Std computations during the overall coefs statistics
    print(" >>>> Report, for each seed, the counts of the coefs, for each best model : ")
    # make collectors needed
    ListCol_number_of_NON_nuls_coefs_fts_in_seed = []
    ListCol_number_of_nuls_coefs_fts_in_seed = []
    ListCol_percentage_of_NON_nuls_coefs_fts_in_seed = []
    ListCol_percentage_of_nuls_coefs_fts_in_seed = []
    # loop on the col respective to the seed (the cols of the seeds coefs are in the same order than the seeds)
    for a_certain_seed_bis in list_seeds:
        a_colname_col_coefs = "Coefficient Estimate Seed " + str(a_certain_seed_bis) # remaking what must be the name of the column that must have accepted the values
        PdCol_fts_contrib_present_seed = PdCol_fts_contrib[[a_colname_col_coefs]] # make the df of the col of the seed only
        PdCol_fts_contrib_present_seed_nonnull_fts = PdCol_fts_contrib_present_seed[PdCol_fts_contrib_present_seed[a_colname_col_coefs] != 0] # a df with the rows (fts) containing non null coefs only
        number_of_NON_nuls_coefs_fts_in_seed = PdCol_fts_contrib_present_seed_nonnull_fts.shape[0]
        number_of_nuls_coefs_fts_in_seed = num_fts_in_df_input - number_of_NON_nuls_coefs_fts_in_seed # we know number_of_total_fts = num_fts_in_df_input
        percentage_of_NON_nuls_coefs_fts_in_seed = (number_of_NON_nuls_coefs_fts_in_seed / num_fts_in_df_input) * 100
        percentage_of_nuls_coefs_fts_in_seed = (number_of_nuls_coefs_fts_in_seed / num_fts_in_df_input) * 100
        # lets supply the collectors used for later Mean, Median and Std computations
        ListCol_number_of_NON_nuls_coefs_fts_in_seed.append(number_of_NON_nuls_coefs_fts_in_seed)
        ListCol_number_of_nuls_coefs_fts_in_seed.append(number_of_nuls_coefs_fts_in_seed)
        ListCol_percentage_of_NON_nuls_coefs_fts_in_seed.append(percentage_of_NON_nuls_coefs_fts_in_seed)
        ListCol_percentage_of_nuls_coefs_fts_in_seed.append(percentage_of_nuls_coefs_fts_in_seed)
        # print the seed results
        print("+ Seed %d : Total features (fts) is %s , number NON null coefs fts is %s (ie %s %% of total ) , number null coefs fts is %s (ie %s %% of total ) " % (a_certain_seed_bis,
                                                                                                                                                                     num_fts_in_df_input,
                                                                                                                                                                     number_of_NON_nuls_coefs_fts_in_seed,
                                                                                                                                                                     percentage_of_NON_nuls_coefs_fts_in_seed,
                                                                                                                                                                     number_of_nuls_coefs_fts_in_seed,
                                                                                                                                                                     percentage_of_nuls_coefs_fts_in_seed))  # %% is % escaped in python strings

    # ------------ a report for the coefs (averages and consensus) :
    print(" >>>> Report, across the seeds, the averages and the consensus of the regression coefs : ")
    # - the overall statistics report
    # for the ListCol_number_of_NON_nuls_coefs_fts_in_seed
    ListCol_number_of_NON_nuls_coefs_fts_in_seed_averaged_as_mean = np.nanmean(ListCol_number_of_NON_nuls_coefs_fts_in_seed)
    ListCol_number_of_NON_nuls_coefs_fts_in_seed_averaged_as_median = np.nanmedian(ListCol_number_of_NON_nuls_coefs_fts_in_seed)
    ListCol_number_of_NON_nuls_coefs_fts_in_seed_averaged_as_std = np.nanstd(ListCol_number_of_NON_nuls_coefs_fts_in_seed)
    # for the ListCol_number_of_nuls_coefs_fts_in_seed
    ListCol_number_of_nuls_coefs_fts_in_seed_averaged_as_mean = np.nanmean(ListCol_number_of_nuls_coefs_fts_in_seed)
    ListCol_number_of_nuls_coefs_fts_in_seed_averaged_as_median = np.nanmedian(ListCol_number_of_nuls_coefs_fts_in_seed)
    ListCol_number_of_nuls_coefs_fts_in_seed_averaged_as_std = np.nanstd(ListCol_number_of_nuls_coefs_fts_in_seed)
    # for the ListCol_percentage_of_NON_nuls_coefs_fts_in_seed
    ListCol_percentage_of_NON_nuls_coefs_fts_in_seed_averaged_as_mean = np.nanmean(ListCol_percentage_of_NON_nuls_coefs_fts_in_seed)
    ListCol_percentage_of_NON_nuls_coefs_fts_in_seed_averaged_as_median = np.nanmedian(ListCol_percentage_of_NON_nuls_coefs_fts_in_seed)
    ListCol_percentage_of_NON_nuls_coefs_fts_in_seed_averaged_as_std = np.nanstd(ListCol_percentage_of_NON_nuls_coefs_fts_in_seed)
    # for the ListCol_percentage_of_nuls_coefs_fts_in_seed
    ListCol_percentage_of_nuls_coefs_fts_in_seed_averaged_as_mean = np.nanmean(ListCol_percentage_of_nuls_coefs_fts_in_seed)
    ListCol_percentage_of_nuls_coefs_fts_in_seed_averaged_as_median = np.nanmedian(ListCol_percentage_of_nuls_coefs_fts_in_seed)
    ListCol_percentage_of_nuls_coefs_fts_in_seed_averaged_as_std = np.nanstd(ListCol_percentage_of_nuls_coefs_fts_in_seed)
    # print
    print("+ The averages of the counts of the coefs (Mean values part) : number NON null coefs fts is %s (ie %s %% of total ) , number null coefs fts is %s (ie %s %% of total ) " % (ListCol_number_of_NON_nuls_coefs_fts_in_seed_averaged_as_mean,
                                                                                                                                                                        ListCol_percentage_of_NON_nuls_coefs_fts_in_seed_averaged_as_mean,
                                                                                                                                                                        ListCol_number_of_nuls_coefs_fts_in_seed_averaged_as_mean,
                                                                                                                                                                        ListCol_percentage_of_nuls_coefs_fts_in_seed_averaged_as_mean))
    print("+ The averages of the counts of the coefs (Median values part) : number NON null coefs fts is %s (ie %s %% of total ) , number null coefs fts is %s (ie %s %% of total ) " % (ListCol_number_of_NON_nuls_coefs_fts_in_seed_averaged_as_median,
                                                                                                                                                                          ListCol_percentage_of_NON_nuls_coefs_fts_in_seed_averaged_as_median,
                                                                                                                                                                          ListCol_number_of_nuls_coefs_fts_in_seed_averaged_as_median,
                                                                                                                                                                          ListCol_percentage_of_nuls_coefs_fts_in_seed_averaged_as_median))
    print("+ The averages of the counts of the coefs (Std values part) : number NON null coefs fts is %s (ie %s %% of total ) , number null coefs fts is %s (ie %s %% of total ) " % (ListCol_number_of_NON_nuls_coefs_fts_in_seed_averaged_as_std,
                                                                                                                                                                       ListCol_percentage_of_NON_nuls_coefs_fts_in_seed_averaged_as_std,
                                                                                                                                                                       ListCol_number_of_nuls_coefs_fts_in_seed_averaged_as_std,
                                                                                                                                                                       ListCol_percentage_of_nuls_coefs_fts_in_seed_averaged_as_std))
    # - the consensus report
    # list_of_cols_coefs = ['Coefficient Estimate Seed 0','Coefficient Estimate Seed 1','Coefficient Estimate Seed 2',....,'Coefficient Estimate Seed 8','Coefficient Estimate Seed 9'] created and updated already
    PdCol_fts_contrib_abs = PdCol_fts_contrib.copy()  # make a copy of the df of coefs to keep the old one as result and be able tosee again if needed the coefs values with their signs
    PdCol_fts_contrib_abs[list_of_cols_coefs] = PdCol_fts_contrib_abs[list_of_cols_coefs].abs()  # to only make as absolute the cols with values and not touch the 1st col containing the names of the fts
    # make a col of mean coefs across the 10 seeds cols of coefs
    PdCol_fts_contrib_abs["Mean Coefficient Estimate 10 Seeds"] = PdCol_fts_contrib_abs[list_of_cols_coefs].mean(axis=1, skipna=True)  # axis = 0 means along the column and axis = 1 means working along the row
    PdCol_fts_contrib_abs_mean_only = PdCol_fts_contrib_abs[["Features", "Mean Coefficient Estimate 10 Seeds"]]
    PdCol_fts_contrib_abs_mean_only_nonnull_fts = PdCol_fts_contrib_abs_mean_only[PdCol_fts_contrib_abs_mean_only["Mean Coefficient Estimate 10 Seeds"] != 0]
    number_of_NON_nuls_coefs_fts = PdCol_fts_contrib_abs_mean_only_nonnull_fts.shape[0]
    number_of_nuls_coefs_fts = num_fts_in_df_input - number_of_NON_nuls_coefs_fts # we know number_of_total_fts = num_fts_in_df_input
    percentage_of_NON_nuls_coefs_fts = (number_of_NON_nuls_coefs_fts / num_fts_in_df_input) * 100
    percentage_of_nuls_coefs_fts = (number_of_nuls_coefs_fts / num_fts_in_df_input) * 100
    # print the seeds consensus results
    print("+ The consensus on the counts of the coefs : Total features (fts) is %s , number NON null coefs fts is %s (ie %s %% of total ) , number null coefs fts is %s (ie %s %% of total ) " % (num_fts_in_df_input,
                                                                                                                                                                           number_of_NON_nuls_coefs_fts,
                                                                                                                                                                           percentage_of_NON_nuls_coefs_fts,
                                                                                                                                                                           number_of_nuls_coefs_fts,
                                                                                                                                                                           percentage_of_nuls_coefs_fts))  # %% is % escaped in python strings

    # lets sort by mean coef value the remaining fts (they have non null coefs)
    PdCol_fts_contrib_abs_mean_only_nonnull_fts_sorted = PdCol_fts_contrib_abs_mean_only_nonnull_fts.sort_values("Mean Coefficient Estimate 10 Seeds", axis=0, ascending=False, kind='mergesort')  # axis=0 ie sort the rows
    # here is a list of the tables to keep for the coefs :
    # - PdCol_fts_contrib (the original list of values)
    # - PdCol_fts_contrib_abs_mean_only_nonnull_fts_sorted (the sorted list of selected fts, sorted by descending mean of absolute values of coefs)

    # fig.savefig(results_dir_for_this_run + 'Output_' + tag_LearningTaskType + "_" + tag_LearningAlgUsed + "-" + tag_ModelBuilt + "_" + tag_cond + "-" + tag_pop + "-" + tag_dataprofile + "_" + tag_NumTrial + '_ROCcurve.png'
    #
    # - saving the list of non null coefs features  # change results_dir_for_this_run line to add the separating /
    print(" >>>> Savings the tables containing views of the regression coefs")
    path_table_sorted_non_null_coefs = results_dir_for_this_run + 'Output_' + tag_LearningTaskType + "_" + tag_LearningAlgUsed + "_" + tag_ModelBuilt + "_" + tag_cond + "_" + tag_pop + "_" + tag_dataprofile + "_" + tag_NumTrial + "_NonNullCoefs.csv"
    path_table_original_list_of_coefs = results_dir_for_this_run + 'Output_' + tag_LearningTaskType + "_" + tag_LearningAlgUsed + "_" + tag_ModelBuilt + "_" + tag_cond + "_" + tag_pop + "_" + tag_dataprofile + "_" + tag_NumTrial + "_RawListOfCoefs.csv"
    PdCol_fts_contrib_abs_mean_only_nonnull_fts_sorted.to_csv(path_table_sorted_non_null_coefs, index=None, header=True)
    PdCol_fts_contrib.to_csv(path_table_original_list_of_coefs, index=None, header=True)
    print("+ A table with the only NON null coefs sorted in descending order is saved in this path : ")
    print(path_table_sorted_non_null_coefs)
    print("+ A table with the raw coefs, sorted in the features order is saved in this path : ")
    print(path_table_original_list_of_coefs)

    # - all the analysis is done : get the runtime
    runtime_analysis = duration_from(globalstart)
    print(bcolors.OKGREEN + " >>>> Analysis done. Time taken is : ", runtime_analysis, bcolors.ENDC)
    # end concatenated with "," instead of "+" to avoid TypeError: unsupported operand type(s) for +: 'datetime.timedelta' and 'str'
    # end of summary
    return



#######################################################################################
# --->

# --->

# ----------------------------------------------------------------------------------------------------------------------

# >>>>>> Part 2 : functions copied or customized from others authors.

# --->

# ----------------------------------------------------------------------------------------------------------------------

# end of file
# ----------------------------------------------------------------------------------------------------------------------