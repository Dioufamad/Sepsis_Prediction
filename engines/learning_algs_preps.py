#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Amad Diouf, amaddioufb13@gmail.com
# Created Date: 27/05/2022
# version ='1.0.0'
# ---------------------------------------------------------------------------
"""  this is the module for all functions related to
preparations needed by some learnings algorithms to run like intended
 """
# ---------------------------------------------------------------------------
# Imports
import os, os.path, sys, warnings, argparse # standard files manipulation set of imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler # for scaling

# ----------------------------------------------------------------------------------------------------------------------

# >>>>>> Part 1 : my own created functions.

# ---> a space of values maker

def space_of_values_maker_V1(space_type, min_val_given, max_val_given, initial_num_val_in_space,type_initial_values,
                             beef_up_space_dec, nameoflist_or_list_val_to_use_for_beef_up,
                             thin_out_space_dec, nameoflist_or_list_val_to_use_for_thin_out,
                             inverting_values_dec, final_sort_dec):
    # + possible values of arguments :
    # - space_type : "geom" or "log"

    # NB : this argument uses logspace() or geomspace() :
    # They both return numbers spaced evenly on a log scale (a geometric progression)
    # > logspace :
    # numpy.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0)
    # Return an array of values that starts at base ** start (base to the power of start) and ends with base ** stop
    # source : https://numpy.org/doc/stable/reference/generated/numpy.logspace.html#numpy.logspace
    # > geomspace :
    # numpy.geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0)
    # This is similar to logspace, but with the array returned starting and finishing with the endpoints specified directly
    # source : https://numpy.org/doc/stable/reference/generated/numpy.geomspace.html


    # - min_val_given : a float, as the minimum value given in order to set what will be the minimal value of the space,
    # taken directly as a value of the space if geomspace() is used or taken by a log function to make a value of the space if logspace() is used
    # - max_val_given : a float, as the maximum value given in order to set what will be the maximal value of the space,
    # taken directly as a value of the space if geomspace() is used or taken by a log function to make a value of the space if logspace() is used
    # - initial_num_val_in_space : an int, as the number of values to be initially in the space just after it is produced
    # - type_initial_values = float or int, used in the argument controlling the dtype of the values produced into the space of values

    # - beef_up_space_dec : "yes" or "no", as the decision to choose whether or not we want to add values (from a specific list) to the produced space
    # - nameoflist_or_list_val_to_use_for_beef_up : a string or a list, to choose the specific list of values to add
    # (a string here points to a specific list of values registered and a list is the list of values to add)
    # - thin_out_space_dec : "yes" or "no", as the decision to choose whether or not we want to remove values (from a specific list) from the produced space
    # - nameoflist_or_list_val_to_use_for_thin_out : a string or a list, to choose the specific list of values to remove
    # (a string here points to a specific list of values registered and a list is the list of values to remove)


    # > NB1 : the values added are usually default values for parameters of learning tasks, and they change from API to API.
    # e.g. : for regularization strength values in regression, sklearn has [1.0, ] and SPAMS has [1.0,0, 0.1, 0.05, 10, 0.001,0.01,0.0001, ].
    # To test these values, this list of values [0.0001, 0.001, 0.05,0.01, 0.1, 1.0, 10.0, 100.0] would be added to
    # beef up the space of values for regularization strength values to test.
    # > NB2 : a value that is typically removed is the 0.0 value. It is also most of the time excluded from the remarkable
    # values to test because it nullifies most of the time a parameter intended to have a value, hence changes the alg intended to run.
    # This can result in errors due to some others parameters that are not making sense anymore. In the event where a null
    # value has to be tested in order to put a param at its lowest strength, we first rely on producing a space of values with values close to 0.

    # - inverting_values_dec : "yes" or "no", as the decision to choose whether or not we want to invert the values in the space
    # > NB : designed to mainly be used for regularization in regression, to go from lambda values to C values (C=1/lambda and is the reg strength param in sklearn)
    # - final_sort_dec : "yes" or "no", as the decision to choose whether or not we want to sort the values in the final stage of the space
    # > NB : adding or removing remarkable values is always ended with a step of sorting the space again, so this only necessary if no remarkable value have been added or removed)

    # + examples of calls for this function :
    # > a call using logspace() can be with these values (11 values obtained) :
    # space_type="log", min_val_given=-4, max_val_given=1, initial_num_val_in_space=5, type_initial_values=float,
    # beef_up_space_dec="yes", nameoflist_or_list_val_to_use_for_beef_up = "MajorTestingValuesInRegul1",
    # thin_out_space_dec="yes", nameoflist_or_list_val_to_use_for_thin_out = "OnlyTheNullVal",
    # inverting_values_dec="no", final_sort_dec="yes"
    # > a call using geomspace() can be with these values (206 values obtained) :
    # space_type="geom", min_val_given=0.0001, max_val_given=100.0, initial_num_val_in_space=200, type_initial_values=float,
    # beef_up_space_dec="yes", nameoflist_or_list_val_to_use_for_beef_up = "MajorTestingValuesInRegul1",
    # thin_out_space_dec="yes", nameoflist_or_list_val_to_use_for_thin_out = "OnlyTheNullVal",
    # inverting_values_dec="no", final_sort_dec="yes"

    # + getting the different values of the space (default is a geomspace)
    if space_type=="log": # the min and max values given will be used in a logspace() to define the extremes of the space (you make a log with them to get the values of the space)
        space_of_values = np.logspace(min_val_given, max_val_given, num=initial_num_val_in_space, dtype=type_initial_values) # the space is log(min_val_given) to log(max_val_given) for a total of initial_num_val_in_space values
    elif space_type=="geom": # space_type=="geom" # the min and max values given will be used in a geomspace() to define the extremes of the space (they are directly values of the space)
        space_of_values = np.geomspace(min_val_given, max_val_given, num=initial_num_val_in_space, dtype=type_initial_values) # the space is min_val_given to max_val_given for a total of initial_num_val_in_space values
    else: # (to make sure a space is created, we set the default as a geomspace)
        space_of_values = np.geomspace(min_val_given, max_val_given, num=initial_num_val_in_space, dtype=type_initial_values)

    # + if desired, adding to the space initially created, a list of remarkable values that we want to also include in it
    # > the idea :
    # step 1 : check if the list of values to add is already registered and indicate by its name, or it is a list given by the user (in any case, a list of val to ad is retrieved)
    # step 2 : update the space of values initially created by adding to it any value that is not already in it
    # NB : with this, final size of space can be different from "initial_size_space + size_list_of_values_to_add"
    # (ie if some values are already in the space of values initially created)
    if beef_up_space_dec == "yes":
        # - the default list of values to add is an empty list (...and the choice of the user changes it or not)
        list_of_values_to_add = []
        # - using a list of values registered here under a specific name
        if isinstance(nameoflist_or_list_val_to_use_for_beef_up, str):
            if nameoflist_or_list_val_to_use_for_beef_up == "MajorTestingValuesInRegul1": # has remarkable regularization strength values common in tests from the librairies sklearn and SPAMS
                list_of_values_to_add = [0.0001, 0.001, 0.05, 0.01, 0.1, 1.0, 10.0, 100.0]
            # # use this to register here another list of values with a different name
            # elif nameoflist_or_list_val_to_use_for_beef_up == "NameOfSpecificListOfValuesRegisteredBelow":
            #     list_of_values_to_add = ListOfValuesRegistered
        # - using a list of values given by the user
        elif isinstance(nameoflist_or_list_val_to_use_for_beef_up, list):
            list_of_values_to_add = nameoflist_or_list_val_to_use_for_beef_up
        # - adding the values...
        for a_value_to_add in list_of_values_to_add:
            if a_value_to_add not in space_of_values:
                space_of_values = np.append(space_of_values, a_value_to_add)
        # - sorting the space of values (to have again the original sorted presentation of the space of values just like as it was produced)
        # NB : this is elegant but can be also usefull as even if the user dont choose to sort the values in the final stage of the function,
        # he is sure to make his tests with values of the param that are increasing
        space_of_values = np.sort(space_of_values)

    # + if desired, inverting the values of the space
    if inverting_values_dec=="yes":
        space_of_values = 1 / space_of_values  # 1/previous_max to 1/previous_min (~initial_num_val_in_space C values including the default value 1=1/previousLambdaIs1)

    # + if desired, removing from the space, after all additions and modifications of values, a list of remarkable values that we want to also include in it
    if thin_out_space_dec == "yes":
        # - the default list of values to remove is an empty list (...and the choice of the user changes it or not)
        list_of_values_to_rm = []
        # - using a list of values registered here under a specific name
        if isinstance(nameoflist_or_list_val_to_use_for_thin_out, str):
            if nameoflist_or_list_val_to_use_for_thin_out == "OnlyTheNullVal":
                list_of_values_to_rm = [0]
            # # use this to register here another list of values with a different name
            # elif nameoflist_or_list_val_to_use_for_thin_out == "NameOfSpecificListOfValuesRegisteredBelow":
            #     list_of_values_to_rm = ListOfValuesRegistered
        # - using a list of values given by the user
        elif isinstance(nameoflist_or_list_val_to_use_for_thin_out, list):
            list_of_values_to_rm = nameoflist_or_list_val_to_use_for_thin_out
        # - removing the values...
        for a_value_to_rm in list_of_values_to_rm:
            space_of_values = [val_i for val_i in space_of_values if val_i != a_value_to_rm]
        # - sorting the space of values (to have again the original sorted presentation of the space of values just like as it was produced)
        space_of_values = np.sort(space_of_values)

    # + keeping only the unique values of the list
    space_of_values_unik_only = np.unique(space_of_values)
    # + changing the space of value from array to list
    space_of_values_unik_only_as_list = list(space_of_values_unik_only)

    # + if desired, making a final sort of the full space of values (only necessary if no remarkable value have been added or removed)
    if final_sort_dec == "yes":
        space_of_values_unik_only_as_list = sorted(space_of_values_unik_only_as_list)

    # + get the final space of values and its size
    final_space_of_values = space_of_values_unik_only_as_list
    size_final_space_of_values = len(final_space_of_values)

    return final_space_of_values, size_final_space_of_values


# ---> A function building a gallery of regularization strengths values setup as "always with lambda1 and/or lambda2 and/or lambda3",
# with the possibility of any of them being in the form of a C=1/lambda

# NB1 : by setting this up like this, we open up the possibilities for the user when composing a gallery of regularization strengths for regression tasks best model selection
# NB2 : in this version 1 of the function, the argument giving "the possibility of any of the lambda_i being in the form of a C=1/lambda_i"
# is present for all 3 lambda_i values but only the lambda1 values has the option to change into a C=1/lambda.
# This is because the C form has only been observed in implementations for the lambda1 value and we dont know how to call it for other lambda_i values.
# In the future, if the C form is met for other lambda_i values and a specific name is given to it, this fonction can be
# copied and modified to be a version 2 that will add the option to change into C value for the lambda_i value concerned by that change.

def dict_lambdas123_space_of_values_maker_V1(space_type1="log", min_val_given1=-4, max_val_given1=1, initial_num_val_in_space1=5, type_initial_values1=float,
                                         beef_up_space_dec1="yes", nameoflist_or_list_val_to_use_for_beef_up1="MajorTestingValuesInRegul1",
                                         thin_out_space_dec1="yes", nameoflist_or_list_val_to_use_for_thin_out1="OnlyTheNullVal",
                                         inverting_values_dec1="no", final_sort_dec1="yes",
                                         space_type2="log", min_val_given2=-4, max_val_given2=1, initial_num_val_in_space2=0, type_initial_values2=float,
                                         beef_up_space_dec2="yes", nameoflist_or_list_val_to_use_for_beef_up2="MajorTestingValuesInRegul1",
                                         thin_out_space_dec2="yes", nameoflist_or_list_val_to_use_for_thin_out2="OnlyTheNullVal",
                                         inverting_values_dec2="no", final_sort_dec2="yes",
                                         space_type3="log", min_val_given3=-4, max_val_given3=1, initial_num_val_in_space3=0, type_initial_values3=float,
                                         beef_up_space_dec3="yes", nameoflist_or_list_val_to_use_for_beef_up3="MajorTestingValuesInRegul1",
                                         thin_out_space_dec3="yes", nameoflist_or_list_val_to_use_for_thin_out3="OnlyTheNullVal",
                                         inverting_values_dec3="no", final_sort_dec3="yes"):

    # + examples of calls for this function :
    # > the default call of the function with dict_lambdas123_space_of_values_maker_V1() gives a gallery of values containing 11 lambda1 values

    # + making an empty dict as a collector of the space of values produced for each lambda_i (we make also a collector of the size of each lambda_i space)
    dict_lambdas123 = {}
    dict_lambdas123_counts = {}

    # + for each of lambda1, lambda2 and lambda3, we do these 3 steps : values are produced and added in this order in the collector dict (their count also is added to the dict of lambda_i counts):
    # > we get a gallery of 2 items from the function space_of_values_maker_V1() : element 1 is the list of lambda_i values and element 2 is its length
    # > we add in the collector of the space of values produced for each lambda_i, the element 1
    # > add in the collector of the size of each lambda_i space, the element 2
    if initial_num_val_in_space1 > 0 :
        lambda1_reg_strength_gallery = space_of_values_maker_V1(space_type1, min_val_given1, max_val_given1, initial_num_val_in_space1,type_initial_values1,
                                                                beef_up_space_dec1, nameoflist_or_list_val_to_use_for_beef_up1,
                                                                thin_out_space_dec1, nameoflist_or_list_val_to_use_for_thin_out1,
                                                                inverting_values_dec1, final_sort_dec1)
        if inverting_values_dec1 == "no":
            dict_lambdas123['lambda1'] = lambda1_reg_strength_gallery[0]
            dict_lambdas123_counts['lambda1'] = lambda1_reg_strength_gallery[1]
        else:
            dict_lambdas123['C'] = lambda1_reg_strength_gallery[0]
            dict_lambdas123_counts['C'] = lambda1_reg_strength_gallery[1]
    if initial_num_val_in_space2 > 0:
        lambda2_reg_strength_gallery = space_of_values_maker_V1(space_type2, min_val_given2, max_val_given2, initial_num_val_in_space2,type_initial_values2,
                                                                beef_up_space_dec2, nameoflist_or_list_val_to_use_for_beef_up2,
                                                                thin_out_space_dec2, nameoflist_or_list_val_to_use_for_thin_out2,
                                                                inverting_values_dec2, final_sort_dec2)
        dict_lambdas123['lambda2'] = lambda2_reg_strength_gallery[0]
        dict_lambdas123_counts['lambda2'] = lambda2_reg_strength_gallery[1]
    if initial_num_val_in_space3 > 0:
        lambda3_reg_strength_gallery = space_of_values_maker_V1(space_type3, min_val_given3, max_val_given3, initial_num_val_in_space3,type_initial_values3,
                                                                beef_up_space_dec3, nameoflist_or_list_val_to_use_for_beef_up3,
                                                                thin_out_space_dec3, nameoflist_or_list_val_to_use_for_thin_out3,
                                                                inverting_values_dec3, final_sort_dec3)
        dict_lambdas123['lambda3'] = lambda3_reg_strength_gallery[0]
        dict_lambdas123_counts['lambda3'] = lambda3_reg_strength_gallery[1]

    return dict_lambdas123, dict_lambdas123_counts

# ---> A function as the SPAMS group membership vector maker
# + NB1 : this is designed for a use in Multitask Regression done with the library SPAMS.
# Here is the principle of multitask and why we need the SPAMS group membership vector...

# - The objective in Multitask learning is that, instead of "using one dataset within a learning task to make a model",
# we "pull together multiple datasets and hopefully get a better model by learning from all those datasets pulled together".
# This is integration of information.
# - Multitask Regression can be done using the library SPAMS, and this requires to feed in a "superdataset" instead of one single dataset.
# - The superdataset (lets call it "D_all") is a reunion of n datasets "d_i", each containing the same x features "f_j".
# - All versions of the same feature "f_j" found, one in each dataset "d_i", are called copies of the feature "f_j".
# ( ie "D_all" has n copies of the feature f_1, n copies of the feature f_2, ...n copies of the feature f_x).
# - This is how the multitask, by using the superdataset "D_all", puts together the information from multiples datasets "d_i" and
# tries to make a better model in comparison to using the feature information from just one single dataset.
# - When "D_all" is supplied to the multitask learning task, there is also another requirement : the learning task have to be able
# to recognize all n copies of a feature "f_j" as information for only the feature "f_j".
# This is done by using a membership vector that is a 1-D array of the same size as the number of fts in "D_all",
# with each entry of the array being an index that stays the same for all copies of a feature (e.g. : the 3rd feature among
# the x features common to all datasets "d_i" will be marked 3 whatever the copy it is, so all the values 3 in the membership vector
# are for that 3rd feature copies)

# + NB 2 : Copies of a feature inside the superdataset "D_all" have specific names in order to differentiate them by dataset of origin (a cohort).
# The default naming that garanties of this specificity is arbitrary and depends on user.
# Here, the following naming is used and considered for the function :
# "NameOfTheFeatureInSuperdataset" = "TrueNameOfTheFeature_in_CohortOfOriginOfTheFeature")

def SPAMS_grp_membership_vector_maker_V1(list_all_fts_df_input, ftname_cohort_sep="_in_"):

    # list_all_fts_df_input = ["_synth_ft","a_in_dt1","b_in_dt1","c_in_dt1","a_in_dt2","b_in_dt2","c_in_dt2","c_in_dt3","b_in_dt3","a_in_dt3"] # to test

    # + making a list of the uniques "TrueNameOfTheFeature" in the order they are met when going through the entire list of features in "D_all"
    list_all_fts_no_tag_cohort = []
    for ft in list_all_fts_df_input:
        ft_no_tag_cohort = ft.split(ftname_cohort_sep)[0]
        if ft_no_tag_cohort not in list_all_fts_no_tag_cohort:
            list_all_fts_no_tag_cohort.append(ft_no_tag_cohort)
    # + making a dict as a unique "TrueNameOfTheFeature" - the number of the position where it has been first found in the entire list of features in "D_all"
    # (can be a value from 1 to length list_all_fts_no_tag_cohort)
    dict_ftnottag_indexplus1 = {}
    for ftnotag in list_all_fts_no_tag_cohort:
        dict_ftnottag_indexplus1[ftnotag] = list_all_fts_no_tag_cohort.index(ftnotag) + 1
    # + making a dict as "NameOfTheFeatureInSuperdataset" - the number of the position where it has been first found in the entire list of features in "D_all"
    dict_ftwtag_index_ftnotag = {}
    list_ftnotag_in_ledger_of_positions = list(dict_ftnottag_indexplus1.keys())
    for ftwtag in list_all_fts_df_input:
        # print("- ftwtag is: ", ftwtag)
        ftwtag_in_wotag_state = ftwtag.split(ftname_cohort_sep)[0]
        # print("- ftwtag_in_wotag_state is: ", ftwtag_in_wotag_state)
        for it_might_the_corresponding_ftnotag in list_ftnotag_in_ledger_of_positions:
            # print("-- it_might_the_corresponding_ftnotag is: ", it_might_the_corresponding_ftnotag)
            if ftwtag_in_wotag_state == it_might_the_corresponding_ftnotag :
                dict_ftwtag_index_ftnotag[ftwtag] = dict_ftnottag_indexplus1[it_might_the_corresponding_ftnotag]
                # print("--- yes, it is the corresponding feature and index atrtibuted is : ", dict_ftnottag_indexplus1[it_might_the_corresponding_ftnotag])
    # + making a list of the number of the position where it has been first found in the entire list of features in "D_all"
    list_of_group_membership_ftswtag = list(dict_ftwtag_index_ftnotag.values())
    # + making the membership array to use for multitask learning
    groups_in_data = np.array(list_of_group_membership_ftswtag, dtype=np.int32)

    return groups_in_data


# ---> A function making the list of seeds used during the model selection for a learning method tested

# NB1 : input is an int, advised to be >=3,
# NB2 : 10 is the default number of seeds to use if an undesirable value is used as input.

def list_seeds_maker(num_seeds):
    # + step 1 :  setting an acceptable number of seeds to use, depending on if an undesirable value is used as input
    if num_seeds < 3:
        # if a number of seeds less than 3 is used, we set the default value at 10 and print a warning
        num_seeds_to_use = 10
        print("Number of seeds set is < 3. Advised number is >=3. Reverting to using default value of 10 seeds !")

    else:
        # if a number of seeds more or equal to 3 is used, we accept this value as the one to use
        num_seeds_to_use = num_seeds
    # + step 2 : making the list of seeds to use
    list_seeds = list(range(num_seeds_to_use))
    return list_seeds

# --->

# --->

# ----------------------------------------------------------------------------------------------------------------------

# >>>>>> Part 2 : functions copied or customized from others authors.

# --->

# ----------------------------------------------------------------------------------------------------------------------

# end of file
# ----------------------------------------------------------------------------------------------------------------------