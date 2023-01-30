#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Amad Diouf, amaddioufb13@gmail.com
# Created Date: 27/05/2022
# version ='1.0.0'
# ---------------------------------------------------------------------------
"""  This is the module for all functions related to defining scorers and computing metrics for learning tasks """
# ---------------------------------------------------------------------------
# Imports
import os, os.path, sys, warnings, argparse # standard files manipulation set of imports
import numpy as np
from sklearn.preprocessing import StandardScaler # for scaling
import pandas as pd # to manipulate dataframes as collectors of features contribution in models
from engines.data_mgmt_engine import dict_val_updater_valueaslist_V1 # to update a dict, precisely for each key update its value that is a list
# to define a scoring to use or compute some metrics
from sklearn.metrics import make_scorer,f1_score,matthews_corrcoef,cohen_kappa_score,balanced_accuracy_score,accuracy_score,precision_score,recall_score,roc_auc_score
from engines.data_mgmt_engine import arr_evenly_spaced_val_maker_V1 # to make arr of evenly spaced values to base interpolation on
# imports dedicated to drawing roc curves
from sklearn.metrics import roc_curve, auc  # to compute fpr,tpr, auc needed to draw roc curves
import matplotlib # change the backend used by matplotlib from the default 'QtAgg' to one that enables interactive plots 'Qt5Agg'
matplotlib.use('Qt5Agg')
matplotlib.get_backend()
import matplotlib.pyplot as plt # used to make plots # for roc curves
import seaborn as sns
# for sns plots, add plt.show() to make the plots appear (replace it with the "%matplotlib inline" in the case of a notebook
# to let the backend solve that issue while displaying the plots inside the notebook)
from textwrap import wrap # to wrap plot titles and make they not go past figures limits
# ----------------------------------------------------------------------------------------------------------------------

# >>>>>> Part 1 : our own created functions.

# ---> F1 score :
# source : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
# NB1 : F1 = 2 * (precision * recall) / (precision + recall)
# NB2 : The F1 score, in most of our analysis will be used as the main metric to rank and select predictive models. This is because,
# even if correlation coefficient based metrics like MCC or the cohen's kappa score appear as the best metrics by default when
# using prediction data from a confusion matrix, they have the weakness of being undefined in some cases (ie when their denominator is null).
# The F1 score is still robust against imbalance of classes in the dependant var while it does not have the weakness of being undefined in some cases.
# Also, for comparative purposes with already published models, the F1 score is most of the time reported and we can directly compare our model performance
# with what is published in contrary of the situation where we would have used less "popular even if better" metrics like the MCC

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

# ---> Correlation coefficient based metrics :
# when using prediction data from a confusion matrix, these are considered the most robust against imbalance of classes in the dependant var
# and others bias in the data that might impact the metric computation.
# NB : they have the weakness of being undefined in some cases (ie when their denominator is null)

# + Matthew’s correlation coefficient (MCC):
# source : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html#sklearn.metrics.matthews_corrcoef
# NB1 : In the binary (two-class) case, MCC = [(TPxTN)+(FPxFN)] / square_root[(TP+FP)x(TPxFN)x(TNxFP)x(TNxFN)]
# NB2 : The Matthews Correlation Coefficient takes into account true and false positives and negatives and is generally regarded as a balanced measure
# which can be used even if the classes are of very different sizes. The MCC is in essence a correlation coefficient value between -1 and +1.
# A coefficient of +1 represents a perfect prediction, 0 an average random prediction and -1 an inverse prediction.

# to compute the MCC, we use sklearn.metrics.matthews_corrcoef(y_true, y_pred)
def MCC_V1(actual, prediction):
	mcc_v1_val = matthews_corrcoef(actual, prediction)
	return mcc_v1_val

# + cohen_kappa_score :
# source : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html#sklearn.metrics.cohen_kappa_score
# NB1 : Cohen’s kappa score expresses the level of agreement between two annotators on a classification problem. For more on its formula and computations, see source
# NB2 : Cohen’s kappa score is a number between -1 and 1. The maximum value means complete agreement; zero means the lowest chance of agreement and -1 complete disagreement.
# Hence, this can be seen as a "copycat" of the MCC metric.

# Computing  the cohen_kappa_score is done with sklearn.metrics.cohen_kappa_score(y1, y2, weights={None, ‘linear’, ‘quadratic’}) where :
# ...(y1 and y2) are (y_true and y_pred). The kappa statistic is symmetric, so swapping y1 and y2 (ie swapping y_true and y_pred) doesn’t change the value.
# ... weights= is for weighting type to calculate the score. None means no weighted; “linear” means linear weighted; “quadratic” means quadratic weighted.
# In here (https://stats.stackexchange.com/questions/59798/quadratic-weighted-kappa-versus-linear-weighted-kappa), it is explained that
# > linear weighting sets the penalty for a non agreement between 2 annotators as a constant
# >...while the quadratic weighting sets makes the penalty grow larger with the share that the disagreement represents
# so to compute the cohen_kappa_score, we use:
# > sklearn.metrics.cohen_kappa_score(y_true, y_pred, weights=None) # to not account for imbalance in the labels of the dependant var
def CK_non_weighted(actual, prediction):
	ck_non_weighted_val = cohen_kappa_score(actual, prediction,weights=None)
	return ck_non_weighted_val
# > sklearn.metrics.cohen_kappa_score(y_true, y_pred, weights=‘linear’) # to account for imbalance
def CK_weighted_as_linear(actual, prediction):
	ck_weighted_as_linear_val = cohen_kappa_score(actual, prediction,weights="linear")
	return ck_weighted_as_linear_val

# ---> Other traditionnal metrics :

# + accuracy score :
# source : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
# NB1 : computes accuracy classification score ie the fraction of correct predictions among all predictions
# ie accuracy = (TP+TN) / (TP+TN+FP+FN)
# NB2 : best value is 1 and worst value is 0

# to compute the accuracy score, we use :
def acc_V1(actual,prediction):
	acc_v1_val = accuracy_score(actual, prediction, normalize=True)
	return acc_v1_val

# + balanced accuracy score :
# source : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score
# NB1 : computes the balanced accuracy as a version of accuracy in classification problems geared towards dealing with imbalanced datasets.
# The balanced_accuracy_score avoids inflated performance estimates on imbalanced datasets by being the macro-average of recall scores per class
# ie (raw accuracy where each sample is weighted according to the inverse prevalence of its true class). Thus for balanced datasets, the score is equal to accuracy.
# NB2 : balanced accuracy is defined as the average of recall obtained on each class
# ie balanced accuracy = (recall on positive class + recall on negative class) / 2
# and we know that...
#...recall on positive class = TP / (TP + FN) = sensitivity
#... and recall on negative class = TN / (TN + FP) = specificity
#...hence balanced accuracy = (sensitivity + specificity) / 2
# NB3 : best value is 1 and worst value is 0

# to compute the balanced accuracy score, we use :
def bal_acc_V1(actual,prediction):
	bal_acc_v1_val = balanced_accuracy_score(actual, prediction)
	return bal_acc_v1_val


# + precision score :
# source : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score
# NB1 : Compute the precision ie the ability of the classifier not to label as positive a sample that is negative
# NB2 : precision = TP / (TP + FP)
# NB3 : best value is 1 and worst value is 0

# to compute the precision score, we use :
# > for binary classification but not weighted (ie not specially geared towards accounting for imbalance)
def prec_as_binary(actual,prediction):
	prec_as_binary_val = precision_score(actual, prediction, average='binary')
	return prec_as_binary_val
# > for each label and weighted (ie geared towards accounting for imbalance by calculating the metric for each label and finding its average weighted by support)
def prec_as_weighted(actual,prediction):
	prec_as_weighted_val = precision_score(actual, prediction, average='weighted')
	return prec_as_weighted_val

# + recall score :
# source : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score
# NB1 : Compute the recall ie the ability of the classifier to find all the positive samples (when the focus is on the positive class)
# NB2 : recall = TP / (TP + FN)
# NB3 : a version of the recall can be computed for each class.
# Depending on the class we are focusing on, the recall is same as sensitivity or specificity  :
# ...if the focus is on the positive class, the recall can be called "sensitivity in the classification problem" and
# we have recall = sensitivity = TP / (TP + FN)
# ...if the focus is on the negative class, another version of the recall can be computed for this negative class and
# it is equal to the "specificity in the classification problem"; we have recall_on_negative_class = specificity = TN / (TN + FP)
# NB4 : best value is 1 and worst value is 0

# to compute the recall score, we use :
# > for binary classification but not weighted (ie not specially geared towards accounting for imbalance)
def rec_as_binary(actual,prediction):
	rec_as_binary_val = recall_score(actual, prediction, average='binary')
	return rec_as_binary_val
# > for each label and weighted (ie geared towards accounting for imbalance by calculating the metric for each label and finding its average weighted by support)
def rec_as_weighted(actual,prediction):
	rec_as_weighted_val = recall_score(actual, prediction, average='weighted')
	return rec_as_weighted_val

# + A point on how to differenciate accuracy, precision and recall as metrics in a classification problem :
# - accuracy :  is "the number of correct predictions made divided by the total number of predictions made"
# ie estimates the "correctness of the model, all classes considered"
# - precision : estimates the "correctness of the model, when looking at everything it called as positive"
# ie "how precise it predicts, when it predicts a class"
# - recall : estimates the "correctness of the model, when looking at everything that was really inside a whole class"
# ie "how much of members of the class of interest, it can retrieve among everything that looks like truly or falsely like the class of interest"

# + roc_auc_score :
# source : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
# NB1 : compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
# NB2 : the metric with binary as it is but some restrictions apply for multiclass and multilabel classification (see source)
# NB3 : Even if a ROC Curve is always drawn and outputed for the classification problems, we use this metric as "a value that summarizes
# the performance displayed on the ROC Curve of the model"
# NB4 : best value is 1 and worst value is 0

# to compute the roc_auc_score, we use :
# > for binary classification but not weighted (ie not specially geared towards accounting for imbalance) # computes the metrics for each label and find the unweighted mean
def roc_auc_score_as_macro(actual,prediction):
	roc_auc_score_as_macro_val = roc_auc_score(actual, prediction, average='macro')
	return roc_auc_score_as_macro_val
# > for each label and weighted (ie geared towards accounting for imbalance by calculating the metric for each label and finding its average weighted by support)
def roc_auc_score_as_weighted(actual,prediction):
	roc_auc_score_as_weighted_val = roc_auc_score(actual, prediction, average='weighted')
	return roc_auc_score_as_weighted_val

# ---> Custom metrics :

# + Mixed Metric of 4 (MM4)

# NB : this stems from our own attempt to reduce, on the overall metric value returned, the impact of MCC zero values from undefined situations (when there is a zero division).
# The idea was to make a metric that will have the strength of the MCC when MCC is defined but reduce the weakness of the MCC when it is undefined.
# How : the metric is an average of 4 metrics :
# - 2 metrics are the MCC and the weighted for imbalanced version cohen kappa score (a metric similar to MCC) that are both giving a lot of MCC-like strength
# - 2 metrics are the weighted for imbalanced version of f1score and the balanced accuracy score
# (both metrics not as powerful as the MCC but still valuable against imbalance) that are both less at risks of a zero division
# - we choose to a tak the absolute value for the MCC and the cohen kappa score. And we make an average of the 4 metrics obtained
# to get a value ranging from 0 (worst value) to 1 (best value)

# sources for the metrics used :
# - for MCC : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html#sklearn.metrics.matthews_corrcoef
# - for cohen kappa score : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html#sklearn.metrics.cohen_kappa_score
# - for the weighted for imbalanced version of f1score : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
# - for the balanced accuracy  score : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score

# NB1 : issues of this custom metric
# - issue 1 with this metric : it is a "0 to 1 metric" and not a symmetrical centered on 0 metric hence lacks the symmetrical aspect of the correlation coefficient based metrics.
# - issue 2 with the metric : this metric has not been field tested very much in order to sort out all of the caviats and blindspots it could have...

# NB2 : how to more adequately make the average of multiples metrics to get a custom metric ? how to use a custom metric like this in the gridsearchcv ?
# source : https://stackoverflow.com/questions/31615190/sklearn-gridsearchcv-scoring-function-error

# to compute the MM4, we use :
def MM4_bespoke_score(actual,prediction):
	mcc_abs_val = abs(MCC_V1(actual, prediction))
	ck_abs_val = abs(CK_weighted_as_linear(actual, prediction))
	f1_sc_weighted_val = f1score_as_weighted(actual, prediction)
	bal_acc_val = bal_acc_V1(actual, prediction)
	MM4_bespoke_score_value = (mcc_abs_val + ck_abs_val + f1_sc_weighted_val + bal_acc_val) / 4
	return MM4_bespoke_score_value


# ---> define our scorer for GridSearchCV()
# NB : to be used within GridSearchCV() with argument scoring (scoring=grid_scorer)
def defining_the_scorer_in_gridsearch(tag_scorer):
	if tag_scorer == "f1score_as_weighted":  # for f1score_as_weighted as the most advised for the gridscorer due to its better management of unbalance of datasets
		grid_scorer = make_scorer(f1score_as_weighted, greater_is_better=True)
	elif tag_scorer == "f1score_as_binary":  # for f1score_as_binary" as the most commonly used f1 when in a binary classification problem
		grid_scorer = make_scorer(f1score_as_binary, greater_is_better=True)
	elif tag_scorer == "MM4_bespoke_score":  # for "Mixed Metric of 4", an experimental metric including correlation coefficient based metrics MCC and weighted Cohen's kapp, the f1 weighted and the balanced accuracy
		grid_scorer = make_scorer(MM4_bespoke_score, greater_is_better=True)
	else:
		grid_scorer = make_scorer(f1score_as_weighted, greater_is_better=True) # default scorer is f1score_as_weighted()
		print("Warning : Non recognized tag for a scorer to use for model selection in gridsearch.")
		print("Please supply a tag pointing to a scorer implemented ! If not done, proceeding analysis will be carried out with f1score_as_weighted as scorer")
	return grid_scorer


# ---> A group of functions geared towards collecting and computing all values estimating the qualities of the models built

# + The idea :
# -  each model built have 2 different qualities that we want to estimate :
# > the features contribution in the model
# > the metrics values recorded to estimate the performance of the model

# - For each of those 2 aspects, we need to :
# > before running any iteration of the model, create the collectors that will receive the values recorded at each iteration
# > after running each iteration of the model, we supply the collectors with the value recorded for each corresponding aspect
# > after running all iterations of the model, a collectors might need to have some final actions done on it
# (e.g. : making computations from its content, drawing plot from its content, etc.)

# NB:  In terms of functions to create for this, we have these remarks :
# > these functions are initially dedicated to classifications tasks (using a classification alg or a logistic regression alg)
# > the feature contributions are differently defined from logistic regressions to classifications, so for creating the collector as well as supplying it,
# we prefer to have a dedicated function for regressions and one for classifications
# > the metrics values recorded are generally the same for classifications tasks (logistic regression included) so,
# whether it is a classification alg or a logistic regression alg, the function to create the collector is the same and
# the function to supply the collector is the same also.


# + functions for making the collector of the qualities of a model built (features contribution and metrics values recorded) :

# - 1/2 : for making the collector of the features contribution in the model
def collector_fts_contrib_type_RegrFtsCoefs_maker_V1(list_all_fts_df_input):
	# + NB : this version of the feature contributions collector is designed for
	# fts coefficients values obtained from regressions,
	# hence here feature contributions are fts coefs

	# + Collector_1a  : a df that later will have columns set up like this
	# "col 1 contains the list of fts", "col seed_1 to col last_seed contain each the fts coefs of the best model for the seed corresponding"
	PdCol_fts_contrib = pd.DataFrame()  # create the df
	PdCol_fts_contrib["Features"] = list_all_fts_df_input  # make the col of the fts
	# + Collector_1b :  a list of the colnames containing coefs at each seed
	# ( this is useful later to access directly the cols with fts coefs and change them into absolute values)
	list_of_cols_fts_contrib = []
	return PdCol_fts_contrib, list_of_cols_fts_contrib

def collector_fts_contrib_type_ClassifFtsImportance_maker_V1(list_all_fts_df_input):
	# + NB : this version of the feature contributions collector is designed for
	# fts importance value obtained from classifications,
	# hence here feature contributions are fts importance

	##! make the following in the fashion of regr but adapt it for classif
	#~~~
	# + Collector_1a  : a df that later will have columns set up like this
	# "col 1 contains the list of fts", "col seed_1 to col last_seed contain each the fts importance of the best model for the seed corresponding"
	PdCol_fts_contrib = pd.DataFrame()  # create the df
	PdCol_fts_contrib["Features"] = list_all_fts_df_input  # make the col of the fts
	# + Collector_1b :  a list of the colnames containing fts importance at each seed
	# ( this is useful later to access directly the cols with fts importance and change them into absolute values)
	list_of_cols_fts_contrib = []
	# ~~~
	return PdCol_fts_contrib, list_of_cols_fts_contrib

# - 2/2 : for making the collector of the metrics values recorded ("one-value-metrics" and "metrics values used to plot ROC curves")
def collector_metrics_type_MadeFromTheConfMatrix_maker_V1():
	# + NB1 : this will collect, at each seed, metrics computed using values of the confusion matrix.
	# This includes these 2 types of metrics :
	# - the ones with the resulting value of the metric being one value (e.g. : F1, MCC, balanced accuracy, recall, etc)
	# - the ones used when plotting ROC curves (fpr values, tpr values, auc value)
	# + NB2 : this is dedicated to metrics but will also collect the hyperparameter values that specify the model being estimated

	# ---> Part 1/2 : making the collector for the hyperparameter values that specify the model being estimated
	# + Collector_2a :  a dict with
	# key as "a name of a hp"
	# and value as "a list of values that have been the best value for the said hp at a seed"
	DictCol_bestval_HPsExplored = {}
	# NB : the list looped on to know which are the HPs explored is "list_names_HPs_explored" and is created from the param_grid used

	# ---> Part 2/2 : making the collector for the metrics values recorded

	# + Collector_2b :
	# a list where to stash the validation score obtained at each seed by the best model (from gridscorer)
	ListCol_val_scores = []

	# + Collector_2c :  a dict with
	# key as "tag of a metric to compute for testscore" and
	# value as "a list of values that are the test scores of the best models, one at each seed"
	DictCol_test_scores = {}
	# NB : the list looped on to know which metrics to compute is "list_tags_metrics_computed_for_testscore" and is defined by operator

	# + Collector_2d : the collectors of the metrics values used to plot a ROC curve at each seed
	# This includes, at each seed, 3 pieces of info :
	# - the array of fpr values,
	# - the array of tpr values,
	# - and the auc value
	# + NB2 : What each piece of info is needed for ?
	# - for each seed, all 3 pieces of info are useful to draw the ROC curve at that seed.
	# - for the average ROC curve over all seeds, the arrays of tpr values collected are used to compute the array of the average tpr values;
	# the auc values are used to compute the average auc value; the array of fpr values of the average ROC curve is arbitrarily chosen by the operator,
	# with the only requirement being that it is from 0 to 1, 0 and 1 included.
	fprs_col_by_seed_one_alg = []  # a list to stash the arrays of fpr values for each iteration
	tprs_col_by_seed_one_alg = []  # a list to stash the arrays of tpr values for each iteration
	aucs_col_by_seed_one_alg = []  # a list to stash the auc values for each iteration

	return DictCol_bestval_HPsExplored, ListCol_val_scores, DictCol_test_scores, fprs_col_by_seed_one_alg, tprs_col_by_seed_one_alg, aucs_col_by_seed_one_alg


# + functions for supplying the collector of the qualities of a model built (features contribution and metrics values recorded) :
# NB1 : in each function, the objective is to use the model built at the present iteration (best model at the seed)
# in order to extract or compute the info needed, then stash the info needed in the corresponding collector
# NB2 : these functions are used as updaters of previously created entities hence they dont return nothing.

# - 1/2 : for supplying the collector of the features contribution in the model
def collector_fts_contrib_type_RegrFtsCoefs_supplier_V1(modelselector_by_GSCV, PdCol_fts_contrib, list_of_cols_fts_contrib, a_seed):

	# ---> make the computations to stash the features contribution for the seed

	# + stash the coefficients of the best estimator
	# - put the coefs into a structure that is appropriate for entry into their collector
	# NB : best_estimator_.coef_ is here is a 2D array of shape (num dep var, num fts), ie if we have 1 dep var its is a 2D array of shape (1, num fts).
	# Knowing that the coefs collector is a df "1 col is for coefs of best model of a seed", we need to flatten the 2D array of shape (1, num fts)
	# into a 1D array of shape (num fts,)to make it usable as the values inside a column,
	coefs_best_estimator_as_2d_arr = modelselector_by_GSCV.best_estimator_.coef_
	coefs_best_estimator_as_1d_arr = coefs_best_estimator_as_2d_arr.flatten()
	# - making the name of the column that will accept the coefs values
	colname_fts_contrib_values = "Coefficient Estimate Seed " + str(a_seed)
	list_of_cols_fts_contrib.append(colname_fts_contrib_values)
	PdCol_fts_contrib[colname_fts_contrib_values] = pd.Series(coefs_best_estimator_as_1d_arr)
	return

##! make here collector_fts_contrib_type_ClassifFtsImportance_supplier_V1(modelselector_by_GSCV, PdCol_fts_contrib, list_of_cols_fts_contrib, a_seed)

# - 2/2 : for supplying the collector of the metrics values recorded ("one-value-metrics" and "metrics values used to plot ROC curves")
def collector_metrics_type_MadeFromTheConfMatrix_supplier_V1(modelselector_by_GSCV, X_test, y_test, class_of_interest,
															 DictCol_bestval_HPsExplored, list_names_HPs_explored,
															 ListCol_val_scores,
															 DictCol_test_scores, list_tags_metrics_computed_for_testscore,
															 fprs_col_by_seed_one_alg, tprs_col_by_seed_one_alg, aucs_col_by_seed_one_alg):

	# ---> Part 1/2 : make the computations for the "one-value-metrics" of this seed

	# + stash, for the best estimator, the val of each hp explored
	for a_hp_explored in list_names_HPs_explored:
		a_hp_explored_best_val = modelselector_by_GSCV.best_params_[a_hp_explored]
		dict_val_updater_valueaslist_V1(DictCol_bestval_HPsExplored, a_hp_explored, a_hp_explored_best_val)

	# + stash, for the best estimator, the validation score
	ListCol_val_scores.append(modelselector_by_GSCV.best_score_)

	# + compute, for the best estimator, the test score and stash it
	# - Part 1 : compute the test score for each metric (each metric is optionally chosen as to be computed)
	# we isolate the best model
	model_retained_asbest = modelselector_by_GSCV.best_estimator_

	# NB1 : A predictive model produced first use is to make predictions. Here, our best model will be used to make predictions on
	# the "unseen data" X_test and the predictions can come in two forms :
	# > either the predicted classes for X_test or the probabilities for each class to be predicted
	# We will use one of these two aspects, depending on how the metric is computed, to estimate predictive performance of our best model
	# > Most metrics, for their computation, use the confusion matrix content (ie "counts of True Positives, True Negatives, False Positives, False Negatives"
	# that are coming from the comparison "true classes vs predicted classes") and for this, the predicted classes for X_test (X_test_preds) is used.
	# > But, the probabilities for each class to be predicted (X_test_probs_all_classes) are needed to compute some other metrics (e.g. : roc_auc_score, auc, etc.)

	# we use the best model to get the predicted classes for X_test
	X_test_preds = model_retained_asbest.predict(X_test)
	# ...as well as what were the probabilities for each class to be predicted from X_test
	X_test_probs_all_classes = model_retained_asbest.predict_proba(X_test)

	# NB2 : what is precisely needed in metrics computations using probabilities of classes is the probability for the positive class
	# (class 1 when classes are 0 and 1, or a specific class of interest). We call it "X_test_probs_of_class1_when_classes_are_bin"

	# NB3 : the function predict_proba, in the present version of scikit-learn and with most ML algs, gives a table (X_test_probs_all_classes)
	# with "col_0 to col_n" for "probability_for_class_0 to probability_of_class_n". But in some cases of analysis (e.g. : with SPAMS custom estimator used in multitask regression),
	# predict_proba gives only 1 col that is "X_test_probs_of_class1_when_classes_are_bin"
	# > So, we make sure to get "X_test_probs_of_class1_when_classes_are_bin" whatever the number of classes present in the table of probabilities given by the function predict_proba

	# to isolate "X_test_probs_of_class1_when_classes_are_bin"...
	# ...we get index last col because it's where the positive class probabilities are always (whether one or multiples classes probs are computed)
	index_last_col_table_of_probs = X_test_probs_all_classes.shape[1] - 1
	# ... then we get the probability for the positive class
	X_test_probs_of_class1_when_classes_are_bin = X_test_probs_all_classes[:, index_last_col_table_of_probs]

	# ...for each metric that we had in our list of metrics to compute, we get the test score of the best model
	for a_metric_of_testscore_to_compute in list_tags_metrics_computed_for_testscore:
		if a_metric_of_testscore_to_compute == "f1score_as_binary":
			model_retained_asbest_testscore = f1score_as_binary(y_test, X_test_preds)
		elif a_metric_of_testscore_to_compute == "f1score_as_weighted":
			model_retained_asbest_testscore = f1score_as_weighted(y_test, X_test_preds)
		elif a_metric_of_testscore_to_compute == "MCC_V1":
			model_retained_asbest_testscore = MCC_V1(y_test, X_test_preds)
		elif a_metric_of_testscore_to_compute == "CK_non_weighted":
			model_retained_asbest_testscore = CK_non_weighted(y_test, X_test_preds)
		elif a_metric_of_testscore_to_compute == "CK_weighted_as_linear":
			model_retained_asbest_testscore = CK_weighted_as_linear(y_test, X_test_preds)
		elif a_metric_of_testscore_to_compute == "acc_V1":
			model_retained_asbest_testscore = acc_V1(y_test, X_test_preds)
		elif a_metric_of_testscore_to_compute == "bal_acc_V1":
			model_retained_asbest_testscore = bal_acc_V1(y_test, X_test_preds)
		elif a_metric_of_testscore_to_compute == "prec_as_binary":
			model_retained_asbest_testscore = prec_as_binary(y_test, X_test_preds)
		elif a_metric_of_testscore_to_compute == "prec_as_weighted":
			model_retained_asbest_testscore = prec_as_weighted(y_test, X_test_preds)
		elif a_metric_of_testscore_to_compute == "rec_as_binary":
			model_retained_asbest_testscore = rec_as_binary(y_test, X_test_preds)
		elif a_metric_of_testscore_to_compute == "rec_as_weighted":
			model_retained_asbest_testscore = rec_as_weighted(y_test, X_test_preds)
		elif a_metric_of_testscore_to_compute == "roc_auc_score_as_macro":
			model_retained_asbest_testscore = roc_auc_score_as_macro(y_test, X_test_probs_of_class1_when_classes_are_bin)
		elif a_metric_of_testscore_to_compute == "roc_auc_score_as_weighted":
			model_retained_asbest_testscore = roc_auc_score_as_weighted(y_test,	X_test_probs_of_class1_when_classes_are_bin)
		elif a_metric_of_testscore_to_compute == "MM4_bespoke_score":
			model_retained_asbest_testscore = MM4_bespoke_score(y_test, X_test_preds)
		else:  # tag of a metric for which a computation has not been implemented
			model_retained_asbest_testscore = 0.0
			print("Warning! the tag", a_metric_of_testscore_to_compute,
				  "does not correspond to an implemented metric computation. Test score of 0.0 is given to it by default")
		# - Part 2 : stash the test score obtained for each metric computed in a dict "metric as key and list of testscores as value")
		dict_val_updater_valueaslist_V1(DictCol_test_scores, a_metric_of_testscore_to_compute, model_retained_asbest_testscore)

	# ---> Part 2/2 : make the computations for the "metrics values used to plot ROC curves" of this seed

	# + computing the FPRs and TPRs values needed to plot the ROC curve at a seed
	# NB1 : we use the sklearn.metrics.roc_curve(array_of_true_binary_labels, array_of_pred_prob_estimates_of_the_positive_class, pos_label_as_int_or_str)
	# to get the FPRs, TPRs and "thresholds_gallery used to get the FPR and the TPR"
	# - array_of_true_binary_labels = array of shape (n_samples,)
	# containing the true binary labels (ie y_test or the col with true classes in test set)
	# - array_of_pred_prob_estimates_of_the_positive_class = array of shape (n_samples,)
	# containing prediction probability estimates of the positive class or clas of interest
	# - pos_label_as_int_or_str = the int or the str used as the positive class in the binary classification problem
	# (important to give if classes are not {0,1} or {-1,1})
	# > what is the function roc_curve() doing : a gallery of thresholds is made and each value of it is used to call the
	# predicted classes of the samples in the test set; by comparing them with the true classes, a value of TPR and FPR can be computed for the threshold.
	# Hence, a gallery of T thresholds give T TPR values and T FPR values.

	# NB2 : The "fixate a starting point tactic" :
	# the gallery of thresholds contains decreasing values ie as we go through the thresholds, the values of TPR computed are increasing.
	# The first value in the gallery of thresholds will seem out of the classic range 0-1 of probabilities.
	# That first value "thresholds[0]" represents the case " no sample will be predicted as the positive class" and to do this,
	# that value is arbitrarily set to "max(prediction probability estimates of the positive class) + 1" (e.g.: 1.888 when max_prob=0.888).
	# This is done to "anchor" the minimal extreme point of the roc curve at (0,0) : by calling in prediction all samples of the test set as the negative class,
	# it is simulated a situation of TP = 0 ie TPR = 0 and FP=0 ie FPR=0, and that way, each roc curve drawn has a starting point at TPR=0 and FPR=0.

	# NB3 : The previous "fixate a starting point tactic" poses the question of "why not also fixate the extreme end point of (1,1) and if yes, when ?"...
	# Well, the best model is chosen for each seed and each "best model by seed" has a "by seed roc curve".
	# A "mean roc curve" averaging all the "by seed roc curves" will be made.
	# The extreme end point of (1,1) is not needed for each "by seed roc curve" that will be drawn as we decide to let them end wherever they do for the moment.
	# Though, that extreme point will be defined at the time of drawing the "mean roc curve".

	fpr_mdl_one_iter, tpr_mdl_one_iter, thresholds_mdl_one_iter = roc_curve(y_test, X_test_probs_of_class1_when_classes_are_bin, pos_label=class_of_interest)

	# + stash the array of TPR and the array of FPR values for when drawing the "by seed roc curve"
	# NB : to draw the "by seed roc curve", "fpr values on the x-axis used" are in fpr_mdl_one_iter and "tpr values on the y-axis used" are tpr_mdl_one_iter.
	fprs_col_by_seed_one_alg.append(fpr_mdl_one_iter)
	tprs_col_by_seed_one_alg.append(tpr_mdl_one_iter)

	# + compute the auc value for this seed and stash it
	# - we compute the auc using sklearn.metrics.auc(fpr_values_array,tpr_values_array)
	roc_auc_of_mdl_one_iter = auc(fpr_mdl_one_iter, tpr_mdl_one_iter)
	# - we stash it in the auc collector
	aucs_col_by_seed_one_alg.append(roc_auc_of_mdl_one_iter)

	return

# ---> Functions dedicated to plotting ROC Curves

# + A point on ROC Curves :
# source : https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
# - Definition :
# The Receiver Operating Characteristic (ROC) metric is used to evaluate classifier output quality.
# ROC curves typically feature true positive rate (TPR) on the Y axis, and false positive rate (FPR) on the X axis.
# For manual computations of the TPR and the FPR, we give this :
# > TPR = TP / all_positives = TP / (TP + FN) (for a better model, has to be maximized)
# > FPR = FP / all_negatives = FP / (TN + FP) (for a better model, has to be minimized)
# This means that the top left corner of the plot is the “ideal” point (ie FPR=0 and TPR=1).
# This is not very realistic, but it does mean that a larger area under the curve (AUC) is usually better.
# The “steepness” of ROC curves is also important, since it is ideal to maximize the true positive rate while minimizing the false positive rate.
# NB : ROC curves are typically used in binary classification to study the output of a classifier.
# In order to extend ROC curve and ROC area to multi-label classification, it is necessary to binarize the output.
# One ROC curve can be drawn per label, but one can also draw a ROC curve by considering each element of the label indicator matrix
# as a binary prediction (micro-averaging).
# - How to draw them :
# > for each algorithm that we ran a gridsearchcv for, we do it on multiple iterations (seeds) that will be averaged.
# > for each seed, the best model selected with be used to get predictions probabilities of the positive class or
# the class of interest; this will be used for get TPRs and FPRs in order to plot a roc curve specific to this seed best model
# > after having been through all seeds, the TPRs obtained at each seed are averaged to obtain the TPRs of
# an average roc curve for all seeds; the FPRs of such a curve are arbitrarily chosen by the operator,
# with the only requirement being that they are from 0 to 1, 0 and 1 included


# + a function that plots, on the same figure, all roc curves, after all iterations of a model selection

def roc_curve_type_AfterAllIterOfModelSelection_maker_V1(fprs_col_by_seed_one_alg,
												 tprs_col_by_seed_one_alg,
												 aucs_col_by_seed_one_alg,
												 list_seeds,
												 num_values_from_interpolation_for_roc_curves,
												 tag_LearningTaskType, tag_LearningAlgUsed, tag_ModelBuilt,
												 tag_cond, tag_pop, tag_dataprofile,
												 tag_NumTrial, results_dir_for_this_run):



	# --->  Create the figure that will receive the roc curves
	# NB1 : the figure has only on subplot (thus we create the_figure,subplot_ax)
	# NB2 : this one subplot will contain all the roc curves
	# (ie if we do a gridsearch for 1 alg over multiple seeds,
	# we have 1 "by seed roc curve" * number_of_seeds+ 1 "mean roc curve" over all seeds)
	figure_all_roc_curves, subplot_ax1outof1 = plt.subplots()
	print("- Started drawing the roc_curves_figure for the entire best model search.")

	# ---> add to the figure the line for the random prediction
	subplot_ax1outof1.plot([0, 1], [0, 1], linestyle='--', lw=2, alpha=.8, color='r', label='Random predictor')
	print("- line for the random prediction, added on the roc_curves_figure.")

	# ---> add to the figure, for each seed, 1 "by seed roc curve"
	for seed_explored in list_seeds:
		# + getting the index of each seed
		# NB : it will be used to retrieve "the info needed to draw a roc curve corresponding to the seed"
		# ie "the FPRs array, the TPRs array and the AUC value stashed for the seed")
		index_seed_explored = list_seeds.index(seed_explored)
		# + retrieving the FPRs array stashed for the seed
		FPRs_arr_retrieved_for_the_seed = fprs_col_by_seed_one_alg[index_seed_explored]
		# + retrieving the TPRs array stashed for the seed
		TPRs_arr_retrieved_for_the_seed = tprs_col_by_seed_one_alg[index_seed_explored]
		# + retrieving the AUC value stashed for the seed
		AUC_val_retrieved_for_the_seed = aucs_col_by_seed_one_alg[index_seed_explored]
		# + adding to the figure, the "by seed roc curve" for this seed (as well as adding a legend giving "roc curve iteration id" and "auc value")
		subplot_ax1outof1.plot(FPRs_arr_retrieved_for_the_seed, TPRs_arr_retrieved_for_the_seed,
							   lw=1, alpha=0.3,
							   label='ROC curve seed %d (AUC = %0.2f)' % (seed_explored, AUC_val_retrieved_for_the_seed))
		print("- by_seed_roc_curve added on the roc_curves_figure, for seed", seed_explored,"ie",(index_seed_explored+1),"out of",len(list_seeds), "...done !")

	# ---> add to the figure, the "mean roc curve"

	# NB1 : to draw the "mean roc curve" :
	# > "fpr values on the x-axis used" are in an array of values of fpr going from 0 to 1 that have been decided
	# arbitrarily by the user (the values are evenly spaced and their number is chosen by the user)
	# > "tpr values on the y-axis used" are the mean of the "tpr values on the y-axis used" for each seed (tpr_mdl_one_iter) .

	# NB2 : How to make "fpr values on the x-axis used" and "tpr values on the y-axis used" correspond
	# just like "fpr_mdl_one_iter" and "tpr_mdl_one_iter" did ? :
	# Instead of using the present version of tpr_mdl_one_iter in the average,
	# we can interpolate tpr_mdl_one_iter to obtain another version of it that corresponds to "fpr values on the x-axis used"

	# NB3 : this has also the advantage to let us choose the number of "tpr values on the y-axis used" corresponding to
	# the same number of "fpr values on the x-axis used"
	# (e.g. : if we had 10 values only in tpr_mdl_one_iter, we can interpolate tpr_mdl_one_iter to contain
	# 100 equally spaced "tpr values on the y-axis used" corresponding to 100 "fpr values on the x-axis used"
	# and this is still respecting the 10 initial values correspondence with what was on the x-axis for them)
	# > hence, more precisely "tpr values on the y-axis used" are the mean of the interpolated version
	# of the "tpr values on the y-axis used" for each seed

	# + making the "fpr values on the x-axis used" (mean_fpr_by_seed_one_alg)
	# NB1 : the number of values in "fpr values on the x-axis used" ie "num_values_from_interpolation_for_roc_curves" is arbitrarily chosen by the operator
	# and is a array of evenly spaced values to base interpolation on
	# NB2 : "num_values_from_interpolation_for_roc_curves" is an int, this is approx the number of points on the "mean roc curve" drawn
	# NB3 : we advise giving "num_values_from_interpolation_for_roc_curves" >=100 for visually better curves.
	mean_fprs_by_seed_one_alg = arr_evenly_spaced_val_maker_V1(0, 1, num_values_from_interpolation_for_roc_curves)


	# + making the "tpr values on the y-axis used" (mean_tprs_by_seed_one_alg)
	# - step 1/4 : create a collector of the interpolated version of the TPRs array obtained for each seed
	interpolated_version_of_tprs_col_by_seed_one_alg = []
	# - step 2/4 : loop on the seeds and make the interpolated version of the TPRs array obtained for each seed and stash it
	for one_seed in list_seeds:
		# > getting the index of each seed
		# NB : it will be used to retrieve the stashed FPRs array and TPRs arrays corresponding to the seed
		index_one_seed = list_seeds.index(one_seed)
		# > retrieving the stashed FPRs array corresponding to the seed
		FPRs_arr_retrieved_for_one_seed = fprs_col_by_seed_one_alg[index_one_seed]
		# > retrieving the stashed TPRs array corresponding to the seed
		TPRs_arr_retrieved_for_one_seed = tprs_col_by_seed_one_alg[index_one_seed]
		# > making the interpolated version of the "tpr values on the y-axis used" for each seed
		interpolated_version_of_TPRs_arr_retrieved_for_one_seed = np.interp(mean_fprs_by_seed_one_alg, FPRs_arr_retrieved_for_one_seed, TPRs_arr_retrieved_for_one_seed)
		# > ensuring the "fixate a starting point tactic"
		# NB : we know that the x-axis of the mean roc curve values (mean_fprs_by_seed_one_alg) starts with a 0.
		# So to make sure first point of the roc curve is (0.0), we just have to make sure the first data point has TPR = 0
		interpolated_version_of_TPRs_arr_retrieved_for_one_seed[0] = 0.0
		# > stashing it (the interpolated version of the "tpr values on the y-axis used" for each seed) in a collector (to compute later, a mean of it, over all seeds)
		interpolated_version_of_tprs_col_by_seed_one_alg.append(interpolated_version_of_TPRs_arr_retrieved_for_one_seed)
	# - step 3/4 : compute a mean of the collector of the interpolated version of the TPRs array obtained for each seed
	mean_tprs_by_seed_one_alg = np.mean(interpolated_version_of_tprs_col_by_seed_one_alg, axis=0)
	# - step 4/4 : fixate the extreme end point value of TPRs for the mean curve as 1
	# NB : this is to guarantee that the end point of the curve is (1,1). We already have the last value on the x-axis being 1,
	# as the function making "fpr values on the x-axis used" makes 1 as last value
	mean_tprs_by_seed_one_alg[-1] = 1.0

	# + compute mean_auc
	# NB : this is same as using auc function with one fpr array and one tpr array to get one auc value
	# but with mean_auc = auc(mean_fpr_array, mean_tpr_array)
	mean_auc_from_cols_of_mdl = auc(mean_fprs_by_seed_one_alg, mean_tprs_by_seed_one_alg)
	# + compute the equivalent of a standard deviation for mean_auc
	# NB : the mean_auc (for the average roc curve) is not obtained from multiple values but in one time so it does not have a std-dev;
	# to report the std_dev that gives best this intuition, we use the std_dev of the multiple auc values obtained across seeds used to make the average roc curve.
	std_auc_from_aucs_col_by_seed_one_alg = np.std(aucs_col_by_seed_one_alg)
	# + plot the mean roc curve and what will be in the legend of the figure for it (mean_auc value and std_dev value)
	subplot_ax1outof1.plot(mean_fprs_by_seed_one_alg, mean_tprs_by_seed_one_alg,
						   lw=2, alpha=.8,
						   color='b', label=r'Mean ROC curve (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_from_cols_of_mdl, std_auc_from_aucs_col_by_seed_one_alg))
	print("- mean_roc_curve added on the roc_curves_figure, for seeds in a total of", len(list_seeds), "... done !")

	# ---> Editing the remaining visuals of our roc curves figure
	# + NB : How the "mean_roc_curve" is differenciated visually from all the "roc_curve_by_seed"
	# - the line width is larger on the "mean_roc_curve" in comparison to the "roc_curve_by_seed" (resp. "lw=2" vs "lw=1")
	# - the opacity is higher (ie less transparent) on the the "mean_roc_curve" in comparison to the "roc_curve_by_seed" ("alpha=.8" vs "alpha=.3")
	# - the line for the random prediction has same width and opacity as the "mean_roc_curve" because it is also
	# a remarkable aspect of the figure. But, it is the only plot in dashes (linestyle='--') and is colored in red (with color='r').

	# + limits of the x-axis and y-axis
	# NB : to see very well that our roc curves in all the space 0-1 of each axis, we extend a bit each of our axis limits
	subplot_ax1outof1.set_xlim([-0.05, 1.05])
	subplot_ax1outof1.set_ylim([-0.05, 1.05])
	# + labels of the x-axis and y-axis
	subplot_ax1outof1.set_xlabel('False Positive Rate')
	subplot_ax1outof1.set_ylabel('True Positive Rate')
	# + title of the figure
	subplot_ax1outof1.set_title("\n".join(wrap('ROC curve of %(Task)s using %(Alg)s-%(Model)s model, on case %(Cond)s-%(Pop)s, %(Profile)s profile, %(Trial)s' %
								 {"Task": tag_LearningTaskType, "Alg": tag_LearningAlgUsed, "Model": tag_ModelBuilt, "Cond": tag_cond, "Pop": tag_pop, "Profile": tag_dataprofile, "Trial": tag_NumTrial})))
	# + position of legend
	# NB : as this is a roc curve, the upper left corner is where most of the interesting action is happening
	# ("the better the model, the more it accumulates area under the curve while pointing towards the upper left corner"
	# so a lot of good models's roc curves will be observed competing on that upper left section),
	# so we force the legend to be on the opposite corner to avoid disturbing observations
	subplot_ax1outof1.legend(loc="lower right")
	# + save the roc curves figure
	# NB : using bbox_inches='tight' removes the extra white space around the figure
	fullname_file_of_figure_all_roc_curves = results_dir_for_this_run+"/"+'Output_' + tag_LearningTaskType + "_" + tag_LearningAlgUsed + "_" + tag_ModelBuilt + "_" + tag_cond + "_" + tag_pop + "_" + tag_dataprofile + "_" + tag_NumTrial + '_ROCcurve.png'
	figure_all_roc_curves.savefig(fullname_file_of_figure_all_roc_curves, bbox_inches='tight')
	# + close it
	# NB : the figure is opened automatically at its creation, but we have no need to keep it opened
	# as it is saved in the specific result dir for this analysis
	plt.close() # could also use "plt.close(figure_all_roc_curves)"
	print("- remaining visuals edited on the roc_curves_figure and figure saved... done !")
	print("- Finished drawing the roc_curves_figure for the entire best model search.")
	return





# --->

# --->

# ----------------------------------------------------------------------------------------------------------------------

# >>>>>> Part 2 : functions copied or customized from others authors.

# --->

# ----------------------------------------------------------------------------------------------------------------------

# end of file
# ----------------------------------------------------------------------------------------------------------------------