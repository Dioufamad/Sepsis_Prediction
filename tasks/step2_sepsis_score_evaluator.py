#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Amad Diouf, amaddioufb13@gmail.com
# Created Date: 27/05/2022
# version ='1.0.0'
# ---------------------------------------------------------------------------
"""  similar to evaluate_sepsis_score.py """
# ---------------------------------------------------------------------------
# Imports
import numpy as np, os, os.path, sys, warnings
# from engines.data_mgmt_engine import load_column
# from engines.metrics_engine2 import compute_auc
# from engines.metrics_engine2 import compute_accuracy_f_measure
# from engines.metrics_engine2 import compute_prediction_utility
from engines.metrics_engine2 import evaluate_sepsis_score

# This file contains functions for evaluating algorithms for the 2019 PhysioNet/
# CinC Challenge. You can run it as follows:
#
#   python evaluate_sepsis_score.py labels predictions scores.psv
#
# where 'labels' is a directory containing files with labels, 'predictions' is a
# directory containing files with predictions, and 'scores.psv' (optional) is a
# collection of scores for the predictions.


if __name__ == '__main__':
    auroc, auprc, accuracy, f_measure, utility = evaluate_sepsis_score(sys.argv[1], sys.argv[2])

    output_string = 'AUROC|AUPRC|Accuracy|F-measure|Utility\n{}|{}|{}|{}|{}'.format(auroc, auprc, accuracy, f_measure, utility)
    if len(sys.argv) > 3:
        with open(sys.argv[3], 'w') as f:
            f.write(output_string)
    else:
        print(output_string)
