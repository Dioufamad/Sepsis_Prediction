#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Amad Diouf, amaddioufb13@gmail.com
# Created Date: 27/05/2022
# version ='1.0.0'
# ---------------------------------------------------------------------------
""" script similar to driver.py"""
# to run, use "python step1_predictions_maker.py input_directory output_directory id_model"
# makes predictions on clinical time-series data when given 2 arguments : the input data folder, the output folder and an id of a saved trained model
# 1- performs all file input (management the input data) and output (writing the predictions made by the saved model on the data given)
# 2- calls load_sepsis_model once and get_sepsis_score many times.
# 2-a- load_sepsis_model function : add any code to the load_sepsis_model function that you only need to run once, such as loading weights for your model.
# 2-b- get_sepsis_score function : it was the main point of the get_sepsis_score.py. It produces the prediction corresponding to the observation (one row of a table ie one patient at a certain point of time)

# ---------------------------------------------------------------------------
# Imports
import numpy as np, os, sys
from engines.data_mgmt_engine import load_challenge_data, save_challenge_predictions
from engines.models_engine import load_sepsis_model
from engines.metrics_engine1 import get_sepsis_score


if __name__ == '__main__':
    # Parse arguments.
    if len(sys.argv) != 3: #? change also if added another argument
        raise Exception('Include the input and output directories as arguments, e.g., python driver.py input output.')

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    # id_model = sys.argv[3] #?

    # Find files.
    files = []
    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('psv'):
            files.append(f)

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    # Load model.
    print('Loading sepsis model...')
    model = load_sepsis_model()

    # Iterate over files.
    print('Predicting sepsis labels...')
    num_files = len(files)
    for i, f in enumerate(files):
        print('    {}/{}...'.format(i+1, num_files))

        # Load data.
        input_file = os.path.join(input_directory, f)
        data = load_challenge_data(input_file)

        # Make predictions.
        num_rows = len(data)
        scores = np.zeros(num_rows)
        labels = np.zeros(num_rows)
        for t in range(num_rows):
            current_data = data[:t+1]
            current_score, current_label = get_sepsis_score(current_data, model)
            scores[t] = current_score
            labels[t] = current_label

        # Save results.
        output_file = os.path.join(output_directory, f)
        save_challenge_predictions(output_file, scores, labels)

    print('Done.')