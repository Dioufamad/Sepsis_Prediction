#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Amad Diouf, amaddioufb13@gmail.com
# Created Date: 27/05/2022
# version ='1.0.0'
# ---------------------------------------------------------------------------
"""  script 2 to launch initial tests"""
# ---------------------------------------------------------------------------
# Imports
import pandas as pd
import matplotlib # change the backend used by maplotlib from the default 'QtAgg' to one that enables interactive plots 'Qt5Agg'
matplotlib.use('Qt5Agg')
matplotlib.get_backend()
import matplotlib.pyplot as plt
import seaborn as sns
# for sns plots, add plt.show() to make the plots appear (replace it with the "%matplotlib inline" in the case of a notebook to let the backend solve that issue while displaying the plots inside the notebook)
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
# ---------------------------------------------------------------------------
# IMPORTING OUR DATA