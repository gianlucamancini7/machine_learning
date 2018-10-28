# -*- coding: utf-8 -*-

# Useful starting lines
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
%load_ext autoreload
%autoreload 2
#allows to print the dataframe nicely
from IPython.core import display as ICD

# import additional packages to insepct data and clean them
import pandas as pd
import os 
import random 
from zipfile import ZipFile
import datetime

# import helping functions from the implementation file
# from proj1_helpers import load_csv_data
from proj1_helpers import *
import implementations
from additional_implementations import *