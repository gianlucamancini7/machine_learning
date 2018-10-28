# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from additional_implementations import *

def perform_PCA(df,number_pa):
    features = df.columns.values
    df_std=pd.DataFrame()
    df_std, mean, std=standardize_personal(df)
    cov_df_std=np.cov(df_std.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_df_std)
    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort()
    eig_pairs.reverse()
    #Find number of feature condsidered
    number_features=len(features)
    #define matrix to be filled in
    matrix_w=np.ones((number_features, number_pa))
    for i in range(number_pa):
        matrix_w[:,i] = eig_pairs[i][1]
    df_std_transf = df_std.dot(matrix_w)
    return df_std_transf, matrix_w, mean, std

def apply_PCA_to_test(df_test,matrix_w, mean_train, std_train):
    features = df_test.columns.values
    df_test_std=pd.DataFrame()
    df_test_std=standardize_test(df_test, mean_train, std_train)
    df_test_std_transf = df_test_std.dot(matrix_w)
    return df_test_std_transf