# -*- coding: utf-8 -*-
"""Main file with all the processing for the best submission.
    This method includes physical selection of features, search 
    of the best hyperparameters and cross validated ridge regression."""

# Useful starting lines
import numpy as np
#allows to print the dataframe nicely

# import additional packages to insepct data and clean them
import pandas as pd
from zipfile import ZipFile

# import helping functions from the implementation file
# from proj1_helpers import load_csv_data
from proj1_helpers import *
#↨import implementations
from additional_implementations import *
import physics as phy

# import zipped files from the github repository
data_folder='./data/'
zip_file = ZipFile(data_folder+'all.zip')

# now we want to access the 'filename' property in the zipfile variable
# and we create a dictionary of dataframe
dfs = {text_file.filename: pd.read_csv(zip_file.open(text_file.filename))
       for text_file in zip_file.infolist()
       if text_file.filename.endswith('.csv')}
df_train=dfs['train.csv']
df_test=dfs['test.csv']

#Clean the prediction and Id column in df training
df_train_selection=clean_pred_Id(df_train)
#Apply the required feature selection
df_train_selection, median = phy.physics_train(df_train_selection)
#Rewrite the prediction label
y_train_selection=assign_y_values(df_train)

#Run Optimization algorithm 
lambdas=np.logspace(-10,-1,20)
degrees=np.linspace(1,15,15).astype('int')
k_fold=4
mses= np.zeros((len(degrees), len(lambdas),2))
print('STARTING OPTIMIZATION')
for ind_degree,degree in enumerate(degrees):
    tx_train_selection_polynomial=polynomial_features_simple(df_train_selection, degree)
    print('Degree under optimization: ',degree)
    for ind_lambda,lambda_ in enumerate(lambdas):
        w, loss_tr, loss_te = cross_validation_ridge_loop(y_train_selection, tx_train_selection_polynomial, lambda_, k_fold, seed=1)
        mses[ind_degree, ind_lambda][0]=loss_tr
        mses[ind_degree, ind_lambda][1]=loss_te
        
mse_tr_final, degree_final,lambda_final, min_row, min_col=get_best_parameters(degrees, lambdas, mses[:,:,0],return_idx=True)
mse_te_final=mses[min_row,min_col,1]
tx_train_selection_polynomial=polynomial_features_simple(df_train_selection, degree_final)
w_final,loss=implementations.ridge_regression(y_train_selection,tx_train_selection_polynomial,lambda_final)
print('')
print('MSE train: ',mse_tr_final,'   MSE test: ',mse_te_final)
print('Degree:    ',degree_final,'   Lambda:   ',lambda_final)

y_pred=predict_labels(w_final,tx_train_selection_polynomial)
print('')
print('Learning Performance: ',list(y_pred*y_train_selection).count(1.)/len(tx_train_selection_polynomial))

#Clean the prediction and Id column in df test
df_test_selection=clean_pred_Id(df_test)
#Apply the required feature selection
df_test_selection=phy.physics_test(df_test_selection, median)
#Create the test matrix
tx_test_selection_polynomial=polynomial_features_simple(df_test_selection, degree_final)

tx_test=tx_test_selection_polynomial
y_pred=predict_labels(w_final,tx_test)
create_csv_submission(df_test['Id'], y_pred, 'Group Higgs_Garrix Submission')
























