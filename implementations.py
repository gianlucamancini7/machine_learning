# -*- coding: utf-8 -*-
"""some helper functions."""

import numpy as np


def build_poly(input_data, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    
    ravel_input_data=np.ravel(input_data.T)
    tx=np.vander(ravel_input_data, N=degree+1, increasing=True)
    
    #number of points, number of degrees plus 1, number of features
    tx=np.reshape(tx,(input_data.shape[1],input_data.shape[0],degree+1))
    return tx

def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

def least_squares(y, tx):
    """calculate the least squares."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)

def ridge_regression(y, tx, lambda_):
# definition of Lambda as in lectures
    """implement ridge regression."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)

def polynomial_regression(degrees):
    """Constructing the polynomial basis function expansion of the data,
       and then running least squares regression."""
    
    # define the structure of the figure
    num_row = 2
    num_col = 2
    f, axs = plt.subplots(num_row, num_col)

    for ind, degree in enumerate(degrees):
        # form dataset to do polynomial regression.
        tx = build_poly(x, degree)

        # least squares
        weights = least_squares(y, tx)

        # compute RMSE
        rmse = np.sqrt(2 * compute_mse(y, tx, weights))
        print("Processing {i}th experiment, degree={d}, rmse={loss}".format(
              i=ind + 1, d=degree, loss=rmse))
        # plot fit
        plot_fitted_curve(
            y, x, weights, degree, axs[ind // num_col][ind % num_col])
    plt.tight_layout()
    plt.savefig("visualize_polynomial_regression")
    plt.show()





