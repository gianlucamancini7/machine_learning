# -*- coding: utf-8 -*-
"""some helper functions."""

import numpy as np
import matplotlib.pyplot as plt

def polynomial_features_simple(tx, order):
    mat=np.polynomial.polynomial.polyvander(tx, order)
    mat=np.reshape(mat,(tx.shape[0],(1+order)*tx.shape[1]))
    mat=np.delete(mat,np.arange(order+1,(1+order)*tx.shape[1],order+1),axis=1)
    return mat

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
#     print(np.info(a))
#     print(np.info(b))
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


#Stochstics gradient functions
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            

            
def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx,batch_size=batch_size, num_batches=1):
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_mse(y, tx, w)
            # store w and loss
            ws.append(w)
            losses.append(loss)

        print("SGD({bi}/{ti}): loss={l}, wavg={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws

#Cross Validation
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation_ridge(y, x, k_indices, k, lambda_):
    """return the loss of ridge regression."""

    test_ind=k_indices[k]
    total_ind=np.ravel(k_indices)
    xi_test=x[test_ind]
    yi_test=y[test_ind]
    train_ind=total_ind[np.logical_not(np.isin(total_ind,test_ind))]
    xi_train=x[train_ind]
    yi_train=y[train_ind]

#     txi_test=build_poly(xi_test,degree)
#     txi_train=build_poly(xi_train,degree)

    wsi_train=ridge_regression(yi_train,xi_train,lambda_)

    loss_tr=np.sqrt(2*compute_mse(yi_train,xi_train,wsi_train))
    loss_te=np.sqrt(2*compute_mse(yi_test,xi_test,wsi_train))
    
    return loss_tr, loss_te,wsi_train

def cross_validation_least_squares(y, x, k_indices, k):
    """return the loss of ridge regression."""

    test_ind=k_indices[k]
    total_ind=np.ravel(k_indices)
    xi_test=x[test_ind]
    yi_test=y[test_ind]
    train_ind=total_ind[np.logical_not(np.isin(total_ind,test_ind))]
    xi_train=x[train_ind]
    yi_train=y[train_ind]

#     txi_test=build_poly(xi_test,degree)
#     txi_train=build_poly(xi_train,degree)

    wsi_train=least_squares(yi_train,xi_train,)

    loss_tr=np.sqrt(2*compute_mse(yi_train,xi_train,wsi_train))
    loss_te=np.sqrt(2*compute_mse(yi_test,xi_test,wsi_train))
    
    return loss_tr, loss_te,wsi_train



def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")
    
def standardize_personal(x):
    
    x=(x-np.mean(x, axis=0))/np.std(x, axis=0)
    return x

#Logistic
def sigmoid(t):
    """apply sigmoid function on t."""
    return (np.exp(t))/(1+(np.exp(t)))

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    ### RMSE or other error??    
    loss=np.sum(np.log(1+np.exp(tx.dot(w)))-y*(tx.dot(w)))
    return loss

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return tx.T.dot(sigmoid(tx.dot(w))-y)

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    # calculate hessian
    S=np.diag((sigmoid(tx.dot(w))*(1-sigmoid(tx.dot(w)))).T[0])
#     a=sigmoid(tx.dot(w))*(1-sigmoid(tx.dot(w)))
    H=tx.T.dot(S).dot(tx) 
    return H

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    # return loss, gradient, and hessian: TODO
    loss=calculate_loss(y, tx, w)+(lambda_*0.5*(w.T.dot(w)))[0][0]
    grad=calculate_gradient(y, tx, w)+lambda_*w
    H=calculate_hessian(y, tx, w)+lambda_*np.eye(tx.shape[1])   
    return loss, grad, H

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    # return loss, gradient and hessian
    loss, grad, H=penalized_logistic_regression(y, tx, w,lambda_)
    
    # update w: TODO
#     w=w-gamma*np.linalg.solve(H, grad)
    w=w-gamma*grad
    #print ('w',w)
    return loss, w

