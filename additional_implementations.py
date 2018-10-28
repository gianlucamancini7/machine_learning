# -*- coding: utf-8 -*-
"""All the fuction needed and coded for predicting the values 
    in the run files add for all the other trials performed to 
    finalize the answer."""

import numpy as np
import matplotlib.pyplot as plt
import implementations

def assign_y_values(df, logistic=False):
    y=np.array(df['Prediction'])
    y[np.where(y=='b')] = -1.
    y[np.where(y=='s')] = 1.
    if logistic:
        y[np.where(y==-1)] = 0    
    y=y.astype('float')
    return y

def clean_pred_Id(df):
    return df.drop(columns=['Prediction', 'Id'])

def get_best_parameters(w0, w1, losses,return_idx=False):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    if return_idx:
        return losses[min_row, min_col], w0[min_row], w1[min_col] , min_row, min_col
    return losses[min_row, min_col], w0[min_row], w1[min_col]

def polynomial_features_simple(tx, order):
    mat=np.polynomial.polynomial.polyvander(tx, order)
    mat=np.reshape(mat,(tx.shape[0],(1+order)*tx.shape[1]))
    mat=np.delete(mat,np.arange(order+1,(1+order)*tx.shape[1],order+1),axis=1)
    return mat

def poly_various_features_simple(tx, order, other_order, gon=True, log=True):
    mat=np.polynomial.polynomial.polyvander(tx, order)
    mat=np.reshape(mat,(tx.shape[0],(1+order)*tx.shape[1]))
    mat=np.delete(mat,np.arange(order+1,(1+order)*tx.shape[1],order+1),axis=1)
    if log:
        for i in other_order:
            mat=np.hstack((mat,np.log(tx*i)))
    if gon:
        for i in other_order:
            mat=np.hstack((mat,np.sin(tx*i)))
            mat=np.hstack((mat,np.cos(tx*i)))
    return mat

def polynomial_features(tx, deg):
    tx = np.asarray(tx).T[np.newaxis]
    n = tx.shape[1]
    p_mat = np.tile(np.arange(deg + 1), (n, 1)).T[..., np.newaxis]
    tX = np.power(tx, p_mat)
    I = np.indices((deg + 1, ) * n).reshape((n, (deg + 1) ** n)).T
    Fin = np.product(np.diagonal(tX[I], 0, 1, 2), axis=2)
    return Fin.T

def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse



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


    wsi_train,_=implementations.ridge_regression(yi_train,xi_train,lambda_)

    loss_tr=compute_mse(yi_train,xi_train,wsi_train)
    loss_te=compute_mse(yi_test,xi_test,wsi_train)
    
    return loss_tr, loss_te,wsi_train

def cross_validation_ridge_loop(y, tx, lambda_, k_fold, seed=1):
    k_indices=build_k_indices(y, k_fold, seed)
    mse_tr = []
    mse_te = []
    wsi_train_lst=[]
    for k in range(k_fold):
        loss_tr, loss_te,wsi_train=cross_validation_ridge(y, tx, k_indices, k, lambda_)
        mse_tr.append(loss_tr)
        mse_te.append(loss_te)
        wsi_train_lst.append(wsi_train)
    return  np.mean(wsi_train_lst, axis=0), np.mean(mse_tr), np.mean(mse_te)

def cross_validation_least_squares(y, x, k_indices, k):
    """return the loss of ridge regression."""

    test_ind=k_indices[k]
    total_ind=np.ravel(k_indices)
    xi_test=x[test_ind]
    yi_test=y[test_ind]
    train_ind=total_ind[np.logical_not(np.isin(total_ind,test_ind))]
    xi_train=x[train_ind]
    yi_train=y[train_ind]

    wsi_train=least_squares(yi_train,xi_train,)

    loss_tr=compute_mse(yi_train,xi_train,wsi_train)
    loss_te=compute_mse(yi_test,xi_test,wsi_train)
    
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
    mean=np.mean(x, axis=0)
    std=np.std(x, axis=0)
    x=(x-mean)/std
    return x, mean, std

def standardize_test(x, mean_train, std_train):
    x=(x-mean_train)/std_train
    return x

#Logistic
#def sigmoid(t):
#    """apply sigmoid function on t."""
#    return (np.exp(t))/(1+(np.exp(t)))
#
#def calculate_loss(y, tx, w):
#    """compute the cost by negative log likelihood."""
#    ### RMSE or other error??    
#    loss=np.sum(np.log(1+np.exp(tx.dot(w)))-y*(tx.dot(w)))
#    return loss
#
#def calculate_gradient(y, tx, w):
#    """compute the gradient of loss."""
#    return tx.T.dot(sigmoid(tx.dot(w))-y)
#
#def calculate_hessian(y, tx, w):
#    """return the hessian of the loss function."""
#    # calculate hessian
#    S=np.diag((sigmoid(tx.dot(w))*(1-sigmoid(tx.dot(w)))).T[0])
##     a=sigmoid(tx.dot(w))*(1-sigmoid(tx.dot(w)))
#    H=tx.T.dot(S).dot(tx) 
#    return H
#
#def penalized_logistic_regression(y, tx, w, lambda_):
#    """return the loss, gradient, and hessian."""
#    # return loss, gradient, and hessian: TODO
#    loss=calculate_loss(y, tx, w)+(lambda_*0.5*(w.T.dot(w)))[0][0]
#    grad=calculate_gradient(y, tx, w)+lambda_*w
#    H=calculate_hessian(y, tx, w)+lambda_*np.eye(tx.shape[1])   
#    return loss, grad, H
#
#def predict_labels_modified(weights, data):
#    """Generates class predictions given weights, and a test data matrix"""
#    y_pred = np.dot(data, weights)
#    y_pred[np.where(y_pred <= 0)] = -10
#    y_pred[np.where(y_pred > 0)] = 10
#    
#    return y_pred
