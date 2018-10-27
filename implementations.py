# -*- coding: utf-8 -*-
"""some helper functions."""

import numpy as np

#REQUIRED IMPLEMENTED ML METHODS

def least_squares_GD(y, tx, initial_w, max_iters, gamma,keeptrack=False, log_info=False):
    """Gradient descent"""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, _ = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        # update w through the gradient update
        w = w - gamma * grad
        # log info
        if log_info:
            if n_iter % 100 == 0:
                print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))  
        # store w and loss
        ws.append(w)
        losses.append(loss)
    if keeptrack:
        return ws, losses
    else:
        return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size=1,keeptrack=False,log_info=False):
    """Stochastic gradient descent"""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx,batch_size=batch_size, num_batches=1):
            # compute loss, gradient
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            loss = compute_mse(y, tx, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # log info
            if log_info:
                if n_iter % 100 == 0:
                    print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))  
            # store w and loss
            ws.append(w)
            losses.append(loss)
    if keeptrack:
        return ws, losses
    else:
        return w, loss

    
def least_squares(y, tx):
    """Least squares"""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression"""
    # Definition of Lambda as in lectures
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma, keeptrack=False, convergence_stop=False,threshold = 1e-8, log_info=False):
    """Logistic regression using Newton method"""   
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    #Start the logistic regression
    for n_iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_newton_method(y, tx, w, gamma)
        # log info
        if log_info:
            if n_iter % 100 == 0:
                print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))   
        # store w and loss
        ws.append(w)
        losses.append(loss)
        # converge criterion
        if convergence_stop:
            if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
                break
    if keeptrack:
        return ws, losses
    else:
        return w, loss
    

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, keeptrack=False, convergence_stop=False,threshold = 1e-8, log_info=False):
    """Regularized logistic regression using Newton method"""   
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    #Start the regularized logistic regression
    for n_iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        if log_info:
            if n_iter % 100 == 0:
                print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))   
        # store w and loss
        ws.append(w)
        losses.append(loss)
        # converge criterion
        if convergence_stop:
            if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
                break
    if keeptrack:
        return ws, losses
    else:
        return w, loss


### ADDITIONAL FUNCTION FOR ML METHODS

#General additional function
def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse


#Stochstics gradient additional functions
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

#Logistic regression additional functions
def sigmoid(t):
    """apply sigmoid function on t."""
    return (np.exp(t))/(1+(np.exp(t)))

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""   
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
    # return loss, gradient, and hessian
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
    # update w
    w=w-gamma*grad
    return loss, w

def learning_by_newton_method(y, tx, w):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """
    # return loss, gradient and hessian
    loss, grad, H=logistic_regression_newton(y, tx, w)  
    # update w: TODO
    w=w-np.linalg.solve(H, grad)
    return loss, w

def logistic_regression_newton(y, tx, w):
    """return the loss, gradient, and hessian."""
    # return loss, gradient, and hessian: TODO
    loss=calculate_loss(y, tx, w)
    grad=calculate_gradient(y, tx, w)
    H=calculate_hessian(y, tx, w)    
    return loss, grad, H
