import numpy as np
from scipy.optimize import fmin_bfgs


def calculate_cost(Theta, X, y):
    cost = 0.0 # you must return the correct value for cost

    N = len(X) # get number of rows
    X = np.c_[np.ones(N), X] # add column of ones at beginning to accommodate theta0

    # calculate the cost of a particular choice of Theta using the logistic regression
    cost =  (-1 / N) * (y.dot(np.log(sigmoid_function(X, Theta))) + (1 - y).dot(np.log(1 - sigmoid_function(X, Theta))))
    return cost


def regularized_calculate_cost(Theta, X, y, l_mbda):
    cost = 0.0 # you must return the correct value for cost

    N = len(X) # get number of rows
    X = np.c_[np.ones(N), X] # add column of ones at beginning to accommodate theta0

    # calculate the cost of a particular choice of Theta using the logistic regression
    cost =  (-1 / N) * (y.dot(np.log(sigmoid_function(X, Theta))) + (1 - y).dot(np.log(1 - sigmoid_function(X, Theta)))) + regularize(Theta, l_mbda)
    return cost


def regularize(Theta, l_mbda):
    N = len(Theta) # get number of rows
    Theta1 = Theta[1:] # remove theta0 since we are not regularizing thetas
    return l_mbda / (2 * N) * (Theta1.dot(Theta1))


def minBFGS(Theta, X, y):
    return fmin_bfgs(calculate_cost, Theta, args=(X,y))


def gradient_descent(X, y, Theta, alpha, num_iters):
    N = len(X) # get number of rows
    T = len(Theta)
    origX = X # original X for use in J_array calculation
    X = np.c_[np.ones(N), X]
    temps = np.empty(T)
    J_array = np.zeros(num_iters)
    for i in range(num_iters):
        for t in range(T):
            cost = ((sigmoid_function(X, Theta) - y) * X[:,t]).sum()
            temps[t] = Theta[t] - (alpha * cost)
        Theta = temps
        J_array[i] = calculate_cost(Theta, origX, y)
    # return the correct values for Theta and the array of cost values for each iteration
    return Theta, J_array


def regularized_gradient_descent(X, y, Theta, alpha, num_iters, l_mbda):
    N = len(X) # get number of rows
    T = len(Theta)
    origX = X # original X for use in J_array calculation
    X = np.c_[np.ones(N), X]
    temps = np.empty(T)
    J_array = np.zeros(num_iters)
    for i in range(num_iters):
        for t in range(T):
            if t == 0: # theta0
                cost = ((sigmoid_function(X, Theta) - y) * X[:,t]).sum()
                temps[t] = Theta[t] - (alpha / N) * cost
            else:
                cost = ((sigmoid_function(X, Theta) - y) * X[:,t] - ((l_mbda / N) * Theta[t])).sum()
                temps[t] = Theta[t] - (alpha / N) * cost
        Theta = temps
        J_array[i] = regularized_calculate_cost(Theta, origX, y, l_mbda)
    # return the correct values for Theta and the array of cost values for each iteration
    return Theta, J_array


def sigmoid_function(X, Theta):
    return (1 / (1 + np.exp(-1 * (X.dot(Theta)))))


def predict_grad_desc(X, Theta):
    N = len(X)
    X = np.c_[np.ones(N), X]
    predictions = sigmoid_function(X, Theta)
    for p in range(len(predictions)):
      if predictions[p] < .5:
        predictions[p] = 0
      else:
        predictions[p] = 1
    return predictions


def normalize(X):
    Xnorm = X
    for x in range(X.shape[1]):
        Xnorm[:,x] -= np.mean((X[:,x]), dtype=np.float64)
        Xnorm[:,x] /= np.std(X[:,x])
    return Xnorm
