# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 20:16:13 2014

@author: jonathankroening
"""
import numpy as np

def calculate_cost(X, Y, Theta):
    cost = 0.0 # you must return the correct value for cost

    # add y-intercept term with a column of ones
    dimX = X.shape # get dimensions of X as a tuple (rows, columns)
    N = dimX[0] # get number of rows
    X = np.c_[np.ones(N), X] # add column of ones at beginning to accommodate theta0

    # calculate the cost of a particular choice of Theta using the least squares method
    total_cost = 0
    for n in range(N):
        # calculate the predicted value (using the calculate_prediction function)
        # for a row in the X matrix, subtract the corresponding actual value in Y,
        # square the result, and add to a running sum.
        single_cost = (calculate_prediction(X[n], Theta)- Y[n]) ** 2
        total_cost += single_cost[0]

    cost = (1.0 / (2.0 * N)) * total_cost
    return cost


def linear_regression(X, Y, Theta, alpha, num_iters):
    N = len(X) # get number of rows
    T = len(Theta)
    origX = X # original X for use in J_array calculation
    X = np.c_[np.ones(N), X]
    temps = [0,0]
    J_array = np.zeros(num_iters)
    for i in range(num_iters):
        # perform a single update to our Theta vector
        for t in range(T):
            total_cost = 0
            for n in range(N):
                single_cost = (calculate_prediction(X[n], Theta) - Y[n]) * X[n][t]
                total_cost += single_cost
            temps[t] = Theta[t][0] - (alpha * (1.0 / N) * total_cost)
        for t in range(T):
            Theta[t][0] = temps[t]
        J_array[i] = calculate_cost(origX, Y, Theta)
    # return the correct values for Theta and the array of cost values for each iteration
    return Theta, J_array


def calculate_prediction(x_val, Theta):
    # calculate the predicted value of y given the feature values in x_val
    # (one training example at a time) and parameters in theta
    y = sum(p * q for p, q in zip(x_val, Theta))
    # return the correct value for y.
    return y
