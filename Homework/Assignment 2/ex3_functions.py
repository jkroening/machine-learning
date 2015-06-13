# -*- coding: utf-8 -*-
"""
Created on Wed Jan  29 10:50:13 2014

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
    total_cost = ((X.dot(Theta) - Y) ** 2).sum() # take the dot product of the X matrix (Nx2) and the Theta matrix (2x1)
                                                                              # then flatten the result because the dot product produces 2-D type results instead of 1-D arrays
                                                                              # subtract the actual Y, square the results and sum them together to get total_cost
    cost = (1.0 / (2.0 * N)) * total_cost
    return cost


def gradient_descent(X, Y, Theta, alpha, num_iters):
    N = len(X) # get number of rows
    T = len(Theta)
    origX = X # original X for use in J_array calculation
    X = np.c_[np.ones(N), X]
    temps = np.empty(T)
    J_array = np.zeros(num_iters)
    for i in range(num_iters):
        for t in range(T):
            total_cost = (((X.dot(Theta) - Y)) * X[:,t]).sum()
            temps[t] = Theta[t] - (alpha * (1.0 / N) * total_cost)
        Theta = temps
        J_array[i] = calculate_cost(origX, Y, Theta)
    # return the correct values for Theta and the array of cost values for each iteration
    return Theta, J_array


def normalize(X):
    Xnorm = X
    for x in range(X.shape[1]):
        Xnorm[:,x] -= np.mean(X[:,x])
        Xnorm[:,x] /= np.std(X[:,x])
    return Xnorm
