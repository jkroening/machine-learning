import csv as csv
import numpy as np
import pandas as pd
import functions as fn
from sklearn import preprocessing
import pylab as pl
from sklearn.cross_validation import train_test_split


#Open up the csv file in to a Python object
train_df = pd.read_csv('train.csv')

# convert categorical variables to indicator variables
train_df['Embarked'] = train_df['Embarked'].replace(['C', 'Q', 'S'], [1, 2, 3])
train_df['Sex'] = train_df['Sex'].replace(['male', 'female'], [0, 1])
for i in range(len(train_df['Cabin'])):
  if pd.isnull(train_df['Cabin'][i]):
    train_df['Cabin'][i] = 0
  else:
    train_df['Cabin'][i] = train_df['Cabin'][i][:1]
train_df['Cabin'] = train_df['Cabin'].replace(['C', 'E', 'G', 'D', 'A', 'B', 'F', 'T'], [1, 2, 3, 4, 5, 6, 7, 8])

# impute missing data
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].median())
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())

# all columns except id, survived, name, and ticket number
# features left, in order:  pclass, sex, age, sibsp, parch, fare, cabin, embarked
# convert from a list to an array.
X = np.array(train_df[[2,4,5,6,7,9,10,11]], dtype=np.float64)
# response variable: survived or perished
# it's necessary to flatten the y to an array to perform vector subtraction in gradient descent
y = np.array(train_df[[1]], dtype=np.float64).flatten()

### Normalize and Scale train data ###

# normalize using my own function of mean over standard deviation
# X = fn.normalize(X)

# scale and normalize across all features using sklearn package
min_max_scaler = preprocessing.MinMaxScaler()
scaled_train = preprocessing.scale(X)
X = scaled_train
X = preprocessing.normalize(X, norm='l2')

# split original train data into train and test sets at a 70/30 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

# ### The following section is implemented if you're using a separate test file, rather than splitting the train data ###
# ### Do the same for the test data ###

# #Open up the csv file in to a Python object
# test_df = pd.read_csv('test.csv')

# # convert categorical variables to indicator variables
# test_df['Embarked'] = test_df['Embarked'].replace(['C', 'Q', 'S'], [1, 2, 3])
# test_df['Sex'] = test_df['Sex'].replace(['male', 'female'], [0, 1])
# for i in range(len(test_df['Cabin'])):
#   if pd.isnull(test_df['Cabin'][i]):
#     test_df['Cabin'][i] = 0
#   else:
#     test_df['Cabin'][i] = test_df['Cabin'][i][:1]
# test_df['Cabin'] = test_df['Cabin'].replace(['C', 'E', 'G', 'D', 'A', 'B', 'F', 'T'], [1, 2, 3, 4, 5, 6, 7, 8])

# # impute missing data
# test_df['Embarked'] = test_df['Embarked'].fillna(test_df['Embarked'].median())
# test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())
# test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

# # all columns except id, survived, name, and ticket number
# # features left, in order:  pclass, sex, age, sibsp, parch, fare, cabin, embarked
# X_test_df= test_df[[1,3,4,5,6,8,9,10]]

# #Then convert from a list to an array.
# X_test = np.array(X_test_df, dtype=np.float64)

### Normalize and Scale test data ###

# normalize using my own function of mean over standard deviation
# X_test = fn.normalize(X_test)

# # scale and normalize across all features using sklearn package
# min_max_scaler = preprocessing.MinMaxScaler()
# scaled_test = preprocessing.scale(X_test)
# X_test = scaled_test
# X_test = preprocessing.normalize(X_test, norm='l2')

dimX_train = X_train.shape  # get dimensions of X as a tuple (rows, columns)
N = dimX_train[1]  # of columns in X; number of features
Theta = np.zeros(N + 1)  # add a column for theta0
l_mbda = 1 # regularization parameter

# the non-regularized results
# print "Cost before gradient descent: " , fn.calculate_cost(Theta, X_train, y_train)
# Results = fn.gradient_descent(X_train, y_train, Theta, .01, 1000)
# print "Thetas: " , Results[0]
# Theta = Results[0]
# print "Cost after gradient descent: ", fn.calculate_cost(Theta, X_train, y_train)

# the regularized results
print "Cost before gradient descent: " , fn.regularized_calculate_cost(Theta, X_train, y_train, l_mbda)
Results = fn.regularized_gradient_descent(X_train, y_train, Theta, .1, 5000, l_mbda)
print "Thetas: " , Results[0]
Theta = Results[0]
print "Cost after gradient descent: ", fn.regularized_calculate_cost(Theta, X_train, y_train, l_mbda)

# predict using our Theta results on the test set
predictions = fn.predict_grad_desc(X_test, Theta)
# # print predictions vector
# print "Predictions: ", predictions
# # print expected binary response and respective prediction, side by side
# for i in range(len(predictions)):
#     print int(y_test[i]), int(predictions[i])

# manual score calculation
correct = 0
wrong = 0
total = 0
for i in range(len(predictions)):
    x = predictions[i]
    y = y_test[i]
    if int(x) == int(y):
        correct += 1
    else:
        wrong += 1
    total += 1
    accuracy = (float(correct) / float(total))
print accuracy

# # reformat predict array, adding back in the PassengerId to satisfy submission file requirements
# predict = np.transpose(np.array((test_df['PassengerId'].values, predictions), dtype=np.int))

# # write predictions to csv file
# np.savetxt("submission.csv", predict, delimiter=",", fmt='%i', header='PassengerId,Survived', comments='')

# perform scipy optimize version of logistic regression minimization
scipy_thetas = fn.minBFGS(np.zeros(N + 1), X_train, y_train)
print "SciPy Thetas: ", scipy_thetas

# Plot cost versus iterations to check if it converges
pl.xlabel("Iterations")
pl.ylabel("Cost")
pl.plot(Results[1])

pl.show()

