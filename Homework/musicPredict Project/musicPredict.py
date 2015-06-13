import sys
import numpy as np
import pylab as pl
import csv
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn import svm
from sklearn import feature_selection
from sklearn import metrics
from sklearn import linear_model
from sklearn import cluster
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import pandas as pd
import operator
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import scipy
import os

randomstate = 42

def load_data(ratings, albums):
    # print albums.ix[:,[0,3:]]
    ratings = ratings.drop(['artist','title'], 1)

    # join albums (one) to ratings (many) on primary key 'albumID'
    ratings = albums.merge(ratings, on='albumID')
    ratings['plays'] = ratings['plays'].fillna(0)

    return ratings

def load_user_data(ratings, albums, userID):

    # subset by userID
    ratings = ratings[ratings['userID']==userID]

    # remove any albums without an albumID, because they can't be joined
    mask = ratings['albumID'].str.len() > 1
    ratings = ratings.loc[mask]

    # remove any albums without an albumID, because they can't be joined
    mask = albums['albumID'].str.len() > 1
    albums = albums.loc[mask]

    # print albums.ix[:,[0,3:]]
    ratings = ratings.drop(['artist','title'], 1)

    # join albums (one) to ratings (one -- because it's only for one user) on primary key 'albumID', using left outer join so we keep unrated albums too
    ratings = albums.merge(ratings, on='albumID', how='left')
    ratings['plays'] = ratings['plays'].fillna(0)

    return ratings

def numpyize(df):
    # for test method, model evaluation

    # drop artist and title
    df = df.drop(['artist','title'], 1)

    # subset only albums that have been rated
    df = df[pd.notnull(df['rating'])]

    # extract Converted column as y (response) vector
    y = np.array(df.ix[:,'rating'].values, dtype='float')

    # save userID in new array so you can see who it is later
    user_list = list(df['userID'])

    # save albumID in new array so you can see who it is later
    album_list = list(df['albumID'])

    # drop albumID and rating columns from data frame, and remaining features are the independent variables
    df = df.drop('albumID', 1)
    df = df.drop('userID', 1)
    df = df.drop('rating', 1)
    df = df.drop('plays', 1)
    X = np.array(df.values, dtype='float')

    return X, y, album_list

def unrated_separate(df):

    # subset albums that haven't been rated
    unrated = df[pd.isnull(df['rating'])]
    unrated = unrated.drop_duplicates(cols='albumID')

    # subset the albums that have been rated
    df = df[pd.notnull(df['rating'])]
    df = df.drop_duplicates(cols='albumID')

    y = df.ix[:,'rating'].values
    df = df.drop('plays',1)
    df = df.drop('userID', 1)
    df = df.drop('rating', 1)
    unrated = unrated.drop('rating',1)
    unrated = unrated.drop('plays',1)
    unrated = unrated.drop('userID',1)

    return df, y, unrated

def numpyize_by_user(df, y, userID):
    # for predict method

    # # subset only albums that have been rated
    # df = df[pd.notnull(df['rating'])]

    # extract Converted column as y (response) vector
    # y = np.array(df.ix[:,'rating'].values, dtype='float')
    y = y.astype('float')

    # save albumID in new array so you can see who it is later
    album_list = []

    for i, value in enumerate(df):
        album_list.append([value[0], value[1], value[2]])

    # drop albumID and rating columns from data frame, and remaining features are the independent variables
    df = np.delete(df,[0,1,2],1)

    X = df

    return X, y, album_list

def buildUnrated(unrated, albumsDB):
    # unimplimented method -- not necessary anymore

    # these two lines makes it so only albums found in allMusic are included in prediction
    mask = unrated['albumID'].str.len() > 1
    unrated = unrated.loc[mask]

    # join albums (one) to unrated (one -- because it's only for one user) on primary key 'albumID', using left outer join (ratings)
    unrated = unrated.merge(albumsDB, on='albumID', how='left')
    # unrated = unrated.drop_duplicates(cols='albumID')
    unrated = np.array(unrated)

    return unrated

def randomInitializeWeights(weights, factor):
##### This is implemented for you. Think: Why do we do this? #################################

    W = np.random.random((weights[0], weights[1]))
    #normalize so that it spans a range of twice epsilon
    W = W * 2.0 * factor # applied element wise
    #shift so that mean is at zero
    W = W - factor #L_in is the number of input units, L_out is the number of output
    #units in layer

    return W

def PCAize(X, pca_X, pca_Y, j, p, save=True):
    pca = PCA(n_components=p)
    X_new = pca.fit(X).transform(X)
    if save:
        pca_X, pca_Y = PCAeval(pca, pca_X, pca_Y, j, p)
        return X_new, pca_X, pca_Y
    else:
        return X_new, None, None

def PCAeval(pca, pca_X, pca_Y, j, p):
    pca_X[j-1].append(p)
    pca_Y[j-1].append(np.sum(pca.explained_variance_ratio_))
    with open('pca_eval.txt', 'ab') as f:
        f.write("User %s -- Components %s\n" % (j, p))
        f.write("%s\n" % np.sum(pca.explained_variance_ratio_))
    return pca_X, pca_Y

def PCAplot(pca_X, pca_Y, n_components):
    # for i, j in enumerate(pca_X):
    #     print pca_X[i], pca_Y[i]
    pl.plot(pca_X, pca_Y)
    pl.xlim([0,n_components])
    pl.ylim([0,1])
    pl.show()

def featureCorrelation(X):
    corr = np.corrcoef(np.array(X, dtype=float))
    for index, value in np.ndenumerate(corr):
        if 0.95 < value < 0.99999999999999:
            with open('feat_corr_eval.txt', 'ab') as f:
                f.write("Features:   %s  .  %s    --> Correlation: %f\n" % (X.columns[index[0]], X.columns[index[1]], value))
        if -0.999999999999 < value < -0.95:
            with open('feat_corr_eval.txt', 'ab') as f:
                f.write("Features:   %s  .  %s    --> Correlation: %f\n" % (X.columns[index[0]], X.columns[index[1]], value))
            if value < 1.0 and value > -1.0:
                with open('feat_corr_eval.txt', 'ab') as f:
                    f.write("Components: %s -- Index: %s -- Correlation: %f\n" % (p, index, value))

def propOdds(X_train, y_train, X_test, y_test, i, show=True, a=None):
    # proportional odds model, a.k.a. ordinal logistic regression, a.k.a. rank loss
    import modules.logistic as lg

    w, theta = lg.ordinal_logistic_fit(X_train, y_train, verbose=True, solver='TNC')
    pred = lg.ordinal_logistic_predict(w, theta, X_test)
    error = metrics.mean_absolute_error(y_test, pred)/2  # divide by 2 because the original y star ratings were multiplied by 2
    print('ERROR (ORDINAL)                 fold %s: %s' % (i, error))
    correct = 0
    for k in range(len(y_test)):
        if show:
            print a[k], float(y_test[k])/2, float(pred[k])/2
        if y_test[k] == pred[k]:
            correct += 1
    score = float(correct)/float(len(y_test))
    if show:
        print "Mean Absolute Error: ", error
    print 'Score: ', score
    return error, score, w, theta

def logReg(X_train, y_train, X_test, y_test, i, show=True, a=None):
    # logistic regression model
    from sklearn import linear_model

    clf = linear_model.LogisticRegression(C=1.0)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    error = metrics.mean_absolute_error(y_test, pred)/2  # divide by 2 because the original y star ratings were multiplied by 2
    print('ERROR (LOGISTIC)                 fold %s: %s' % (i, error))
    if show:
        for k in range(len(y_test)):
            print a[k], float(y_test[k])/2, float(pred[k])/2
    score = clf.score(X_test, y_test)
    if show:
        print "Mean Absolute Error: ", error
    print 'Score: ', score
    return error, score, clf

def ridgeReg(X_train, y_train, X_test, y_test, i, show=True, a=None):
    # linear regression model with Tikhonov regularization, a.k.a ridge regression
    from sklearn import linear_model

    clf = linear_model.Ridge(alpha=1.0)
    clf.fit(X_train, y_train)
    pred = np.round(clf.predict(X_test))
    error = metrics.mean_absolute_error(y_test, pred)/2  # divide by 2 because the original y star ratings were multiplied by 2
    print('ERROR (RIDGE)                      fold %s: %s' % (i, error))
    correct = 0
    for k in range(len(y_test)):
        if show:
            print a[k], float(y_test[k])/2, float(pred[k])/2
        if y_test[k] == pred[k]:
            correct += 1
    score = float(correct)/float(len(y_test))
    if show:
        print "Mean Absolute Error: ", error
    print 'Score: ', score
    return error, score, clf

def decisionTree(X_train, y_train, X_test, y_test, i=None, show=True, a=None):
    # decision tree classifier model
    from sklearn import tree

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    error = metrics.mean_absolute_error(y_test, pred)/2  # divide by 2 because the original y star ratings were multiplied by 2
    if i is not None:
        print('ERROR (DECISION TREE)        fold %s: %s' % (i, error))
    if show:
        for k in range(len(y_test)):
            print a[k], float(y_test[k])/2, float(pred[k])/2
    score = clf.score(X_test, y_test)
    if show:
        print "Mean Absolute Error: ", error
    print 'Score: ', score
    return error, score, clf

def randomForest(X_train, y_train, X_test, y_test, i, n_trees=10, show=True, a=None):
    # random forest ensemble model of decision trees
    from sklearn import ensemble

    print "Number of trees: %d" % n_trees
    clf = ensemble.RandomForestClassifier(n_estimators=n_trees)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    error = metrics.mean_absolute_error(y_test, pred)/2  # divide by 2 because the original y star ratings were multiplied by 2
    if i is not None:
        print('ERROR (RANDOM FOREST)    fold %s: %s' % (i, error))
    if show:
        for k in range(len(y_test)):
            print a[k], float(y_test[k])/2, float(pred[k])/2
    score = clf.score(X_test, y_test)
    if show:
        print "Mean Absolute Error: ", error
    print 'Score: ', score
    return error, score, clf

def neuralNetwork(X_train, y_train, X_test, y_test, i, show=True, a=None):
    # neural network of logistic regression models
    import modules.NeuralNet_ as nn

    random_thetas1 = randomInitializeWeights((int((X_train.shape[1]+len(np.unique(y_train)))/2), X_train.shape[1]+1), .12)
    random_thetas2 = randomInitializeWeights((10, int((X_train.shape[1]+len(np.unique(y_train)))/2)+1), .12)
    params = np.concatenate([random_thetas1.flatten(), random_thetas2.flatten()])
    input_layer_size = X_train.shape[1]
    hidden_layer_size = int((X_train.shape[1]+len(np.unique(y_train)))/2)
    num_labels = 10  # this is set to 10 because y can take on 10 different values, adjust accordingly
    lambd = 1.0  # regularization parameter
    args = (input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambd)
    result = scipy.optimize.fmin_cg(nn.getCost, params, fprime=nn.getGrad, gtol=.001, args=args, maxiter=1000)
    pred = nn.forwardPropAndAccuracy(result, input_layer_size, hidden_layer_size, num_labels, X_test, y_test)[0]
    error = metrics.mean_absolute_error(y_test, pred)/2  # divide by 2 because the original y star ratings were multiplied by 2
    print('ERROR (NEURAL NETWORK)  fold %s: %s' % (i, error))
    correct = 0
    for k in range(len(y_test)):
        if show:
            print a[k], float(y_test[k])/2, float(pred[k])/2
        if y_test[k] == pred[k]:
            correct += 1
    score = float(correct)/float(len(y_test))
    if show:
        print "Mean Absolute Error: ", error
    print 'Score: ', score
    return error, score, result, input_layer_size, hidden_layer_size, num_labels


def main(method='predict', userID=1, n_components=0):
    # read in the data file
    albumsDB = pd.read_csv('musicPredict Data/albumsDB.csv')
    ratingsDB = pd.read_csv('musicPredict Data/ratingsDB_all.csv')

    # save highly correlated feature relationships to txt file
    albums_feats_only = albumsDB.drop(['albumID','artist','title'], 1)
    albums = np.array(albums_feats_only.values, dtype='float')
    featureCorrelation(albums_feats_only)

    if method == 'test':
    #################################################################
    ######################### model evaluation #########################

        # join the two tables on albumID key
        ratings = load_data(ratingsDB, albumsDB)

        total_error_logistic = []
        total_error_ordinal_logistic = []
        total_error_ridge = []
        total_error_decision_tree = []
        total_error_random_forest = []
        total_error_neural_network = []

        total_score_logistic = []
        total_score_ordinal_logistic = []
        total_score_ridge = []
        total_score_decision_tree = []
        total_score_random_forest = []
        total_score_neural_network = []

        num_users = max(ratings['userID'])
        pca_X = [[] for i in range(num_users)]
        pca_Y = [[] for i in range(num_users)]

        # for different numbers of pca components
        for p in range(0,n_components+1):

            if p == 0 and n_components != 0:  # skip 0, PCA n_components of 0 is nonsense
                continue

            if n_components != 0:  # if PCA n_components is not 0, use PCA
                print '-------------------- PCA Components = %d' % p, '--------------------'
            else:
                p = albums.shape[1]  # set number of features used to ALL

            # for each user
            for j in range(1,num_users+1):

                print '-------------------- User %d' % j, 'of %d' % num_users, '--------------------'

                user_ratings = ratings[ratings['userID']==j]
                X, y, album_list = numpyize(user_ratings)

                if n_components != 0:
                    # transform X to a reduced dimensionality of n PCA components
                    # if save is True then plot explained variance and save the results to txt file for evaluation purposes
                    X, pca_X, pca_Y = PCAize(X, pca_X, pca_Y, j, p=p, save=True)

                # multiple y ratings by 2 to convert half-star ratings into integers
                y = y*2
                y.astype(int)

                # build cross validation sets for evaluation
                from sklearn import cross_validation
                cv = cross_validation.ShuffleSplit(y.size, n_iter=10, test_size=.3, random_state=0)

                error_logistic = []
                error_ordinal_logistic = []
                error_ridge = []
                error_decision_tree = []
                error_random_forest = []
                error_neural_network = []
                score_logistic = []
                score_ordinal_logistic = []
                score_ridge = []
                score_decision_tree = []
                score_random_forest = []
                score_neural_network = []

                # run models on each cross validation set
                for i, (train, test) in enumerate(cv):

                    print '-------------------- Fold #%d' % (i+1), '--------------------'

                    if not np.all(np.unique(y[train]) == np.unique(y)):
                        # we need the train set to have all of the available classes
                        continue
                    assert np.all(np.unique(y[train]) == np.unique(y))

                    ##### proportional odds model #####
                    if 'ordinal' in model:
                        # scale y ratings by minimum value present
                        y_po = y - y.min()
                        # the rank loss function is concerned with order (rank), so we sort the sets
                        idx = np.argsort(y_po)
                        X_po = X[idx]
                        y_po = y_po[idx]
                        train_po = np.sort(train)
                        test_po = np.sort(test)
                        error, score, or_clf, theta = propOdds(X_po[train_po], y_po[train_po], X_po[test_po], y_po[test_po], i+1, show=False)
                        error_ordinal_logistic.append(error)
                        score_ordinal_logistic.append(score)

                    ##### logistic regression model #####
                    if 'logistic' in model:
                        error, score, lg_clf = logReg(X[train], y[train], X[test], y[test], i+1, show=False)
                        error_logistic.append(error)
                        score_logistic.append(score)

                    # ##### linear ridge regression model #####
                    if 'linear' in model:
                        error, score, lr_clf = ridgeReg(X[train], y[train], X[test], y[test], i+1, show=False)
                        error_ridge.append(error)
                        score_ridge.append(score)

                    # ##### decision tree model #####
                    if 'tree' in model:
                        error, score, dt_clf = decisionTree(X[train], y[train], X[test], y[test], i+1, show=False)
                        error_decision_tree.append(error)
                        score_decision_tree.append(score)

                    # ##### random forest model #####
                    if 'forest' in model:
                        error, score, rf_clf = randomForest(X[train], y[train], X[test], y[test], i+1, n_trees=1000, show=False)
                        error_random_forest.append(error)
                        score_random_forest.append(score)

                    # ##### neural network model #####
                    if 'neural' in model:
                        error, score, result, input_layer_size, hidden_layer_size, num_labels = neuralNetwork(X[train], y[train], X[test], y[test], i+1, show=False)
                        error_neural_network.append(error)
                        score_neural_network.append(score)

                total_error_ordinal_logistic.append(np.mean(error_ordinal_logistic))
                total_error_logistic.append(np.mean(error_logistic))
                total_error_ridge.append(np.mean(error_ridge))
                total_error_decision_tree.append(np.mean(error_decision_tree))
                total_error_random_forest.append(np.mean(error_random_forest))
                total_error_neural_network.append(np.mean(error_neural_network))

                total_score_ordinal_logistic.append(np.mean(score_ordinal_logistic))
                total_score_logistic.append(np.mean(score_logistic))
                total_score_ridge.append(np.mean(score_ridge))
                total_score_decision_tree.append(np.mean(score_decision_tree))
                total_score_random_forest.append(np.mean(score_random_forest))
                total_score_neural_network.append(np.mean(score_neural_network))

                # write mean error and score for each model for each user to csv file
                with open("model_eval_all.csv", "ab") as f:
                    c = csv.writer(f)
                    c.writerow([p, j, np.mean(error_ordinal_logistic), np.mean(error_logistic), np.mean(score_logistic), np.mean(error_decision_tree), np.mean(score_decision_tree),
                               np.mean(error_random_forest), np.mean(score_random_forest), np.mean(error_neural_network), np.mean(score_neural_network)])

                print('\nMEAN ABSOLUTE ERROR (ORDINAL LOGISTIC):    %s' % np.mean(error_ordinal_logistic))
                print('MEAN ABSOLUTE ERROR (LOGISTIC REGRESSION): %s' % np.mean(error_logistic))
                print('MEAN ABSOLUTE ERROR (RIDGE REGRESSION):    %s' % np.mean(error_ridge))
                print('MEAN ABSOLUTE ERROR (DECISION TREE):    %s' % np.mean(error_decision_tree))
                print('MEAN ABSOLUTE ERROR (RANDOM FOREST):    %s' % np.mean(error_random_forest))
                print('MEAN ABSOLUTE ERROR (NEURAL NETWORK):    %s' % np.mean(error_neural_network))

                print('\nMEAN ABSOLUTE SCORE (ORDINAL LOGISTIC):    %s' % np.mean(score_ordinal_logistic))
                print('MEAN ABSOLUTE SCORE (LOGISTIC REGRESSION): %s' % np.mean(score_logistic))
                print('MEAN ABSOLUTE SCORE (RIDGE REGRESSION):    %s' % np.mean(score_ridge))
                print('MEAN ABSOLUTE SCORE (DECISION TREE):    %s' % np.mean(score_decision_tree))
                print('MEAN ABSOLUTE SCORE (RANDOM FOREST):    %s' % np.mean(score_random_forest))
                print('MEAN ABSOLUTE SCORE (NEURAL NETWORK):    %s' % np.mean(score_neural_network))

            # write mean error and score for each model averaged over all users to csv file
            with open("model_eval_all.csv", "ab") as f:
                c = csv.writer(f)
                c.writerow([p, 'Total', np.mean(total_error_ordinal_logistic), np.mean(total_error_logistic), np.mean(total_score_logistic), np.mean(total_error_decision_tree), np.mean(total_score_decision_tree),
                           np.mean(total_error_random_forest), np.mean(total_score_random_forest), np.mean(total_error_neural_network), np.mean(total_score_neural_network)])

            print('\nTOTAL MEAN ABSOLUTE ERROR (ORDINAL LOGISTIC):    %s' % np.mean(total_error_ordinal_logistic))
            print('TOTAL MEAN ABSOLUTE ERROR (LOGISTIC REGRESSION): %s' % np.mean(total_error_logistic))
            print('TOTAL MEAN ABSOLUTE ERROR (RIDGE REGRESSION):    %s' % np.mean(total_error_ridge))
            print('TOTAL MEAN ABSOLUTE ERROR (DECISION TREE):    %s' % np.mean(total_error_decision_tree))
            print('TOTAL MEAN ABSOLUTE ERROR (RANDOM FOREST):    %s' % np.mean(total_error_random_forest))
            print('TOTAL MEAN ABSOLUTE ERROR (NEURAL NETWORK):    %s' % np.mean(total_error_neural_network))

            print('\nTOTAL MEAN ABSOLUTE SCORE (ORDINAL LOGISTIC):    %s' % np.mean(total_score_ordinal_logistic))
            print('TOTAL MEAN ABSOLUTE SCORE (LOGISTIC REGRESSION): %s' % np.mean(total_score_logistic))
            print('TOTAL MEAN ABSOLUTE SCORE (RIDGE REGRESSION):    %s' % np.mean(total_score_ridge))
            print('TOTAL MEAN ABSOLUTE SCORE (DECISION TREE):    %s' % np.mean(total_score_decision_tree))
            print('TOTAL MEAN ABSOLUTE SCORE (RANDOM FOREST):    %s' % np.mean(total_score_random_forest))
            print('TOTAL MEAN ABSOLUTE SCORE (NEURAL NETWORK):    %s' % np.mean(total_score_neural_network))

        if n_components != 0:
            for user in range(1, num_users+1):
                PCAplot(pca_X[user], pca_Y[user], n_components)

    if method == 'predict':
    #################################################################
    ######################### model prediction #########################

        # it makes the most sense to predict on a user by user basis, because each has a unique regression or decision tree
        # X, y, album_list = numpyize_by_user(ratings, userID=userID)
        # however, a model can be built on all users at once
        # X, y, user_list, album_list = numpyize(ratings)

        user_ratings = load_user_data(ratingsDB, albumsDB, userID)

        # separate X and y in dataframe
        X, y, unrated = unrated_separate(user_ratings)

        # split data set into test and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=randomstate)

        # # it only makes sense to predict on a user by user basis, because each has a unique regression or decision tree
        X_train, y_train, album_list_train = numpyize_by_user(X_train, y_train, userID=userID)
        X_test, y_test, album_list_test = numpyize_by_user(X_test, y_test, userID=userID)

        X_test = np.array(X_test, dtype='float')
        X_train = np.array(X_train, dtype='float')
        y_test = np.array(y_test*2, dtype='int')
        y_train = np.array(y_train*2, dtype='int')

        if 'ordinal' in model:
            print '\nOrdinal Logistic Regression Model'
            y_train = y_train - y_train.min()
            train_idx = np.argsort(y_train)
            X_train = X_train[train_idx]
            y_train = y_train[train_idx]
            y_test = y_test - y_test.min()
            test_idx = np.argsort(y_test)
            X_test = X_test[test_idx]
            y_test = y_test[test_idx]
            error, score, or_clf, theta = propOdds(X_train, y_train, X_test, y_test, i=None, show=True, a=album_list_test)

        if 'logistic' in model:
            print '\nLogistic Regression Model'
            error, score, lg_clf = logReg(X_train, y_train, X_test, y_test, i=None, show=True, a=album_list_test)

        if 'linear' in model:
            print '\nLinear Regression Model'
            error, score, lr_clf = ridgeReg(X_train, y_train, X_test, y_test, i=None, show=True, a=album_list_test)

        if 'tree' in model:
            print "\nDecision Tree Model"
            error, score, dt_clf = decisionTree(X_train, y_train, X_test, y_test, i=None, show=True, a=album_list_test)

        if 'forest' in model:
            print "\nRandom Forest Model"
            error, score, rf_clf = randomForest(X_train, y_train, X_test, y_test, n_trees=1000, i=None, show=True, a=album_list_test)

        if 'neural' in model:
            print "\nNeural Network Model"
            error, score, result, input_layer_size, hidden_layer_size, num_labels = neuralNetwork(X_train, y_train, X_test, y_test, i=None, show=True, a=album_list_test)


        print "\n---------- Our Best Guesses For You ----------"
        unrated = np.array(unrated)
        unrated, empty_ys, a = numpyize_by_user(unrated, np.empty(len(unrated)), userID=userID)
        unrated = np.array(unrated, dtype='float')

        if 'ordinal' in model:
            print '\nOrdinal Logistic Regression Model'
            import modules.logistic as lg
            g = lg.ordinal_logistic_predict(or_clf, theta, unrated)
            guesses = []
            for k in range(len(g)):
                guesses.append([a[k][0], a[k][1], a[k][2], float(g[k])/2])
            guesses = sorted(guesses, key=operator.itemgetter(3), reverse=True)
            for g in guesses:
                print g

        if 'logistic' in model:
            print '\nLogistic Regression Model'
            g = lg_clf.predict(unrated)
            guesses = []
            for k in range(len(g)):
                guesses.append([a[k][0], a[k][1], a[k][2], float(g[k])/2])
            guesses = sorted(guesses, key=operator.itemgetter(3), reverse=True)
            for g in guesses:
                print g

        if 'linear' in model:
            print '\nLinear Regression Model'
            g = np.round(lr_clf.predict(unrated))
            guesses = []
            for k in range(len(g)):
                guesses.append([a[k][0], a[k][1], a[k][2], float(g[k])/2])
            guesses = sorted(guesses, key=operator.itemgetter(3), reverse=True)
            for g in guesses:
                print g

        if 'tree' in model:
            print "\nDecision Tree Model"
            g = dt_clf.predict(unrated)
            guesses = []
            for k in range(len(g)):
                guesses.append([a[k][0], a[k][1], a[k][2], float(g[k])/2])
            guesses = sorted(guesses, key=operator.itemgetter(3), reverse=True)
            for g in guesses:
                print g

        if 'forest' in model:
            print "\nRandom Forest Model"
            g = rf_clf.predict(unrated)
            guesses = []
            for k in range(len(g)):
                guesses.append([a[k][0], a[k][1], a[k][2], float(g[k])/2])
            guesses = sorted(guesses, key=operator.itemgetter(3), reverse=True)
            for g in guesses:
                print g

        if 'neural' in model:
            import modules.NeuralNet_ as nn
            print "\nNeural Network Model"
            g = nn.forwardPropPredict(result, input_layer_size, hidden_layer_size, num_labels, unrated)
            guesses = []
            for k in range(len(g)):
                guesses.append([a[k][0], a[k][1], a[k][2], float(g[k])/2])
            guesses = sorted(guesses, key=operator.itemgetter(3), reverse=True)
            for g in guesses:
                print g

if __name__=="__main__":
    # method should be either 'test' for model evaluation or 'predict' for ratings predictions
    method = sys.argv[1]

    # which model you want to use (should be one of:  ordinal, logistic, linear, tree, forest, neural, or all -- or a list of models)
    model = sys.argv[2]
    if model == 'all':
        model = ['ordinal', 'logistic', 'linear', 'tree', 'forest', 'neural']

    # if desired, select a number of components for a PCA model, set to 0 to retain all features
    n_components = int(sys.argv[3])

    # userID is ignored for 'test' method, when 'predicting' select the userID for the user for which you'd like to predict ratings
    userID = int(sys.argv[4])

    main(method, userID, n_components)
