81.7%

Using sklearn to split the train.csv into train and test data, with seed at 42, I was able to predict with .8172 accuracy on the test set.


I used 3 different versions:

The fmin_bfgs method from scipy with the following results:
Thetas:  [-0.38268439 -0.96092777  2.70111223 -0.9751297  -1.15452975 -0.0635024 1.34048945  0.84322438 -0.40539488]
Current function value: 286.557377

My own regularized gradient descent algorithm with the following results:
Thetas:  [-0.38635719 -1.00436542  2.69974879 -0.97047272 -1.11586694 -0.0509724 1.22254682  0.84820601 -0.4154562 ]
Cost after gradient descent:  286.579466077

My own non-regularized gradient descent algorithm with the following results:
Thetas:  [-0.38268442 -0.96092805  2.70111221 -0.97512973 -1.15452973 -0.06350234 1.34048895  0.84322433 -0.4053953 ]
Cost after gradient descent:  286.557377318


I did not try different combinations of training features, nor feature selection algorithms to determine the best predictors, I simply used them all.  Although you will see that I did some wrangling: converting categorical variables to indicators, and bucketing Cabin data based on first letter of the cabin ID.  I also removed Name and Ticket Number from the feature set because they are certainly not correlated to survival on a sinking ship -- at least I hope they're not.  That would be weird.  "Sorry Bill, tough luck with that name, you're gonna die today."  ..Although perhaps ticket number could be a predictor if it in some way represented class or cabin location, although I didn't see evidence of that sort of relationship.  And I suppose if there are titles (Mr., Ms.) within Name that could help with more accurate imputation of missing Age or missing Gender data.
