import numpy as np
from setup_problem import load_problem
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, make_scorer
import pandas as pd
from ridge_regression import *

def soft_threshold(a, delta):
    if a < - delta:
        return (a + delta)
    elif a > delta:
        return (a - delta)
    else: 
        return 0
    
def compute_sum_sqr_loss(X, y, w):
    return np.dot((np.dot(X,w)-y).T,np.dot(X,w)-y)
    
def compute_lasso_objective(X, y, w, l1_reg=0):
    return np.dot((np.dot(X,w)-y).T,np.dot(X,w)-y) + l1_reg*np.sum(np.abs(w))
    
def shooting_algorithm(X, y, w0=None, l1_reg = 1., max_num_epochs = 1000, min_obj_decrease=1e-8, random=False):
    if w0 is None:
        w = np.zeros(X.shape[1])
    else:
        w = np.copy(w0)
    d = X.shape[1]
    epoch = 0
    obj_val = compute_lasso_objective(X, y, w, l1_reg)
    obj_decrease = min_obj_decrease + 1.
    while (obj_decrease>min_obj_decrease) and (epoch<max_num_epochs):
        obj_old = obj_val
        # Cyclic coordinates descent
        coordinates = range(d)
        # Randomized coordinates descent
        if random:
            coordinates = np.random.permutation(d)
        for j in coordinates:
            aj = 2*np.dot(X[:,j].T,X[:,j])
            cj = 2*np.dot(X[:,j].T,y-np.dot(X,w)+w[j]*X[:,j])
            w[j] = soft_threshold(cj/aj,l1_reg/aj)    
        obj_val = compute_lasso_objective(X, y, w, l1_reg)
        obj_decrease = obj_old - obj_val
        epoch += 1
    print("Ran for "+str(epoch)+" epochs. " + 'Lowest loss: ' + str(obj_val))
    return w, obj_val, epoch

class LassoRegression(BaseEstimator, RegressorMixin):
    """ Lasso regression"""
    def __init__(self, l1_reg=1.0, randomized=False):
        if l1_reg < 0:
            raise ValueError('Regularization penalty should be at least 0.')
        self.l1_reg = l1_reg
        self.randomized = randomized


    def fit(self, X, y, max_epochs = 1000, coef_init=None):
        # convert y to 1-dim array, in case we're given a column vector
        y = y.reshape(-1)
        if coef_init is None:
            coef_init = get_ridge_solution(X,y, self.l1_reg)

        self.w_, obj_val, epoch = shooting_algorithm(X, y, w0=coef_init, l1_reg=self.l1_reg, max_num_epochs=max_epochs, min_obj_decrease=1e-8, random=self.randomized)
        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "w_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        return np.dot(X, self.w_)

    def score(self, X, y):
        try:
            getattr(self, "w_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        return compute_sum_sqr_loss(X, y, self.w_)/len(y)

def do_grid_search_lasso(X_train, y_train, X_val, y_val):
	# Now let's use sklearn to help us do hyperparameter tuning
	# GridSearchCv.fit by default splits the data into training and
	# validation itself; we want to use our own splits, so we need to stack our
	# training and validation sets together, and supply an index
	# (validation_fold) to specify which entries are train and which are
	# validation.
	X_train_val = np.vstack((X_train, X_val))
	y_train_val = np.concatenate((y_train, y_val))
	val_fold = [-1]*len(X_train) + [0]*len(X_val) #0 corresponds to validation

	# Now we set up and do the grid search over l2reg. The np.concatenate
	# command illustrates my search for the best hyperparameter. In each line,
	# I'm zooming in to a particular hyperparameter range that showed promise
	# in the previous grid. This approach works reasonably well when
	# performance is convex as a function of the hyperparameter, which it seems
	# to be here.
	param_grid = [{'l1reg':np.unique(np.concatenate((10.**np.arange(-6,1,1),
			np.arange(1,3,.3)))) }]

	lasso_regression_estimator = ShootingAlgorithm()
	grid = GridSearchCV(lasso_regression_estimator,
			    param_grid,
			    return_train_score=True,
			    cv = PredefinedSplit(test_fold=val_fold),
			    refit = True,
			    scoring = make_scorer(mean_squared_error,
			    greater_is_better = False))
	grid.fit(X_train_val, y_train_val)

	df = pd.DataFrame(grid.cv_results_)
	# Flip sign of score back, because GridSearchCV likes to maximize,
	# so it flips the sign of the score if "greater_is_better=FALSE"
	df['mean_test_score'] = -df['mean_test_score']
	df['mean_train_score'] = -df['mean_train_score']
	cols_to_keep = ["param_l1reg", "mean_test_score","mean_train_score"]
	df_toshow = df[cols_to_keep].fillna('-')
	df_toshow = df_toshow.sort_values(by=["param_l1reg"])
	return grid, df_toshow

lasso_data_fname = "lasso_data.pickle"
x_train, y_train, x_val, y_val, target_fn, coefs_true, featurize = load_problem(lasso_data_fname)

# Generate features
X_train = featurize(x_train)
X_val = featurize(x_val)

#grid, results = do_grid_search_lasso(X_train, y_train, X_val, y_val)
## Plot validation performance vs regularization parameter
#fig, ax = plt.subplots()
#ax.loglog(results["param_l1reg"], results["mean_test_score"])
#ax.semilogx(results["param_l1reg"], results["mean_test_score"])
#ax.grid()
#ax.set_title("Validation Performance vs L1 Regularization")
#ax.set_xlabel("L1-Penalty Regularization Parameter")
#ax.set_ylabel("Mean Squared Error")
#plt.show(block=False)
#
#plt.figure()
#plt.plot(x_train,y_train,'k.',label='Data')
#x = np.linspace(0,1,1000)
#X = featurize(x)
#plt.plot(x, target_fn(x),color='tab:red',label='Target Function')
#lasso_regression_estimator = ShootingAlgorithm(l1reg=grid.best_params_['l1reg'])
#lasso_regression_estimator.fit(X_train, y_train)
#plt.plot(x, lasso_regression_estimator.predict(X),'tab:blue',label='Prediction Function, lambda=1.3')
#lasso_regression_estimator = ShootingAlgorithm(l1reg=1e-7)
#lasso_regression_estimator.fit(X_train, y_train)
#plt.plot(x, lasso_regression_estimator.predict(X),'orange',label='Prediction Function, lambda=1e-7')
#lasso_regression_estimator = ShootingAlgorithm(l1reg=3)
#lasso_regression_estimator.fit(X_train, y_train)
#plt.plot(x, lasso_regression_estimator.predict(X),'cyan',label='Prediction Function, lambda=3')
#plt.legend()
#plt.show(block=False)
#
#lasso_regression_estimator = ShootingAlgorithm(l1reg=grid.best_params_['l1reg']) # the choosen lambda
#lasso_regression_estimator.fit(X_train, y_train)
#coefs = lasso_regression_estimator.w_
#fig,ax = plt.subplots(2)
#ax[0].bar(range(len(coefs_true)),coefs_true)
#ax[0].set_title('True coefs')
#ax[1].bar(range(len(coefs)),coefs)
#ax[1].set_title('Best fitted coefs')
#plt.show(block=False)
#
lambda_max = 2*np.max(np.abs(X_train.T.dot(y_train)))
w_0 = np.zeros(X_train.shape[1])
loss_hist = []
for i in range(30):
    l1reg = lambda_max*0.8**i
    lasso_regression_estimator = ShootingAlgorithm(l1reg=l1reg) # the choosen lambda
    lasso_regression_estimator.fit(X_train, y_train,w_0)
    w_0 = lasso_regression_estimator.w_
    loss_hist.append(lasso_regression_estimator.score(X_val,y_val))
plt.semilogx(lambda_max*0.8**np.arange(30),loss_hist)
plt.grid()
plt.show(block=False)

plt.figure()
plt.plot(x_train,y_train,'k.',label='Data')
x = np.linspace(0,1,1000)
X = featurize(x)
plt.plot(x, target_fn(x),color='tab:red',label='Target Function')
plt.plot(x, lasso_regression_estimator.predict(X),'cyan',label='Prediction Function, soft starting')
plt.legend()
plt.show()
