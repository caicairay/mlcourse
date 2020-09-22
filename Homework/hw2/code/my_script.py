from ridge_regression import *
import matplotlib.pyplot as plt

lasso_data_fname = "lasso_data.pickle"
x_train, y_train, x_val, y_val, target_fn, coefs_true, featurize = load_problem(lasso_data_fname)
# Generate features
X_train = featurize(x_train)
X_val = featurize(x_val)

def calc_loss(X_train,y_train,X_test,y_test,l2_reg):
    ridge_regression_estimator = RidgeRegression(l2reg=l2_reg)
    ridge_regression_estimator.fit(X_train, y_train)
    loss = ridge_regression_estimator.score(X_test,y_test)
    return loss
lambda_list = [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,1e1]
loss = lambda x: calc_loss(X_train,y_train,X_val,y_val,x)
loss_list = map(loss,lambda_list)
plt.figure()
plt.semilogx(lambda_list,list(loss_list))
plt.show(block=False)

plt.figure()
plt.plot(x_train,y_train,'k.',label='Data')
x = np.linspace(0,1,1000)
X = featurize(x)
plt.plot(x, target_fn(x),color='tab:red',label='Target Function')
ridge_regression_estimator = RidgeRegression(l2reg=0) # unregularized
ridge_regression_estimator.fit(X_train, y_train)
plt.plot(x, ridge_regression_estimator.predict(X),'orange',label='unregularized least square fit')
ridge_regression_estimator = RidgeRegression(l2reg=1e-2) # the choosen lambda
ridge_regression_estimator.fit(X_train, y_train)
plt.plot(x, ridge_regression_estimator.predict(X),'b',label='Prediction Function, lambda=1e-2')
plt.legend()
plt.show(block=False)

ridge_regression_estimator = RidgeRegression(l2reg=1e-2) # the choosen lambda
ridge_regression_estimator.fit(X_train, y_train)
coefs = ridge_regression_estimator.w_
thresholds = [1e-6,1e-3,1e-1]
def thresholding(coefs,threshold):
    coefs_threshold=coefs.copy()
    coefs_threshold[np.abs(coefs)<threshold] = 0
    return coefs_threshold
coefs_th_list = map(lambda x: thresholding(coefs,x),thresholds)
fig, axs = plt.subplots(len(thresholds),1, sharex=True)
for i in range(len(thresholds)):
    axs[i].bar(range(len(coefs)), thresholding(coefs,thresholds[i]))
plt.show()
