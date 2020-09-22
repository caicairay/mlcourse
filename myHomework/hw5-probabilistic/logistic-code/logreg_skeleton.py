import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def f_objective(theta, X, y, l2_param=1):
    '''
    Args:
        theta: 1D numpy array of size num_features
        X: 2D numpy array of size (num_instances, num_features)
        y: 1D numpy array of size num_instances
        l2_param: regularization parameter

    Returns:
        objective: scalar value of objective function
    '''
    y_hat = np.dot(X,theta) # (num_instances, )
    s = np.multiply(y,y_hat)
    R_n = np.mean(np.logaddexp(0, -s))
    J_logistic = R_n + l2_param*theta.dot(theta)
    return J_logistic
    
def fit_logistic_reg(X, y, objective_function, l2_param=1):
    '''
    Args:
        X: 2D numpy array of size (num_instances, num_features)
        y: 1D numpy array of size num_instances
        objective_function: function returning the value of the objective
        l2_param: regularization parameter
        
    Returns:
        optimal_theta: 1D numpy array of size num_features
    '''
    n, num_ftrs = X.shape
    # convert y to 1-dim array, in case we're given a column vector
    y = y.reshape(-1)
    theta_0 = np.zeros(num_ftrs)
    obj_func = lambda theta : f_objective(theta,X,y,l2_param)
    theta_ = minimize(obj_func, theta_0).x
    return theta_

def NLL(X,y,theta):
    y_hat = X.dot(theta) # (num_instances, )
    s = np.multiply(y,y_hat)
    nll = np.sum(np.logaddexp(0, -s))
    return nll
def predict(X,theta):
    y_hat = X.dot(theta) # (num_instances, )
    return 1/(1+np.exp(-y_hat))

x_train = np.loadtxt('X_train.txt',delimiter = ',')
y_train = np.loadtxt('y_train.txt',delimiter = ',')
x_val = np.loadtxt('X_val.txt',delimiter = ',')
y_val = np.loadtxt('y_val.txt',delimiter = ',')
x_total = np.vstack((x_train,x_val))
scaler = StandardScaler()
scaler.fit(x_total)
X_train=scaler.transform(x_train)
X_val  =scaler.transform(x_val)
X_train = np.hstack([np.ones([X_train.shape[0],1]),X_train])
X_val = np.hstack([np.ones([X_val.shape[0],1]),X_val])
Y_train = y_train.copy()
Y_val = y_val.copy()
Y_train[y_train == 0] = -1
Y_val[y_val == 0] = -1

l2_param_list = np.logspace(-3,-1,20)
val_NLL_list = np.empty(len(l2_param_list))
for it,l2_param in enumerate(l2_param_list):
    theta = fit_logistic_reg(X_train,Y_train,f_objective,
            l2_param = l2_param)
    val_NLL_list[it]=NLL(X_val,Y_val,theta)
plt.semilogx(l2_param_list,val_NLL_list,label='val NLL')
plt.legend()
plt.show()
print(l2_param_list[np.argmin(val_NLL_list)]) # ~0.018
# 2e-2 minimize

l2_param = 2e-2
theta = fit_logistic_reg(X_train,Y_train,f_objective,
        l2_param = l2_param)
y_true = y_val
y_prob = predict(X_val,theta)
prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=8)
plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))
ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
ax1.plot(prob_true,prob_pred, "s-")
ax2.hist(y_prob, range=(0, 1), bins=10,
         histtype="step", lw=2)
plt.show()
