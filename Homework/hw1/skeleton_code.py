import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


### Assignment Owner: Tian Wang


#######################################
### Feature normalization
def feature_normalization(train, test):
    """Rescale the data so that each feature in the training set is in
    the interval [0,1], and apply the same transformations to the test
    set, using the statistics computed on the training set.

    Args:
        train - training set, a 2D numpy array of size (num_instances, num_features)
        test - test set, a 2D numpy array of size (num_instances, num_features)

    Returns:
        train_normalized - training set after normalization
        test_normalized - test set after normalization
    """
    # TODO
    min_feature = np.min(train,axis=0)
    gap_feature = np.max(train,axis=0)-np.min(train,axis=0)
    train_normalized = (train-min_feature)/gap_feature
    test_normalized = (test-min_feature)/gap_feature
    return train_normalized,test_normalized


#######################################
### The square loss function
def compute_square_loss(X, y, theta):
    """
    Given a set of X, y, theta, compute the average square loss for predicting y with X*theta.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D array of size (num_features)

    Returns:
        loss - the average square loss, scalar
    """
    loss = 0 #Initialize the average square loss
    #TODO
    m = X.shape[0]
    h = X.dot(theta)
    tmp = h-y
    loss = tmp.dot(tmp)/m
    return loss


#######################################
### The gradient of the square loss function
def compute_square_loss_gradient(X, y, theta):
    """
    Compute the gradient of the average square loss (as defined in compute_square_loss), at the point theta.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)

    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    #TODO
    h = X.dot(theta)
    grad = 2*(h-y).dot(X).T/num_instances
    return grad


#######################################
### Gradient checker
#Getting the gradient calculation correct is often the trickiest part
#of any gradient-based optimization algorithm. Fortunately, it's very
#easy to check that the gradient calculation is correct using the
#definition of gradient.
#See http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
def grad_checker(X, y, theta, epsilon=0.01, tolerance=1e-4):
    """Implement Gradient Checker
    Check that the function compute_square_loss_gradient returns the
    correct gradient for the given X, y, and theta.

    Let d be the number of features. Here we numerically estimate the
    gradient by approximating the directional derivative in each of
    the d coordinate directions:
    (e_1 = (1,0,0,...,0), e_2 = (0,1,0,...,0), ..., e_d = (0,...,0,1))

    The approximation for the directional derivative of J at the point
    theta in the direction e_i is given by:
    ( J(theta + epsilon * e_i) - J(theta - epsilon * e_i) ) / (2*epsilon).

    We then look at the Euclidean distance between the gradient
    computed using this approximation and the gradient computed by
    compute_square_loss_gradient(X, y, theta).  If the Euclidean
    distance exceeds tolerance, we say the gradient is incorrect.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        epsilon - the epsilon used in approximation
        tolerance - the tolerance error

    Return:
        A boolean value indicating whether the gradient is correct or not
    """
    true_gradient = compute_square_loss_gradient(X, y, theta) #The true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features) #Initialize the gradient we approximate
    #TODO
    for i in np.arange(num_features):
        h=np.zeros(num_features)
        h[i]=1
        J1 = compute_square_loss(X,y,theta+epsilon*h)
        J2 = compute_square_loss(X,y,theta-epsilon*h)
        approx_grad[i] = (J1-J2)/(2*epsilon)
    def Euc_dist(a,b):
        return np.sqrt((a-b).dot(a-b))
    dist = Euc_dist(approx_grad,true_gradient)
    return dist<tolerance


#######################################
### Generic gradient checker
def generic_gradient_checker(X, y, theta, objective_func, gradient_func, epsilon=0.01, tolerance=1e-4):
    """
    The functions takes objective_func and gradient_func as parameters. 
    And check whether gradient_func(X, y, theta) returned the true 
    gradient for objective_func(X, y, theta).
    Eg: In LSR, the objective_func = compute_square_loss, and gradient_func = compute_square_loss_gradient
    """
    #TODO
    true_gradient = gradient_func(X, y, theta) #The true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features) #Initialize the gradient we approximate
    for i in np.arange(num_features):
        h=np.zeros(num_features)
        h[i]=1
        J1 = objective_func(X,y,theta+epsilon*h)
        J2 = objective_func(X,y,theta-epsilon*h)
        approx_grad[i] = (J1-J2)/(2*epsilon)
    def Euc_dist(a,b):
        return np.sqrt((a-b).dot(a-b))
    dist = Euc_dist(approx_grad,true_gradient)
    return dist<tolerance


#######################################
### Batch gradient descent
def batch_grad_descent(X, y, alpha=0.1, num_step=1000, grad_check=False, bls=False, c=0.5,tau=0.5):
    """
    In this question you will implement batch gradient descent to
    minimize the average square loss objective.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        num_step - number of steps to run
        grad_check - a boolean value indicating whether checking the gradient when updating

    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size (num_step+1, num_features)
                     for instance, theta in step 0 should be theta_hist[0], theta in step (num_step) is theta_hist[-1]
        loss_hist - the history of average square loss on the data, 1D numpy array, (num_step+1)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_step+1, num_features)) #Initialize theta_hist
    loss_hist = np.zeros(num_step+1) #Initialize loss_hist
    theta = np.zeros(num_features) #Initialize theta
    #TODO
    for steps in np.arange(num_step+1):
        print ('iter_step={}'.format(steps),end='\r')
        loss = compute_square_loss(X,y,theta)
        grad = compute_square_loss_gradient(X,y,theta)
        loss_hist[steps] = loss
        theta_hist[steps,:] = theta
        if grad_check:
            if not grad_checker(X, y, theta):
               print ("algorithm didn't pass gradient check,stoping")
               break
        if bls:
            objective_func = lambda theta:compute_square_loss(X,y,theta)
            alpha = bls_func(c,tau,theta,loss,grad,objective_func)
        theta -= alpha*grad
    return theta_hist,loss_hist

#######################################
### Backtracking line search
#Check http://en.wikipedia.org/wiki/Backtracking_line_search for details
#TODO
def bls_func(c,tau,theta,loss,grad,objective_func):
    grad_norm=np.linalg.norm(grad)
    d = -grad/grad_norm # searching direction, a unit vector
    m = grad.dot(d) # local slope along searching direction
    t = -c*m
    j = 0
    while True:
        theta_add = theta+tau**j*d
        new_loss = objective_func(theta_add)
        if loss - new_loss >= tau**j*t:
            alpha = tau**j/grad_norm
            return alpha
        j+=1


#######################################
### The cost function of regularized batch gradient descent
def compute_regularized_square_loss(X, y, theta,lambda_reg=10**-2):
    """
    Given a set of X, y, theta, compute the average square loss for predicting y with X*theta.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D array of size (num_features)

    Returns:
        loss - the average square loss, scalar
    """
    loss = 0 #Initialize the average square loss
    #TODO
    num_instances = X.shape[0]
    h = X.dot(theta)
    tmp = h-y
    loss = tmp.dot(tmp)/num_instances+theta.dot(theta)*lambda_reg
    return loss


#######################################
### The gradient of regularized batch gradient descent
def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg=10**-2):
    """
    Compute the gradient of L2-regularized average square loss function given X, y and theta

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        lambda_reg - the regularization coefficient

    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    #TODO
    num_instances = X.shape[0]
    h = X.dot(theta)
    grad = 2*(h-y).dot(X).T/num_instances
    grad += 2*lambda_reg*theta
    return grad



#######################################
### Regularized batch gradient descent
def regularized_grad_descent(X, y, alpha=0.05, lambda_reg=10**-2, num_step=1000, grad_check=False, bls=False, c=0.5, tau=0.5):
    """
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        lambda_reg - the regularization coefficient
        num_step - number of steps to run
    
    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size (num_step+1, num_features)
                     for instance, theta in step 0 should be theta_hist[0], theta in step (num_step+1) is theta_hist[-1]
        loss hist - the history of average square loss function without the regularization term, 1D numpy array.
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.zeros(num_features) #Initialize theta
    theta_hist = np.zeros((num_step+1, num_features)) #Initialize theta_hist
    loss_hist = np.zeros(num_step+1) #Initialize loss_hist
    #TODO
    for steps in np.arange(num_step+1):
        print ('iter_step={}'.format(steps),end='\r')
        loss = compute_regularized_square_loss(X,y,theta,lambda_reg)
        grad = compute_regularized_square_loss_gradient(X,y,theta,lambda_reg)
        loss_hist[steps] = loss
        theta_hist[steps,:] = theta
        if grad_check:
            objective_func = lambda X,y,theta: compute_regularized_square_loss(X,y,theta,lambda_reg=lambda_reg)
            grad_func = lambda X,y,theta: compute_regularized_square_loss_gradient(X,y,theta,lambda_reg=lambda_reg)
            grad_check_result=generic_gradient_checker(X,y,theta,objective_func,grad_func)
            if not grad_check_result:
                print ("gradient check not passed, stoping")
                break
        if bls:
            objective_func = lambda theta:compute_regularized_square_loss(X,y,theta,lambda_reg)
            alpha = bls_func(c,tau,theta,loss,grad,objective_func)
        theta -= alpha*grad
    return theta_hist,loss_hist

#######################################
### Stochastic gradient descent
def stochastic_grad_descent(X, y, alpha=0.01, lambda_reg=10**-2, num_epoch=1000):
    """
    In this question you will implement stochastic gradient descent with regularization term

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - string or float, step size in gradient descent
                NOTE: In SGD, it's not a good idea to use a fixed step size. Usually it's set to 1/sqrt(t) or 1/t
                if alpha is a float, then the step size in every step is the float.
                if alpha == "1/sqrt(t)", alpha = 1/sqrt(t).
                if alpha == "1/t", alpha = 1/t.
        lambda_reg - the regularization coefficient
        num_epoch - number of epochs to go through the whole training set

    Returns:
        theta_hist - the history of parameter vector, 3D numpy array of size (num_epoch, num_instances, num_features)
                     for instance, theta in epoch 0 should be theta_hist[0], theta in epoch (num_epoch) is theta_hist[-1]
        loss hist - the history of loss function vector, 2D numpy array of size (num_epoch, num_instances)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones(num_features) #Initialize theta

    theta_hist = np.zeros((num_epoch, num_instances, num_features)) #Initialize theta_hist
    loss_hist = np.zeros((num_epoch, num_instances)) #Initialize loss_hist
    if alpha == "1/sqrt(t)": 
        alpha = lambda t: 0.1/np.sqrt(t)
    elif alpha == "1/t": 
        alpha = lambda t: 0.1/(t)
    elif alpha == "extra": 
        alpha = lambda t: 0.05/(1+0.05*lambda_reg*t)
    #TODO
    for epoch in np.arange(num_epoch):
        print ('iter_epoch={}'.format(epoch),end='\r')
        for i in np.arange(num_instances):
            loss = compute_regularized_square_loss(X,y,theta,lambda_reg)
            h = theta.dot(X[i,:])
            tmp = h-y[i]
            grad = 2*(tmp*X[i,:]+lambda_reg*theta)
            loss_hist[epoch,i] = loss
            theta_hist[epoch,i,:] = theta
            if np.isscalar(alpha):
                theta -= alpha*grad
            else:
                theta -= alpha((epoch+1)*(i+1))*grad
    return theta_hist,loss_hist


def main():
    #Loading the dataset
    print('loading the dataset')

    df = pd.read_csv('data.csv', delimiter=',')
    X = df.values[:,:-1]
    y = df.values[:,-1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =100, random_state=10)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))  # Add bias term
    # TODO
    lambda_reg = 0.1
    #theta_hist,loss_hist = batch_grad_descent(X_train, y_train,alpha=0.05,grad_check=False,bls=True)
    #theta_hist,loss_hist = regularized_grad_descent(X_train, y_train,lambda_reg=lambda_reg,grad_check=False,bls=False)
# QUESTIONS:
# 1. large B without bls will crash, which is not supposed to be. since large B should equivlent to small lambda. Need to find the reason
# 2. review Stochastic gradient descent
def SGD_experiments():
    #Loading the dataset
    print('loading the dataset')

    df = pd.read_csv('data.csv', delimiter=',')
    X = df.values[:,:-1]
    y = df.values[:,-1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =100, random_state=10)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))  # Add bias term
    # TODO
    lambda_reg = 0.1
    for alpha in ['1/t','1/sqrt(t)',5e-2,5e-3,'extra']:
        theta_hist,loss_hist = stochastic_grad_descent(X_train, y_train,alpha=alpha,lambda_reg=lambda_reg)
        plt.semilogy(loss_hist[:,-1],label=alpha)
    plt.legend()
    plt.show()

def find_prop_B_lambda():
    #Loading the dataset
    print('loading the dataset')

    df = pd.read_csv('data.csv', delimiter=',')
    X = df.values[:,:-1]
    y = df.values[:,-1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =100, random_state=10)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    B_list = [1,10,30,100]
    lambda_list=[1e-7,1e-5,1e-3,1e-1,1,10,100]
    train_loss=np.zeros((len(lambda_list),len(B_list)))
    test_loss =np.zeros((len(lambda_list),len(B_list)))
    for j,B in enumerate(B_list):
        X_train = np.hstack((X_train, B*np.ones((X_train.shape[0], 1))))  # Add bias term
        X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))  # Add bias term
        # TODO

        # find suitable lambda_reg
        for i,lambda_reg in enumerate(lambda_list):
            theta_hist,loss_hist = regularized_grad_descent(X_train, y_train,alpha=0.01,lambda_reg=lambda_reg,grad_check=False,bls=True)
            theta = theta_hist[-1,:]
            train_loss[i,j]=(loss_hist[-1])
            test_loss[i,j]=(compute_square_loss(X_test,y_test,theta))
    plt.contour(test_loss)
    plt.show()
if __name__ == "__main__":
    #main()
    #find_prop_B_lambda()
    SGD_experiments()
