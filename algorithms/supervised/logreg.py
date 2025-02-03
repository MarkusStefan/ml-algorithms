from typing import Literal
import warnings
import numpy as np


class LogisticRegression:

    def __init__(self, 
                 epochs:int=1000, 
                 eta:float=0.01,
                 algorithm:Literal['ols', 'gd']='ols',
                 batch_size:int=None):
        self.eta = eta
        self.epochs = epochs # iterations for gd
        self.batch_size = batch_size
        # self.algorithm = algorithm  # to be implemented
        self.beta = None # weights & biases

    def sigmoid(x):
        return 1 / ( 1 + np.exp(-x) )

    def _proba_to_label(self, proba:np.ndarray, threshold:int=0.5):
        """
        Converts probabilities to binary labels.
        """
        return np.where(proba >= threshold, 1, 0)

    def cost_func(self, X, y):
        """
        Computes Misclassification Rate as the cost function (mean squared error).
        """
        n, m = X.shape # n rows, m cols
        linear_predictions = np.dot(X, self.beta) 
        predictions_proba = self.sigmoid(linear_predictions) # convert to probabilities using sigmpid function
        predictions = self._proba_to_label(predictions_proba) # convert pobabilities to binary labels
        mcr_cost = np.sum(predictions != y) / n # percentage of misclassified samples
        return mcr_cost



    def gradient_descent(self, X, y):
        """
        Performs GD to update beta params.
        Default: use entire dataset in each iteration.
        Optional: batch_size can be set for batch GD.
        """
        n, m = X.shape # n rows, m cols
        cost_history = [] # container to store loss history

        if self.batch_size is not None:
            for _ in range(self.epochs):
                # shuffle dataset
                indices = np.random.permutation(n)
                X = X[indices]
                y = y[indices]

                for i in range(0, n, self.batch_size): # start, stop, step
                    # get batch
                    X_i = X[i:i+self.batch_size]
                    y_i = y[i:i+self.batch_size]

                    # predictions
                    y_hat = np.dot(X_i, self.beta) # matmul
                    residuals = y_hat - y_i
                    gradient = np.dot(X_i.T, residuals) / self.batch_size # matmul with transposed X and residuals
                    self.beta -= gradient * self.eta # move towards local minima
                    cost = self.cost_func(X_i, y_i)
                    cost_history.append(cost)
            return cost_history # return cost history for plotting
        else: 
            for _ in range(self.epochs):
                # y_hat are the predicted values
                y_hat = np.dot(X, self.beta) # matmul
                residuals = y_hat - y 
                gradient = np.dot(X.T, residuals) / n # matmul with transposed X and residuals
                self.beta -= gradient * self.eta # move towards local minima
                cost = self.cost_func(X, y)
                cost_history.append(cost)

            return cost_history # return cost history for plotting



    def fit(self, X, y):
        """
        Fits the LR model to the data.
        """
        # add bias term to X (intercept) ... consisting of just 1s
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        n, m_features = X.shape 
        self.beta = np.zeros(m_features) # init beta parameters for all m features
        
        if self.batch_size is not None and self.batch_size > n:
            raise ValueError("Batch size cannot be greater than the number of samples.")
        
        cost_history = self.gradient_descent(X, y) # perform GD for learning betas


        if np.all(np.isnan(self.beta)):
            warnings.warn(f"Learning rate of {self.eta} or number of epochs set at {self.epochs} \
                           may be too high for the dataset.\nConsider adjusting them for better convergence.")
        
        return cost_history # return cost history for plotting



    def predict(self, X):
        """
        Makes predictions on new data.
        """
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1) # add bias term
        return np.dot(X, self.beta) # matmul by learned beta parameters

