from typing import Literal
import warnings
import numpy as np

class LinearRegression():
    """
    Custom LR module which implements GD instead of OLS.
    """
    
    def __init__(self, 
                 epochs:int=1000, 
                 eta:float=0.01, 
                 algorithm:Literal['ols', 'gd']='ols', # implement ols and gd version
                 batch_size:int=None) -> None:
        self.epochs = epochs
        self.eta = eta
        # self.algorithm = algorithm # to be implemented
        self.batch_size = batch_size
        self.beta = None



    def cost_func(self, X, y):
        """
        Computes MSE as the cost function (mean squared error).
        """
        n, m = X.shape # n rows, m cols
        predictions = np.dot(X, self.beta)
        cost = np.sum((predictions - y) ** 2) / (2 * n)
        return cost



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




if __name__ == '__main__':
    print('*'*100)
    print('Linear Regression module.')
    # simple dataset
    X = np.array([[1, 2, 3, 4],
                [2, 3, 4, 5],
                [3, 4, 5, 6],
                [4, 5, 6, 7],
                [5, 6, 7, 8]])
    
    y = np.array([10, 20, 30, 40, 50])

    # init and fit model
    lr = LinearRegression(epochs=10_000, eta=0.01)
    cost_history = lr.fit(X, y)

    print(f"{'Learned parameters:':<40} {lr.beta}")

    preds = lr.predict(X)
    print(f"{'Predictions on the training data:':<40} {preds}")
    print(f"{'Final cost':<40} {cost_history[-1]}")
    print('*'*100)