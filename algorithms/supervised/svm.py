from typing import Literal, Dict
import warnings
import numpy as np



class Kernel:
    
    def __init__(self, type:Literal['linear', 'polynomial', 'rbf', 'sigmoid']):
        self.type = type
        self.kernel = None
        self.kernel_params = None
        self.kernel_matrix = None
        self.is_fitted = False

    def linear(self, X1, X2):
        return np.dot(X1, X2.T)
    
    def polynomial(self, X1, X2, alpha:float|int=0.0, gamma:float|int=0.1, degree:int=3):
        return (alpha + gamma*np.dot(X1, X2.T)) ** degree
    
    def rbf(self, X1, X2, gamma:float=0.1):
        return np.exp(-gamma * np.linalg.norm(X1 - X2) ** 2)
    
    def sigmoid(self, X1, X2, alpha:float|int=0.0, gamma:float|int=0.1):
        return np.tanh(alpha + gamma * np.dot(X1, X2.T))
    


class SVM:

    def __init__(self, eta:int=0.001, epochs:int=1000):
        pass