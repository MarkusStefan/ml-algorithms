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


class SVM:

    def __init__(self, n_iter:int=1000):
        pass