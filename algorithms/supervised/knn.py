import numpy as np
import warnings
from joblib import Parallel, delayed

class KNNRegression:
    def __init__(self, k_neighbors:int=5, 
                 dist_func:str='euclidean', 
                 p:int=2,
                 reuse_kneighbors:bool=True,
                 parallelize:bool=False, 
                 weighting:str='uniform'):
        self.k = k_neighbors
        self.dist_func = dist_func
        self.p = p
        self.parallelize = parallelize
        self.weighting = weighting
        self.is_fitted = False

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_rows, self.m_cols = self.X.shape
        self.is_fitted = True # set flag to True

    @staticmethod
    def uniform_dist(distances):
        """ 
        Same weight for all k-neighbors.
        """
        return np.ones(len(distances))

    @staticmethod
    def weighted_dist(distances):
        """ 
        Decreasing weights for k-neighbours according to magnitude of distance (inverse of distance).
        """
        prevents_zero_div = 1e-7
        return 1 / np.array([d if d != 0.0 else d + prevents_zero_div for d in distances])

    @staticmethod
    def euclidean_dist(x1, x2):
        """
        Square root of sum of squared differences.
        """
        return np.sqrt(np.sum((np.array(x1) - np.array(x2))**2))
     
    @staticmethod
    def manhattan_dist(x1, x2):
        """ 
        Sum of absolute differences (mimicing anti-diagonal/city-block distances).
        """
        return np.sum(np.abs(np.array(x1) - np.array(x2)))
    

    @staticmethod
    def minkowski_dist(x1, x2, p):
        """
        General metric for computing the p-th root of the aggregated differences raised to the p-th power.
        """
        return np.power(np.sum(np.power(np.abs(np.array(x1) - np.array(x2)), p)), 1/p)
    

    def compute_knn(self, test_sample, dist_func): 
            """ 
            Computes distance according to distance function between
            test_sample and ALL train_samples (brute force approach)
            Args:
                test_sample: data point of test set (passed in iteratively)
                dist_func: distance function to use for computing distances
            Returns:
                array of target variables of the k-closest data points of from the fitted training set
            """

            # distances = []
            distances = np.zeros(self.n_rows) # init distancd array with 0
            
            for i, train_sample in enumerate(self.X): # iterate through every sample, i \in {0,...,n_rows-1}

                d = dist_func(test_sample, train_sample) # compute distance
                # distances.append((d, self.y[i])) # tuple containing (distance, value from training data)
                distances[i] = d # add distance to distance array


            # sort according to distance values (ascending) and yield indices  
            # distances.sort(key=lambda x: x[0])  
            indices = distances.argsort() 
            # yield sorted samples of the training set
            y_sorted = self.y[indices] 
            return y_sorted[:self.k] # only return k-nearest targets from training samples

    def compute_knn_weights(self, test_sample, dist_func):
        """
        Computes k-nearest neighbors and their weights
        """
        # compute k-nearest neighbors using compute_knn helper func
        k_nearest = self.compute_knn(test_sample, dist_func)
        
        # select weighting method
        if self.weighting == 'uniform':
            weights = self.uniform_dist(k_nearest)
        elif self.weighting == 'weighted':
            weights = self.weighted_dist(k_nearest)
        else:
            raise ValueError("Invalid weighting method!")
        
        weighted_sum = 0
        weight_sum = 0
        for i in range(self.k):
            weighted_sum += weights[i] * k_nearest[i]
            weight_sum += weights[i]

        return weighted_sum / weight_sum
    
    def predict(self, X_test):
        """
        Computes distance and inference using a brute force approach
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet!")
            
        # select distance func
        if self.dist_func == 'minowski':
            dist_func = lambda x, y: self.minkowski_dist(x, y, p=self.p) # computes euclidean distance by default
        elif self.dist_func == 'euclidean':
            dist_func = self.euclidean_dist
        elif self.dist_func == 'manhattan':
            dist_func = self.manhattan_dist
        else:
            raise ValueError("Invalid distance function!")
        
        if self.parallelize: 
            y_preds = Parallel(n_jobs=-2)(delayed(self.compute_knn_weights)(x_test, dist_func) for x_test in X_test)
            return y_preds
        
        y_preds = [] # init container for predictions
        for x_test in X_test:
            y_preds.append(self.compute_knn_weights(test_sample=x_test, dist_func=dist_func))

        return y_preds


    def predict_(self, X):
        """
        Variant without helper function
        """

        y_pred = []
        for x in X:
            distances = []
            for i, x_train in enumerate(self.X):
                if self.dist_func == 'euclidean':
                    distance = self.euclidean_dist(x, x_train)
                elif self.dist_func == 'manhattan':
                    distance = self.manhattan_dist(x, x_train)
                else:
                    raise ValueError("Invalid distance function")

                distances.append((distance, self.y[i]))

            distances.sort(key=lambda x: x[0])

            if self.weighting == 'uniform':
                weights = self.uniform_dist(distances[:self.k])
            elif self.weighting == 'weighted':
                weights = self.weighted_dist(distances[:self.k])
            else:
                weights = np.ones(self.k)

            weighted_sum = 0
            weight_sum = 0
            for i in range(self.k):
                weighted_sum += weights[i] * distances[i][1]
                weight_sum += weights[i]

            y_pred.append(weighted_sum / weight_sum)

        return y_pred



if __name__ == '__main__':
    print('*'*100)
    print('KNN Regressor module.')
    # simple dataset
    X = np.array([[1, 2, 3, 4],
                [2, 3, 4, 5],
                [3, 4, 5, 6],
                [4, 5, 6, 7],
                [5, 6, 7, 8]])
    
    y = np.array([10, 20, 30, 40, 50])

    # init and fit model
    knn  = KNNRegression(k_neighbors=2, dist_func='manhattan')
    knn.fit(X, y)


    preds = knn.predict(X)
    print(f"{'Predictions on the training data:':<40} {preds}")
    print('*'*100)