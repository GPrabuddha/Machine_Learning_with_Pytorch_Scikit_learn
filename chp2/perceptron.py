import numpy as np

class Perceptron:
    def __init__(self, eta=0.01, n_iter=50, seed=1):
        self.eta = eta
        self.n_iter = n_iter
        self.seed = seed
        
        def fit(self, X, y):
            rgen = np.random.RandomState(self.seed) # Creates a random number generator object
            self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1]) # Size creates a weight for each feature
            self.b_ = np.float_(0.) # underscores are used to indicate that these variables are not initialized with object creation
            self.errors_ = []
            
            for _ in range(n_iter):
                errors = []
                for xi, target in zip(X, y):
                    update = self. eta * (target - self.predict(xi))
                    self.w_ += update * xi
                    self.b_ += update
                    errors += int(update!=0.0)
                self.errors_.append(errors)
            return self
        
        def net_input(self, X):
            return np.dot(X, self.w_) + self.b_
        
        def predict(self, X):
            return np.where(self.net_input(X) >= 0.0, 1, 0)