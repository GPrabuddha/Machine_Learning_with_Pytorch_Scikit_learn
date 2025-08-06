import numpy as np

class Perceptron:
    def __init__(self, n_iter=50, eta=0.01, seed=42):
        self.n_iter = n_iter
        self.eta = eta
        self.rgen = np.random.RandomState(seed)
        
    def fit(self, X, y):
            self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
            self.b_ = np.float_(0.)
            errors_ = [] # use self.errors otherwise it will be lost after training
            # for i in self.n_iter:  ## mistake: it is an int so can't iterate it like this
            for _ in range(self.n_iter):
                errors = 0
                for xi, yi in zip(X, y):
                    y_hat = self.predict(xi)
                    # if y - y_hat == 0:  ### not necessary as if y-y_hat is 0 then there will be no update anyway
                    #     continue
                    # else:
                    update = self.eta * (yi - y_hat)
                    self.w_ += update * xi
                    self.b_ += update
                    # errors += 1   # always incrementing errors but should only increment when there is an error
                    if yi != y_hat:
                        errors += 1
                errors_.append(errors)
             
            return self
                
                
    def predict(self, X):
            # if np.dot(self.w_, X) + self.b_ >= 0:  ### This only works for single sample as for a comparison
                                     ## of 0 with an array is not possible, use np.where
            #     return 1
            # else: 
            #     return 0
            return np.where(np.dot(X, self.w_) + self.b_>=0, 1, 0)