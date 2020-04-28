import torch

class LinearRegressor():
    def __init__(self, lr=0.025):
        self.lr = lr

    def fit(self, x, n_iters=100):
        """ 
        fits the LinearRegressor with input matrix x
        of the form (n_samples, n_features)
        """
        self.n_dim = x.shape[1]
        self.coefs = torch.randn(self.n_dim, requires_grad=True)

        # the last feature is our target and we create
        # a new matrix with the target replaced with a column
        # of ones
        ones_column = torch.ones((x.shape[0]))
        target = x[:, -1]
        X = torch.empty(x.shape[0], x.shape[1])
        X[:, :-1] = x[:, :-1]
        X[:, -1] = ones_column
        
        # Keeping track of the errors during the training process
        self.errors = []

        # fitting process
        for i in range(n_iters):
            y = X.matmul(self.coefs)
            error = self.MSE(y, target)
            error.backward()
            self.coefs = (self.coefs.data - self.lr*self.coefs.grad).requires_grad_(True)
            self.errors.append(error.detach())

    def get_error(self):
        return self.errors

    def predict(self, x):
        X = torch.empty(x.shape[0], x.shape[1]+1)
        X[:, :-1] = x
        X[:, -1] = torch.ones(x.shape[0])
        return X.matmul(self.coefs.detach())

    def MSE(self, y, target):
        return ((y-target)**2).mean()
