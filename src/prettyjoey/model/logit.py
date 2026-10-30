# Logistics Regression Class
import numpy as np
import torch


class LogisticRegression:
    # start with learning rate 0.1
    # will check with loss curve
    # 50 epochs could be enough for loss curve to flatten
    # will also check this later
    # batch side 64 for SGD, many updates per epoch
    # if batch size was = 1 like the lecture 
    def __init__(self, learning_rate=0.1, epochs=50, batch_size=64, seed=0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.loss_past = [] # full-training-set loss after each epoch

    def sigmoid(self, z):
        # from in class
        return 1 / (1 + torch.exp(-z))

    def forward(self, x):
        """h(x) = sigmoid(w0 + x @ w): the predicted probability of class 1"""
        return self.sigmoid(self.w0 + x @ self.w)
    
    # negative log likelyhood model
    # gives simple gradient when combined with sigmoid funct.
    # CLAMP forces log to always be between [0,1]
    def cross_entropy(self, h, y):
        """average cross-entropy; the clamp keeps log() away from 0"""
        h = torch.clamp(h, 1e-12, 1 - 1e-12)
        return -torch.mean(y * torch.log(h) + (1 - y) * torch.log(1 - h))
    
    # autograd requires tensor inputs
    def fit(self, x_train, y_train):
        x = torch.tensor(np.array(x_train), dtype=torch.float64)
        y = torch.tensor(np.array(y_train), dtype=torch.float64).flatten()
        n = x.shape[0]

        # weights and intercept start at zero (because of convex loss)
        self.w = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
        self.w0 = torch.zeros(1, dtype=torch.float64, requires_grad=True)

        # sgd implementation       
        self.optimization = torch.optim.SGD([self.w0, self.w], lr=self.learning_rate)

        gen = torch.Generator().manual_seed(self.seed)
        self.loss_past = []

        for epoch in range(self.epochs):
            order = torch.randperm(n, generator=gen) # shuffle every epoch
            for start in range(0, n, self.batch_size): # one SGD step per mini-batch
                idx = order[start:start + self.batch_size]
                self.optimization.zero_grad()
                loss = self.cross_entropy(self.forward(x[idx]), y[idx])
                loss.backward()
                self.optimization.step()

            with torch.no_grad():
                # record loss on all training data
                self.loss_past.append(self.cross_entropy(self.forward(x), y).item())

        print("w0 (intercept):", round(self.w0.item(), 6))
        for j, wj in enumerate(self.w):
            print(f"w{j+1}:", round(wj.item(), 6))
        print("final training loss:", round(self.loss_past[-1], 6))

    def predict_proba(self, x_new):
        x_new = torch.tensor(np.array(x_new), dtype=torch.float64)
        with torch.no_grad():
            return self.forward(x_new).numpy()

    def predict(self, x_new, threshold=0.5):
        return (self.predict_proba(x_new) >= threshold).astype(int)