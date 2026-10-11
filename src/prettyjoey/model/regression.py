import numpy as np
import torch
import matplotlib.pyplot as plt


class LinearRegression:
    """
    init takes arguments, learning rate, # epochs, initialize w0,w1, SGD optimizer
    and mean-squared erros as loss function, list to store w0,w1 and loss throughout training
    """
    def __init__(self, learning_rate=0.1, epochs=500):
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        # intialize w0,w1
        self.w0 = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        self.w1 = torch.zeros(1, dtype=torch.float64, requires_grad=True)

        # SGD & MSE 
        self.optimization = torch.optim.SGD([self.w0, self.w1], lr=self.learning_rate)
        self.loss = torch.nn.MSELoss()


        # per level historical data 
        self.r2 = None
        self.loss_past = []
        self.w0_past = []
        self.w1_past = []
        # 
        self.x_train = None
        self.y_train = None
        self.x_test = None 
        self.y_test = None 

    def forward(self,x):
        """
        specify linear model and compute response y
        """
        response_y = self.w0 + self.w1*x
        return response_y
    
    def fit(self, x_train, y_train, x_test, y_test):
        """
        ML on training dataset then pass in test data after
        compute R^2 on test data and return it
        """
        # this requires us to convert the inputs to float64 tensors
        # autograd and optimization only work for tensor inputs
        # flatten(), flattens the response to a 1-D tensor so that the linear model fits
        x_train = torch.tensor(np.array(x_train), dtype=torch.float64).flatten() 
        y_train = torch.tensor(np.array(y_train), dtype=torch.float64).flatten() 
        x_test = torch.tensor(np.array(x_test), dtype=torch.float64).flatten() 
        y_test = torch.tensor(np.array(y_test), dtype=torch.float64).flatten() 

        # store data
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

        # start from w0,w1 = 0 every time fit is called
        with torch.no_grad():
            self.w0.zero_()
            self.w1.zero_()
        self.loss_past = []
        self.w0_past = []
        self.w1_past = []

        for i in range(self.epochs):
            self.optimization.zero_grad()
            y_pred = self.forward(x_train)
            loss_value = self.loss(y_pred, y_train)

            # record w0, w1 and loss for this epoch
            self.w0_past.append(self.w0.item())
            self.w1_past.append(self.w1.item())
            self.loss_past.append(loss_value.item())

            loss_value.backward()
            self.optimization.step()

        # R^2 = 1 - ss_res / ss_tot on the test data
        # ss_res = sum(yi - yi_hat)^2
        # ss_tot = sum(yi-y_avg)^2
        with torch.no_grad():
            y_hat = self.forward(x_test)
            ss_res = torch.sum((y_test - y_hat) ** 2)
            ss_tot = torch.sum((y_test - torch.mean(y_test)) ** 2)
            self.r2 = (1 - ss_res / ss_tot).item()

        print("w0 (intercept):", round(self.w0.item(), 6))
        print("w1 (slope):", round(self.w1.item(), 6))
        print("R^2 (test):", round(self.r2, 6))
        
        return self.r2

    def predict(self, x_new):
        """
        predict AnnualProduction for new BCR data
        """
        x_new = torch.tensor(np.array(x_new), dtype=torch.float64).flatten()
        with torch.no_grad():
            return self.forward(x_new).numpy()

    def analysis_plot(self):
        """
        several plots on one figure: data, fitted line, loss, w0, w1
        """
        fig, axis = plt.subplots(2, 2, figsize=(10, 8))

        # original data
        axis[0, 0].scatter(self.x_train, self.y_train, label="train")
        axis[0, 0].scatter(self.x_test, self.y_test, label="test")
        axis[0, 0].set(title="original data", xlabel="BCR", ylabel="AnnualProduction")
        axis[0, 0].legend()

        # fitted regression line 
        x_line = torch.linspace(self.x_train.min().item(), self.x_train.max().item(), 100)
        y_line = self.forward(x_line).detach()
        axis[0, 1].scatter(self.x_train, self.y_train, label="train")
        axis[0, 1].scatter(self.x_test, self.y_test, label="test")
        axis[0, 1].plot(x_line, y_line, color="red", label="fitted line")
        axis[0, 1].set(title="fitted regression line", xlabel="BCR", ylabel="AnnualProduction")
        axis[0, 1].legend()

        # w0 and w1 during training
        axis[1, 0].plot(self.w0_past, label="w0")
        axis[1, 0].plot(self.w1_past, label="w1")
        axis[1, 0].set(title="w0 and w1 during training", xlabel="Epoch")
        axis[1, 0].legend()

        # loss during training
        axis[1, 1].plot(self.loss_past)
        axis[1, 1].set(title="loss during training", xlabel="Epoch", ylabel="MSE")

        plt.tight_layout()
        plt.show()