import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import onnxruntime as ort
from sklearn.preprocessing import StandardScaler

class TorchNet:
    def __init__(self,input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scaler = StandardScaler()
        self.loss_vec = []
        # initialize empty sequential model and add layers sequentially
        self.model = nn.Sequential()
        self.model.add_module('input_layer', nn.Linear(input_dim, 4)) # first hidden layer: 4 neurons
        self.model.add_module('act1', nn.ReLU())
        self.model.add_module('hidden_layer1', nn.Linear(4, 3)) # second hidden layer: 3 neurons
        self.model.add_module('act2', nn.ReLU())
        self.model.add_module('output_layer', nn.Linear(3, output_dim)) # 2 classes

    # 1
    # standardize
    def standardize(self, X):
        X = np.asarray(X, dtype=float)
        self.scaler.fit(X)
        return self.scaler.transform(X)

    # 3
    # train with adam, lr = 0.01, 1000 epochs
    def train(self, X, y):
        X = torch.FloatTensor(self.standardize(X))
        y = torch.LongTensor(np.asarray(y))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        for epoch in range(1000):
            outputs = self.model(X)
            loss = criterion(outputs, y)
            self.loss_vec.append(loss.detach().item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 4
    # loss plot
    def plot_loss(self):
        plt.plot(self.loss_vec)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    # 5
    # save as onnx, took this from the code examples
    def save_onnx(self, path):
        self.model.eval()
        dummy_input = torch.zeros(1, self.input_dim)
        batch = torch.export.Dim("batch_size", min=1)
        torch.onnx.export(
            self.model,
            (dummy_input,),
            path,
            export_params=True,
            opset_version=18,
            input_names=["input"],
            output_names=["output"],
            dynamic_shapes=({0: batch},),
            dynamo=True,
        )

    # 6
    # load the onnx model and predict
    def load_onnx(self, path):
        self.session = ort.InferenceSession(path)

    def predict(self, X):
        X = self.scaler.transform(np.asarray(X, dtype=float).reshape(1, -1))
        logits = self.session.run(None, {'input': X.astype(np.float32)})[0]
        return int(np.argmax(logits, axis=1)[0])