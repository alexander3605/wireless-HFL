from nn_classes import get_net
import torch.nn as nn
import torch
from tqdm import tqdm

class Client():
    def __init__(self, id, config, train_data, initial_weights):
        self.id = id
        self.config = config
        self.train_data = train_data
        self.model = get_net(config).to(config["device"])
        if initial_weights:
            self.set_weights(initial_weights)
        if config["debug"]:
            print("--------Init client", self.id)

    def get_weights(self):
        return self.model.parameters()

    def set_weights(self, weights): 
        for param, param_in in zip(self.model.parameters(), weights):
            param.data[:] = param_in.data[:] + 0


    def learn(self):
        if self.config["debug"]:
            print(f"--- Client {self.id} learning ...")

        if self.config["client_algorithm"] == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), 
                                        lr=self.config["client_lr"], 
                                        weight_decay=0.0001)
        else:
            raise NotImplementedError

        criterion = nn.CrossEntropyLoss()
        device = self.config["device"]
        for epoch in range(self.config["client_n_epochs"]):
            for data in self.train_data:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                predicts = self.model(inputs)
                loss = criterion(predicts, labels)
                loss.backward()
                optimizer.step()