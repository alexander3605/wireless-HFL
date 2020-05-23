from nn_classes import get_net
import torch.nn as nn
import torch
from my_library import count_parameters
import numpy as np
import copy
from tqdm import tqdm

class Client():
    def __init__(self, id, config, train_data, initial_weights, multiGPU=False):
        self.id = id
        self.config = config
        self.train_data = train_data
        model = get_net(config)
        if torch.cuda.device_count() > 1 and multiGPU:
            model = nn.DataParallel(model)
        self.model = model.to(config["device"])
        if initial_weights:
            self.set_weights(initial_weights)
        if self.config["client_algorithm"] == "scaffold":
            self.control_variate = [torch.zeros(p.shape, device=self.config["device"]) for p in self.model.parameters() if p.requires_grad]
            self.c_variate_update = [torch.zeros(p.shape, device=self.config["device"]) for p in self.model.parameters() if p.requires_grad]
        if config["debug"]:
            print("--------Init client", self.id)

    def get_weights(self):
        return self.model.parameters()

    def set_weights(self, weights): 
        for param, param_in in zip(self.model.parameters(), weights):
            param.data[:] = param_in.data[:] + 0


    def learn(self, server_control_variate=None):
        if self.config["debug"]:
            print(f"--- Client {self.id} learning ...")

        if self.config["client_algorithm"] == "sgd":
            self.learn_sgd()
        elif self.config["client_algorithm"] == "scaffold":
            self.learn_scaffold(server_control_variate)
        else:
            raise NotImplementedError


    def learn_sgd(self):
        optimizer = torch.optim.SGD(self.model.parameters(), 
                                    lr=self.config["client_lr"], 
                                    momentum=0.9)

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


    def learn_scaffold(self, server_control_variate):
        GRAD_CLIP_VALUE = 1e6
        # GRAD_CLIP_VALUE = 1e2
        # print(self.control_variate.shape)
        # print(server_control_variate)
        # print(len(self.train_data))
        optimizer = torch.optim.SGD(self.model.parameters(), 
                                    lr=self.config["client_lr"])

        criterion = nn.CrossEntropyLoss()
        device = self.config["device"]

        for epoch in range(self.config["client_n_epochs"]):
            see=0
            for data in self.train_data:
                see += 1
                # print("MINIBATCH", see)
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                predicts = self.model(inputs)
                loss = criterion(predicts, labels)
                # print(loss)
                assert torch.isnan(loss.view(-1)).sum().item()==0
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), GRAD_CLIP_VALUE)
                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                for i,p in enumerate(trainable_params):
                    assert torch.isnan(p.grad.view(-1)).sum().item()==0
                    correction = server_control_variate[i] - self.control_variate[i]
                    assert torch.isnan(correction.view(-1)).sum().item()==0                        
                    p.grad += correction
                    assert torch.isnan(p.grad.view(-1)).sum().item()==0
                optimizer.step()

    
    def update_control_variate(self, server_model, server_control_variate):
        # Update client control variate
        # self.c_variate_update = []
        server_trainable_params =  [p for p in server_model.parameters() if p.requires_grad]
        client_trainable_params =  [p for p in self.model.parameters() if p.requires_grad]
        
        for i,c in enumerate(self.control_variate):
            weights_diff = server_trainable_params[i] - client_trainable_params[i]
            update = - server_control_variate[i] + weights_diff / ((len(self.train_data)*self.config["client_n_epochs"] * self.config["client_lr"])) 
            c += update
            self.c_variate_update[i] = update 
        # for layer_idx in range(len(self.control_variate)):
        #     update = server_control_variate[layer_idx]
        #     update.data -= cumulative_grads[layer_idx].data / (len(self.train_data) * self.config["client_lr"])
        #     self.c_variate_update.append(update)

        # for layer_idx, c_variate in enumerate(self.control_variate):
        #     c_variate.data -= self.c_variate_update[layer_idx].data
   