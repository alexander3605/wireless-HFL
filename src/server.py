from nn_classes import get_net
from my_library import zero_param, count_parameters
import numpy as np
import torch
from client import Client

class Server():

    def __init__(self, id, config, initial_weights=None):
        self.id = id
        self.config = config
        self.model = get_net(config).to(config["device"])
        if initial_weights:
            self.set_weights(initial_weights)
        if config["client_algorithm"] == "scaffold":
            self.control_variate = [torch.zeros(p.shape, device=self.config["device"]) for p in self.model.parameters() if p.requires_grad]


    def get_weights(self):
        return self.model.parameters()

    def set_weights(self, weights):
        for param, param_in in zip(self.model.parameters(), weights):
            param.data[:] = param_in.data[:] + 0

    # Averages the user models in the server
    def set_average_model(self, clients, selected_idx=None, clusters=False):
        try: 
            flattened_clients = clients.flatten()
        except:
            flattened_clients = clients
        # if indeces are not provided, use all client models
        if selected_idx is None:
            selected_idx = range(len(flattened_clients))
        # Get total number of clients
        num_client = len(selected_idx)
        zero_param(self.model)
        if clusters:
            participants_weights = np.array([cluster.n_update_participants for cluster in flattened_clients])
            participants_weights = participants_weights / np.sum(participants_weights)
            for ind in selected_idx:
                for param_user, param_server in zip(flattened_clients[ind].get_weights(), self.model.parameters()):
                    param_server.data += param_user.data[:] * participants_weights[ind]
        else:
            for ind in selected_idx:
                for param_user, param_server in zip(flattened_clients[ind].get_weights(), self.model.parameters()):
                    param_server.data += param_user.data[:] / num_client

    # Copies server weights to recipients models 
    # (recipient can be a single object or a list)
    def download_model(self, recipients):
        # LIST
        if type(recipients) == list:
            for i in range(len(recipients)):
                    recipients[i].set_weights(self.get_weights())
        elif type(recipients) == Client:
                    recipients.set_weights(self.get_weights())
        # NUMPY 2D ARRAY
        else:
            if len(recipients.shape) == 2:
                for i in range(np.prod(recipients.shape)):
                    idx_2D = np.unravel_index(i, recipients.shape)
                    recipients[idx_2D].set_weights(self.get_weights())