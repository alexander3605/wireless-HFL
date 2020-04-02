from nn_classes import get_net
from my_library import zero_param

class Server():

    def __init__(self, id, config, initial_weights=None):
        self.id = id
        self.config = config
        self.model = get_net(config).to(config["device"])
        if initial_weights:
            self.set_weights(initial_weights)


    def get_weights(self):
        return self.model.parameters()

    def set_weights(self, weights):
        for param, param_in in zip(self.model.parameters(), weights):
            param.data[:] = param_in.data[:] + 0

    # Averages the user models in the server
    def set_average_model(self, clients, selected_idx=None):
        zero_param(self.model)
        # if indeces are not provided, use all client models
        if selected_idx is None:
            selected_idx = range(len(clients))
        num_client = len(selected_idx)
        for ind in selected_idx:
            for param_user, param_server in zip(clients[ind].get_weights(), self.model.parameters()):
                param_server.data += param_user.data[:] / num_client + 0

    # Copies server weights to recipients models 
    # (recipient can be a single object or a list)
    def download_model(self, recipients):
        if type(recipients) == list:
            for i in range(len(recipients)):
                    recipients[i].set_weights(self.get_weights())
        else:
            recipients.set_weights(self.get_weights())