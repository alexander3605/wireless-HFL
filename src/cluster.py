from client import Client
from server import Server
from numpy.random import choice
from math import ceil
from latency import rand_comp_time, downlink_latency, uplink_latency
from my_library import count_parameters
from nn_classes import get_net

class Cluster():

    def __init__(self, id, config, n_clients, training_sets, clients_id, initial_weights):
        self.config = config
        self.id = id
        self.sbs = Server(id, config, initial_weights)
        self.round_count = 0
        if config["debug"]:
            print("--- CLUSTER", id)
        if n_clients <= 0:
            raise ValueError("Invalid population value provided.")
        self.clients = []
        for i in range(n_clients):
            self.clients.append(Client(clients_id[i], config, training_sets[i], initial_weights))
        self.n_update_participants = 0
        self.weight_divergence = None


    def learn(self):
        l_rate = self.config["client_lr"]
        if self.config["lr_warmup"]:
            rounds_per_epoch = round(1/self.config["client_selection_fraction"])
            WARMUP_EPOCHS = 5
            rates = [0.1 + 0.9*i/WARMUP_EPOCHS for i in range(WARMUP_EPOCHS)]
            n_block = int((self.round_count-1)/rounds_per_epoch)
            if n_block < len(rates):
                rate =  rates[n_block]
                l_rate = l_rate * rate 
        if self.config["debug"]:
            print(f"- Cluster {self.id} learning ...")
        self.round_count += 1
        if self.config["client_algorithm"] == "scaffold":
            return self.learn_scaffold(l_rate)
        else:
            return self.learn_sgd(l_rate)


    def learn_sgd(self, l_rate):
        # Select clients
        n_clients = len(self.clients)
        selected_clients_inds = choice(n_clients, ceil(n_clients*self.config["client_selection_fraction"]), replace=False)
        # For each selcted client
        for i in selected_clients_inds:
            # Download sbs model
            self.sbs.download_model(self.clients[i])
            # Update model
            self.clients[i].learn(l_rate)
         # Average updates
        if self.config["debug"]:
            print(f"-- Server {self.sbs.id} learning ...")
        self.sbs.set_average_model(self.clients, selected_clients_inds)
        self.n_update_participants += len(selected_clients_inds)
        if len(selected_clients_inds) > 0: 
            round_latency = self.get_round_latency(selected_clients_inds)
            return round_latency
        else:
            return 0


        
    def learn_scaffold(self, l_rate):
        # Select clients
        n_clients = len(self.clients)
        selected_clients_inds = choice(n_clients, ceil(n_clients*self.config["client_selection_fraction"]), replace=False)
        # For each selcted client
        for i in selected_clients_inds:
            # Download sbs model
            self.sbs.download_model(self.clients[i])
            # Update model
            self.clients[i].learn(l_rate, self.sbs.control_variate)
            # Update client control variate
            raise NotImplementedError
            self.clients[i].update_control_variate(self.sbs.model, self.sbs.control_variate)
        # Update sbs control variate
        total_clients = self.config["n_clients"]
        for i in selected_clients_inds:
            for layer_idx, update in enumerate(self.clients[i].c_variate_update):
                self.sbs.control_variate[layer_idx] += update/total_clients
         # Average updates
        if self.config["debug"]:
            print(f"-- Server {self.sbs.id} learning ...")
        self.sbs.set_average_model(self.clients, selected_clients_inds)
        self.n_update_participants += len(selected_clients_inds)
        if len(selected_clients_inds) > 0: 
            round_latency = self.get_round_latency(selected_clients_inds)
            return round_latency
        else:
            return 0


    def get_round_latency(self, selected_clients):
        model_size = count_parameters(get_net(self.config))*16 # HALF-PRECISION floating point... 

        dl_latency = downlink_latency(self.config, len(self.clients), selected_clients, model_size)
        # print("downlink latency", round(dl_latency,1),sep="\t")
        comptime = rand_comp_time(self.config, len(selected_clients))
        ul_latency = uplink_latency(self.config, len(self.clients), selected_clients,comptime,model_size) 
        # print("uplink latency", round(ul_latency,1),sep="\t\t")

        
        return(dl_latency + ul_latency)


    def get_weights(self):
        return self.sbs.get_weights()

    def set_weights(self, weights):
        self.sbs.set_weights(weights)
