from client import Client
from server import Server
from numpy.random import choice
from math import ceil

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


    def learn(self):
        if self.config["debug"]:
            print(f"- Cluster {self.id} learning ...")
        self.round_count += 1

        # Select clients
        n_clients = len(self.clients)
        selected_clients_inds = choice(n_clients, ceil(n_clients*self.config["client_selection_fraction"]))
        # For each selcted client
        for i in selected_clients_inds:
            # Download sbs model
            self.sbs.download_model(self.clients[i])
            # Update model
            self.clients[i].learn()
        # Average updates
        if self.config["debug"]:
            print(f"-- Server {self.sbs.id} learning ...")
        self.sbs.set_average_model(self.clients, selected_clients_inds)
        self.n_update_participants += len(selected_clients_inds)
        



    def get_weights(self):
        return self.sbs.get_weights()

    def set_weights(self, weights):
        self.sbs.set_weights(weights)