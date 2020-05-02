from cluster import Cluster
from server import Server
from my_library import get_split_dataset, evaluate_accuracy, get_clusters_grid_shape
from tqdm import tqdm
import numpy as np
from pprint import pprint
import copy
import threading
import time
from multiprocessing import Process, Pool

class Network():
    
    def __init__(self, config):
        self.config = config
        self.mbs = Server(id="MBS", config=config)

        self.round_count = 0
        # Import dataset and divide training data into partitions
        train_set, training_partitions, test_set = get_split_dataset(self.config)
        self.train_set = train_set
        self.test_set = test_set

        # Create clusters
        if not self.config['n_clusters']:
            raise ValueError("Wrong number of clusters found in configuration.")
        
        self.clusters_grid_shape = get_clusters_grid_shape(self.config['n_clusters'])

        if self.config['clients_distribution'] == 'balanced':
            cluster_population = int(self.config['n_clients'] / self.config['n_clusters'])
            self.clusters = np.empty(shape=self.clusters_grid_shape, dtype=Cluster)
            for i in range(self.config['n_clusters']):
                cluster_index = np.unravel_index(i, self.clusters_grid_shape) # 1D index --> 2D index
                training_sets = training_partitions[i*cluster_population : (i+1)*cluster_population]
                clients_id = list(range(i*cluster_population,(i+1)*cluster_population))
                self.clusters[cluster_index] = (Cluster(id=i, config=config, n_clients=cluster_population,
                                                training_sets=training_sets, clients_id=clients_id, 
                                                initial_weights=self.mbs.get_weights()))
        
        else:
            raise NotImplementedError



    def learn(self):
        self.round_count += 1

        # LEARN IN EACH CLUSTER
        for i in range(self.config["n_clusters"]):
            cluster_index = np.unravel_index(i, self.clusters_grid_shape) # 1D index --> 2D index
            self.clusters[cluster_index].learn()

        # MBS LEARNS FROM SBS
        if self.round_count % self.config["server_global_rate"] == 0:
            if self.config["debug"]:
                print(f"- MBS learning ...")
            self.mbs.set_average_model(self.clusters, clusters=True) # generate new global model
            self.mbs.download_model(self.clusters)    # download new global model to clusters
        
            # if self.config["stdout_verbosity"] >= 1:
            #     print(f"%%%% TRAIN ACCURACY:\t{round(self.evaluate_train(),4)}")
            for i in range(self.config["n_clusters"]):
                cluster_index = np.unravel_index(i, self.clusters_grid_shape) # 1D index --> 2D index
                self.clusters[cluster_index].n_update_participants = 0


    def evaluate(self):
        if self.config["debug"]:
            print("- Evaluating global model on TEST set ...")
        return evaluate_accuracy(self.mbs.model, self.test_set, self.config["device"])
    
    def evaluate_train(self):
        if self.config["debug"]:
            print("- Evaluating global model on TRAIN set ...")
        return evaluate_accuracy(self.mbs.model, self.train_set, self.config["device"])


    def move_clients(self):
        if self.config["clients_mobility"]:
            
            # print("OLD DISTRIBUTION")
            # pprint(np.array([len(cluster.clients) for cluster in self.clusters.flatten()]).reshape(self.clusters.shape))
            

            moving_clients = [[] for _ in self.clusters.flatten()]
            destinations = [[] for _ in self.clusters.flatten()]

            proximities = np.zeros(shape= (np.prod(self.clusters.shape), np.prod(self.clusters.shape)))
            
            for i in range(proximities.shape[0]):
                for j in range(proximities.shape[1]):
                    if i != j:
                        proximities[i][j] = 1/np.linalg.norm(
                            np.subtract(
                                list(np.unravel_index(i, shape=self.clusters.shape)),
                                list(np.unravel_index(j, shape=self.clusters.shape))
                            ), ord=2)
                    # Normalize distances so they sum up to 1 for each row
                proximities[i] /= np.sum(proximities[i])

            for i, cluster in enumerate(self.clusters.flatten()):
                for client in cluster.clients:
                    if np.random.random() < client.config["mobility_rate"]:
                        moving_clients[i].append(client)
                        destinations[i].append(np.random.choice(list(range(self.config["n_clusters"])), 
                                                                p=proximities[i]))
            
            for source, clients in enumerate(moving_clients):
                source_2D = np.unravel_index(source, self.clusters.shape)
                for i, client in enumerate(clients):
                    destination = destinations[source][i]
                    destination_2D = np.unravel_index(destination, self.clusters.shape)
                    self.clusters[destination_2D].clients.append(client)
                    self.clusters[source_2D].clients.remove(client)

        if self.config["debug"]:
            print("--- NEW DISTRIBUTION")
            pprint(np.array([len(cluster.clients) for cluster in self.clusters.flatten()]).reshape(self.clusters.shape))          