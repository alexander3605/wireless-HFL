from cluster import Cluster
from server import Server
from my_library import get_split_dataset, evaluate_accuracy, get_clusters_grid_shape, get_weight_divergence
from tqdm import tqdm
import numpy as np
from pprint import pprint
import copy


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

        self.weight_divergence = None


            
            
    def learn(self):
        self.round_count += 1
        round_latency = 0
        # LEARN IN EACH CLUSTER
        for i in range(self.config["n_clusters"]):
            cluster_index = np.unravel_index(i, self.clusters_grid_shape) # 1D index --> 2D index
            cluster_latency = self.clusters[cluster_index].learn()
            # TODO: use cluster_latency to compute network latency across rounds
            # NOTE: synchronization of clusters is only needed when there is a global update! 
            #       But what about clients that have moved but are still in use in their previous cluster? 
            round_latency = max(round_latency, cluster_latency)
        # MBS LEARNS FROM SBS
        if self.round_count % self.config["server_global_rate"] == 0:
            if self.config["debug"]:
                print(f"- MBS learning ...")
            self.mbs.set_average_model(self.clusters, clusters=True) # generate new global model

            # COMPUTE WEIGHT DIVERGENCE
            divergence_list = []
            for cluster in self.clusters.flatten():
                cluster.weight_divergence = get_weight_divergence(cluster.sbs.model, self.mbs.model) 
                divergence_list.append(cluster.weight_divergence)
            self.weight_divergence = float(np.mean(divergence_list))

            # DOWNLOAD NEW GLOBAL MODEL TO CLUSTERS 
            self.mbs.download_model(self.clusters)    # download new global model to clusters

            # IF SCAFFOLD, GENERATE AND DISTRIBUTE GLOBAL CONTROL VARIATE
            if self.config["client_algorithm"] == "scaffold":
                for i in range(len(self.mbs.control_variate)):
                    self.mbs.control_variate[i] = sum([c.sbs.control_variate[i] for c in self.clusters.flatten()])/self.config["n_clusters"]
                    for c in self.clusters.flatten():
                        c.sbs.control_variate[i].data = self.mbs.control_variate[i].data
                        # print(hex(id(c.sbs.control_variate[i])))
                        # print(hex(id(self.mbs.control_variate[i])))


            # Get clusters internal variables ready for next set of rounds
            for i in range(self.config["n_clusters"]):
                cluster_index = np.unravel_index(i, self.clusters_grid_shape) # 1D index --> 2D index
                self.clusters[cluster_index].n_update_participants = 0


            
        return round_latency


    def evaluate(self):
        if self.config["debug"]:
            print("- Evaluating global model on TEST set ...")
        return evaluate_accuracy(self.mbs.model, self.test_set, self.config["device"])
    
    def evaluate_train(self):
        if self.config["debug"]:
            print("- Evaluating global model on TRAIN set ...")
        return evaluate_accuracy(self.mbs.model, self.train_set, self.config["device"])


    def move_clients(self):
        if self.config["clients_mobility"] and self.config["n_clusters"]>1:
            
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
            


    def log(self):
        log_dict = {}
        
        # # Measure accuracy of sbs models
        # clusters_test_acc = []
        # for cluster in self.clusters.flatten():
        #     test_acc = evaluate_accuracy(cluster.sbs.model, self.test_set, self.config["device"])
        #     clusters_test_acc.append(test_acc)
        # log_dict["clusters_test_accuracy"] = clusters_test_acc
        
        
        log_dict["weight_divergence"] = self.weight_divergence
        return log_dict

            
            
