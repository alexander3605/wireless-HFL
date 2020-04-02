from cluster import Cluster
from server import Server
from my_library import get_split_dataset, evaluate_accuracy
from tqdm import tqdm

class Network():
    
    def __init__(self, config):
        self.config = config
        self.mbs = Server(id="MBS", config=config)

        self.round_count = 0
        # Import dataset and divide training data into partitions
        training_partitions, test_set = get_split_dataset(self.config)
        
        self.test_set = test_set

        # Create clusters
        if not self.config['n_clusters']:
            return ValueError("Wrong number of clusters found in configuration.")
        
        if self.config['clients_distribution'] == 'balanced':
            cluster_population = int(self.config['n_clients'] / self.config['n_clusters'])
            self.clusters = []
            for i in range(self.config['n_clusters']):
                training_sets = training_partitions[i*cluster_population : (i+1)*cluster_population]
                clients_id = list(range(i*cluster_population,(i+1)*cluster_population))
                self.clusters.append(Cluster(id=i, config=config, n_clients=cluster_population, 
                                             training_sets=training_sets, clients_id=clients_id, 
                                             initial_weights=self.mbs.get_weights()))
        
        else:
            raise NotImplementedError



    def learn(self):
        self.round_count += 1

        # LEARN IN EACH CLUSTER
        for i in tqdm(range(len(self.clusters))):
            self.clusters[i].learn()

        # MBS LEARNS FROM SBS
        if self.round_count % self.config["server_global_rate"] == 0:
            if self.config["debug"]:
                print(f"- MBS learning ...")
            self.mbs.set_average_model(self.clusters) # generate new global model
            self.mbs.download_model(self.clusters)    # download new global model to clusters



    def evaluate(self):
        if self.config["debug"]:
            print("- Evaluating global model ...")
        return evaluate_accuracy(self.mbs.model, self.test_set, self.config["device"])