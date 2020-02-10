from cluster import Cluster
from server import Server
from torchvision.datasets import MNIST
class Network():
    
    def __init__(self, config):
        self.config = config
        self.mbs = Server(self.config)


        # Import dataset and divide into partitions
        if self.config['dataset_name']=='MNIST' and self.config['dataset_distribution']=='IID':
            pass
        else:
            raise NotImplementedError

        # Create clusters
        if not self.config['n_clusters']:
            return ValueError("Wrong number of clusters found in configuration.")
        
        if self.config['clients_distribution'] == 'balanced':
            cluster_population = int(self.config['n_clients'] / self.config['n_clusters'])
            self.clusters = []
            for _ in range(self.config['n_clusters']):
                self.clusters.append(Cluster(config, n_clients=cluster_population))
        
        else:
            raise NotImplementedError



