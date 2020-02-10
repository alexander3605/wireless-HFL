from client import Client

class Cluster():

    def __init__(self, config, population):
        self.config = config
        self.sbs = Server(config)
        
        if population <= 0:
            return ValueError("Invalid population value provided.")
        self.clients = []
        for _ in range(population):
            self.clients.append(Client())