from network import Network
import torch
import matplotlib.pyplot as plt
from my_library import update_log
import os

class Simulation():

    
    def __init__(self, id, config, log_file):
        self.id = id
        self.config = config
        self.log_file = log_file


    def configure(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config["device"]=device
        if os.path.exists(self.log_file):
            os.system(f"rm {self.log_file}")
        self.network = Network(self.config)
        self.round_count = None
        self.test_accuracy = []

    def start(self):
        self.round_count = 0
        if self.config['stop_condition'] == 'rounds':
            for _ in range(self.config['stop_value']):
                self.round_count += 1
                if self.config["stdout_verbosity"] >= 1:
                    print(f"----\tRound {self.round_count} of {self.config['stop_value']}\t----")
                self.network.learn()
                self.network.move_clients()
                if self.round_count % self.config["server_global_rate"] == 0:
                    self.test_accuracy.append(self.network.evaluate())
                    print(f"%%%% TEST ACCURACY:\t{self.test_accuracy[-1]}")
                self.log()
        else:
            raise NotImplementedError
        print()
        print(self.test_accuracy)


    def log(self):
        update = {}
        update["round"] = self.round_count
        # update["network"] = self.network.log()
        if self.test_accuracy:
            update["test_accuracy"] = self.test_accuracy[-1]
        else:
            update["test_accuracy"] = None
        update_log(self.log_file, update)


