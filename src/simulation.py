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
#         self.config["device"]=device
        if os.path.exists(self.log_file):
            os.system(f"rm {self.log_file}")
        self.config["n_clients"] -= self.config["n_clients"] % self.config["n_clusters"]
        self.config["client_lr"] *= self.config["client_batch_size"] / 32
        if "lr_warmup" not in self.config.keys():
            self.config["lr_warmup"] = False

        if "epochs_delay_localSGD" not in self.config.keys():
            self.config["epochs_delay_localSGD"] = 0
            self.config["rounds_delay_localSGD"] = 0
        elif self.config["epochs_delay_localSGD"] >= 0:
            self.config["rounds_delay_localSGD"] = round(self.config["epochs_delay_localSGD"] / self.config["client_selection_fraction"])

        if self.config["model_type"] == "mobileNet":
            self.config["save_memory"] = True
        else:
            self.config["save_memory"] = False

        if "move_to_neighbours" not in self.config.keys():
            self.config["move_to_neighbours"] = False

        self.network = Network(self.config)
        self.round_count = None
        self.train_accuracy = []
        self.test_accuracy = []
        self.latency = []

    def start(self):
        self.round_count = 0
        if self.config['stop_condition'] == 'rounds':
            for _ in range(self.config['stop_value']):
                self.round_count += 1
                if self.config["stdout_verbosity"] >= 1:
                    print(f"----\tRound {self.round_count} of {self.config['stop_value']}\t----")
                round_latency = self.network.learn()
                self.latency.append(round_latency)
                print(f"%%%% ROUND LATENCY:\t{round(self.latency[-1],4)}")
                self.network.move_clients()
                if self.round_count % self.config["server_global_rate"] == 0:
                    self.train_accuracy.append(self.network.evaluate_train())
                    print(f"%%%% TRAIN ACCURACY:\t{round(self.train_accuracy[-1],4)}")
                    self.test_accuracy.append(self.network.evaluate())
                    print(f"%%%% TEST ACCURACY:\t{self.test_accuracy[-1]}")
                    print(f"%%%% WEIGHT DIVERGENCE:\t{round(float(self.network.weight_divergence),4)}")
                self.log()
        else:
            raise NotImplementedError
        print()
        print(f"Train_acc:\t{self.train_accuracy}")
        print(f"Test_acc: \t{self.test_accuracy}")
        print(f"Latency:  \t{self.latency}")


    def log(self):
        update = {}
        ##
        update["round"] = self.round_count
        ##
        if self.train_accuracy:
            update["train_accuracy"] = self.train_accuracy[-1]
        else:
            update["train_accuracy"] = None
        ##
        if self.test_accuracy:
            update["test_accuracy"] = self.test_accuracy[-1]
        else:
            update["test_accuracy"] = None
        ##
        if self.latency:
            update["latency"] = self.latency[-1]
        else:
            update["latency"] = None
        ##
        update["weight_divergence"] = self.network.weight_divergence
        # network_log = self.network.log()
        # if network_log:
        #     update["network"] = network_log

        # print(update)
        # exit()
        #######################################
        update_log(self.log_file, update)