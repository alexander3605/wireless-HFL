from network import Network
import torch
import matplotlib.pyplot as plt
class Simulation():


    def configure(self, config):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config["device"]=device
        self.config = config
        self.log_file = config['log_file']
        self.network = Network(self.config)
        self.round_count = None
        self.test_accuracy = []

    def start(self):
        self.round_count = 0
        if self.config['stop_condition'] == 'rounds':
            for _ in range(self.config['stop_value']):
                self.round_count += 1
                if self.config["stdout_verbosity"] > 0:
                    print(f"----\tRound {self.round_count} of {self.config['stop_value']}\t----")
                self.network.learn()
                self.test_accuracy.append(self.network.evaluate())
                print(f"%%%% TEST ACCURACY: {self.test_accuracy[-1]}")
                # self.log()
        else:
            raise NotImplementedError
        
        plt.figure()
        plt.plot(self.test_accuracy)
        plt.show()



