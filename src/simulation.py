from network import Network

class Simulation():


    def configure(self, config):
        self.config = config
        self.log_file = config['log_file']
        self.network = Network(self.config)
        self.round_count = None

    def start(self):
        self.round_count = 0
        if self.config['stop_condition'] == 'rounds':
            for _ in range(self.config['stop_value']):
                self.round_count += 1
                if self.config["stdout_verbosity"] > 0:
                    print(f"----\tRound {self.round_count} of {self.config['stop_value']}\t----")
                self.network = self.network.learn()
                self.log()
        else:
            return NotImplementedError


