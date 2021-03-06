from simulation import Simulation
import json
import os

class Simulator():
    ##### Attributes
    simulations = []


    ##### METHODS
    def __init__(self, config_files, n_experiments, device):
        if not len(config_files):
            return ValueError("No config_files found in folder. Program is going to close.")
        ## Create simulations
        print(f"====\tConfiguring {len(config_files)} simulations ...\t====")
        log_files = [[] for _ in config_files]
        for i, config_file in enumerate(config_files):
            for k in range(n_experiments):
                with open(config_file, 'r') as cfg:
                    config = json.load(cfg)
                    config["device"] = device
                log_file = os.path.join(os.getcwd(), f"{config['log_file']}_{k}.json")
                sim = Simulation(id=i*n_experiments+k, config=config, log_file=log_file)
                self.simulations.append(sim)
                log_files[i].append(log_file)
        self.config_files = config_files
        self.log_files = log_files


    ## Start simulations (they will run until they end)
    ## TODO: run each simulation in a separate thread to parallelize
    def start(self):
        sim_count = 0
        for i,sim in enumerate(self.simulations):
            sim_count += 1
            print(f"====\tExecuting simulation {sim_count} of {len(self.simulations)} ...\t====")
            sim.configure()
            sim.start()
            print(f"====\tSimulation {sim_count} of {len(self.simulations)} finished.\t====")
            self.simulations[i] = None
        print("====\tAll simulations finished.\t====")