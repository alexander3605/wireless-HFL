from simulation import Simulation
import json
class Simulator():
    ##### Attributes
    simulations = []


    ##### METHODS
    def __init__(self, config_files):
        if not len(config_files):
            return ValueError("No config_files found in folder. Program is going to close.")
        ## Create simulations
        print(f"====\tConfiguring {len(config_files)} simulations ...\t====")
        for config_file in config_files:
            with open(config_file, 'r') as cfg:
                config = json.load(cfg)
            sim = Simulation()
            sim.configure(config)
            self.simulations.append(sim)


    ## Start simulations (they will run until they end)
    ## TODO: run each simulation in a separate thread to parallelize
    def start(self):
        sim_count = 0
        for sim in self.simulations:
            sim_count += 1
            print(f"====\tExecuting simulation {sim_count} of {len(self.simulations)} ...\t====")
            sim.start()
            print(f"====\tSimulation {sim_count} of {len(self.simulations)} finished.\t====")
        
        print("====\tAll simulations finished.\t====")