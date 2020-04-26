import sys, os
sys.path.insert(1, os.path.join(os.getcwd(), "src"))
from os import listdir
from os.path import isfile, join
from simulator import Simulator
import torch
import warnings
warnings.filterwarnings('ignore')


CONFIG_FILES_DIR = join(os.getcwd(), "config_files")
N_EXPERIMENTS = 10

def run_simulator(config_location, folder=True, 
                  device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")):
    if folder:
        config_files = [join(config_location,f) for f in listdir(config_location) if isfile(join(config_location, f))]
        print("====", "Executing simulations in folder:", config_location, sep="\t")
        print("====", "Configuration files found:\t", len(config_files), sep="\t")
        print()
        simulator = Simulator(config_files, N_EXPERIMENTS, device)
        simulator.start()
    else:
        config_files = [join(CONFIG_FILES_DIR, config_location)]
        simulator = Simulator(config_files, N_EXPERIMENTS, device)
        simulator.start()

if __name__ == "__main__":
    if len(sys.argv) > 2:
        run_simulator(sys.argv[1], False, sys.argv[2])
    elif len(sys.argv) > 1:
        run_simulator(sys.argv[1], False)
    else:
        run_simulator(CONFIG_FILES_DIR)
