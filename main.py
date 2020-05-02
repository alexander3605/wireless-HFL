import sys, os
sys.path.insert(1, os.path.join(os.getcwd(), "src"))
from os import listdir
from os.path import isfile, join
from simulator import Simulator
import torch
import warnings
from parameters import args_parser
warnings.filterwarnings('ignore')


CONFIG_FILES_DIR = join(os.getcwd(), "config_files")

def run_simulator(config_location, is_folder, device,n_experiments):
    if is_folder:
        config_files = [join(config_location,f) for f in listdir(config_location) if isfile(join(config_location, f))]
        print("====", "Executing simulations in folder:", config_location, sep="\t")
        print("====", "Configuration files found:\t", len(config_files), sep="\t")
        print()
        simulator = Simulator(config_files, n_experiments, device)
        simulator.start()
    else:
        config_files = [join(CONFIG_FILES_DIR, config_location)]
        simulator = Simulator(config_files, n_experiments, device)
        simulator.start()

if __name__ == "__main__":
    args = args_parser()
    if args.config:
        run_simulator(config_location=args.config, is_folder=args.is_folder, 
                        device=args.device, n_experiments=args.n_experiments)
    else:
        run_simulator(config_location=CONFIG_FILES_DIR, is_folder=True, 
                      device=args.device, n_experiments=args.n_experiments)
