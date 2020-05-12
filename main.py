import sys, os
sys.path.insert(1, os.path.join(os.getcwd(), "src"))
from os import listdir
from os.path import isfile, join
from simulator import Simulator
import torch
import warnings
from parameters import args_parser
from my_library import combine_results, save_combined_results
warnings.filterwarnings('ignore')


CONFIG_FILES_DIR = join(os.getcwd(), "config_files")

def run_simulator(config_location, is_folder, device,n_experiments):
    config_files = None
    simulator = None
    if is_folder:
        config_files = [join(config_location,f) for f in listdir(config_location) if isfile(join(config_location, f))]
        print("====", "Executing simulations in folder:", config_location, sep="\t")
        print("====", "Configuration files found:\t", len(config_files), sep="\t")
        print()
        simulator = Simulator(config_files, n_experiments, device)
    else:
        config_files = [join(CONFIG_FILES_DIR, config_location)]
        simulator = Simulator(config_files, n_experiments, device)
    
    simulator.start()
    print("Combining results files...")
    combined_res = combine_results(simulator.config_files, simulator.log_files)
    # save_combined_results(combined_res, simulator.log_files, delete_old_logs=True)
    save_combined_results(combined_res, simulator.log_files, delete_old_logs=False)
    print("DONE.")

if __name__ == "__main__":
    args = args_parser()
    if args.config:
        run_simulator(config_location=args.config, is_folder=args.is_folder, 
                        device=args.device, n_experiments=args.n_experiments)
    else:
        run_simulator(config_location=CONFIG_FILES_DIR, is_folder=True, 
                      device=args.device, n_experiments=args.n_experiments)
