from sklearn.model_selection import ParameterGrid
import os, sys
sys.path.insert(1, os.path.join(os.getcwd(), "src"))
from random import shuffle
from pprint import pprint
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
import torch
from my_library import write_log
#SET RANDOM SEED FOR REPRODUCIBILITY OF EXPERIMENTS
np.random.seed(2020)
torch.manual_seed(2020)

from parallelExecutor import parallelRun
import os
import time
from os import listdir
from os.path import isfile, join
import random
from pprint import pprint



######### CONSTANTS ###########

TEST_RESULTS_DIR = "log_files"
CONFIG_DIR = "config_files"
MAX_THREADS = 6
N_EXPERIMENTS = 9
# N_EXPERIMENTS = 1 # DEBUG

def run_test(config_file, n_exp=N_EXPERIMENTS):
    device = f"cuda:{random.randint(0,1)}"
    # # DEBUG
    # device = "cuda:0"
    return os.system(f"python3 main.py --config={config_file} --device={device} --n_experiments={n_exp} >/dev/null 2>&1")
    
    
# MAIN LOOP
if __name__ == "__main__":
    '''
    Define the parameters values that you want
    to test here.
    If a parameter list is left empty, the default
    value will be used for all experiments.
    '''
    
    n_clusters = [25]
    n_clients = [250]
    mobility_rate = [0.0, 0.1, 0.25, 0.5]
    model_type = ["mnist"]
    dataset_name = ["mnist"]
    dataset_distribution = ["iid", "non_iid"]
    client_algorithm = ["sgd"]
    client_batch_size = [32]
    client_n_epochs = [2,4]
    client_lr = [0.0100]
    server_global_rate = [1]
    client_selection_fraction = [0.3]
    lr_warmup = [False, True]
    epochs_delay_localSGD = [5,10]
#  "log_file": "log_files/test_log"
    log_verbosity = [2]
    log_frequency = [1]
    stdout_verbosity = [2]
    stdout_frequency = [1]
    debug = [False]

    # Fixed parameters
    clients_mobility = [True]
    clients_distribution = ["balanced"]
    model_init = ["random"]
    stop_condition = ["rounds"]
    stop_value = [250]
    # stop_value = [2] # DEBUG

    '''
    Define the fraction of all the possible combinations
    that you want to run.
    (Set to 1 to run all)
    '''
    fraction_to_run = 1.0

    '''
    Define name of the test results file.
    If set to "" or None, the *current epoch time* will be used.
    WARNING!! - 1) There is currently no sanitation of the name
                   string provided.
                2) If the file name is already in use, the old file
                   will be overwritten. (In that case leaving the
                   field blank is a safer option)
    '''
    
    results_name_root = "mnist_post-local-sgd"
    
    #################################################################
    #################################################################


    params = {}
    params["n_clusters"] = n_clusters
    params["n_clients"] = n_clients
    params["clients_mobility"] = clients_mobility
    params["mobility_rate"] = mobility_rate
    params["model_type"] = model_type
    params["dataset_name"] = dataset_name
    params["dataset_distribution"] = dataset_distribution
    params["client_algorithm"] = client_algorithm
    params["client_batch_size"] = client_batch_size
    params["client_n_epochs"] = client_n_epochs
    params["client_lr"] = client_lr
    params["server_global_rate"] = server_global_rate
    params["client_selection_fraction"] = client_selection_fraction
    params["lr_warmup"] = lr_warmup
    params["epochs_delay_localSGD"] = epochs_delay_localSGD
#  "log_file": "log_files/test_log"
    params["log_verbosity"] = log_verbosity
    params["log_frequency"] = log_frequency
    params["stdout_verbosity"] = stdout_verbosity
    params["stdout_frequency"] = stdout_frequency
    params["debug"] = debug
    params["clients_distribution"] = clients_distribution
    params["model_init"] = model_init
    params["stop_condition"] = stop_condition
    params["stop_value"] = stop_value



    combinations = list(ParameterGrid(params))
    if fraction_to_run < 1 and fraction_to_run > 0:
        print(f"Combinations found:\t{len(combinations)}")
        comb_to_try = int(len(combinations)*fraction_to_run)

        ## SHUFFLE COMBINATIONS AND CHOOSE SUBSET
        shuffle(combinations)
        combinations = combinations[:comb_to_try]
        print(f"Combinations selected:\t{len(combinations)}")
    elif fraction_to_run == 1.0:
        print(f"Combinations selected:\t{len(combinations)}")

    if not combinations:
        print("NO EXPERIMENTS TO RUN.\nExiting...")
        exit()


    # MAKE CONFIG FILES
    config_dir = os.path.join(os.getcwd(), CONFIG_DIR)
    config_files = []
    for i,p in enumerate(combinations):
        p["clients_mobility"] = bool(p["mobility_rate"])
        filename = f"{results_name_root}-{i}"
        p["log_file"] = os.path.join(TEST_RESULTS_DIR,filename)
        config_path = os.path.join(config_dir,f"{filename}.json")
        write_log(config_path, p)
        config_files.append(f"{filename}.json")
    pprint(config_files)    

    # RUN EXPERIMENTS
    parallelRun(function=run_test,
            args_list=config_files,
            max_threads=MAX_THREADS,
            wait_time=10,
            fool_proof=True,
            wait_time_retry=100)
