from parallelExecutor import parallelRun
import os
import time
from os import listdir
from os.path import isfile, join
import random
from pprint import pprint

N_EXPERIMENTS = 3

def run_test(config_file, n_exp=N_EXPERIMENTS):
    device = f"cuda:{random.randint(0,1)}"
    return os.system(f"python3 main.py --config={config_file} --device={device} --n_experiments={n_exp} >/dev/null 2>&1")
    
    
    
    
    

if __name__ == "__main__":
    
    config_location = join(os.getcwd(), "config_files")
    files = [f for f in listdir(config_location) if isfile(join(config_location, f))]
    # files.remove("test_CIFAR.json")
    files = [f for f in files if "CIFAR" in f]
    # pprint(len(files))
    # exit()
    parallelRun(function=run_test,
                args_list=files,
                max_threads=6,
                wait_time=5,
                fool_proof=True,
                wait_time_retry=100)
