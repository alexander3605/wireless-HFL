from parallelExecutor import parallelRun
import os
import time
from os import listdir
from os.path import isfile, join
import random


def run_test(config_file):
    device = f"cuda:{random.randint(0,1)}"
    return os.system(f"python3 main.py --config={config_file} --device={device} >/dev/null 2>&1")
    
    
    
    
    

if __name__ == "__main__":
    
    config_location = join(os.getcwd(), "config_files")
    files = [f for f in listdir(config_location) if isfile(join(config_location, f))]
    print(files)
    
    parallelRun(function=run_test,
                args_list=files,
                max_threads=10,
                wait_time=5,
                fool_proof=True,
                wait_time_retry=10)