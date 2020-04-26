from sklearn.model_selection import ParameterGrid
import os, sys
from random import shuffle
from pprint import pprint
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
from pickle_functions import load_data

import threading, _thread
import queue
from threading import Thread
from TimeEstimator import TimeEstimator
MAX_THREADS = 15     # Define maximum number of testing threads that can run concurrently 


TEST_RESULTS_DIR = "test_results"




def make_cmd(command, gpu_id, verbose):
    cmd = command  
    cmd += f" --gpu_id={gpu_id}"
    if verbose:
        cmd += " 2>&1"
    else:
        cmd += " >/dev/null 2>&1"
    return cmd


def run_sim(n, command, verbose):
#         print(f"starting simulation {n}...")
    ret = -1
    gpu_id = 0
    ret = os.system(make_cmd(command,gpu_id,verbose))
    while(ret != 0):
#             print(f"retrying to start simulation {n}...")
        time.sleep(np.random.randint(10))
        gpu_id = (gpu_id+1)%2
        ret = os.system(make_cmd(command,gpu_id,verbose))
#     print(f"simulation {n} DONE")



def run_experiments(dataset_name, nn_name, dataset_dist,
                    num_users, bs, local_epochs,
                    comm_rounds, lr, acc_freq,
                    sched_frac_uplink, sched_frac_downlink, power_bs,
                    power_usr, noise_psd, bandwidth,
                    radius, path_loss, 
                    fraction_to_run, test_name, verbose):


    params = {}
    if dataset_name:
        params["dataset_name"] = dataset_name
    if nn_name:
        params["nn_name"] = nn_name
    if dataset_dist:
        params["dataset_dist"] = dataset_dist
    if num_users:
        params["num_users"] = num_users
    if bs:
        params["bs"] = bs
    if local_epochs:
        params["local_epochs"] = local_epochs
    if comm_rounds:
        params["comm_rounds"] = comm_rounds
    if lr:
        params["lr"] = lr
    if acc_freq:
        params["acc_freq"] = acc_freq
    if sched_frac_uplink:
        params["sched_frac_uplink"] = sched_frac_uplink
    if sched_frac_downlink:
        params["sched_frac_downlink"] = sched_frac_downlink
    if power_bs:
        params["power_bs"] = power_bs
    if power_usr:
        params["power_usr"] = power_usr
    if noise_psd:
        params["noise_psd"] = noise_psd
    if bandwidth:
        params["bandwidth"] = bandwidth
    if radius:
        params["radius"] = radius
    if path_loss:
        params["path_loss"] = path_loss
    
    if params == {}:
        print("WARNING --- No parameters supplied.")
        print("            Running script only with default values.")

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

    pprint(combinations)

    # PREPARE OUTPUT DIRECTORY
    results_dir = os.path.join(os.getcwd(), TEST_RESULTS_DIR)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # PREPARE OUTPUT FILE
    if test_name:
        results_file = os.path.join(results_dir, f"{test_name}.csv")

    else:
        results_file = os.path.join(results_dir, f"{int(time.time())}.csv")
    print(f"Output file -> {results_file}")

    

        
     
    # RUN EXPERIMENTS
    start_time = int(time.time())
    timeEst = TimeEstimator(len(combinations))
    timeEst.start()
    t_list = []
    
    for idx,p in enumerate(combinations):
        while threading.active_count() == 1 + MAX_THREADS:
            time.sleep(1)
            fprocessing = threading.active_count() - 1
            fdone = int(idx) - fprocessing
            fleft = len(combinations) - fdone
            print("Running tests - Left {} - Processing {} - Done {}         {}      ".format(fleft, fprocessing, fdone, timeEst.progress(fdone)), end="\r")
        # when a thread slot is free
        command = "python3 main.py"
        for k in list(p.keys()):
            command += f" --{k}={p[k]}" 
        command += f" --results_id={idx+1}"
        thread1 = threading.Thread(target = run_sim, 
                                   args = (idx,command, verbose), 
                                   daemon=True)
        thread1.start()
        t_list.append(thread1)
        time.sleep(np.random.randint(5))

    
    ### WAIT FOR ALL THE SIMULATIONS TO FINISH
    while threading.activeCount() > 1:
        time.sleep(0.5)
        fprocessing = threading.active_count() - 1
        fdone = len(combinations) - fprocessing
        print(f"Waiting for {fprocessing} threads out of {len(combinations)} to terminate...  [{timeEst.progress(fdone)}]       ", end="\r")

    # wait for all frames to be generated
    for t in t_list:
        t.join()
    

    print("All simulations finished.")
    results = []
    for idx in range(len(combinations)):
        avg_delays = load_data(f"avg_delays_{idx+1}")
        avg_accs  = np.load(f"avg_accs_{idx+1}.npy")
        # SAVE RESULTS
        res = {"avg_delay": avg_delays['avg_delay'],
               "avg_delay_dl": avg_delays['avg_delay_dl'],
               "avg_delay_ul": avg_delays['avg_delay_ul'],
               "avg_accs": avg_accs}
        res.update(combinations[idx]) # add parameters tested to results
        results.append(res)
    results_df = pd.DataFrame(results)    
    results_df.to_csv(results_file)

    print(results_df)
    # Delete temp files
    for idx in range(len(combinations)):
        os.system(f"rm avg_delays_{idx+1}")
        os.system(f"rm avg_accs_{idx+1}.npy")
