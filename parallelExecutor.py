from numpy.random import randint
import time
import threading, _thread
from threading import Thread
from TimeEstimator import TimeEstimator
import time

def parallelRun(function, args_list, max_threads, wait_time=1, 
                fool_proof=False, wait_time_retry=0):
    
    
    start_time = int(time.time())
    timeEst = TimeEstimator(len(args_list))
    timeEst.start()
    t_list = []
    
    for idx,arguments in enumerate(args_list):
        while threading.active_count() == 1 + max_threads:
            time.sleep(wait_time)
            fprocessing = threading.active_count() - 1
            fdone = int(idx) - fprocessing
            fleft = len(args_list) - fdone
            print("Running tests - Left {} - Processing {} - Done {}         {}      ".format(fleft, fprocessing, fdone, timeEst.progress(fdone)), end="\r")
        # when a thread slot is free
        t=None
        if fool_proof:
            t = threading.Thread(target = foolProofRun, 
                        args = (function, arguments, wait_time_retry), 
                        daemon=True)

        else:
            t = threading.Thread(target = function, 
                                    args = arguments, 
                                    daemon=True)
        t.start()
        t_list.append(t)
        time.sleep(wait_time)
    print(t_list)
    ### WAIT FOR ALL THE SIMULATIONS TO FINISH
    while threading.activeCount() > 1:
        time.sleep(0.5)
        fprocessing = threading.active_count() - 1
        fdone = len(args_list) - fprocessing
        print(f"Waiting for {fprocessing} threads out of {len(args_list)} to terminate...  [{timeEst.progress(fdone)}]       ", end="\r")

    ### WAIT FOR ALL THE THREADS TO FINISH
    for t in t_list:
        t.join()
    

def foolProofRun(function, args, wait_time_retry):
    ret = -1
    ret = function(args)
    while(ret != 0):
        time.sleep(randint(wait_time_retry))
        ret = function(args)

