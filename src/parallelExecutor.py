from numpy.random import randint
import time
import threading, _thread
from threading import Thread

def parallelRun(function, args_list, max_threads, wait_time=1, 
                fool_proof=False, wait_time_retry=0):
    t_list = []
    for args in args_list:
        while threading.active_count() == 1 + max_threads:
            time.sleep(wait_time)
        # when a thread slot is free
        if fool_proof:
            t = threading.Thread(target = foolProofRun, 
                        args = (function, args, wait_time_retry), 
                        daemon=True)

        else:
            t = threading.Thread(target = function, 
                                    args = args, 
                                    daemon=True)
        t.start()
        t_list.append(t)
        time.sleep(wait_time)

    ### WAIT FOR ALL THE THREADS TO FINISH
    for t in t_list:
        t.join()
    

def foolProofRun(thread_args):
    function = thread_args[0]
    args = thread_args[1]
    wait_time_retry = thread_args[2]
    ret = -1
    ret = function(args)
    while(ret != 0):
        time.sleep(randint(wait_time_retry))
        ret = function(args)

