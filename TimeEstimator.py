import time

class TimeEstimator:
    def __init__(self, max_iterations):
        self.max_iterations = max_iterations
        self.start_time = None

    def start(self):
        self.start_time = int(time.time())

    def progress(self, curr_iteration):
        t_now = int(time.time())
        t_passed = t_now - self.start_time
        done = curr_iteration/self.max_iterations
        t_left = 0
        if done:
            t_left = int(t_passed/done * (1-done))

        (h,m,s) = breakTime(t_left)
        progress_str = str(round(done*100,2)) + '%' +' done - Time left '
        progress_str += "{:02d}:{:02d}:{:02d}".format(h,m,s)
        return progress_str



def breakTime(t):
    seconds = t % 60
    minutes = int((t - seconds)/60)%60
    hours = int((t-minutes*60-seconds)/3600)
    return(hours, minutes, seconds)
    
