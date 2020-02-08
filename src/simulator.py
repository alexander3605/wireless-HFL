class simulator():
    ##### Attributes
    simulations = None


    ##### METHODS
    def __init__(config_files, log_files):
        if len(config_files) != len(log_files):
            return ValueError("Different number of config_files and log_files provided.")
        for i in range(len(config_files)):
            simulations.append(start(config_files[i], log_files[i]))

    def start(config_file, log_file):
        return NotImplementedError