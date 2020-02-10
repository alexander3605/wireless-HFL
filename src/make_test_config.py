from my_library import *
import os
CONFIG_FILES_DIR = os.path.join(os.getcwd(), "config_files")
LOG_FILES_DIR = os.path.join(os.getcwd(), "log_files")

###################################################################
# DEFINE CONFIG
config = {}
### NETWORK STRUCTURE
config['n_clusters'] = 1
config['n_clients'] = 1
config['clients_distribution'] = 'balanced'
### MODEL STRUCTURE
config['model_type'] = 'FCNN'
config['model_init'] = 'random' 
### DATA STRUCTURE
config['dataset_name'] = 'MNIST'
config['dataset_distribution'] = 'IID'
### CLIENTS LEARNING PARAMETERS
config['client_algorithm'] = 'sgd'
config['client_batch_size'] = 32
config['client_n_epochs'] = 1 
### SERVER LEARNING PARAMETERS
config['server_global_rate'] = 1
config['server_local_rate'] = 1
### SIMULATION STRUCTURE
config['stop_condition'] = 'rounds'
config['stop_value'] = 2 
### LOGGING DIRECTIVES
config['log_file'] = os.path.join(LOG_FILES_DIR, 'test_log.json')
config['log_verbosity'] = 2
config['log_frequency'] = 1
### TERMINAL OUTPUT DIRECTIVES
config['stdout_verbosity'] = 2
config['stdout_frequency'] = 1

###################################################################
# WRITE CONFIG TO TEST FILE
config_file = os.path.join(CONFIG_FILES_DIR, 'test_config.json')
save_config_to_file(config, config_file)