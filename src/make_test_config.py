from my_library import *
import os
CONFIG_FILES_DIR = os.path.join(os.getcwd(), "config_files")
LOG_FILES_DIR = os.path.join(os.getcwd(), "log_files")

###################################################################
# DEFINE CONFIG
config = {}
### NETWORK STRUCTURE
config['n_clusters'] = 9
config['n_clients'] = 200
config['clients_distribution'] = 'balanced'
config['clients_mobility'] = True
config['mobility_rate'] = 0.1
### MODEL STRUCTURE
config['model_type'] = 'mnist'
config['model_init'] = 'random' 
### DATA STRUCTURE
config['dataset_name'] = 'mnist'
config['dataset_distribution'] = 'non_iid'
### CLIENTS LEARNING PARAMETERS
config['client_algorithm'] = 'sgd'
config['client_batch_size'] = 32
config['client_n_epochs'] = 1 
config['client_lr'] = 0.03
### SERVER LEARNING PARAMETERS
config['server_global_rate'] = 2
config['client_selection_fraction'] = 0.3
### SIMULATION STRUCTURE
config['stop_condition'] = 'rounds'
config['stop_value'] = 10
### LOGGING DIRECTIVES
config['log_file'] = os.path.join(LOG_FILES_DIR, 'test_log.json')
config['log_verbosity'] = 2
config['log_frequency'] = 1
### TERMINAL OUTPUT DIRECTIVES
config['stdout_verbosity'] = 2
config['stdout_frequency'] = 1
config['debug'] = 0

###################################################################
# WRITE CONFIG TO TEST FILE
config_file = os.path.join(CONFIG_FILES_DIR, 'test_config.json')
save_config_to_file(config, config_file)