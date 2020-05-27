import sys, os
import json
from pathlib import Path
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np
import torch
from nn_classes import *
import data_loader as dl
from pprint import pprint
from tqdm import tqdm
import json
import copy
from functools import reduce
from math import sqrt

#########################################################
#########################################################
# Create a json file based on config (python dict)
# and save it in a specified destination (absolute address)
def save_config_to_file(config, destination, indent=1):
    destination = Path(destination)
    # create containing folder is necessary
    if not destination.parent.exists():
        os.makedirs(destination.parent)
    # if destination file exists, delete it
    if destination.exists():
        os.remove(str(destination))
    # write config object
    with open(str(destination), 'w') as outfile:
        json.dump(config, outfile, indent=indent)


#########################################################
#########################################################
# Enter function description
def get_split_dataset(config):
    trainset, testset = dl.get_dataset(config)
    sample_inds = dl.get_indices(trainset, config)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config["client_batch_size"], shuffle=False, num_workers=0)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config["client_batch_size"], shuffle=False, num_workers=0)
    train_partitions = []
    # ##### DEBUG
    # train_partitions = [trainset for _ in range(config["n_clients"])]
    # return train_partitions, testloader
    # ##### END
    config["data_per_client"] = len(sample_inds[0])
    if config["stdout_verbosity"]>=2:
            print("Creating dataset partitions")
    if len(sample_inds) == 1:
        train_partitions = [torch.utils.data.DataLoader(trainset, 
                                                batch_size=config["client_batch_size"], 
                                                shuffle=True, 
                                                num_workers=0)]
        return trainloader, train_partitions, testloader
    for partition_n in tqdm(range(len(sample_inds))):
        # if config["stdout_verbosity"]>=2:
        #     print(f"Creating dataset partition {partition_n+1} of {len(sample_inds)} ...")
        x_shape = list(np.array(trainset[0][0]).shape)
        n_samples = [len(sample_inds[partition_n])]
        images = np.empty(shape=n_samples+x_shape, dtype=float)
        labels = np.empty(n_samples, dtype=int)
        for i, idx in enumerate(sample_inds[partition_n]):
            image, label = trainset[idx]
            images[i] = image
            labels[i] = label
        images_t = torch.FloatTensor(images)
        labels_t = torch.LongTensor(labels)
        # if config["debug"]:
        #     print("--- train X:", images_t.shape)
        #     print("--- train Y:", labels_t.shape)
        partition_data = torch.utils.data.TensorDataset(images_t,labels_t)
        partition = torch.utils.data.DataLoader(partition_data, 
                                                batch_size=config["client_batch_size"], 
                                                shuffle=True, 
                                                num_workers=0)
        train_partitions.append(partition)
    
    

    return trainloader, train_partitions, testloader    

#########################################################
#########################################################
# Return number of trainable parameters in a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#########################################################
#########################################################
# Set all parameters of a model to zero
def zero_param(model):
    for param in model.parameters():
        param.data.data.mul_(0)


#########################################################
#########################################################
# Evaluate accuracy of the model
def evaluate_accuracy(model, testloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total



#########################################################
#########################################################
# Read log file
def read_log(filename): 
    if os.path.isfile(filename):
        with open(filename, "r+") as JSONfile:
            data = json.load(JSONfile)
        return data
    else:
        raise ValueError


#########################################################
#########################################################
# Write data to log file.
# If called with mode='a', it is
# equivalent to calling update_log.
def write_log(filename, data, mode="w"):
    if mode == "w":
        if os.path.exists(filename):
            os.system(f"rm {filename}")
        update = json.dumps(data, indent=2)
        with open(filename, mode) as JSONfile:
            JSONfile.write(update)
    elif mode == "a":
        update_log(filename, data)

#########################################################
#########################################################
# Updates log file with new data.
# If the log file does not exist,
# the data is written to a new one.
def update_log(filename, data):
    try:
        log_data = read_log(filename)
    except ValueError:
        write_log(filename, data)
        return
    if type(log_data) is not list:
        current_data = copy.deepcopy(log_data)
        log_data = []
        log_data.append(current_data)
    log_data.append(data)
    writable_log = json.dumps(log_data, indent=2)
    with open(filename, "w") as JSONfile:
        JSONfile.write(writable_log)


#########################################################
#########################################################
# Return all factors of input number (as a set)
def factors(n):
        step = 2 if n%2 else 1
        return set(reduce(list.__add__,
                    ([i, n//i] for i in range(1, int(sqrt(n))+1, step) if n % i == 0)))


#########################################################
#########################################################
# Return tuple of 2-D arrangement of clusters
def get_clusters_grid_shape(n_clusters):
    factors_l = list(factors(n_clusters))
    factors_l.sort()
    n_factors = len(factors_l)
    if n_factors % 2: # odd factors --> square shaped
        return (factors_l[int((n_factors-1)/2)], factors_l[int((n_factors-1)/2)])
    else: # even factors --> rectangle shaped
        return (factors_l[int(n_factors/2 - 1)], factors_l[int(n_factors/2)])

#########################################################
#########################################################
# Average results from different experiments 
def combine_results(config_files, log_files):
    combined = []
    for config_idx, config_file in enumerate(config_files):
        config = read_log(config_file)
        comm_rounds = config["stop_value"] if config["stop_condition"]=="rounds" else None
        n_logs = len(log_files[config_idx])
        # Get data and combine (average)
        train_accs = [0 for _ in range(comm_rounds)]
        test_accs = [0 for _ in range(comm_rounds)]
        latency = [0 for _ in range(comm_rounds)]
        weight_divergence = [0 for _ in range(comm_rounds)]
        latency_median_ls = []
        for log_file in log_files[config_idx]:
            log = read_log(log_file)
            for n_round in range(comm_rounds):
                train = log[n_round]["train_accuracy"]
                train = train if train is not None else 0
                train_accs[n_round] += train / n_logs

                test = log[n_round]["test_accuracy"]
                test = test if test is not None else 0
                test_accs[n_round] += test / n_logs
                
                latency[n_round] += log[n_round]["latency"] / n_logs
                latency_median_ls.append(log[n_round]["latency"])

                div = log[n_round]["weight_divergence"]
                div = div if div is not None else 0
                weight_divergence[n_round] += div / n_logs

        # Save combined results
        combined_results = {}
        combined_results["config"] = config
        combined_results["results"] = {}
        combined_results["results"]["rounds"] = list(range(1,comm_rounds+1))
        combined_results["results"]["train_accuracy"] = train_accs
        combined_results["results"]["test_accuracy"] = test_accs
        combined_results["results"]["latency"] = latency
        combined_results["results"]["latency_median"] = np.median(latency_median_ls)
        combined_results["results"]["weight_divergence"] = weight_divergence
        combined.append(combined_results)
    return combined

#########################################################
#########################################################
# Enter function description
def save_combined_results(results, old_logs,  delete_old_logs=False):
    for i,res in enumerate(results):
        combined_log_name = f"{old_logs[i][0][:-6]}avg.json"
        write_log(combined_log_name, res)
        if delete_old_logs:
            for filename in old_logs[i]:
                os.system(f"rm {filename}")



#########################################################
#########################################################
# Compute weight divergence between model parameters and
# server model parameters
def get_weight_divergence(model, server_model):

    weights = [torch.flatten(p) for p in model.parameters() if p.requires_grad]
    server_weights = [torch.flatten(p) for p in server_model.parameters() if p.requires_grad]
    diff = [weights[i]-server_weights[i] for i in range(len(weights))]

    norm_diff = np.linalg.norm(torch.cat(diff).detach().cpu(), ord=1)
    norm_server = np.linalg.norm(torch.cat(server_weights).detach().cpu())

    return norm_diff / norm_server

#########################################################
#########################################################
# Enter function description
def filter_results(metadata, results, filters, sort_by=[]):
    selected_meta = copy.deepcopy(metadata)
    for f in filters.keys():
        f_vals = filters[f]
        if f_vals:
            selected_meta = selected_meta[selected_meta[f].isin(f_vals)]
    if sort_by:
        selected_meta = selected_meta.sort_values(sort_by)
    selected_res = results.loc[list(selected_meta.index)]
    return selected_meta, selected_res

#########################################################
#########################################################
# Enter function description
def f():
    pass
