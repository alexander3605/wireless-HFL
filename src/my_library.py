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
    testloader = torch.utils.data.DataLoader(testset, batch_size=config["client_batch_size"], shuffle=False, num_workers=2)

    train_partitions = []

    # ##### DEBUG
    # train_partitions = [trainset for _ in range(config["n_clients"])]
    # return train_partitions, testloader
    # ##### END
    if config["stdout_verbosity"]>=2:
            print("Creating dataset partitions")
    for partition_n in tqdm(sample_inds):
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
                                                num_workers=2)
        train_partitions.append(partition)
    
    

    return train_partitions, testloader    

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
# Enter function description
def f():
    pass