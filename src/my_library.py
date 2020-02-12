import sys, os
import json
from pathlib import Path
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np

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

    MAX_PARTITION_SIZE = 6000
    MAX_VARIATION = 2000

    data_dir = os.path.join(os.getcwd(), 'data')
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,),(0.3081,))
                ]),
    if config['dataset_name']=='MNIST' and config['dataset_distribution']=='IID':
        train = MNIST(data_dir, train=True, transform=transform, download=True)
        test = MNIST(data_dir, train=False, transform=transform, download=True)
        
        train_partitions = []
        for _ in range(config['n_clients']):
            data_amount = np.random.randint(MAX_PARTITION_SIZE-MAX_VARIATION, MAX_PARTITION_SIZE+MAX_VARIATION)
            # partition = [train[np.random.randint(len(train))] for _ in range(data_amount)]
            partition = []
            print(f'DATA AMOuNT : {data_amount}')
            for _ in range(data_amount):
                index = np.random.randint(len(train))
                datapoint = train.data[index]
                partition.append(datapoint)
            train_partitions.append(partition)
        print(f"---- train_partitions size: {len(train_partitions)},{len(train_partitions[0])},{len(train_partitions[0][0])},{len(train_partitions[0][0][0])}")
    else:
        raise NotImplementedError
    
    return (train_partitions, test)
    

#########################################################
#########################################################
# Enter function description
def f():
    pass