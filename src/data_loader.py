""""""
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np


def get_mnist_dataset():
    """returns trainset and testsets for MNIST dataset"""

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

    return trainset, testset


def get_fmnist_dataset():
    """returns trainset and testsets for Fashion MNIST dataset"""

    transform_train = transforms.Compose([
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)

    return trainset, testset


def get_cifar10_dataset():
    """returns trainset and testsets for Fashion CIFAR10 dataset"""

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    return trainset, testset


def get_dataset(config):

    dataset_name = config["dataset_name"]

    if dataset_name == 'mnist':
        trainset, testset = get_mnist_dataset()
    elif dataset_name == 'fmnist':
        trainset, testset = get_fmnist_dataset()
    elif dataset_name == 'cifar10':
        trainset, testset = get_cifar10_dataset()
    else:
        raise ValueError('dataset name can only be mnist, fmnist or cifar10')

    return trainset, testset

def get_indices(trainset, config):
    """returns the indices of sample for each worker in either iid, or non_iid manner provided in args"""

    if config["dataset_distribution"] == 'iid':
        inds = get_iid_index(trainset, config)
    elif config["dataset_distribution"] == 'non_iid':
        inds = get_non_iid_index(trainset, config)
    else:
        raise ValueError('Dataset distribution can only be iid or non_iid')
    return inds


def get_non_iid_index(trainset, config):
    """Returns the indexes of samples for each user such that the distributions of data for each user
    have a non_iid distribution. Sorts the indexs that have a lablel 0 to label 10. Then equally splits
     the indexes for each user"""

    dataset_name = config["dataset_name"]
    num_users = config["n_clients"]

    if dataset_name == 'mnist':
        num_samples = trainset.train_labels.shape[0]
        labels = trainset.train_labels.numpy()
    elif dataset_name == 'cifar10':
        labels = trainset.targets
        num_samples = len(labels)

    inds_sorted = np.argsort(labels) # sort indicies based on labels
    num_sample_perworker = int(num_samples / num_users)

    indx_sample = {n: [] for n in range(num_users)}

    for user in range(num_users):
        indx_sample[user] = inds_sorted[user * num_sample_perworker: (user + 1) * num_sample_perworker] # assign indices to each user

    return indx_sample


def get_iid_index(trainset, config):
    """Returns the indexes of samples for each user such that the distributions of data for each user
    have a iid distribution. Then equally splits
     the indexes for each user"""

    dataset_name = config["dataset_name"]
    num_users = config["n_clients"]

    if dataset_name == 'mnist':
        num_samples = trainset.train_labels.shape[0]
        labels = trainset.train_labels.numpy()
    elif dataset_name == 'cifar10':
        labels = trainset.targets
        num_samples = len(labels)
    num_sample_perworker = int(num_samples / num_users)
    inds = [*range(num_samples)]
    inds_split = np.random.choice(inds, [num_users, num_sample_perworker], replace=False)
    indx_sample = {n: [] for n in range(num_users)}
    for user in range(num_users):
        indx_sample[user] = list(inds_split[user])

    return indx_sample


class DatasetSplit(Dataset):
    def __init__(self, dataset, indxs):
        self.dataset = dataset
        self.indxs = indxs

    def __len__(self):
        return len(self.indxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.indxs[item]]
        return image, label
