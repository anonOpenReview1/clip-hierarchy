from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST
import torchvision
from torchvision import transforms
from typing import Callable, Optional, List
from torch.utils.data import Subset
import numpy as np
import torch
import torch.utils.data as data
from src.simple_utils import load_pickle
import pathlib
import json
import os
import logging 
import pickle
from PIL import Image
from robustness.tools.breeds_helpers import make_living17, make_entity13, make_entity30, make_nonliving26
from robustness.tools.helpers import get_label_mapping
from robustness.tools import folder
from torchvision.datasets import ImageFolder

from wilds import get_dataset as get_dataset_wilds

# log = logging.getLogger(__name__)
log = logging.getLogger("app")

osj = os.path.join

def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })

class ImageNetDS(data.Dataset):
    """`Downsampled ImageNet <https://patrykchrabaszcz.github.io/Imagenet32/>`_ Datasets.
    Args:
        root (string): Root directory of dataset where directory
            ``ImagenetXX_train`` exists.
        img_size (int): Dimensions of the images: 64,32,16,8
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    base_folder = 'Imagenet{}_train'
    train_list = [
        ['train_data_batch_1', ''],
        ['train_data_batch_2', ''],
        ['train_data_batch_3', ''],
        ['train_data_batch_4', ''],
        ['train_data_batch_5', ''],
        ['train_data_batch_6', ''],
        ['train_data_batch_7', ''],
        ['train_data_batch_8', ''],
        ['train_data_batch_9', ''],
        ['train_data_batch_10', '']
    ]

    test_list = [
        ['Imagenet32_val_data', ''],
    ]

    def __init__(self, root, img_size, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.img_size = img_size

        self.base_folder = self.base_folder.format(img_size)

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                with open(file, 'rb') as fo:
                    entry = pickle.load(fo)
                    self.train_data.append(entry['data'])
                    self.train_labels += [label - 1 for label in entry['labels']]
                    self.mean = entry['mean']

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((self.train_data.shape[0], 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, f)
            fo = open(file, 'rb')
            entry = pickle.load(fo)
            self.test_data = entry['data']
            self.test_labels = [label - 1 for label in entry['labels']]
            fo.close()
            self.test_data = self.test_data.reshape((self.test_data.shape[0], 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

def get_breeds_mapping(dataset_name, data_dir): 
    path = f"{data_dir}/imagenet/imagenet_hierarchy/"

    if dataset_name.startswith("living17"): 
        ret = make_living17(path, split="good")
    elif dataset_name.startswith("entity13"):
        ret = make_entity13(path, split="good")
    elif dataset_name.startswith("entity30"):
        ret = make_entity30(path, split="good")
    elif dataset_name.startswith("nonliving26"):
        ret = make_nonliving26(path, split="good")

    label_mapping = get_label_mapping('custom_imagenet', np.concatenate((ret[1][0], ret[1][1]), axis=1)) 

    return label_mapping

def get_dataset(data_dir, dataset, train, transform): 

    Imagenet_Folder_with_indices = dataset_with_indices(ImageFolder)
    ImageFolder_with_indices = dataset_with_indices(folder.ImageFolder)
    ImageNetDS_with_indices = dataset_with_indices(ImageNetDS)
    c100_idx = dataset_with_indices(CIFAR100)
    fm_idx = dataset_with_indices(FashionMNIST)

    if dataset.lower() == "cifar100":
        data = c100_idx(root = data_dir + "/cifar100/", train=train, transform=transform, download=True)    
    elif dataset.lower() == "imagenet-sketch":
        data = Imagenet_Folder_with_indices(data_dir + "/imagenet/imagenet-sketch/sketch", transform = transform)
    elif dataset.lower() == "fruits360":
        data = Imagenet_Folder_with_indices(data_dir + "/fruits-360/Training", transform = transform)
    elif dataset.lower() == "food-101":
        data = Imagenet_Folder_with_indices(data_dir + "/food-101/images", transform = transform) 
    elif dataset.lower() == "lsun-scene":
        data = Imagenet_Folder_with_indices(data_dir + "/lsun/scene", transform = transform) 
    elif dataset.lower() in ["fashion1M", 'fashion1m']:
        data = Imagenet_Folder_with_indices(data_dir + "/fashion1M/clean_data", transform = transform)
    elif dataset.lower() == "imagenet":
        data = Imagenet_Folder_with_indices(data_dir + "/imagenet/imagenetv1/val", transform = transform)
    elif dataset.lower() == "objectnet":
        data = Imagenet_Folder_with_indices(data_dir + "/objectnet-1.0/images", transform = transform)
    elif dataset.lower() == "imagenet-c1":
        data = Imagenet_Folder_with_indices(data_dir + "/imagenet/imagenet-c/fog/1", transform = transform)
    elif dataset.lower() == "imagenet-c2":
        data = Imagenet_Folder_with_indices(data_dir + "/imagenet/imagenet-c/contrast/2", transform = transform)
    elif dataset.lower() == "imagenet-c3":
        data = Imagenet_Folder_with_indices(data_dir + "/imagenet/imagenet-c/snow/3", transform = transform)
    elif dataset.lower() == "imagenet-c4":
        data = Imagenet_Folder_with_indices(data_dir + "/imagenet/imagenet-c/gaussian_blur/4", transform = transform)
    elif dataset.lower() == "imagenet-c5":
        data = Imagenet_Folder_with_indices(data_dir + "/imagenet/imagenet-c/saturate/5", transform = transform)
    elif dataset.lower() == "imagenetv2":
        data = Imagenet_Folder_with_indices(data_dir + "/imagenet/imagenetv2/imagenetv2-matched-frequency-format-val", transform = transform)
    elif dataset.lower() == "office31-amazon":
        data = Imagenet_Folder_with_indices(data_dir + "/office31/amazon/images/", transform = transform)
    elif dataset.lower() == "office31-dslr":
        data = Imagenet_Folder_with_indices(data_dir + "/office31/dslr/images/", transform = transform)
    elif dataset.lower() == "office31-webcam":
        data = Imagenet_Folder_with_indices(data_dir + "/office31/webcam/images/", transform = transform)
    elif dataset.lower() == "officehome-product":
        data = Imagenet_Folder_with_indices(data_dir + "/officehome/Product/", transform = transform)
    elif dataset.lower() == "officehome-realworld":
        data = Imagenet_Folder_with_indices(data_dir + "/officehome/RealWorld/", transform = transform)
    elif dataset.lower() == "officehome-art":
        data = Imagenet_Folder_with_indices(data_dir + "/officehome/Art/", transform = transform)
    elif dataset.lower() == "officehome-clipart":
        data = Imagenet_Folder_with_indices(data_dir + "/officehome/Clipart/", transform = transform)  
    elif dataset.lower() == "fashion-mnist": 
        data = fm_idx(root = data_dir, train=train, transform=transform, download=True)
    else: 
        raise NotImplementedError("Please add support for %s dataset" % dataset)
    return data


def split_idx(y_true, num_classes = 1000): 

    classes_idx = []

    y_true = np.array(y_true)
    
    for i in range(num_classes): 
        classes_idx.append(np.where(y_true==i)[0])

    return classes_idx
