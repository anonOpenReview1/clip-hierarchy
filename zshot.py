import argparse
import os
import sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os, glob
import time
from torchvision.io import read_image, ImageReadMode
import torch
import numpy as np
import torchvision
import nltk
import torch.nn.functional as F
from nltk.corpus import wordnet as wn
import torch.nn as nn
import pickle as pk
import matplotlib.pyplot as plt
import time
import openai
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import json
from models.clip_models import *
from src.zshot_utils import *
from src.constants import *
from src.get_inputs import *

openai.api_key = os.environ.get("OPENAI_API_KEY")

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default='nonliving26')
parser.add_argument("--domain", type=str, default='imagenet')
parser.add_argument("--model", type=str, default='ClipViTL14')
parser.add_argument("--experiment", type=str, default='true')
parser.add_argument("--data-dir", type=str)
parser.add_argument("--label-set-size", type=int, default=10)
parser.add_argument("--temp", type=float, default=0.7)
parser.add_argument("--rerun-gpt", action='store_true')
parser.add_argument("--superclass-set-ens", action='store_true')



def check_args(args):
    if args.dataset not in DATASETS:
        raise NotImplementedError(f"Invalid dataset: {args.dataset}")
    if args.dataset in DOMAINS.keys() and args.domain not in DOMAINS[args.dataset]:
        raise NotImplementedError(f"Invalid domain: {args.domain}")
    if args.model not in MODELS:
        raise NotImplementedError(f"Invalid model: {args.model}")
    if args.experiment not in EXPERIMENTS:
        raise NotImplementedError(f"Invalid experiment : {args.experiment}")
    if args.dataset not in TRUESETS and args.experiment == 'true':
        raise NotImplementedError(f"No ground-truth subsets for {args.dataset}")


if __name__ == '__main__':
    args = parser.parse_args()
    check_args(args)
    print(args)
    mod = eval(args.model)()
    features, labels, sub2super, super_classes = get_inputs(args)
    out = run(mod, features, labels, sub2super, super_classes, args)
    print(out)



