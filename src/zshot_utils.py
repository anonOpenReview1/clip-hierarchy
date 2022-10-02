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
from torchvision.datasets import ImageFolder

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
Imagenet_Folder_with_indices = dataset_with_indices(ImageFolder)

def query_gpt_prompt(prompt):
    
    completion = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0.15,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    sub_str = completion.choices[0].text
    #sub_list = [x.strip() for x in re.sub('[^a-zA-Z, ]+', '', sub_str).split(",")]
    return sub_str.strip("\n")

def get_CLIP_inputs_from_dict(cl_set_dict, ord_class):
    '''
    cl_set_dict: any dict that is of the form {class: [list of class words]}
    ord_class: ORDERED list of classnames that must match dataset
    '''
    subclasses = [''] * sum([len(v) for k,v in cl_set_dict.items()])

    counter = 0
    sub_to_super = {}
    for idx, cl in enumerate(ord_class):
        subs = cl_set_dict[cl]
        for sub in subs:
            subclasses[counter] = sub
            sub_to_super[counter] = idx
            counter += 1
    
    return subclasses, sub_to_super

def get_idir(data_dir, dist):
    if dist == 'imagenet':
        imagenet_dir = data_dir + "/imagenet/imagenetv1/train/"
    elif dist == 'imagenet-sketch':
        imagenet_dir = data_dir + "/imagenet/imagenet-sketch/sketch/"
    elif dist == 'imagenet-c1':
        imagenet_dir = data_dir +'/imagenet/imagenet-c/fog/1'
    elif dist == 'imagenet-c2':
        imagenet_dir = data_dir +'/imagenet/imagenet-c/contrast/2'
    elif dist == 'imagenet-c3':
        imagenet_dir = data_dir +'/imagenet/imagenet-c/snow/3'
    elif dist == 'imagenet-c4':
        imagenet_dir = data_dir +'/imagenet/imagenet-c/gaussian_blur/4'
    elif dist == 'imagenet-c5':
        imagenet_dir = data_dir +'/imagenet/imagenet-c/saturate/5'
    elif dist == 'imagenetv2':
        imagenet_dir = data_dir +'/imagenet/imagenetv2/imagenetv2-matched-frequency-format-val'
    return imagenet_dir


def query_gpt(word, n=10, temp=0.1):
    try:
        prompt = f"Generate a comma separated list of {n} types of the following:\n\n>{word}:"
        
        completion = openai.Completion.create(
            model="text-davinci-002",
            prompt=prompt,
            temperature=temp,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        sub_str = completion.choices[0].text
        sub_list = [x.strip() for x in re.sub('[^a-zA-Z, ]+', '', sub_str).split(",")]
        return sub_list
    except:
        time.sleep(10)
        return query_gpt(word, n, temp)
    

def get_cleaned_gpt_sets(sup_classes, n=10, temp=0.1):
    out_d = {}
    for cl in sup_classes:
        if out_d.get(cl) is None:
            out_d[cl] = query_gpt(cl, n=n, temp=temp)
    new_d = out_d.copy()
    for k, v in out_d.items():
        new_v = []
        if k not in v:
            new_v.append(k)
        for sub in v:
            if k.lower() not in sub.lower():
                new_v.append(sub + f" {k}")
            else:
                new_v.append(sub)
        new_d[k] = new_v
    return new_d

def get_breeds_all_subs(hier, dg, dataset):
    if dataset == 'living17':
        ret = dg.get_superclasses(level=5,
                      Nsubclasses=4,
                      split=None,
                      ancestor='n00004258',
                      balanced=False)
    elif dataset == 'nonliving26':
        ret = dg.get_superclasses(level=5,
                      Nsubclasses=4,
                      split=None,
                      ancestor='n00021939',
                      balanced=False)
    elif dataset == 'entity13':
        ret = dg.get_superclasses(level=3,
                      Nsubclasses=20,
                      split=None,
                      ancestor=None,
                      balanced=False)
    else:
        ret = dg.get_superclasses(level=4,
                      Nsubclasses=8,
                      split=None,
                      ancestor=None,
                      balanced=False)
    superclasses, subclass_split, label_map = ret
    df = print_dataset_info(superclasses,
                    subclass_split,
                    label_map,
                    hier.LEAF_NUM_TO_NAME)
    breeds_classes = [x.split(",")[0].strip() for x in df['superclass']]
    sub2super = {k: v for k, v in zip(breeds_classes, df['subclasses'].apply(lambda z: [re.sub(r'\([^)]*\)', '', y).strip() for y in sorted(z, key=lambda x: int(re.search(r'\([^)]*\)', x).group(0).strip("()")))]))}
    return sub2super

def conf_pred(raw_logits, super_logits, raw2super_map):
    inv_map = {}
    for k,v in raw2super_map.items():
        if inv_map.get(v) is None:
            inv_map[v] = [k]
        else:
            inv_map[v].append(k)
    
    for k,v in inv_map.items():
        inv_map[k] = sorted(v)
    raw_probs = F.softmax(raw_logits, dim=1)
    super_probs = F.softmax(super_logits, dim=1)
    output = torch.zeros_like(raw_probs)

    for out_cl, in_cls in inv_map.items():
        sub_probs = raw_probs[:, in_cls]

        if len(sub_probs.shape) == 1:
            sub_probs = sub_probs.reshape(-1, 1)
        

        raw_probs[:, in_cls] = raw_probs[:, in_cls] * super_probs[:, out_cl].reshape(-1, 1)
    return raw_probs

def run(clip_mod, features, labels, sub2super, breeds_classes, args):
    raw_classes, reset_raw_to_super_mapping = get_CLIP_inputs_from_dict(sub2super, breeds_classes)

    if args.superclass_set_ens:
        clip_mod.text_features = clip_mod.zeroshot_classifier_set_templates(breeds_classes, clip_mod.templates)
        super_out = clip_mod.emb_forward(torch.tensor(features).cuda())['logits']
        super_preds = torch.argmax(super_out, dim=1).detach().cpu().numpy()
        super_preds = np.array([clip_mod.temp_map[x] for x in super_preds])
    else:
        clip_mod.text_features = clip_mod.zeroshot_classifier(breeds_classes, clip_mod.templates)
        super_out = clip_mod.emb_forward(torch.tensor(features).cuda())['logits']
        super_preds = torch.argmax(super_out, dim=1).detach().cpu().numpy()

    if args.experiment in ['true', 'gpt', 'true_noise']:
        clip_mod.text_features = clip_mod.zeroshot_classifier(raw_classes, clip_mod.templates)
        raw_out = clip_mod.emb_forward(torch.tensor(features).cuda())['logits']

        conf_preds = torch.argmax(conf_pred(raw_out, super_out, reset_raw_to_super_mapping), dim=1).detach().cpu().numpy()
        raw_preds = torch.argmax(raw_out, dim=1).detach().cpu().numpy()
        raw_preds = np.array([reset_raw_to_super_mapping[x] for x in raw_preds])
        conf_preds = np.array([reset_raw_to_super_mapping[x] for x in conf_preds])

        out_d = {
            'Superclass': (super_preds == labels).sum() / len(labels),
            'CHiLSNoRW': (raw_preds == labels).sum() / len(labels),
            'CHiLS': (conf_preds == labels).sum() / len(labels)
        }
        return out_d
    else:
        assert args.experiment in ['true_lin', 'gpt_lin']
        # no confidence prediction, as linear ensembling for subclasses outputs in superclass space
        clip_mod.text_features = clip_mod.zeroshot_classifier_ens(raw_classes, clip_mod.templates)
        raw_out = clip_mod.emb_forward(torch.tensor(features).cuda())['logits']
        raw_preds = torch.argmax(raw_out, dim=1).detach().cpu().numpy()
        out_d = {
            'Superclass': (super_preds == labels).sum() / len(labels),
            'CHiLSNoRW': (raw_preds == labels).sum() / len(labels),
        }
        return out_d




def find_classes(dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


def get_mapping(data_dir, dataset_name, idir): 
    
    ret = eval(f"make_{dataset_name}")(data_dir, split='good')
    # if dataset_name.startswith("living17"): 
    #     ret = make_living17(data_dir, split="good")
    # elif dataset_name.startswith("entity13"):
    #     ret = make_entity13(data_dir, split="good")
    # elif dataset_name.startswith("entity30"):
    #     ret = make_entity30(data_dir, split="good")
    # elif dataset_name.startswith("nonliving26"):
    #     ret = make_nonliving26(data_dir, split="good")
        
    label_mapping = get_label_mapping('custom_imagenet', np.concatenate((ret[1][0], ret[1][1]), axis=1)) 

    classes, old_class_to_idx = find_classes(imagenet_dir)
    classes, new_class_to_idx = label_mapping(classes, old_class_to_idx)

    new_map  = defaultdict(lambda : -1)

    for key in new_class_to_idx: 
        new_map[old_class_to_idx[key]] = new_class_to_idx[key]

    return new_map

def get_idx(dataset_name, y_true, idir): 

    label_map = get_mapping(dataset_name, idir)

    labels = label_map.keys()

    idx = []
    for label in labels: 
        idx.append(np.where(y_true==label)[0])

    idx = np.concatenate(idx, axis = 0)

    return idx, label_map

def transform_labels(label_map, labels): 
    return np.array([label_map[i] for i in labels])


def prep_data(data, dataset, idir):
    print(idir)
    features, labels, outputs, indices = data["features"], data["labels"], data["outputs"], data["indices"]
    idx, label_map = get_idx(dataset, labels, idir)
    label_map = {k:v for k,v in sorted(label_map.items(), key=lambda x:(x[1], x[0]))}
    features = features[idx]
    labels = transform_labels(label_map, labels[idx])
    outputs = outputs[idx]
    indices = indices[idx]
    return features, labels, outputs, indices