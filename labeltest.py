import glob
import math
import numpy as np
import os
import os.path as osp
import string 
import pickle
import json
import os

from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as T

def preprocess(cap):
    words = str(cap).lower().translate(str.maketrans('', '', string.punctuation)).strip().split()
    return words
    
def get_all_labels(split, split_file):
    split_info = json.load(open(split_file))
    annos = {}

    split_include = []
    if split == "train": 
        split_include = ["train", "restval"]
    else: split_include = [split]

    for item in split_info["images"]:
        if item['split'] in split_include:
            annos[item['cocoid']] = item
    
    all_captions = []
    for anno in annos:
        captions = [preprocess(caption["raw"]) for caption in anno["sentences"]]
        all_captions.extend(captions)
    return all_captions
    
