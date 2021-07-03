import glob
import math
import numpy as np
import os
import os.path as osp
import string 
import pickle
import json
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as T


class coco_test_loader(Dataset):
    def __init__(self, coco_root, split_info):
        self.coco_root = coco_root
        self.get_split_info(split_info)

        worddict_tmp = pickle.load(open("data/wordlist.p", "rb"))
        wordlist = [l for l in worddict_tmp if l != "</S>"]
        self.wordlist = ["EOS"] + sorted(wordlist)
        self.vocab_size = len(self.wordlist)
        print(f"[DEBUG] Vocabulary size {self.vocab_size}")
        
        self.img_transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean = [ 0.485, 0.456, 0.406 ], 
                std = [ 0.229, 0.224, 0.225 ])])
        
    def get_split_info(self, split_file):
        split_info = json.load(open(split_file))
        self.ids   = list(split_info["images"])
        print('Found %d images in split'%(len(self.ids)))
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]["id"]
        filename = self.ids[idx]["file_name"]
        #get path to image
        imgpath = os.path.join(self.coco_root, filename)

        #load image
        img = Image.open(os.path.join(imgpath)).convert('RGB')
        img = self.img_transforms(img)
        
        return img, torch.tensor(img_id).view((1,))

    def __len__(self):
        return len(self.ids)