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

class coco_loader(Dataset):
    def __init__(self, coco_root, split="train", max_tokens=15, ncap_per_img=5):
        self.max_tokens = max_tokens
        self.ncap_per_img = ncap_per_img
        self.coco_root = coco_root
        self.split = split
        self.get_split_info('data/dataset_coco.json')

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
        annos = {}

        split_include = []
        if split == "train": split_include = ["train", "restval"]
        else: split_include = [split]

        for item in split_info["images"]:
            if item['split'] in split_include:
                annos[item['cocoid']] = item
        
        self.annos = annos
        self.ids   = list(self.annos.keys())
        print('Found %d images in split: %s'%(len(self.ids), self.split))
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        anno   = self.annos[img_id]

        captions = [caption["raw"] for caption in anno["sentences"]]
        #get path to image
        imgpath = os.path.join(self.coco_root, anno["filepath"], anno["filename"])

        #load image
        img = Image.open(os.path.join(imgpath)).convert('RGB')
        img = self.img_transforms(img)

        if self.split != "train":
            #if not train split, pick a random caption
            r = np.random.randint(0, len(captions))
            captions = [captions[r]]
        else:
            #if train split, pick ncap_per_img random captions
            if len(captions) > self.ncap_per_img:
                ids = np.random.permutation(len(captions))[:self.ncap_per_img]
                captions_sel = [captions[l] for l in ids]
                captions = captions_sel
            assert len(captions) == self.ncap_per_img
        
        #create tensor to store word index
        wordclass = torch.zeros(len(captions), self.max_tokens, dtype=torch.long)
        #mask used to compute loss
        sentence_mask = torch.zeros(len(captions), self.max_tokens, dtype=torch.uint8)

        for i, caption in enumerate(captions):
            #split raw caption, get rid of punctuation
            words = str(caption).lower().translate(str.maketrans('', '', string.punctuation)).strip().split()
            #add start token at first position
            words = ["<S>"] + words
            #maximum length for a caption is max_tokens - 1 token and an EOS token
            num_words = min(len(words), self.max_tokens - 1)
            sentence_mask[i, :(num_words + 1)] = 1
            for word_i, word in enumerate(words):
                if word_i >= num_words: break #stop if index out length
                if word not in self.wordlist: word = "UNK"  #set unknown word
                wordclass[i, word_i] = self.wordlist.index(word) 
        
        return img, wordclass, sentence_mask, torch.tensor(img_id).view((1,))

    def __len__(self):
        return len(self.ids)