import argparse
import numpy as np 
import time 
import pickle 
import itertools
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm 

from beamsearch import beamsearch 
from coco_loader import coco_loader
from convcap import Convcap
from vggfeats import Vgg16Feats
from evaluate import language_eval
from device import DeviceDataLoader, get_default_device, to_device
default_device = get_default_device()

def repeat_img_feats(conv_feats, lin_feats, beam_size=5):
    """ Repeat features up to ncap_per_img"""
    # conv_feats has shape [batchsize, 512, 7, 7], lin_feats has shape [batchsize, 4096]
    # output have shape [batchsize * ncap_per_img, 512, 7, 7] and [batchsize * ncap_per_img, 4096]

    bs, n_channels, width, height = conv_feats.size()
    conv_feats = conv_feats.unsqueeze(1).expand(bs, beam_size, n_channels, width, height)
    conv_feats = conv_feats.contiguous().view(-1, n_channels, width, height)

    bs, n_feats= lin_feats.size()
    lin_feats = lin_feats.unsqueeze(1).expand(bs, beam_size, n_feats)
    lin_feats = lin_feats.contiguous().view(-1, n_feats)

    return conv_feats, lin_feats

def test_beam(convcap_model=None, image_model=None, split="val", \
              coco_root="data/coco/", batchsize=20, beam_size=5):

    data = coco_loader(coco_root, split=split, ncap_per_img=1)
    data_loader = DataLoader(dataset=data, num_workers=2, batch_size=batchsize, shuffle=False, drop_last=True)
    data_loader = DeviceDataLoader(data_loader, default_device)
    convcap_model.train(False)
    image_model.train(False)

    max_tokens = data.max_tokens
    pred_captions = []

    for imgs, word_indices, sentence_masks, img_id in tqdm(data_loader):
        imgs = imgs.view(batchsize, 3, 224, 224)

        conv_feats, lin_feats = image_model(imgs)
        #shape (batchsize * beamsize, height, width), (batchsize_cap, 4096)
        conv_feats, lin_feats = repeat_img_feats(conv_feats, lin_feats, beam_size) 

        _, nfeats, height, width = conv_feats.size()
        
        wordclass_feed       = np.zeros((batchsize * beam_size, max_tokens), dtype='int64')
        wordclass_feed[:, 0] = data.wordlist.index("<S>")

        outcaps = [[] for b in range(batchsize)]
        beam_searcher = beamsearch(beam_size, batchsize, max_tokens)
        for j in range(max_tokens - 1):
            wordclass = to_device(torch.from_numpy(wordclass_feed), default_device)
            logits, _ = convcap_model(conv_feats, lin_feats, wordclass)

            #bs_cap: beamsize * batchsize
            logits = logits[:, :, :-1] #shape to (bs_cap, vocabulary_size, maxtokens - 1)
            logits_step_j = logits[..., j] #shape to (bs_cap, vocabulary_size)

            #beam_indices: indices of chosen beam
            #wordclass_indices: predicted word in chosen beam with same index
            beam_indices, wordclass_indices = beam_searcher.expand_beam(logits_step_j)  
            if len(beam_indices) == 0 or j == (max_tokens - 2):
                generated_captions = beam_searcher.get_result()
                for k in range(batchsize):
                    g = generated_captions[:, k]
                    outcaps[k] = [data.wordlist[x] for x in g]
            else:
                #keep chosen beams
                wordclass_feed = wordclass_feed[beam_indices]

                #keep chosen beam's features (remaining beam may be different shape)
                conv_feats = conv_feats.index_select(0, torch.cuda.LongTensor(beam_indices))
                lin_feats  = lin_feats.index_select(0, torch.cuda.LongTensor(beam_indices))
                
                #add predicted word to input of next step
                for i, wordclass_idx in enumerate(wordclass_indices):
                    wordclass_feed[i, j + 1] = wordclass_idx

        for j in range(batchsize):
            num_words = len(outcaps[j])
            if "EOS" in outcaps[j]:
                num_words = outcaps[j].index("EOS")
            outcap = ' '.join(outcaps[j][:num_words])
            pred_captions.append({'image_id': img_id[j].item(), 'caption': outcap})
    
    scores = language_eval(pred_captions, ".", "result_val.json", split)
    return scores
