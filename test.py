import os
import numpy as np 
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
 
from coco_loader import coco_loader
from torchvision import models                                                                     
from convcap import Convcap
from vggfeats import Vgg16Feats
from evaluate import language_eval

from device import DeviceDataLoader, get_default_device, to_device

default_device = get_default_device()


def save_test_json(preds, resFile):
    print('Writing %d predictions' % (len(preds)))
    json.dump(preds, open(resFile, 'w')) 

def test(convcap_model=None, image_model=None, split="val", \
         coco_root="data/coco/", batchsize=20):
    data = coco_loader(coco_root, split=split, ncap_per_img=1)
    data_loader = DataLoader(dataset=data, num_workers=2, batch_size=batchsize, shuffle=False, drop_last=True)

    convcap_model.train(False)
    image_model.train(False)

    pred_captions = []
    max_tokens = data.max_tokens

    for imgs, word_indices, sentence_masks, img_id in tqdm(data_loader):
        imgs = imgs.view(batchsize, 3, 224, 224)

        conv_feats, lin_feats = image_model(imgs)
        _, nfeats, height, width = conv_feats.size()

        wordclass_feed       = np.zeros((batchsize, max_tokens), dtype='int64')
        wordclass_feed[:, 0] = data.wordlist.index("<S>")

        outcaps = [[] for b in range(batchsize)]
        for j in range(max_tokens - 1):
            wordclass = to_device(torch.from_numpy(wordclass_feed), default_device)
            logits, _ = convcap_model(conv_feats, lin_feats, wordclass)

            logits = logits[:, :, :-1]
            #change shape to (bs_cap, maxtokens, vocabulary_size) 
            # then to (bs_cap * maxtokens, vocabulary_size) 
            logits = logits.permute(0, 2, 1).contiguous().view(batchsize*(max_tokens-1), -1)

            wordprobs = F.softmax(logits, dim=-1).cpu().numpy()
            wordids   = np.argmax(wordprobs, axis=-1)

            for k in range(batchsize):
                word = data.wordlist[wordids[j+k*(max_tokens-1)]]
                outcaps[k].append(word)
                if(j < max_tokens-1):
                    wordclass_feed[k, j+1] = wordids[j+k*(max_tokens-1)]

        for j in range(batchsize):
            num_words = len(outcaps[j]) 
            if 'EOS' in outcaps[j]:
                num_words = outcaps[j].index('EOS')
            outcap = ' '.join(outcaps[j][:num_words])
            pred_captions.append({'image_id': img_id[j].item(), 'caption': outcap})

    scores = language_eval(pred_captions, ".", split)
    return scores