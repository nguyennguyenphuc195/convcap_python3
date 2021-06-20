import os
import numpy as np 
import json
import time
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
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

def load_wordlist():
    worddict_tmp = pickle.load(open("data/wordlist.p", "rb"))
    wordlist = [l for l in worddict_tmp if l != "</S>"]
    wordlist = ["EOS"] + sorted(wordlist)
    vocab_size = len(wordlist)
    print(f"[DEBUG] Vocabulary size {vocab_size}")
    return wordlist

def load_image(img_path):
    img_transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
            mean = [ 0.485, 0.456, 0.406 ], 
            std = [ 0.229, 0.224, 0.225 ])])

    #load image
    img = Image.open(os.path.join(img_path)).convert('RGB')
    img = img_transforms(img)
    img = img.unsqueeze(0)

    img = to_device(img, default_device)
    return img

def regular_plot(img, cap):
    plt.imshow(img)
    plt.title(f"Caption: {cap}")
    plt.axis("off")
    plt.show()

def upsample(alpha, size):
    sz = alpha.shape[0]
    k  = size // sz
    
    res = np.zeros((size, size, 1))

    for i in range(sz):
        for j in range(sz):
            res[k * i : k * (i + 1), k * j : k * (j + 1), :] = alpha[i, j]
            
    return res

def plot_with_attention(img, attentions, outcap, bright=0.1):
    plt.figure(figsize=(10, 10))
    words = outcap.split()

    # temp = torch.cat(attentions, dim=0)
    # minnum = temp.min().item()
    # maxnum = temp.max().item()
    plt.subplot(3, (len(words) + 1) // 3 + 1, 1)
    plt.imshow(img)
    plt.title(f"Image")
    plt.axis("off")
    for i, word in enumerate(words):
        plt.subplot(3, (len(words) + 1) // 3 + 1, i + 2)
        score = attentions[i].squeeze(0).detach().cpu().numpy()
        minnum = score.min()
        maxnum = score.max()
        score = (score - minnum) / (maxnum - minnum) + bright
        # print(score.round(2))
        alpha = upsample(score, 224)

        alpha = gaussian_filter(alpha, sigma=10)
        img_t = img * alpha
        img_t = np.clip(img_t, 0, 1)
        plt.imshow(img_t)
        plt.title(f"{word}")
        plt.axis("off")
    plt.savefig("plot_result.png", bbox_inches='tight')
    plt.show()

def caption_image(convcap_model, image_model, img_root, max_tokens=15, plot=1, bright=0.1, att_idx=-1):
    convcap_model.train(False)
    image_model.train(False)
    
    wordlist = load_wordlist()

    pred_captions = []
    captions      = []
    attention_scores = [[] for x in range(convcap_model.n_layers)]

    wordclass_feed       = np.zeros((1, max_tokens), dtype='int64')
    wordclass_feed[:, 0] = wordlist.index("<S>")

    img = load_image(img_root)
    img = img.view(1, 3, 224, 224)

    conv_feats, lin_feats = image_model(img)
    _, nfeats, height, width = conv_feats.size()

    wordclass_feed       = np.zeros((1, max_tokens), dtype='int64')
    wordclass_feed[:, 0] = wordlist.index("<S>")

    outcap = []
    for j in range(max_tokens - 1):
        wordclass = to_device(torch.from_numpy(wordclass_feed), default_device)
        logits, all_attention_scores = convcap_model(conv_feats, lin_feats, wordclass, return_all_attention=True)

        if plot == 2:
            for l in range(len(attention_scores)):
                attention_score = all_attention_scores[l]
                attention_score = attention_score.view(1, max_tokens, 7, 7)
                attention_scores[l].append(attention_score[:, j, :, :])

        logits = logits[:, :, :-1]
        #change shape to (bs_cap, maxtokens, vocabulary_size) 
        #then to (bs_cap * maxtokens, vocabulary_size) 
        logits = logits.permute(0, 2, 1).contiguous().view((max_tokens-1), -1)

        wordprobs = F.softmax(logits, dim=-1).detach().cpu().numpy()
        wordids   = np.argmax(wordprobs, axis=-1)

        word = wordlist[wordids[j]]
        outcap.append(word)

        if word == "EOS":
            break

        if (j < max_tokens-1):
            wordclass_feed[0, j + 1] = wordids[j]

    num_words = len(outcap) 
    if 'EOS' in outcap:
        num_words = outcap.index('EOS')
    outcap = ' '.join(outcap[:num_words])

    img_plot = None
    if plot != 0:
        img_plot = img.squeeze(0).transpose(0, -1).transpose(0, 1).cpu().detach().numpy()
        mean = np.array([ 0.485, 0.456, 0.406 ]).reshape(1, 1, 3)
        std  = np.array([ 0.229, 0.224, 0.225 ]).reshape(1, 1, 3)
        img_plot = img_plot * std + mean
        img_plot = np.clip(img_plot, 0, 1)

    if plot == 1:
        regular_plot(img_plot, outcap)
    
    if plot == 2:
        plot_with_attention(img_plot, attention_scores[att_idx], outcap, bright)

    return outcap, attention_scores