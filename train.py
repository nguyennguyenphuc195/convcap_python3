import os
import argparse
import numpy as np 
import json
import time
import pickle
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader                                                                    

from coco_loader import coco_loader
from convcap import Convcap
from vggfeats import Vgg16Feats
from device import DeviceDataLoader, get_default_device, to_device
from test import test 

default_device = get_default_device()

def repeat_img_feats(conv_feats, lin_feats, ncap_per_img=5):
    """ Repeat features up to ncap_per_img"""
    # conv_feats has shape [batchsize, 512, 7, 7], lin_feats has shape [batchsize, 4096]
    # output have shape [batchsize * ncap_per_img, 512, 7, 7] and [batchsize * ncap_per_img, 4096]

    bs, n_channels, width, height = conv_feats.size()
    conv_feats = conv_feats.unsqueeze(1).expand(bs, ncap_per_img, n_channels, width, height)
    conv_feats = conv_feats.contiguous().view(-1, n_channels, width, height)

    bs, n_feats= lin_feats.size()
    lin_feats = lin_feats.unsqueeze(1).expand(bs, ncap_per_img, n_feats)
    lin_feats = lin_feats.contiguous().view(-1, n_feats)

    return conv_feats, lin_feats


def train(data_root="./data/coco/", epochs=30, batchsize=20, ncap_per_img=5, num_layers=3,\
     is_attention=True, learning_rate=5e-5, lr_step_size=15, finetune_after=8,\
     model_dir=".", ImageCNN=Vgg16Feats, checkpoint=None, stats_savedir=".", checkpoint_savedir="."):
    train_ds = coco_loader(data_root, split="train", ncap_per_img=ncap_per_img)
    print("[DEBUG] Data loaded size")

    #create torch data loader
    train_dl = DataLoader(train_ds, batch_size=batchsize, shuffle=True, pin_memory=True, num_workers=2, drop_last=True)
    train_dl = DeviceDataLoader(train_dl, default_device)

    #load pretrained image encoder
    image_model = ImageCNN()
    image_model = to_device(image_model, default_device)
    image_model.train()

    #convcap model
    convcap_model = Convcap(train_ds.vocab_size, num_layers, is_attention)
    convcap_model = to_device(convcap_model, default_device)
    optimizer = optim.RMSprop(convcap_model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=.1)
    img_optimizer = None
    img_scheduler = None 

    batchsize_cap = batchsize * ncap_per_img
    bestscore = 0
    start_epoch = 0
    max_tokens = train_ds.max_tokens
    if checkpoint != None:
        # load_checkpoint()
        check = torch.load(checkpoint)

        image_model.load_state_dict(check['img_state_dict'])
        convcap_model.load_state_dict(check['state_dict'])
        optimizer.load_state_dict(check['optimizer'])
        scheduler.load_state_dict(check['scheduler'])
        bestscore = check["best_score"]
        start_epoch = check["epoch"] + 1

        if check['img_optimizer']:
            img_optimizer = optim.RMSprop(image_model.parameters(), lr=1e-5)
            img_scheduler = lr_scheduler.StepLR(img_optimizer, step_size=lr_step_size, gamma=.1)
            img_optimizer.load_state_dict(check['img_optimizer'])
            img_scheduler.load_state_dict(check['img_scheduler'])
    
    for epoch in range(start_epoch, epochs):
        loss_train  = 0
        emb0_grad_norm   = 0
        clf1_grad_norm   = 0
        batch_count = 0
        if epoch == finetune_after:
            img_optimizer = optim.RMSprop(image_model.parameters(), lr=1e-5)
            img_scheduler = lr_scheduler.StepLR(img_optimizer, step_size=lr_step_size, gamma=.1)
        
        convcap_model.train()
        image_model.train()
        for imgs, wordclass, mask, img_id in tqdm(train_dl):
            imgs = imgs.view(batchsize, 3, 224, 224)
            wordclass = wordclass.view(batchsize_cap, max_tokens)
            mask = mask.view(batchsize_cap, max_tokens)

            optimizer.zero_grad()
            if img_optimizer != None:
                img_optimizer.zero_grad()

            conv_feats, lin_feats = image_model(imgs) #shape (batchsize, height, width), (batchsize, 4096)
            conv_feats, lin_feats = repeat_img_feats(conv_feats, lin_feats, ncap_per_img) #shape (batchsize_cap, height, width), (batchsize_cap, 4096)
            _, _, height, width = conv_feats.size()

            if is_attention:
                #logits shape (batch_size_cap, vocabulary_size, maxtoken), attn shape (batchsize_cap, max_tokens, height x width)
                logits, attn = convcap_model(conv_feats, lin_feats, wordclass)
                attn = attn.view(batchsize_cap, max_tokens, height, width) #attn shape (batchsize_cap, max_tokens, height, width)
            else:
                logits, _    = convcap_model(conv_feats, lin_feats, wordclass)
            
            #ignore last prediction (when input is EOS)
            logits = logits[:, :, :-1]
            #ignore first token <S>, do not exist in prediction
            wordclass = wordclass[:, 1:]
            mask      = mask[:, 1:].contiguous()

            #change shape to (bs_cap, maxtokens, vocabulary_size) 
            # then to (bs_cap * maxtokens, vocabulary_size) 
            logits = logits.permute(0, 2, 1).contiguous().view(batchsize_cap * (max_tokens - 1), -1)
            # change shape to (bs_cap * (maxtokens - 1),)
            wordclass = wordclass.contiguous().view(-1)
            mask      = mask.view(-1)
            # assert wordclass.size(0) == batchsize_cap * (max_tokens - 1)

            maskids = torch.nonzero(mask.view(-1)).view(-1)
            #test
            # maskids_test = torch.nonzero(mask.view(-1)).cpu().numpy().reshape(-1)
            # assert torch.allclose(maskids.cpu(), torch.from_numpy(maskids_test))

            if is_attention:
                loss = F.cross_entropy(logits[maskids, ...], wordclass[maskids, ...]) + (torch.sum(torch.pow(1. - torch.sum(attn, 1), 2))) / (batchsize_cap * height * width)
            else:
                loss = F.cross_entropy(logits[maskids, ...], wordclass[maskids, ...]) 

            loss_train = loss_train + loss.item()
            batch_count = batch_count + 1

            loss.backward()

            optimizer.step()
            if img_optimizer != None:
                img_optimizer.step()

            sum_square = 0
            for params in convcap_model.emb_0.parameters():
                sum_square += params.grad.data.norm(2).item() ** 2
            emb0_grad_norm += sum_square ** (1 / 2)

            sum_square = 0
            for params in convcap_model.classifier_1.parameters():
                sum_square += params.grad.data.norm(2).item() ** 2
            clf1_grad_norm += sum_square ** (1 / 2)
        
        scheduler.step()
        if img_optimizer:
            img_scheduler.step()

        loss_train = loss_train / batch_count
        emb0_grad_norm = emb0_grad_norm / batch_count
        clf1_grad_norm = clf1_grad_norm / batch_count

        # print('[DEBUG] Training epoch %d has loss %f' % (epoch, loss_train))

        checkpoint_path = os.path.join(".", "checkpoint", "model.pth")
        if img_optimizer:
            img_optimizer_state = img_optimizer.state_dict()
            img_scheduler_state = img_scheduler.state_dict()
        else:
            img_optimizer_state = None
            img_scheduler_state  = None

        scores, _ = test(convcap_model=convcap_model, image_model=image_model, fn=f"result_val_{epoch}.json", savedir=stats_savedir) 
        score  = scores["CIDEr"]
        print('[DEBUG] Training epoch %d has loss %f and score %f' % (epoch, loss_train, score))
        if img_optimizer:
            img_optimizer_state = img_optimizer.state_dict()
            img_scheduler_state = img_scheduler.state_dict()
        else:
            img_optimizer_state = None
            img_scheduler_state  = None
        
        if(score > bestscore):
            bestscore = score
            print('[DEBUG] Saving model at epoch %d with CIDer score of %f'% (epoch, score))
            bestmodelfn = os.path.join(checkpoint_savedir, "bestmodel.pth")
            torch.save({
                'best_score' : bestscore,
                'loss' : loss_train,
                'epoch': epoch,
                'state_dict': convcap_model.state_dict(),
                'img_state_dict': image_model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'img_optimizer' : img_optimizer_state,
                'scheduler' : scheduler.state_dict(), 
                'img_scheduler' : img_scheduler_state, 
            }, bestmodelfn)

        checkpoint_path = os.path.join(checkpoint_savedir, "model.pth")
        torch.save({
            'best_score' : bestscore,
            'loss' : loss_train,
            'epoch': epoch,
            'state_dict': convcap_model.state_dict(),
            'img_state_dict': image_model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'img_optimizer' : img_optimizer_state,
            'scheduler' : scheduler.state_dict(), 
            'img_scheduler' : img_scheduler_state, 
        }, checkpoint_path)

        #for experiments
        checkpoint_record = os.path.join(stats_savedir, f"record_{epoch}.pth")
        torch.save({
            'score' : score,
            'loss' : loss_train,
            'epoch': epoch,
            'clf1_gradient_norm' : clf1_grad_norm,
            'emb0_gradient_norm' : emb0_grad_norm,
        }, checkpoint_record)