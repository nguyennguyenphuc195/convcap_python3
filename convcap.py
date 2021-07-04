import sys

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def Conv1D(in_channels, out_channels, kernel_size, padding, dropout=0.):
    m = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
    std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.normal_(0, 0.1)
    return m

def Linear(in_features, out_features, dropout=0.):
    m = nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)

def ReduceConv1D(in_channels, mid_channels, out_channels, kernel_size, padding, dropout=0.):
    first_conv  = Conv1D(in_channels, mid_channels, 1, 0, dropout)
    relu        = nn.ReLU()
    second_conv = Conv1D(mid_channels, out_channels, kernel_size, padding, dropout)
    return nn.Sequential(first_conv, relu, second_conv)

class AttentionLayer(nn.Module):
    def __init__(self, conv_channels, embed_dim):
        super(AttentionLayer, self).__init__()
        self.in_projection = Linear(conv_channels, embed_dim)
        self.out_projection = Linear(embed_dim, conv_channels)
        # self.bmm = torch.bmm
    """
    def forward(self, x, wordemb, imgsfeats):
        #x with shape (batchsize, maxlen, conv_channels)
        #wordemb with shape (batchsize, maxlen, embed_dim)
        residual = x

        x = (self.in_projection(x) + wordemb) * math.sqrt(0.5) #x with shape (batchsize, maxlen, embed_dim)

        batchsize, embed_dim, height, width = imgsfeats.size()
        y = imgsfeats.view(batchsize, embed_dim, height * width) #y with shape (batchsize, embed_dim, height * width)
        x = self.bmm(x, y) # x with shape (batchsize, maxlen, height * width)

        batchsize, maxlen, n_vector = x.size()
        x = F.softmax(x.view(batchsize * maxlen, n_vector), dim=-1)
        x = x.view(batchsize, maxlen, n_vector)
        attn_scores = x # attn_scores (batchsize, maxlen, height * width)
        # assert attn_scores.size() == (batchsize, maxlen, n_vector)

        y = y.permute(0, 2, 1) # shape (batchsize, n_vector, embed_dim)
        x = self.bmm(x, y) # shape (batchsize, maxlen, embed_dim)

        x = x * (n_vector * math.sqrt(1.0 / n_vector)) # shape (batchsize, maxlen, embed_dim)
        x = (self.out_projection(x) + residual) * math.sqrt(0.5) # shape (batchsize, maxlen, conv_channels)

        return x, attn_scores
    """
    def forward(self, x, word_embed, conv_feats):
        h   = (self.in_projection(x) + word_embed) * np.sqrt(0.5)
        h, attention_score = self.attend(h, conv_feats)
        out = (self.out_projection(h) + x) * np.sqrt(0.5)
   
        return out, attention_score

    def attend(self, x, conv_feats):
        #x with shape (batchsize, seq_len, dim)
        #conv_feats with shape (batchsize, dim, height, width)
        batchsize, n_channels, height, width = conv_feats.size()

        #Turn conv feats to height x width vectors
        resized_conv_feats = conv_feats.reshape(batchsize, n_channels, height * width)

        #attention_score matrix with shape (batchsize, seq_len, height x width)
        # print(x.shape, resized_conv_feats.shape)
        attention_score = torch.bmm(x, resized_conv_feats)
        attention_score = F.softmax(attention_score, dim=-1)

        #out with shape (batchsize, seq_len, dim)
        #change shape from (batchsize, n_channels, height * width) to (batchsize, height * width, n_channels)
        resized_conv_feats = resized_conv_feats.permute(0, 2, 1)
        out = torch.bmm(attention_score, resized_conv_feats)

        num_feats = height * width
        out = out * (num_feats * np.sqrt(1.0 / num_feats))
        
        return out, attention_score

class Convcap(nn.Module):
    def __init__(self, num_wordclass, num_layers=1, is_attention=True, nfeats=512, dropout=.1, kernel_size=5, nimgfeats=4096, positional_emb=False, maxtokens=15, reduce_dim=False):
        super(Convcap, self).__init__()
        self.nimgfeats = nimgfeats
        self.is_attention = is_attention
        self.nfeats = nfeats
        self.dropout = dropout
        self.emb_0 = Embedding(num_wordclass, nfeats, padding_idx=0)
        self.emb_1 = Linear(nfeats, nfeats, dropout=dropout)

        self.imgproj = Linear(self.nimgfeats, self.nfeats, dropout=dropout)
        self.resproj = Linear(nfeats * 2, self.nfeats, dropout=dropout)

        n_in  = 2 * self.nfeats
        n_out = self.nfeats

        self.n_layers = num_layers
        self.convs = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.kernel_size = kernel_size
        self.pad         = self.kernel_size - 1

        self.positional_emb = None
        if positional_emb == True:
            self.positional_emb = nn.Parameter(torch.rand(maxtokens, nfeats, requires_grad=True), requires_grad=True)
            self.positional_emb.data.normal_(0, 0.1)


        for i in range(self.n_layers):
            if reduce_dim == True:
                conv = ReduceConv1D(n_in, 128, 2 * n_out, self.kernel_size, self.pad, dropout)
            else:
                conv = Conv1D(n_in, 2 * n_out, self.kernel_size, self.pad, dropout)
            self.convs.append(conv)
            if self.is_attention:
                self.attention.append(AttentionLayer(n_out, nfeats))
            n_in = n_out

        self.classifier_0 = Linear(self.nfeats, (nfeats // 2))
        self.classifier_1 = Linear((nfeats // 2), num_wordclass, dropout=dropout)  
           
    def forward(self, imgsfeats, imgsfc7, wordclass, return_all_attention=False):

        attn_buffer = None
        wordemb = self.emb_0(wordclass)
        wordemb = self.emb_1(wordemb) # (batch_size, max_tokens, nfeats)
        if self.positional_emb != None:
            wordemb = (wordemb + self.positional_emb) * math.sqrt(0.5)
        # print(wordemb.size())
        x = wordemb.transpose(2, 1) # (batch_size, nfeats, max_tokens)
        batchsize, wordembdim, maxtokens = x.size()

        y = F.relu(self.imgproj(imgsfc7)) # (batch_size, nfeats)
        y = y.unsqueeze(2).expand(batchsize, self.nfeats, maxtokens) # (batch_size, nfeats, maxtokens)
        x = torch.cat([x, y], dim=1) # (batch_size, 2 * nfeats, maxtokens)

        if return_all_attention == True:
            all_attentions = []

        for i, conv in enumerate(self.convs):
            if i == 0:
                x = x.transpose(2, 1) # (batch_size, maxtokens, 2 * nfeats)
                residual = self.resproj(x) # (batch_size, maxtokens, nfeats)
                residual = residual.transpose(2, 1) # (batch_size, nfeats, maxtokens)
                x = x.transpose(2, 1) # (batch_size, 2 * nfeats, maxtokens)
            else:
                residual = x # (batch_size, nfeats, maxtokens)

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x) # (batch_size, 2 * nfeats, maxtoken + pad)
            x = x[:, :, :-self.pad] # (batch_size, 2 * nfeats, maxtoken)
            x = F.glu(x, dim=1) # (batch_size, nfeats, maxtoken)
            if (self.is_attention):
                attn = self.attention[i]
                x = x.transpose(2, 1) # (batch_size, maxtoken, nfeats)
                x, attn_buffer = attn(x, wordemb, imgsfeats)
                if return_all_attention == True:
                    all_attentions.append(attn_buffer)
                x = x.transpose(2, 1) # (batch_size, nfeats, maxtoken)

            x = (x + residual) * math.sqrt(0.5) # (batch_size, nfeats, maxtoken)
        
        x = x.transpose(2, 1) # (batch_size, maxtoken, nfeats)
        x = self.classifier_0(x) # (batch_size, maxtoken, nfeats / 2)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier_1(x) # (batch_size, maxtoken, vocabulary_size)

        x = x.transpose(2, 1) # (batch_size, vocabulary_size, maxtoken)
        if return_all_attention == True:
            return x, all_attentions
            
        return x, attn_buffer