import torch
import torch.nn as nn
from torchvision.models import resnet50

resnet = resnet50(pretrained=True)

class resnetFeats(nn.Module):
    def __init__(self):
        """
        conv_feats will have shape (bs, 512, 28, 28), lin_feats will have shape (bs, 2048)
        """
        super(resnetFeats, self).__init__()
        self.to_conv_feats = nn.Sequential()
        self.to_lin_feats  = nn.Sequential()

        #get rid of last layer
        layer_names = list(resnet._modules)[:-1]
        idx = layer_names.index("layer2")
        for i, name in enumerate(layer_names):
            if i <= idx:
                self.to_conv_feats.add_module(name, resnet._modules[name])
            else:
                self.to_lin_feats.add_module(name, resnet._modules[name])

    def forward(self, x):
        #last conv layer before avg_pool
        conv_feats = self.to_conv_feats(x)

        #after linear layer
        lin_feats = self.to_lin_feats(conv_feats)
        lin_feats = lin_feats.view(lin_feats.size(0), -1)

        #conv_feats will have shape (bs, 1024, 14, 14), lin_feats will have shape (bs, 2048)
        return conv_feats, lin_feats

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False