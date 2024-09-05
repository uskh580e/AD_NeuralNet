import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
random.seed(0)
np.random.seed(0)




class CustomfcNetwork(nn.Module):
    def __init__(self, layers_node, n, masking):
        super(CustomfcNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.masks = []
        self.gamma = 0.0001

        for i in range(n-1):
            #print(-1-i)
            #print(layers_node[-1-i])
            in_features = len(layers_node[-1-i])
            out_features = len(layers_node[-1-i-1])
            self.layers.append(nn.Linear(in_features, out_features, bias=False))
            self.masks.append(masking[-1-i])

        self.layers.append(nn.Linear(len(layers_node[-n]), len(layers_node[0])-1))
        self.masks = [torch.tensor(mask, dtype=torch.float32) for mask in self.masks]


    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            mask = self.masks[i]
            #print(mask[0].shape)
            #layer.weight.data *= mask
            #print(layer(x).shape)
            x = layer(x)
            
            '''
            x = x.unsqueeze(2)
            print('x shape before multi: ',x.shape)
            print('shape of mask: ', mask.shape )
            x =  x*mask
            x = x.squeeze(2)
            x = torch.tanh(x)
            '''
            x = torch.tanh(x)
            #print('x shape: ',x.shape)
        #print(x.shape)
        
        x = self.layers[-1](x)
        return x