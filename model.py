import argparse
import os
if os.name == 'nt': # only for windows
    from ctypes.wintypes import LONG
import datetime
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.models import VGAE
from torch_geometric.nn import Sequential
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.aggr import SumAggregation


from decoder.model import *

class GCNEncoder(nn.Module):
    # encoder model based on GCNConv
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.gcn_shared = GCNConv(in_channels, hidden_channels)
        self.gcn_mu = Sequential('x, edge_index', [(GCNConv(hidden_channels, out_channels), 'x, edge_index -> x'),SumAggregation()])
        self.gcn_logvar = Sequential('x, edge_index', [(GCNConv(hidden_channels, out_channels), 'x, edge_index -> x'), SumAggregation()])

    def forward(self, x, edge_index):

        edge_weight=torch.ones(len(edge_index[0]), dtype=torch.float) 
        x = F.relu(self.gcn_shared(x, edge_index, edge_weight=edge_weight))
        mu = self.gcn_mu(x, edge_index)
        logvar = self.gcn_logvar(x, edge_index)
        return mu, logvar


class DGMG_VAE(VGAE):
    def __init__(self, args):
        super(DGMG_VAE, self).__init__(encoder=GCNEncoder(-1,
                                                          args['enc_hidden_channels'],
                                                          args['enc_out_channels']),
                                       decoder=DGMG(args['max_size'], int(args['enc_out_channels']/2), 2),
                                       )
        self.reg = args['reg']
        self.out_dim = args['enc_out_channels']
    
    def forward(self, x, edge_index, actions=None):
        z = self.encode(x, edge_index)
        loss_rec = self.decoder.forward(actions = actions, latent_z=z) # reconstrution loss
        loss_kl = self.kl_loss() # Kubler-Leibler loss
        #print(loss_rec, loss_kl)
        return - loss_rec + self.reg*loss_kl, loss_rec, loss_kl 

    def test_generation(self):
        z = torch.normal(torch.zeros(self.out_dim),torch.ones(self.out_dim)).view(1,-1)
        generated_sample = self.decoder.forward(latent_z=z)
        return generated_sample
