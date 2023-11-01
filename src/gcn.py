# -*- coding: utf-8 -*-
import json, os
import math

import numpy
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCNModel(nn.Module):
       
    def __init__(self, args):
        super(GCNModel, self).__init__()
        self.args = args
        self.gc1 = GraphConvolution(2*args.lstm_dim, 2*args.lstm_dim)
        self.gc2 = GraphConvolution(2*args.lstm_dim, 2*args.lstm_dim)
        self.gc3 = GraphConvolution(2 * args.lstm_dim, 2 * args.lstm_dim)
        self.fc = nn.Linear(2 * args.lstm_dim,args.hidden_dim)
    def forward(self, lstm_features, sentence_adjs , mask):
        """
        dep_matrix:[batch, batch*sentence_nums,55,55]
        lstm_features:[batch,batch*sentence_nums,55,lstm_dim*2] 
        pos_embedding:[batch,sentence_nums,55,emb_dim] 
        """
        outputs = []
        for i,lstm_feature in enumerate(lstm_features):
            inputs = lstm_feature #[10,55,600]
            adjs = sentence_adjs[i] #[10,55,55]
            x = F.relu(self.gc1(inputs, adjs))
            x = F.relu(self.gc2(x, adjs))
            x = x * mask.unsqueeze(2).float().expand_as(x)
            x = self.fc(x).to(x.device)
            outputs.append(x)
        return outputs #[batch,sentence_nums,max_len,hidden_dim]:[1,10,55,hidden_dim]