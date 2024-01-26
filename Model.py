"""
Created on Wed Sep 14 15:08:32 2022

@author: Hojun
"""

from typing import List

import torch
import torch.nn as nn

class Rainfall_linearnet(nn.Module):
    
    def __init__(
            self, 
            num_layers: int, 
            layer_widths: List,
            dropout_p: float=.0,
            net_type: str='r',
            activation: str='ReLU',
        ):
        
        super().__init__()
        self.num_layers = num_layers
        self.layer_widths = layer_widths
        self.net_type = net_type
        self.activation = activation
        if self.activation == 'ReLU':
            activation_layer = nn.ReLU()
        elif self.activation == 'Tanh':
            activation_layer = nn.Tanh()
        elif self.activation == 'Sigmoid':
            activation_layer = nn.Sigmoid()
        else:
            raise ValueError('{self.activation} function not supported for activation layer')

        layers = []
        for i in range(self.num_layers):
            layers.append(nn.Linear(self.layer_widths[i], self.layer_widths[i+1]))
            if i!=(self.num_layers-1):
                layers.append(activation_layer)
                layers.append(nn.Dropout(p=dropout_p))
            
        self.linear_model = nn.Sequential(*layers)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        output = self.linear_model(x)
        if self.net_type == 'r':
            output = torch.exp(output)
            
        return output
