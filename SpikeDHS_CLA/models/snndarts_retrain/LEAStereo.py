import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.snndarts_search.SNN import *
from models.snndarts_search.decoding_formulas import network_layer_to_space
from models.snndarts_retrain.new_model_2d import newFeature
import time

class LEAStereo(nn.Module):
    def __init__(self, init_channels=3, args=None):
        super(LEAStereo, self).__init__()
        p=0.0
        network_path_fea = [0,0,1,1,1,2,2,2]
        network_path_fea = np.array(network_path_fea)
        network_arch_fea = network_layer_to_space(network_path_fea)

        cell_arch_fea = [[1, 1],
                            [0, 1],
                            [3, 2],
                            [2, 1],
                            [7, 1],
                            [8, 1]]
        cell_arch_fea = np.array(cell_arch_fea)

        self.feature = newFeature(init_channels,network_arch_fea, cell_arch_fea, args=args)
        self.global_pooling = SpikingAdaptiveAvgPool2d(1)
        self.classifier = SpikingLinear(576, 10)
        # self.classifier = SpikingLinear(3*32*32, 10)
        self._time_step = 6
        
    def arch_parameters(self):
        return [param for name, param in self.named_parameters() if 'alpha_diffb' in name]

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'alpha_diffb' not in name]

    def forward(self, input): 
        input = input.expand(self._time_step, -1, -1, -1, -1)
        shape = input.shape[:2]

        feature_out, logits_aux = self.feature(input) 
        pooling_out = self.global_pooling(feature_out) 
        shape = pooling_out.shape[:2]
        logits_buf = self.classifier(pooling_out.view(*shape,-1)) 

        logits = logits_buf.mean(0)
        if logits_aux is not None:
            logits_aux = logits_aux.mean(0)

        if self.training:
            return logits, logits_aux
        else:
            return logits, None


def check_spike(input):
    input = input.cpu().detach().clone().reshape(-1)
    all_01 = torch.sum(input == 0) + torch.sum(input == 1)
    print(all_01 == input.shape[0])

