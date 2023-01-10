import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.snndarts_search.genotypes_2d import PRIMITIVES
from models.snndarts_search.genotypes_2d import Genotype
from models.snndarts_search.operations_2d import *
import torch.nn.functional as F
import numpy as np
import pdb
from spikingjelly.activation_based import layer

decay = 0.2

class MixedOp(nn.Module):
    def __init__(self):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()


    def forward(self, x, weights, left_or_right):
        opt_outs = []
        for i in range(3):
            opt_out = self._ops[i](x, left_or_right)
            opt_out = weights[i] * opt_out
            opt_outs.append(opt_out)
        return sum(opt_outs)  

class Cell(nn.Module):
    def __init__(self, steps, block_multiplier, prev_prev_fmultiplier,
                 prev_filter_multiplier, cell_arch, network_arch,
                 filter_multiplier, downup_sample, args=None):
        super(Cell, self).__init__()
        self.cell_arch = cell_arch

        self.C_in = block_multiplier * filter_multiplier
        self.C_out = filter_multiplier
        self.C_prev = int(block_multiplier * prev_filter_multiplier)
        self.C_prev_prev = int(block_multiplier * prev_prev_fmultiplier)
        self.downup_sample = downup_sample
        self._steps = steps
        self.block_multiplier = block_multiplier
        self._ops = nn.ModuleList()
        if downup_sample == -1:
            self.scale = 0.5
        elif downup_sample == 1:
            self.scale = 2

        self.cell_arch = torch.sort(self.cell_arch,dim=0)[0].to(torch.uint8)
        for x in self.cell_arch:
            primitive = PRIMITIVES[x[1]]
            if x[0] in [0,2,5]:
                op = OPS[primitive](self.C_prev_prev, self.C_out, stride=1, signal=1)
            elif x[0] in [1,3,6]:
                op = OPS[primitive](self.C_prev, self.C_out, stride=1, signal=1)
            else:
                op = OPS[primitive](self.C_out, self.C_out, stride=1, signal=1)

            self._ops.append(op)

        self.spikes = nn.ModuleList([getSpikingNode()
                                    for _ in range(self._steps)])

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))


    def forward(self, prev_prev_input, prev_input):
        s0 = prev_prev_input
        s1 = prev_input
        ## BUG?
        if self.downup_sample != 0:
            feature_size_h = self.scale_dimension(s1.shape[3], self.scale)
            feature_size_w = self.scale_dimension(s1.shape[4], self.scale)
            interpolate = layer.SeqToANNContainer(Nearest([feature_size_h, feature_size_w]))
            s1 = interpolate(s1)
        if (s0.shape[3] != s1.shape[3]) or (s0.shape[4] != s1.shape[4]):
            interpolate = layer.SeqToANNContainer(Nearest([s1.shape[3], s1.shape[4]]))

            s0 = interpolate(s0)
            # s0 = F.interpolate(s0, (s1.shape[2], s1.shape[3]),
                                            # mode='nearest')

        device = prev_input.device

        states = [s0, s1]
        offset = 0
        ops_index = 0
        for i in range(self._steps):
            new_states = []
            for j, h in enumerate(states):
                branch_index = offset + j
                if branch_index in self.cell_arch[:, 0]:
                    if prev_prev_input is None and j == 0:
                        ops_index += 1
                        continue
                    new_state = self._ops[ops_index](h)
                    new_states.append(new_state)
                    ops_index += 1
            s = sum(new_states)
            s = self.spikes[i](s)
            offset += len(states)
            states.append(s)
        # for i in range(len(states)):
        #     print('state %s shape'%i,states[i].shape)
        # print('self.block_multiplier',self.block_multiplier)
        concat_feature = torch.cat(states[-self.block_multiplier:], dim=2) 
        return prev_input, concat_feature

def check_spike(input):
    input = input.cpu().detach().clone().reshape(-1)
    all_01 = torch.sum(input == 0) + torch.sum(input == 1)
    print(all_01 == input.shape[0])


class newFeature(nn.Module):
    def __init__(self, frame_rate, network_arch, cell_arch, cell=Cell, args=None):
        super(newFeature, self).__init__()
        self.args = args
        self.cells = nn.ModuleList()
        self.network_arch = torch.from_numpy(network_arch)
        self.cell_arch = torch.from_numpy(cell_arch)
        self._step = args.fea_step
        self._num_layers = args.fea_num_layers
        self._block_multiplier = args.fea_block_multiplier
        self._filter_multiplier = args.fea_filter_multiplier

        f_initial = int(self._filter_multiplier)
        half_f_initial = int(f_initial/2)
        self._num_end = self._filter_multiplier*self._block_multiplier

        self.stem0_original = SNN_2d_Super(frame_rate, f_initial * self._block_multiplier, kernel_size=3, stride=1, padding=1,b=3) # DGS
        self.stem0 = SpikingConv2d(frame_rate, f_initial * self._block_multiplier, kernel_size=3, stride=1, padding=1, b=3)
        self.auxiliary_head = AuxiliaryHeadCIFAR(576, 10)

        filter_param_dict = {0: 1, 1: 2, 2: 4, 3: 8}

        for i in range(self._num_layers):
            level_option = torch.sum(self.network_arch[i], dim=1)
            prev_level_option = torch.sum(self.network_arch[i - 1], dim=1)
            prev_prev_level_option = torch.sum(self.network_arch[i - 2], dim=1)
            level = torch.argmax(level_option).item()
            prev_level = torch.argmax(prev_level_option).item()
            prev_prev_level = torch.argmax(prev_prev_level_option).item()

            if i == 0:
                downup_sample = - torch.argmax(torch.sum(self.network_arch[0], dim=1))
                _cell = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             self._filter_multiplier,
                             self.cell_arch, self.network_arch[i],
                             int(self._filter_multiplier * filter_param_dict[level]),
                             downup_sample, self.args)

            else:
                three_branch_options = torch.sum(self.network_arch[i], dim=0)
                downup_sample = torch.argmax(three_branch_options).item() - 1
                if i == 1:
                    _cell = cell(self._step, self._block_multiplier,
                                 self._filter_multiplier,
                                 int(self._filter_multiplier * filter_param_dict[prev_level]),
                                 self.cell_arch, self.network_arch[i],
                                 int(self._filter_multiplier * filter_param_dict[level]),
                                 downup_sample, self.args)

                else:
                    _cell = cell(self._step, self._block_multiplier,
                                 int(self._filter_multiplier * filter_param_dict[prev_prev_level]),
                                 int(self._filter_multiplier * filter_param_dict[prev_level]),
                                 self.cell_arch, self.network_arch[i],
                                 int(self._filter_multiplier * filter_param_dict[level]), downup_sample, self.args)

            self.cells += [_cell]
        




    def forward(self, x):
        # print('input shape',x.shape)
        
        # print('expand shape',x.shape)
        stem0 = self.stem0(x) 
        # print('stem0 shape',x.shape)
        
        stem1 = stem0
        out = (stem0,stem1)
        
        for i in range(self._num_layers):
            out = self.cells[i](out[0], out[1])
            # print('cell ',out[-1].shape)
            '''
            cell torch.Size([50, 144, 32, 32])
            cell torch.Size([50, 144, 32, 32])
            cell torch.Size([50, 288, 16, 16])
            cell torch.Size([50, 288, 16, 16])
            cell torch.Size([50, 288, 16, 16])
            cell torch.Size([50, 576, 8, 8] -> auxiliary [50, 10]
            cell torch.Size([50, 576, 8, 8])
            cell torch.Size([50, 576, 8, 8])
            '''
            if i == 2*8//3:
                if self.training:
                    logits_aux = self.auxiliary_head(out[-1])

        last_output = out[-1]

        if self.training:
            return last_output, logits_aux
        else:
            return last_output, None
            
    def get_params(self):
        bn_params = []
        non_bn_params = []
        for name, param in self.named_parameters():
            if 'bn' in name or 'downsample.1' in name:
                bn_params.append(param)
            else:
                bn_params.append(param)
        return bn_params, non_bn_params

class AuxiliaryHeadCIFAR(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR, self).__init__()
    
    # self.pooling = SNN_Avgpooling(5, stride=3, padding=0)
    self.pooling  = SpikingAvgPool2d(kernel_size=5, stride=3, padding=0)
    self.conv1 = SpikingConv2d(C, 128, 1, padding=0, b=3)
    self.conv2 = SpikingConv2d(128, 768, 2, padding=0, b=3)
    self.classifier = SpikingLinear(768, num_classes)

    # self.conv1 = SNN_2d(C, 128, 1, padding=0, b=3)
    # self.conv2 = SNN_2d(128, 768, 2, padding=0, b=3)
    # self.classifier = SNN_2d_fc(768, num_classes)

  def forward(self, x):
    x = self.pooling(x)
    spike1 = self.conv1(x)
    spike2 = self.conv2(spike1)
    shape = spike2.shape[:2]
    # print(spike2.shape)
    # print(spike2.view(*shape,-1).shape)
    result = self.classifier(spike2.view(*shape,-1))
    return result