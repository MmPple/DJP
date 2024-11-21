import torch
from torch import Tensor, nn
from models.submodules.sparse import ConvBlock, Mask, multi_step_sigmoid
from typing import List
import math

import numpy as np

# def multi_step_sigmoid(x, num_steps=20, steepness=10, scale=1.0):
#     """
#     实现多级阶梯激活函数的替代版本，使用多个Sigmoid函数。
    
#     参数：
#     x (numpy array or scalar): 输入值。
#     num_steps (int): 阶梯数量。
#     steepness (float): Sigmoid的陡峭程度，值越大，Sigmoid过渡越陡峭。
#     scale (float): 输出值的缩放参数。
    
#     返回：
#     numpy array or scalar: 激活函数的输出值。
#     """
#     step_size = 1 / num_steps  # 每一级阶梯的宽度
    
#     output = 0
#     for i in range(num_steps):
#         step_position = i * step_size
#         output += torch.sigmoid(steepness * (x - step_position))
    
#     output = scale * output / num_steps  # 归一化输出值到[0, scale]
#     return output



# def multi_step_sigmoid(x, num_steps=10, steepness=10, scale=1.0):
#     """
#     实现多级阶梯激活函数的替代版本，使用多个Sigmoid函数。
    
#     参数：
#     x (numpy array or scalar): 输入值。
#     num_steps (int): 阶梯数量。
#     steepness (float): Sigmoid的陡峭程度，值越大，Sigmoid过渡越陡峭。
#     scale (float): 输出值的缩放参数。
    
#     返回：
#     numpy array or scalar: 激活函数的输出值。
#     """
#     # 生成所有的阶梯位置（每个阶梯的起始点）
#     step_positions = torch.linspace(0, 1, num_steps, device=x.device)
    
#     # 使用广播机制计算所有的sigmoid值，这里避免了显存占用大的for循环
#     sigmoids = torch.sigmoid(steepness * (x.unsqueeze(-1) - step_positions))
    
#     # 将所有sigmoid值求和，并进行归一化
#     output = scale * sigmoids.mean(dim=-1)
    
    # return output


class PenaltyTerm(nn.Module):
    def __init__(self, model: nn.Module, lmbda: float) -> None:
        super(PenaltyTerm, self).__init__()
        self.layers: List[ConvBlock] = []
        self.model = model
        for m in model.modules():
            if isinstance(m, ConvBlock):
                self.layers.append(m)
        self.lmbda = lmbda
        model.calc_c()

    def forward(self) -> Tensor:
        loss = 0
        for layer in self.layers:
            if layer.sparse_neurons:
                # loss = loss + (self.lmbda * layer.neuron_mask.lmbda) * ( torch.sigmoid(
                #     layer.neuron_mask.mask_value * layer.neuron_mask.temp)).sum()
                loss = loss + (self.lmbda * layer.neuron_mask.lmbda * multi_step_sigmoid(
                    layer.neuron_mask.mask_value, steepness = layer.neuron_mask.temp)).sum()
                # loss = loss + 10 * (self.lmbda * layer.neuron_mask.lmbda) * torch.norm(torch.sigmoid(
                #     layer.neuron_mask.mask_value * layer.neuron_mask.temp), p=2).sum()
            if layer.sparse_weights:
                loss = loss + (self.lmbda * layer.weight_mask.lmbda * torch.sigmoid(
                    layer.weight_mask.mask_value * layer.weight_mask.temp)).sum()
            if(self.model.sparse_f):
                loss = loss + self.lmbda * (self.tlmbda * layer.norm.weight).abs().sum()


            # if self.model.sparse_spikes:
            # print(loss)
            # if layer.sparse_neurons and self.model.sparse_spikes:
                # loss = loss +  * (self.lmbda * layer.neuron_mask.lmbda) * torch.norm(nn.functional.leaky_relu(nn.functional.hardtanh(layer.neuron_mask.mask_value, -1.5, 1.5), negative_slope=0.01), p=2).sum()
                # loss = loss +  * (self.lmbda * layer.neuron_mask.lmbda) * torch.norm(nn.functional.leaky_relu(layer.neuron_mask.mask_value, negative_slope=0.01), p=2).sum()
                # print("now", loss)
                # loss = loss + (self.lmbda * layer.neuron_mask.lmbda) * (torch.sigmoid(
                #     layer.neuron_mask.mask_value * layer.neuron_mask.temp)).sum()

                # if(layer.norm):
                #     loss = loss + self.lmbda * 100000000 * (torch.norm(layer.norm.weight, p=2)).sum()

        return loss

