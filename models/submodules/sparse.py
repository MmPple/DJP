import torch
from torch import nn
from spikingjelly.clock_driven import neuron, functional
from typing import List


def multi_step_sigmoid(x, num_steps=20, steepness=10, scale=1.0):
    """
    实现多级阶梯激活函数的替代版本，使用多个Sigmoid函数。
    
    参数：
    x (numpy array or scalar): 输入值。
    num_steps (int): 阶梯数量。
    steepness (float): Sigmoid的陡峭程度，值越大，Sigmoid过渡越陡峭。
    scale (float): 输出值的缩放参数。
    
    返回：
    numpy array or scalar: 激活函数的输出值。
    """
    step_size = 1 / num_steps  # 每一级阶梯的宽度
    
    output = 0
    for i in range(num_steps):
        step_position = i * step_size
        output += torch.sigmoid(steepness * (x - step_position))
    
    output = scale * output / num_steps  # 归一化输出值到[0, scale]
    return output



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
    
#     return output


class TemporalBatchNorm(nn.Module):
    def __init__(self, num_channels, num_timesteps=4):
        super(TemporalBatchNorm, self).__init__()
        self.num_channels = num_channels
        self.num_timesteps = num_timesteps
        self.temp = 5
        # 定义每个时间步的缩放因子（gamma）和偏置（beta）
        self.weight = nn.Parameter(torch.ones(num_timesteps, num_channels))
        self.bias = nn.Parameter(torch.zeros(num_timesteps, num_channels))
        self.scalarw = nn.Parameter(torch.ones(1, num_channels) * num_timesteps)
        # self.scalarb = nn.Parameter(torch.zeros(1, num_channels) * 10)
        # 定义用于计算均值和标准差的BatchNorm层
        self.bn = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        # x shape: [timesteps, batch_size, channels, height, width]
        if(len(x.size())==5):
            _, timesteps, channels, height, width = x.shape
        else:
            _, channels, height, width = x.shape
        out = self.bn(x)  # BN计算
        if(_ == 1):
            return out
        if(len(x.size())==5):
            weight = (self.scalarw * nn.functional.softmax(self.weight, dim=0)).view(self.num_timesteps, 1, channels, 1, 1)
        # weight = torch.sigmoid(self.temp * self.weight).view(self.num_timesteps, 1, channels, 1, 1)
        # weight = self.weight.view(self.num_timesteps, 1, channels, 1, 1)
        # bias = (self.scalarb * nn.functional.softmax(self.bias, dim=0)).view(self.num_timesteps, 1, channels, 1, 1)
            bias = self.bias.view(self.num_timesteps, 1, channels, 1, 1)
            out = weight * out + bias
            out = out.flatten(0,1)
        
        return out



class Mask(nn.Module):
    def __init__(self, spike_mask=False, neuron = False):
        super().__init__()
        self.mask_value = None
        self.pruning = False
        self.lmbda = 0
        self.temp = 1
        self.spike_mask = spike_mask
        self.neuron = neuron

    def _init_mask(self, shape, mean: float, std: float = 0):
        if(self.spike_mask):
            # self.mask_value = torch.nn.parameter.Parameter(0.5 * torch.ones(shape, device='cuda'))
            self.mask_value = torch.nn.parameter.Parameter(
            torch.normal(mean, std, size=shape, device='cuda'))
            # self.mask_value = torch.nn.parameter.Parameter(
            # torch.normal(mean, std, size=[1,1,shape[2],1,1], device='cuda'))

        else:
            self.mask_value = torch.nn.parameter.Parameter(
            torch.normal(mean, std, size=shape, device='cuda'))
            # self.mask_value = torch.ones(shape, device='cuda').detach()
        return self.mask_value

    def _pruning(self, flag: bool):
        self.pruning = flag

    def _set_temp(self, temp: float):
        self.temp = temp
        if self.neuron:
            self.temp = 5

    def mask(self):
        if self.spike_mask:
            if self.mask_value is None:
                return None
            elif self.pruning:
                # return nn.functional.leaky_relu(nn.functional.hardtanh(self.mask_value.detach(), -1.5, 1.5), negative_slope=0.01)
                # return nn.functional.leaky_relu(self.mask_value.detach(), negative_slope=0.01)
                # return 0.25 * torch.sigmoid(self.temp * self.mask_value.detach())
                return multi_step_sigmoid(self.mask_value.detach(), steepness=self.temp)

            else:
                return torch.where(self.mask_value > 0, 1, 0).float()

        if self.mask_value is None:
            return None
        elif self.pruning:
            return torch.sigmoid(self.temp * self.mask_value.detach())
            # return torch.where(self.mask_value > 0, 1, 0).float()
        else:
            return torch.where(self.mask_value > 0, 1, 0).float()

    def left(self):
        if self.mask_value is None:
            return 0., 0.
        else:
            return self.mask().sum().item(), self.mask_value.numel()

    def forward(self, x):
        if self.spike_mask:
            if self.mask_value is None:
                return x
            elif self.pruning:
                # return nn.functional.leaky_relu(nn.functional.hardtanh(self.mask_value, -1.5, 1.5), negative_slope=0.01) * x
                # return nn.functional.leaky_relu(self.mask_value, negative_slope=0.01) * x
                # return 0.25 * torch.sigmoid(self.temp * self.mask_value) * x
                return multi_step_sigmoid(self.mask_value, steepness=self.temp) * x

            else:
                # return torch.relu(nn.functional.hardtanh(self.mask_value, -1.5, 1.5)) * x
                zeros = torch.zeros(x.shape).cuda(x.device)
                # x = 0.25 * torch.sigmoid(self.temp * self.mask_value) * x 
                x = multi_step_sigmoid(self.mask_value, steepness=self.temp) * x
                return torch.where(self.mask_value > 0, x, zeros)

        if self.mask_value is None:
            return x
        elif self.pruning:
            return torch.sigmoid(self.temp * self.mask_value) * x

            # zeros = torch.zeros(x.shape).cuda(x.device)
            # return torch.where(self.mask_value > 0, x, zeros)

        else:
            zeros = torch.zeros(x.shape).cuda(x.device)
            # x = torch.sigmoid(self.temp * self.mask_value) * x 
            return torch.where(self.mask_value > 0, x, zeros)

    def single_step_forward(self, x):
        if self.mask_value is None:
            return x
        mask_value = self.mask_value
        if len(mask_value.shape) == 5:
            mask_value = self.mask_value.squeeze(0)
        if self.pruning:
            return torch.sigmoid(self.temp * mask_value) * x
            # return multi_step_sigmoid(mask_value, steepness=self.temp) * x

            # zeros = torch.zeros(x.shape).cuda(x.device)
            # return torch.where(mask_value > 0, x, zeros)

        else:
            zeros = torch.zeros(x.shape).cuda(x.device)
            # x =  torch.sigmoid(self.temp * mask_value) * x
            x = multi_step_sigmoid(mask_value, steepness=self.temp) * x
            return torch.where(mask_value > 0, x, zeros)


class ConvBlock(nn.Module):

    spike_list = []

    def __init__(self, conv: nn.Conv2d, norm: nn.BatchNorm2d, node: neuron.BaseNode,
                 static: bool = False, T: int = None, sparse_weights: bool = False,
                 sparse_neurons: bool = False) -> None:
        super(ConvBlock, self).__init__()
        assert isinstance(conv, nn.Conv2d)
        # assert norm is None or isinstance(norm, nn.BatchNorm2d)
        assert node is None or isinstance(node, neuron.BaseNode)
        self.conv = conv
        self.norm = norm
        self.node = node
        self.static = static
        self.T = T
        self.sparse_weights = sparse_weights
        self.sparse_neurons = sparse_neurons
        self.weight_mask = Mask()
        self.neuron_mask = Mask(spike_mask=True, neuron = False)

    def init_mask(self, weights_mean: float, neurons_mean: float, weights_std: float = 0,
                  neurons_std: float = 0):
        masks = []
        if self.sparse_weights:
            masks.append(
                self.weight_mask._init_mask(self.conv.weight.shape, weights_mean, weights_std))
        if self.sparse_neurons:
            masks.append(
                self.neuron_mask._init_mask(
                    self.node.v.unsqueeze(0).shape, neurons_mean, neurons_std))
        return masks

    def _pruning(self, flag: bool):
        if self.sparse_weights:
            self.weight_mask._pruning(flag)
        if self.sparse_neurons:
            self.neuron_mask._pruning(flag)

    def set_temp(self, temp: float):
        self.set_weight_temp(temp)
        self.set_neuron_temp(temp)

        if isinstance(self.norm, TemporalBatchNorm):
            self.norm.temp = temp

    def set_weight_temp(self, temp: float):
        if self.sparse_weights:
            self.weight_mask._set_temp(temp)

    def set_neuron_temp(self, temp: float):
        if self.sparse_neurons:
            self.neuron_mask._set_temp(temp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight_mask(self.conv.weight)
        # self.conv.weight.data = self.weight_mask(self.conv.weight.data)
        # weight = self.conv.weight
        if self.static:
            x = torch.nn.functional.conv2d(x, weight, bias=self.conv.bias, stride=self.conv.stride,
                                           padding=self.conv.padding, dilation=self.conv.dilation,
                                           groups=self.conv.groups)
            if self.norm is not None:
                x = self.norm(x)
            x.unsqueeze_(0)
            x = x.repeat(self.T, 1, 1, 1, 1)
        else:
            x_shape = [x.shape[0], x.shape[1]]
            x = x.flatten(0, 1)
            x = torch.nn.functional.conv2d(x, weight, bias=self.conv.bias, stride=self.conv.stride,
                                           padding=self.conv.padding, dilation=self.conv.dilation,
                                           groups=self.conv.groups)
            if self.norm is not None:
                x = self.norm(x)
            x_shape.extend(x.shape[1:])
            x = x.view(x_shape)
        if self.node is not None:
            x = self.node(x)
        x = self.neuron_mask(x)
        if(self.conv.weight.size(-1) != 1):
            ConvBlock.spike_list.append(x)
        return x

    def connects(self, sparse, dense):
        with torch.no_grad():
            if self.sparse_weights:
                weight = self.weight_mask.mask()
            else:
                weight = torch.ones_like(self.conv.weight)
            sparse = torch.nn.functional.conv2d(sparse, weight, bias=None, stride=self.conv.stride,
                                                padding=self.conv.padding,
                                                dilation=self.conv.dilation,
                                                groups=self.conv.groups)
            dense = torch.nn.functional.conv2d(dense, torch.ones_like(self.conv.weight), bias=None,
                                               stride=self.conv.stride, padding=self.conv.padding,
                                               dilation=self.conv.dilation, groups=self.conv.groups)
            sparse = self.neuron_mask.single_step_forward(sparse)
            conn = sparse.sum().item()
            total = dense.sum().item()
            sparse = (sparse != 0).float()
            if self.sparse_neurons:
                neuron_mask = self.neuron_mask.mask()
                if len(neuron_mask.shape) == 5:
                    neuron_mask.squeeze_(0)
                sparse = sparse * neuron_mask
            dense = torch.ones_like(sparse)
            return conn, total, sparse, dense

    def calc_c(self, x: torch.Tensor, prev_layers: List = []):
        # x: [1, C, H, W]
        assert self.conv.stride[0] == self.conv.stride[1]
        stride = self.conv.stride[0]
        assert self.conv.dilation[0] == self.conv.dilation[1] == 1
        assert self.conv.groups == 1
        with torch.no_grad():
            # calc lambda_n for prev layers
            # weight: [C_out, C_in, h, w]
            self.weight_mask.lmbda = x.shape[2] * x.shape[3] / (stride * stride)
            c_prev = self.conv.weight.shape[0] * self.conv.weight.shape[2] * self.conv.weight.shape[
                3] / (stride * stride)
            for layer in prev_layers:
                layer: ConvBlock
                # reshape or flatten
                layer.neuron_mask.lmbda += c_prev
                layer.tlmbda = layer.neuron_mask.lmbda
            y = torch.nn.functional.conv2d(x, self.conv.weight, None, self.conv.stride,
                                           self.conv.padding, self.conv.dilation, self.conv.groups)
            return torch.zeros_like(y)
