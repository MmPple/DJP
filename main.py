import logging
import os
import random
from typing import Union
import torch
import torch.utils.data
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from thop import profile
import math
from math import ceil, sqrt
from torch.cuda import amp
import torch.distributed
import argparse
import matplotlib

import matplotlib.pyplot as plt
from matplotlib import colors

from models import cifar10dvs, sew_resnet, cifar10
from models.submodules.sparse import ConvBlock, Mask, TemporalBatchNorm
from sparsity.penalty_term import PenaltyTerm
from sparsity.temp_scheduler import SplitTemperatureScheduler, TemperatureScheduler
# from models import spiking_resnet, sew_resnet
from utils import RecordDict, GlobalTimer, Timer
from utils import DatasetSplitter, CriterionWarpper, DVStransform, SOPMonitor, CIFAR10Policy, Cutout, Augment, DatasetWarpper
from utils import left_neurons, left_weights, init_mask, set_pruning_mode, update_lmbda
from utils import is_main_process, save_on_master, search_tb_record, finetune_tb_record, accuracy, safe_makedirs
from spikingjelly.clock_driven import functional


def parse_args():
    parser = argparse.ArgumentParser(description='Training')

    # training options
    parser.add_argument('--seed', default=12450, type=int)
    parser.add_argument('-gpu', '--GPUID', default=0, type=int)
    parser.add_argument('--epoch-search', default=280, type=int)
    parser.add_argument('--epoch-finetune', default=40, type=int,
                        help='when to fine tune, -1 means will not fine tune')
    parser.add_argument('--not-prune-weight', action='store_true')
    parser.add_argument('--not-prune-neuron', action='store_true')
    parser.add_argument('--prune-f', action='store_true')
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument('--T', default=4, type=int, help='simulation steps')
    parser.add_argument('--model', default='sew_resnet18_cifar', help='model type')
    parser.add_argument('--dataset', default='CIFAR100', help='dataset type')
    parser.add_argument('--augment', action='store_true', help='Additional augment')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--search-lr', default=0.025, type=float, help='initial learning rate')
    parser.add_argument('--finetune-lr', default=0.025, type=float, help='finetune learning rate')
    parser.add_argument('--prune-lr', type=float, help='initial learning rate of pruning')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Adam or SGD')
    parser.add_argument('--prune-optimizer', type=str, help='Adam or SGD')
    parser.add_argument('--weight-decay', default=0, type=float, help='weight decay (default: 0)')
    parser.add_argument('--prune-weight-decay', default=0, type=float)
    parser.add_argument('--criterion', type=str, default='CE', help='MSE or CE')
    parser.add_argument(
        '--search-lr-scheduler', type=str, nargs='+', default=['Cosine'],
        help='''--lr-scheduler Cosine [<T0> <Tt> <Tmax(period of cosine)>]
            or --lr-scheduler Step [minestones]...''')
    parser.add_argument(
        '--finetune-lr-scheduler', type=str, nargs='+', default=['Cosine'],
        help='''--lr-scheduler Cosine [<T0> <Tt> <Tmax(period of cosine)>]
            or --lr-scheduler Step [minestones]...''')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='Number of times a debug message is printed in one epoch')
    parser.add_argument('--tb-interval', type=int, default=10)
    parser.add_argument('--data-path', default='./datasets', help='dataset')
    parser.add_argument('--output-dir', default='./logs/temp')
    parser.add_argument('--resume', type=str, help='resume from checkpoint')
    parser.add_argument('--resume-type', type=str, default='test', help='search, finetune or test')
    parser.add_argument('--distributed-init-mode', type=str, default='env://')

    # mask init
    parser.add_argument(
        '--mask-init-factor', type=float, nargs='+', default=[0, 0, 0, 0],
        help='--mask-init-factor <weights mean> <neurons mean> <weights std> <neurons std>')

    # penalty term
    parser.add_argument('--penalty-lmbda', type=float, default=1e-9)

    parser.add_argument(
        '--temp-scheduler', type=float, nargs='+', default=[5, 1000],
        help='''--temp-scheduler <init temp> <final temp>
                or --temp-scheduler <init temp> <final temp> <T0> <Tmax>
                or --temp-scheduler <init temp of weight> <init temp of neuron> 
                <final temp of weight> <final temp of neuron> <T0> <Tmax>''')
    # deprecated
    parser.add_argument('--accumulate-step', type=int, default=1)

    # argument of sew resnet
    parser.add_argument('--zero-init-residual', action='store_true',
                        help='zero init all residual blocks')
    parser.add_argument(
        "--cache-dataset", action="store_true",
        help="Cache the datasets for quicker initialization. It also serializes the transforms")
    parser.add_argument("--sync-bn", action="store_true", help="Use sync batch norm")
    parser.add_argument("--test-only", action="store_true", help="Only test the model")
    parser.add_argument('--amp', action='store_true', help='Use AMP training')

    # argument of TET
    parser.add_argument('--TET', action='store_true', help='Use TET training')
    parser.add_argument('--TET-phi', type=float, default=1.0)
    parser.add_argument('--TET-lambda', type=float, default=0.0)

    parser.add_argument('--save-latest', action='store_true')

    args = parser.parse_args()
    return args


def setup_logger(output_dir):
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s]%(message)s',
                                  datefmt=r'%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(os.path.join(output_dir, 'log.log'))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    return logger


def init_distributed(logger: logging.Logger, distributed_init_mode):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        logger.info('Not using distributed mode')
        return False, 0, 1, 0

    torch.cuda.set_device(local_rank)
    backend = 'nccl'
    logger.info('Distributed init rank {}'.format(rank))
    torch.distributed.init_process_group(backend=backend, init_method=distributed_init_mode,
                                         world_size=world_size, rank=rank)
    # only master process logs
    if rank != 0:
        logger.setLevel(logging.WARNING)
    return True, rank, world_size, local_rank


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


class Pruner():
    curr_prune_iter = 0
    total_prune_iter = 40

    def _normalize_scores(scores):
        """
        Normalizing scheme for LAMP.
        """
        # sort scores in an ascending order
        sorted_scores,sorted_idx = scores.view(-1).sort(descending=False)
        # compute cumulative sum
        scores_cumsum_temp = sorted_scores.cumsum(dim=0)
        scores_cumsum = torch.zeros(scores_cumsum_temp.shape,device=scores.device)
        scores_cumsum[1:] = scores_cumsum_temp[:len(scores_cumsum_temp)-1]
        # normalize by cumulative sum
        sorted_scores /= (scores.sum() - scores_cumsum)
        # tidy up and output
        new_scores = torch.zeros(scores_cumsum.shape,device=scores.device)
        new_scores[sorted_idx] = sorted_scores
        
        return new_scores.view(scores.shape)


    def weight_pruning(net, pruning, epoch):
        l2 = [np.log(x) for x in l2]
        if(not pruning or not (epoch % 7 == 0)):
            return
        if(Pruner.curr_prune_iter > Pruner.total_prune_iter):
            return
        curr_prune_iter = Pruner.curr_prune_iter
        total_prune_iter = Pruner.total_prune_iter
        prune_decay = (1 - (curr_prune_iter / total_prune_iter)) ** 3
        curr_prune_rate =  0.95 * (1 - prune_decay)
        
        print(f"---- {curr_prune_iter}nd Pruning----    Prune Rate:{curr_prune_rate} ")

        weight_abs = []
        cnt = 0
        for name, m in net.named_modules():
                if name in net.skip:
                    continue            
                if isinstance(m, ConvBlock):
                    # weight_abs.append(Pruner._normalize_scores(m.conv.weight)**2)
                    # weight_abs.append(m.conv.weight.abs() * (1.0/l2[cnt]))
                    weight_abs.append(m.conv.weight.abs())
                    cnt += 1
        
        all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
        num_params_to_keep = int(len(all_scores) * (1 - curr_prune_rate))
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]

        cnt = 0
        for name, m in net.named_modules():
                if name in net.skip:
                    continue            
                if isinstance(m, ConvBlock):
                    m.weight_mask.mask_value = ((weight_abs[cnt]) > acceptable_score).float().detach()
                    cnt += 1
        Pruner.curr_prune_iter += 1
        return 


import torch
import math
import numpy as np

class newPruner():
    curr_prune_iter = 0
    total_prune_iter = 40
    initial_rate = 0.5  # 初始的剪枝率
    T_max = 40  # 余弦衰减的总周期数
    eta_min = 0.005  # 余弦衰减的最小学习率

    # 初始化余弦衰减参数
    rate = initial_rate

    @staticmethod
    def update_rate():
        # 余弦衰减计算 Pruner.rate 的新值
        cosine_decay = 0.5 * (1 + math.cos(math.pi * Pruner.curr_prune_iter / Pruner.T_max))
        Pruner.rate = (Pruner.initial_rate - Pruner.eta_min) * cosine_decay + Pruner.eta_min
        

    @staticmethod
    def weight_pruning(net, pruning, epoch):
        layer_frequency_list  = [np.log(x) for x in layer_frequency_list]
    
        if not pruning or not (epoch % 7 == 0):
            return
        if Pruner.curr_prune_iter > Pruner.total_prune_iter:
            return
        
        # 更新剪枝率
        Pruner.update_rate()

        curr_prune_iter = Pruner.curr_prune_iter
        total_prune_iter = Pruner.total_prune_iter
        prune_decay = (1 - (curr_prune_iter / total_prune_iter)) ** 3
        curr_prune_rate = 0.95 * (1 - prune_decay)
        
        print(f"---- {curr_prune_iter}nd Pruning----    Prune Rate:{curr_prune_rate}, Extra Prune Rate:{Pruner.rate} ")

        '''
        # Step 1: Initial pruning based on magnitude
        weight_abs = []
        num_nonzeros_dict = {}  # 用于保存每层的 num_nonzeros
        cnt = 0
        for name, m in net.named_modules():
            if name in net.skip:
                continue
            if isinstance(m, ConvBlock):
                weight_abs.append(m.conv.weight.abs())
                cnt += 1

        all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
        num_params_to_keep = int(len(all_scores) * (1 - curr_prune_rate))
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]

        cnt = 0
        for name, m in net.named_modules():
            if name in net.skip:
                continue
            if isinstance(m, ConvBlock):
                mask = (weight_abs[cnt] > acceptable_score).float().detach()
                m.weight_mask.mask_value = mask

                # 保存初始剪枝后的 num_nonzeros
                num_nonzeros_dict[name] = mask.sum().item()
                cnt += 1
        '''

        weight_abs = []
        num_nonzeros_dict = {}  # 用于保存每层的 num_nonzeros
        cnt = 0

        # Step 1: 收集所有的 weight 绝对值
        for name, m in net.named_modules():
            if name in net.skip:
                continue
            if isinstance(m, ConvBlock):
                weight_abs.append(m.conv.weight.abs())
                cnt += 1

        # Step 2: 获取所有权重的最大值和最小值
        all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
        max_val = torch.max(all_scores)
        min_val = torch.min(all_scores)

        # Step 3: 定义频率的映射表
        frequency_mapping = {1024.0: 1, 256.0: 2, 64.0: 3, 16.0: 4}

        # 计算排序值
        def get_sorted_value(weight, frequency, max_val, min_val):
            normalized_weight = (weight - min_val) / (max_val - min_val)  # 归一化到 0 到 1
            # 将归一化后的值映射到 0-99 级
            level = (normalized_weight * 50).clamp(0, 49).floor()  # 使用 clamp 确保范围，并 floor() 转为整数
            frequency_score = frequency_mapping[frequency]  # 直接使用映射后的频率得分
            remaining_value = normalized_weight - (level / 50)  # 取出剩余的部分
            # 组合成排序值5
            sorted_value = level + frequency_score * 1e-1 + remaining_value
            return sorted_value



        # 计算每层的排序值
        sorted_values = []
        for i, weight in enumerate(weight_abs):
            frequency = layer_frequency_list[i]
            sorted_value = get_sorted_value(weight, frequency, max_val, min_val)
            sorted_values.append(sorted_value)
        weight_abs = sorted_values

        # Step 4: 使用 acceptable_score 对权重进行剪枝
        all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
        num_params_to_keep = int(len(all_scores) * (1 - curr_prune_rate))
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]

        cnt = 0
        for name, m in net.named_modules():
            if name in net.skip:
                continue
            if isinstance(m, ConvBlock):
                # 根据三级排序值 mask
                mask = (weight_abs[cnt] > acceptable_score).float().detach()
                m.weight_mask.mask_value = mask

                # 保存初始剪枝后的 num_nonzeros
                num_nonzeros_dict[name] = mask.sum().item()
                cnt += 1

        '''
        # Step 2: Extra pruning to make room for regrowth
        cnt = 0
        for name, m in net.named_modules():
            if name in net.skip:
                continue
            if isinstance(m, ConvBlock):
                weight = weight_abs[cnt]
                mask = m.weight_mask.mask_value

                # 读取初始剪枝后的 num_nonzeros
                num_nonzeros = num_nonzeros_dict[name]
                num_zeros = mask.numel() - num_nonzeros

                # Perform extra pruning (magnitude death) based on updated non-zero count
                num_remove = math.ceil(Pruner.rate * num_nonzeros)
                if num_remove > 0:
                    k = math.ceil(num_zeros + num_remove)
                    x, _ = torch.sort(weight.view(-1))
                    threshold = x[k-1].item()
                    extra_mask = (torch.abs(weight.data) > threshold).float()
                else:
                    extra_mask = weight.data != 0.0
                
                m.weight_mask.mask_value = extra_mask.float()
                cnt += 1

        # Step 3: Regrow weights based on gradient information
        for name, m in net.named_modules():
            if name in net.skip:
                continue
            if isinstance(m, ConvBlock):
                weight = m.conv.weight
                mask = m.weight_mask.mask_value

                # 读取初始剪枝后的 num_nonzeros
                num_nonzeros = num_nonzeros_dict[name]

                # 计算 total_regrowth
                total_regrowth = num_nonzeros - mask.sum().item()

                # Get gradients for weights
                grad = weight.grad.clone() if weight.grad is not None else torch.zeros_like(weight)
                grad = grad * (mask == 0).float()  # Consider only pruned weights

                # Select weights to regrow based on the highest gradients
                _, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
                regrowth_indices = idx[:int(total_regrowth)]
                mask.view(-1)[regrowth_indices] = 1.0

                m.weight_mask.mask_value = mask.float()
        '''
        Pruner.curr_prune_iter += 1


def load_data(dataset_dir, cache_dataset, dataset_type, distributed: bool, augment: bool,
              logger: logging.Logger, T: int):

    if dataset_type == 'CIFAR10':
        if augment:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                Cutout(n_holes=1, length=16), ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

        dataset = torchvision.datasets.CIFAR10(root=os.path.join(dataset_dir), train=True,
                                               download=True)
        dataset_test = torchvision.datasets.CIFAR10(root=os.path.join(dataset_dir), train=False,
                                                    download=True)
    elif dataset_type == 'CIFAR100':
        if augment:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]],
                                     std=[n / 255. for n in [68.2, 65.4, 70.4]]),
                Cutout(n_holes=1, length=8), ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]],
                                     std=[n / 255. for n in [68.2, 65.4, 70.4]]), ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]],
                                 std=[n / 255. for n in [68.2, 65.4, 70.4]]), ])

        dataset = torchvision.datasets.CIFAR100(root=os.path.join(dataset_dir), train=True,
                                                download=True)
        dataset_test = torchvision.datasets.CIFAR100(root=os.path.join(dataset_dir), train=False,
                                                     download=True)
    elif dataset_type == 'CIFAR10DVS':
        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
        if augment:
            transform_train = DVStransform(transform=transforms.Compose([
                transforms.Resize(size=(48, 48), antialias=True),
                Augment()]))
        else:
            transform_train = DVStransform(
                transform=transforms.Compose([transforms.Resize(size=(48, 48), antialias=True)]))
        transform_test = DVStransform(transform=transforms.Resize(size=(48, 48), antialias=True))

        dataset = CIFAR10DVS(dataset_dir, data_type='frame', frames_number=T, split_by='number')
        dataset, dataset_test = DatasetSplitter(dataset, 0.9,
                                                True), DatasetSplitter(dataset, 0.1, False)
    elif dataset_type == 'ImageNet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        logger.info('Loading training data')
        traindir = os.path.join(dataset_dir, 'train')
        valdir = os.path.join(dataset_dir, 'val')
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize, ])
        transform_test = transforms.Compose([
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize, ])
        with Timer('Load training data', logger):
            cache_path = _get_cache_path(traindir)
            if cache_dataset and os.path.exists(cache_path):
                # Attention, as the transforms are also cached!
                dataset, _ = torch.load(cache_path)
                logger.info("Loaded training dataset from {}".format(cache_path))
            else:
                dataset = torchvision.datasets.ImageFolder(traindir)
                if cache_dataset:
                    safe_makedirs(os.path.dirname(cache_path))
                    save_on_master((dataset, traindir), cache_path)
                    logger.info("Cached training dataset to {}".format(cache_path))
                logger.info("Loaded training dataset")

        logger.info("Loading validation data")
        with Timer('Load validation data', logger):
            cache_path = _get_cache_path(valdir)
            if cache_dataset and os.path.exists(cache_path):
                # Attention, as the transforms are also cached!
                dataset_test, _ = torch.load(cache_path)
                logger.info("Loaded test dataset from {}".format(cache_path))
            else:
                dataset_test = torchvision.datasets.ImageFolder(valdir)
                if cache_dataset:
                    safe_makedirs(os.path.dirname(cache_path))
                    save_on_master((dataset_test, valdir), cache_path)
                    logger.info("Cached test dataset to {}".format(cache_path))
                logger.info("Loaded test dataset")
    else:
        raise ValueError(dataset_type)

    dataset_train = DatasetWarpper(dataset, transform_train)
    dataset_test = DatasetWarpper(dataset_test, transform_test)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset_train, dataset_test, train_sampler, test_sampler


def train_one_epoch(model, criterion, penalty_term, optimizer_train, optimizer_prune,
                    data_loader_train, temp_scheduler, logger, epoch, print_freq, factor,
                    scaler=None, accumulate_step=1, prune=False, one_hot=None, TET=False):
    model.train()
    metric_dict = RecordDict({'loss': None, 'acc@1': None, 'acc@5': None})
    timer_container = [0.0]

    set_pruning_mode(model, prune)
    model.zero_grad()
    for idx, (image, target) in enumerate(data_loader_train):
        with GlobalTimer('iter', timer_container):
            image, target = image.float().cuda(), target.cuda()
            if scaler is not None:
                with amp.autocast():
                    output = model(image)
                    if one_hot:
                        loss = criterion(output, F.one_hot(target, one_hot).float())
                    else:
                        loss = criterion(output, target)
            else:
                output = model(image)
                if one_hot:
                    loss = criterion(output, F.one_hot(target, one_hot).float())
                else:
                    loss = criterion(output, target)
            metric_dict['loss'].update(loss.item())
            if prune:
                loss = loss + penalty_term()
            

            if temp_scheduler is not None:
                # model.temp_value = temp_scheduler.get_temp()
                model.temp_value = temp_scheduler.linear_factor()
            
            if(model.sparse_f):               
                l2_coefficients = []
                # for i, s in enumerate(model.spike_list):
                for i, s in enumerate(ConvBlock.spike_list):
                    # [1e-7 1, big 10, normal 20, 5e-10 200]
                    # loss = loss + (1 * model.temp_value) * 1e-12 * l2_coefficients[i] * (s * s).sum()
                    # loss = loss + (1 * model.temp_value) * 5e-11 * l2_coefficients[i] * (torch.norm(s, p=2, dim=[3,4])**2).sum()
                    loss = loss + (1 * model.temp_value) * 1e-11 * l2_coefficients[i] * (s**2).sum()
                    # loss = loss + (1 * model.temp_value) * 5e-8 * l2_coefficients[i] * torch.norm(s.sum(dim=0), p=2)
            # if(model.sparse_t):               
                # l2_coefficients = [2304, 2304, 2304, 2304, 2304, 2048, 100]
                # for m in model.modules():
                    # [1e-7 1, big 10, normal 20, 5e-10 200]
                    # if isinstance(m, (TemporalBatchNorm)):
                        # loss = loss + (1 * model.temp_value) * penalty_term.lmbda * (torch.norm(m.scalarw, p=2) ** 2)
                        # loss = loss + (1 * model.temp_value) * 1e-10 * (torch.norm(m.weight.sum(axis=[0]), p=2) ** 2)


            loss = loss / accumulate_step

            if scaler is not None:
                scaler.scale(loss).backward()
                if (idx + 1) % accumulate_step == 0:
                    if prune:
                        scaler.step(optimizer_prune)
                    scaler.step(optimizer_train)
                    scaler.update()

                    # if(idx == (len(data_loader_train)-1)):
                    #     Pruner.weight_pruning(model, prune, epoch)

                    model.zero_grad()
                    if temp_scheduler is not None:
                        temp_scheduler.step()

            else:
                loss.backward()
                if (idx + 1) % accumulate_step == 0:
                    if prune:
                        optimizer_prune.step()
                    optimizer_train.step()

                    # if(idx == (len(data_loader_train)-1)):
                    #     Pruner.weight_pruning(model, prune, epoch)

                    model.zero_grad()
                    if temp_scheduler is not None:
                        temp_scheduler.step()
            
            functional.reset_net(model)
            acc1, acc5 = accuracy(output.mean(0), target, topk=(1, 5))
            acc1_s = acc1.item()
            acc5_s = acc5.item()

            batch_size = image.shape[0]
            metric_dict['acc@1'].update(acc1_s, batch_size)
            metric_dict['acc@5'].update(acc5_s, batch_size)


        if print_freq != 0 and ((idx + 1) % int(len(data_loader_train) / (print_freq))) == 0:
            #torch.distributed.barrier()
            metric_dict.sync()
            logger.debug(' [{}/{}] it/s: {:.5f}, loss: {:.5f}, acc@1: {:.5f}, acc@5: {:.5f}'.format(
                idx + 1, len(data_loader_train),
                (idx + 1) * batch_size * factor / timer_container[0], metric_dict['loss'].ave,
                metric_dict['acc@1'].ave, metric_dict['acc@5'].ave))
            # for i, s in enumerate(model.spike_list):
            #     print(i, " layer: ", s.sum().item())
    

    #torch.distributed.barrier()
    metric_dict.sync()
    return metric_dict['loss'].ave, metric_dict['acc@1'].ave, metric_dict['acc@5'].ave


def evaluate(model, criterion, data_loader, print_freq, logger, prune, one_hot, args, over_se = False):
    model.eval()
    set_pruning_mode(model, prune)
    mon = SOPMonitor(model)
    mon.enable()
    sops = 0
    metric_dict = RecordDict({'loss': None, 'acc@1': None, 'acc@5': None})
    with torch.no_grad():
        for idx, (image, target) in enumerate(data_loader):
            image = image.float().to(torch.device('cuda'), non_blocking=True)
            target = target.to(torch.device('cuda'), non_blocking=True)
            output = model(image)
            if one_hot:
                loss = criterion(output, F.one_hot(target, one_hot).float())
            else:
                loss = criterion(output, target)
            metric_dict['loss'].update(loss.item())
            functional.reset_net(model)

            acc1, acc5 = accuracy(output.mean(0), target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_dict['acc@1'].update(acc1.item(), batch_size)
            metric_dict['acc@5'].update(acc5.item(), batch_size)
                        
            if print_freq != 0 and ((idx + 1) % int(len(data_loader) / print_freq)) == 0:
                #torch.distributed.barrier()
                metric_dict.sync()
                logger.debug(' [{}/{}] loss: {:.5f}, acc@1: {:.5f}, acc@5: {:.5f}'.format(
                    idx + 1, len(data_loader), metric_dict['loss'].ave, metric_dict['acc@1'].ave,
                    metric_dict['acc@5'].ave))
        if(over_se):
            for name in mon.monitored_layers:
                sublist = mon[name]
                sop = torch.cat(sublist).mean().item()
                sops = sops + sop
            sops = sops / (1000**2)
            # input is [N, C, H, W] or [T*N, C, H, W]
            sops = sops / args.batch_size
            logger.info('Avg SOPs: {:.5f} M'.format(sops))
        mon.remove_hooks()
        #torch.distributed.barrier()
        metric_dict.sync()
    return metric_dict['loss'].ave, metric_dict['acc@1'].ave, metric_dict['acc@5'].ave, sops


def main():

    ##################################################
    #                       setup
    ##################################################

    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    safe_makedirs(args.output_dir)
    logger = setup_logger(args.output_dir)

    distributed, rank, world_size, local_rank = init_distributed(logger, args.distributed_init_mode)

    logger.info(str(args))
    
    torch.cuda.set_device(args.GPUID)

    # load data

    dataset_type = args.dataset
    if dataset_type == 'CIFAR10' or dataset_type == 'CIFAR10DVS':
        num_classes = 10
    elif dataset_type == 'CIFAR100':
        num_classes = 100
    elif dataset_type == 'ImageNet':
        num_classes = 1000

    dataset_train, dataset_test, train_sampler, test_sampler = load_data(
        args.data_path, args.cache_dataset, dataset_type, distributed, args.augment, logger, args.T)
    logger.info('dataset_train: {}, dataset_test: {}'.format(len(dataset_train), len(dataset_test)))

    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                                    sampler=train_sampler, num_workers=args.workers,
                                                    pin_memory=True, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                                   sampler=test_sampler, num_workers=args.workers,
                                                   pin_memory=True, drop_last=False)

    # model

    model: Union[cifar10.Cifar10Net, cifar10dvs.VGGSNN, sew_resnet.SEWResNet_ImageNet,
                 sew_resnet.SEWResNet_CIFAR, sew_resnet.ResNet19]
    if args.model in cifar10.__dict__:
        model = cifar10.__dict__[args.model](T=args.T, num_classes=num_classes).cuda()
    elif args.model in cifar10dvs.__dict__:
        model = cifar10dvs.__dict__[args.model]().cuda()
    elif args.model in sew_resnet.__dict__:
            model = sew_resnet.__dict__[args.model](zero_init_residual=args.zero_init_residual,
                                                T=args.T, num_classes=num_classes, norm_layer=TemporalBatchNorm).cuda()

    else:
        raise NotImplementedError(args.model)

    if args.not_prune_weight:
        for m in model.modules():
            if isinstance(m, ConvBlock):
                m.sparse_weights = False
    if args.not_prune_neuron:
        for m in model.modules():
            if isinstance(m, ConvBlock):
                m.sparse_neurons = False

    model.cuda()
    if distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # optimzer

    param_without_masks = list(model.parameters())

    if args.optimizer == 'SGD':
        optimizer_train = torch.optim.SGD(param_without_masks, lr=args.search_lr, momentum=0.9,
                                          weight_decay=args.weight_decay, nesterov=True)
    elif args.optimizer == 'Adam':
        optimizer_train = torch.optim.Adam(param_without_masks, lr=args.search_lr,
                                           betas=(0.9, 0.999), weight_decay=args.weight_decay)
    else:
        raise ValueError(args.optimizer)

    # init mask
    set_pruning_mode(model, True)
    model.sparse_f = False
    if dataset_type == 'CIFAR10' or dataset_type == 'CIFAR100':
        inputs = torch.rand(1, 3, 32, 32).cuda()
    elif dataset_type == 'CIFAR10DVS':
        inputs = torch.rand(1, 1, 2, 48, 48).cuda()
    elif dataset_type == 'ImageNet':
        inputs = torch.rand(1, 3, 224, 224).cuda()
    _ = model(inputs)

    masks = init_mask(model, *args.mask_init_factor)
    set_pruning_mode(model, False)
    functional.reset_net(model)

    if not (args.not_prune_weight and args.not_prune_neuron):
        if args.prune_optimizer is None:
            args.prune_optimizer = args.optimizer
        if args.prune_lr is None:
            args.prune_lr = args.search_lr
        if args.prune_optimizer == 'SGD':
            optimizer_prune = torch.optim.SGD(masks, lr=args.prune_lr, momentum=0.9,
                                              weight_decay=args.prune_weight_decay, nesterov=True)
        elif args.prune_optimizer == 'Adam':
            optimizer_prune = torch.optim.Adam(masks, lr=args.prune_lr, betas=(0.9, 0.999),
                                               weight_decay=args.prune_weight_decay)
        else:
            raise ValueError(args.prune_optimizer)

    # loss_fn

    if dataset_type == 'CIFAR10' or dataset_type == 'CIFAR10DVS':
        one_hot = 10
    elif dataset_type == 'CIFAR100':
        one_hot = 100
    elif dataset_type == 'ImageNet':
        one_hot = None

    if args.criterion == 'MSE':
        criterion = nn.MSELoss()
    elif args.criterion == 'CE':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(args.criterion)
    criterion = CriterionWarpper(criterion, args.TET, args.TET_phi, args.TET_lambda)

    # penalty term

    if not (args.not_prune_weight and args.not_prune_neuron):
        penalty_term = PenaltyTerm(model, args.penalty_lmbda)

    # amp speed up

    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None

    # lr scheduler

    milestones = []
    lr_scheduler_train, lr_scheduler_prune = None, None
    lr_scheduler_T0, lr_scheduler_Tmax = 0, args.epoch_search
    if not (args.not_prune_weight and args.not_prune_neuron):
        if len(args.search_lr_scheduler) != 0:
            if args.search_lr_scheduler[0] == 'Step':
                for i in range(1, len(args.search_lr_scheduler)):
                    milestones.append(int(args.search_lr_scheduler[i]))
                lr_scheduler_train = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer=optimizer_train, milestones=milestones, gamma=0.1)
                lr_scheduler_prune = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer=optimizer_prune, milestones=milestones, gamma=0.1)
            elif args.search_lr_scheduler[0] == 'Cosine':
                if len(args.search_lr_scheduler) > 1:
                    lr_scheduler_T0, lr_scheduler_Tmax, T_max = int(
                        args.search_lr_scheduler[1]), int(args.search_lr_scheduler[2]), int(
                            args.search_lr_scheduler[3])
                else:
                    T_max = lr_scheduler_Tmax - lr_scheduler_T0
                print("set CosineLR")
                lr_scheduler_train = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer_train, T_max=T_max)
                lr_scheduler_prune = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer_prune, T_max=T_max)
            else:
                raise ValueError(args.search_lr_scheduler)

    # DDP

    model_without_ddp = model
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                          find_unused_parameters=True)
        model_without_ddp = model.module

    # threshold scheduler

    if not (args.not_prune_weight and args.not_prune_neuron):
        iter_per_epoch = len(data_loader_train) // args.accumulate_step
        if len(args.temp_scheduler) == 2:
            (args.temp_scheduler).append(0)
            (args.temp_scheduler).append(args.epoch_search)
        if len(args.temp_scheduler) == 4:
            temp_scheduler = TemperatureScheduler(model, args.temp_scheduler[0],
                                                  args.temp_scheduler[1],
                                                  int(args.temp_scheduler[2]) * iter_per_epoch,
                                                  int(args.temp_scheduler[3]) * iter_per_epoch)
        elif len(args.temp_scheduler) == 6:
            temp_scheduler = SplitTemperatureScheduler(model, args.temp_scheduler[0],
                                                       args.temp_scheduler[1],
                                                       args.temp_scheduler[2],
                                                       args.temp_scheduler[3],
                                                       int(args.temp_scheduler[4]) * iter_per_epoch,
                                                       int(args.temp_scheduler[5]) * iter_per_epoch)
        else:
            raise ValueError(args.temp_scheduler)

    # resume

    if args.resume and args.resume_type == 'search':
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer_train.load_state_dict(checkpoint['optimizer_train'])
        optimizer_prune.load_state_dict(checkpoint['optimizer_prune'])
        start_epoch = checkpoint['epoch']
        max_acc1 = checkpoint['max_acc1']
        if lr_scheduler_train is not None:
            lr_scheduler_train.load_state_dict(checkpoint['lr_scheduler_train'])
            lr_scheduler_prune.load_state_dict(checkpoint['lr_scheduler_prune'])
        logger.info('Resume from epoch {}'.format(start_epoch))
        start_epoch += 1
        temp_scheduler.current_step = start_epoch * len(data_loader_train)
    else:
        start_epoch = 0
        max_acc1 = 0

    logger.debug(str(model))

    ##################################################
    #                   test only
    ##################################################

    if args.test_only:
        if args.resume and args.resume_type == 'test':
            checkpoint = torch.load(args.resume, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])
        logger.info('Test start')
        if is_main_process():
            test(model, dataset_type, data_loader_test, inputs, args, logger)
        return

    ##################################################
    #                   search
    ##################################################

    # tb_writer = None
    # if is_main_process():
    #     tb_writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'),
    #                               purge_step=start_epoch)

    logger.info("Search start")
    for epoch in range(start_epoch, args.epoch_search):
        if args.resume and args.resume_type == 'finetune':
            break
        if distributed:
            train_sampler.set_epoch(epoch)
        logger.info('Epoch [{}] Start, lr {:.6f}, {}'.format(epoch,
                                                             optimizer_train.param_groups[0]["lr"],
                                                             str(temp_scheduler)))

        with Timer(' Train', logger):
            logger.debug('[Training]')
            train_loss, train_acc1, train_acc5 = train_one_epoch(
                model, criterion, penalty_term, optimizer_train, optimizer_prune, data_loader_train,
                temp_scheduler, logger, epoch, args.print_freq, world_size, scaler,
                args.accumulate_step, True, one_hot)
            if lr_scheduler_train is not None and lr_scheduler_T0 <= epoch < lr_scheduler_Tmax:
                lr_scheduler_train.step()
                lr_scheduler_prune.step()
        
        set_pruning_mode(model, False)

        logger.debug('[Test]')
        test_loss_s, test_acc1_s, test_acc5_s, _ = evaluate(model, criterion, data_loader_test,
                                                            args.print_freq, logger, False,
                                                            one_hot, args)
        set_pruning_mode(model, True)

        logger.info(' Test  Acc@1: {:.5f}, Acc@5: {:.5f}'.format(
            test_acc1_s, test_acc5_s))

        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer_train': optimizer_train.state_dict(),
            'optimizer_prune': optimizer_prune.state_dict(),
            'epoch': epoch,
            'max_acc1': max_acc1, }
        if lr_scheduler_train is not None:
            checkpoint['lr_scheduler_train'] = lr_scheduler_train.state_dict()
            checkpoint['lr_scheduler_prune'] = lr_scheduler_prune.state_dict()

        if args.save_latest:
            save_on_master(checkpoint, os.path.join(args.output_dir, 'checkpoint_latest.pth'))

        if (epoch + 1) == args.epoch_search:
            save_on_master(checkpoint, os.path.join(args.output_dir, 'checkpoint_sparsified.pth'))

    logger.info('Search finish.')

    ##################################################
    #                   finetune
    ##################################################

    ##### reset utils #####

    # reset lr
    if args.finetune_lr is None:
        args.finetune_lr = args.search_lr
    for param_group in optimizer_train.param_groups:
        param_group['lr'] = args.finetune_lr

    # lr scheduler

    milestones = []
    lr_scheduler_train = None
    lr_scheduler_T0, lr_scheduler_Tmax = 0, args.epoch_finetune
    if len(args.finetune_lr_scheduler) != 0:
        if args.finetune_lr_scheduler[0] == 'Step':
            for i in range(1, len(args.finetune_lr_scheduler)):
                milestones.append(int(args.finetune_lr_scheduler[i]))
            lr_scheduler_train = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_train,
                                                                      milestones=milestones,
                                                                      gamma=0.1)
        elif args.finetune_lr_scheduler[0] == 'Cosine':
            if len(args.finetune_lr_scheduler) > 1:
                lr_scheduler_T0, lr_scheduler_Tmax, T_max = int(args.finetune_lr_scheduler[1]), int(
                    args.finetune_lr_scheduler[2]), int(args.finetune_lr_scheduler[3])
            else:
                T_max = lr_scheduler_Tmax - lr_scheduler_T0
            lr_scheduler_train = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer_train, T_max=T_max)
        else:
            raise ValueError(args.finetune_lr_scheduler)

    # resume

    if args.resume and args.resume_type == 'finetune':
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer_train.load_state_dict(checkpoint['optimizer_train'])
        start_epoch = checkpoint['epoch']
        max_acc1 = checkpoint['max_acc1']
        if lr_scheduler_train is not None:
            lr_scheduler_train.load_state_dict(checkpoint['lr_scheduler_train'])
        logger.info('Resume from epoch {}'.format(start_epoch))
        start_epoch += 1
    else:
        start_epoch = 0

    ##### finetune #####

    min_sop = 0
    logger.info("Finetune start")
    for epoch in range(start_epoch, args.epoch_finetune):
        save_max = False
        if distributed:
            train_sampler.set_epoch(epoch)
        logger.info('Epoch [{}] Start, lr {:.6f}'.format(epoch,
                                                         optimizer_train.param_groups[0]["lr"]))

        with Timer(' Train', logger):
            logger.debug('[Training]')
            train_loss, train_acc1, train_acc5 = train_one_epoch(
                model, criterion, None, optimizer_train, None, data_loader_train, None, logger,
                epoch, args.print_freq, world_size, scaler, args.accumulate_step, False, one_hot)
            if lr_scheduler_train is not None and lr_scheduler_T0 <= epoch < lr_scheduler_Tmax:
                lr_scheduler_train.step()

        with Timer(' Test', logger):
            logger.debug('[Test]')
            test_loss, test_acc1, test_acc5, sops = evaluate(model, criterion, data_loader_test,
                                                       args.print_freq, logger, False, one_hot, args, True)


        logger.info(' Test Acc@1: {:.5f}, Acc@5: {:.5f}'.format(test_acc1, test_acc5))

        if max_acc1 < test_acc1:
            max_acc1 = test_acc1
            min_sop = sops
            save_max = True

        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer_train': optimizer_train.state_dict(),
            'epoch': epoch,
            'max_acc1': max_acc1, }
        if lr_scheduler_train is not None:
            checkpoint['lr_scheduler_train'] = lr_scheduler_train.state_dict()

        if args.save_latest:
            save_on_master(checkpoint, os.path.join(args.output_dir, 'checkpoint_latest.pth'))

        if save_max:
            save_on_master(checkpoint, os.path.join(args.output_dir, 'checkpoint_max_acc1.pth'))

    logger.info('Finetune finish.')
    logger.info('Best Top-1 Acc: {:.2f}, SOPs: {:.5f} M'.format(max_acc1, min_sop))

if __name__ == "__main__":
    main()
