from __future__ import print_function
import argparse
import os
import sys

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegressionCV
#import models.densenet as dn
#import models.wideresnet as wn
import util.svhn_loader as svhn
import numpy as np
import time
import util.svhn_loader as svhn
from scipy import misc
from util.mahalanobis_lib import sample_estimator, get_Mahalanobis_score

import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
__all__ = ['ResNet', 'resnet18', 'resnet50', ]

torch.manual_seed(1)
#torch.cuda.manual_seed(1)
np.random.seed(1)

parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')

parser.add_argument('--in-dataset', default="imagenet", type=str, help='in-distribution dataset')
parser.add_argument('--name', default = "resnet50", type=str,
                    help='neural network name and training set')
parser.add_argument('--model-arch', default='resnet50', type=str, help='model architecture')

parser.add_argument('--gpu', default = '0', type = str,
		    help='gpu index')
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')

parser.add_argument('-b', '--batch-size', default=10, type=int,
                    help='mini-batch size')

parser.add_argument('--layers', default=100, type=int,
                    help='total number of layers (default: 100)')

parser.add_argument('--depth', default=40, type=int,
                    help='depth of resnet')
parser.add_argument('--width', default=4, type=int,
                    help='width of resnet')

parser.set_defaults(argument=True)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

normalization = nn.BatchNorm2d

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Identity(nn.Module):
    def forward(self, input):
        return input + 0.0

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = normalization(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = normalization(planes)
        self.shortcut = Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.shortcut(out)
        out = self.relu(out)

        return out

    def forward_masked(self, x, mask_weight=None, mask_bias=None):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.shortcut(out)
        out = self.relu(out)

        if mask_weight is not None:
            out = out * mask_weight[None,:,None,None]
        if mask_bias is not None:
            out = out + mask_bias[None,:,None,None]
        return out

    def forward_threshold(self, x, threshold=1e10):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        b, c, w, h = out.shape
        mask = out.view(b, c, -1).mean(2) < threshold
        out = mask[:, :, None, None] * out
        # print(mask.sum(1).float().mean(0))
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = normalization(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = normalization(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = normalization(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.shortcut = Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.shortcut(out)
        out = self.relu(out)

        return out

    def forward_masked(self, x, mask_weight=None, mask_bias=None):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        if mask_weight is not None:
            out = out * mask_weight[None,:,None,None]
        if mask_bias is not None:
            out = out + mask_bias[None,:,None,None]
        out = self.relu(out)
        return out

    def forward_threshold(self, x, threshold=1e10):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)
        b, c, w, h = out.shape
        mask = out.view(b, c, -1).mean(2) < threshold
        out = mask[:, :, None, None] * out

        return out


class AbstractResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(AbstractResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = normalization(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def _initial_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                normalization(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def load_state_dict(self, state_dict, strict=True):
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(self)

        if strict:
            error_msg = ''
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in missing_keys)))

        if len(error_msgs) > 0:
            print('Warning(s) in loading state_dict for {}:\n\t{}'.format(self.__class__.__name__, "\n\t".join(error_msgs)))


class ResNet(AbstractResNet):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__(block, layers, num_classes)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        #self.fc = nn.Linear(512 * block.expansion, num_classes)
        self._initial_weight()

    def forward_masked(self, x, mask_weight=None, mask_bias=None):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer3(self.layer2(self.layer1(x)))
        x = self.layer4[:-1](x)
        x = self.layer4[-1].forward_masked(x, mask_weight=mask_weight, mask_bias=mask_bias)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def forward_threshold(self, x, threshold=1e10):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x= self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x)
        x = x.clip(max=threshold)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_list(self, x):
        out_list = []
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        # out_list.append(out)
        out = self.layer1(out)
        # out_list.append(out)
        out = self.layer2(out)
        # out_list.append(out)
        out = self.layer3(out)
        # out_list.append(out)
        out = self.layer4(out)
        #out = self.avgpool(out)

        print(self.fc)
        out_list.append(out)
        out = out.view(out.size(0), -1)
        print("x here", out.shape[1])
        self.fc = nn.Linear(out.shape[1], 1000)
        print(out.shape)
        y = self.fc(out)
        
        
        #y = self.fc(out)
        return y, out_list

    def intermediate_forward(self, x, layer_index):
    # if layer_index >= 0:
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
    # if layer_index >= 1:
        out = self.layer1(out)
    # if layer_index >= 2:
        out = self.layer2(out)
    # if layer_index >= 3:
        out = self.layer3(out)
    # if layer_index >= 4:
        out = self.layer4(out)
        out = self.avgpool(out)
        # out = out.clip(max=1.0)
        return out


def c_resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

def print_results(results, stypes):
    mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']

    for stype in stypes:
        print(' OOD detection method: ' + stype)
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results[stype]['FPR']), end='')
        print(' {val:6.2f}'.format(val=100.*results[stype]['DTERR']), end='')
        print(' {val:6.2f}'.format(val=100.*results[stype]['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*results[stype]['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*results[stype]['AUOUT']), end='')
        print('')

def tune_mahalanobis_hyperparams():

    print('Tuning hyper-parameters...')
    stypes = ['mahalanobis']

    save_dir = os.path.join('output/mahalanobis_hyperparams/', args.in_dataset, args.name, 'tmp')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    if args.in_dataset == "CIFAR-10":
        normalizer = transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0))

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        trainset= torchvision.datasets.CIFAR10('./datasets/cifar10', train=True, download=True, transform=transform)
        trainloaderIn = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=False, download=True, transform=transform)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        num_classes = 10
    elif args.in_dataset == "CIFAR-100":
        normalizer = transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0))

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        trainset= torchvision.datasets.CIFAR100('./datasets/cifar100', train=True, download=True, transform=transform)
        trainloaderIn = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR100(root='./datasets/cifar100', train=False, download=True, transform=transform)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        num_classes = 100

    elif args.in_dataset == "SVHN":

        normalizer = None
        trainloaderIn = torch.utils.data.DataLoader(
            svhn.SVHN('datasets/svhn/', split='train',
                                      transform=transforms.ToTensor(), download=False),
            batch_size=args.batch_size, shuffle=True)
        testloaderIn = torch.utils.data.DataLoader(
            svhn.SVHN('datasets/svhn/', split='test',
                                  transform=transforms.ToTensor(), download=False),
            batch_size=args.batch_size, shuffle=True)

        args.epochs = 20
        num_classes = 10

    elif args.in_dataset == "imagenet":
        root = 'datasets/id_data/imagenet'
        transform_train_largescale = transforms.Compose([
            transforms.Resize(256),
	    transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
	    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.244, 0.225])
        ])
        transform_test_largescale = transforms.Compose([
            transforms.Resize(256),
	    transforms.CenterCrop(224),
            transforms.ToTensor(),
	    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.244, 0.225])
        ])


        # Data loading code
        trainloaderIn = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(os.path.join(root, 'train'), transform = transform_train_largescale),
            batch_size=args.batch_size, shuffle=True)
        testloaderIn = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(os.path.join(root, 'val'), transform = transform_test_largescale),
            batch_size=args.batch_size, shuffle=True)
        num_classes = 1000
    '''
    if args.model_arch == 'densenet':
        model = dn.DenseNet3(args.layers, num_classes, normalizer=normalizer)
    elif args.model_arch == 'wideresnet':
        model = wn.WideResNet(args.depth, num_classes, widen_factor=args.width, normalizer=normalizer)
    else:
        assert False, 'Not supported model arch: {}'.format(args.model_arch)
    '''
    #checkpoint = torch.load("./checkpoints/{in_dataset}/{name}/checkpoint_{epochs}.pth.tar".format(in_dataset=args.in_dataset, name=args.name, epochs=args.epochs))
    #model.load_state_dict(checkpoint['state_dict'])
    if args.model_arch == 'resnet50':
        model = c_resnet50(num_classes=num_classes, pretrained=True)
    elif args.model_arch == 'mobilenet':
        from models.mobilenet import mobilenet_v2
        model = mobilenet_v2(num_classes=num_classes, pretrained=True)
    


    model.eval()
    #model.cuda()

    # set information about feature extaction
    temp_x = torch.rand(2,3,32,32)
    temp_x = Variable(temp_x)
    #temp_x = Variable(temp_x).cuda()
    temp_list = model.feature_list(temp_x)[1]
    #num_output = len(temp_list)
    num_output = 1
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1

    print('get sample mean and covariance')
    sample_mean, precision = sample_estimator(model, num_classes, feature_list, trainloaderIn)

    print('train logistic regression model')
    m = 500

    train_in = []
    train_in_label = []
    train_out = []

    val_in = []
    val_in_label = []
    val_out = []

    cnt = 0
    for data, target in testloaderIn:
        data = data.numpy()
        target = target.numpy()
        for x, y in zip(data, target):
            cnt += 1
            if cnt <= m:
                train_in.append(x)
                train_in_label.append(y)
            elif cnt <= 2*m:
                val_in.append(x)
                val_in_label.append(y)

            if cnt == 2*m:
                break
        if cnt == 2*m:
            break

    print('In', len(train_in), len(val_in))

    criterion = nn.CrossEntropyLoss()
    #criterion = nn.CrossEntropyLoss().cuda()
    adv_noise = 0.05

    for i in range(int(m/args.batch_size) + 1):
        if i*args.batch_size >= m:
            break
        data = torch.tensor(train_in[i*args.batch_size:min((i+1)*args.batch_size, m)])
        target = torch.tensor(train_in_label[i*args.batch_size:min((i+1)*args.batch_size, m)])
        #data = data.cuda()
        #target = target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        self.fc = nn.Linear(output.shape[1], 1000)
        output = model(data)
        print("x here", output.shape[1])
        
        print(output.shape)
        y = self.fc(output)

        model.zero_grad()
        inputs = Variable(data.data, requires_grad=True)
	#inputs = Variable(data.data, requires_grad=True).cuda()
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()

        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float()-0.5)*2

        adv_data = torch.add(input=inputs.data, other=gradient, alpha=adv_noise)
        adv_data = torch.clamp(adv_data, 0.0, 1.0)

        train_out.extend(adv_data.cpu().numpy())

    for i in range(int(m/args.batch_size) + 1):
        if i*args.batch_size >= m:
            break
        data = torch.tensor(val_in[i*args.batch_size:min((i+1)*args.batch_size, m)])
        target = torch.tensor(val_in_label[i*args.batch_size:min((i+1)*args.batch_size, m)])
        #data = data.cuda()
        #target = target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        model.zero_grad()
        inputs = Variable(data.data, requires_grad=True)
	#inputs = Variable(data.data, requires_grad=True).cuda()
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()

        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float()-0.5)*2

        adv_data = torch.add(input=inputs.data, other=gradient, alpha=adv_noise)
        adv_data = torch.clamp(adv_data, 0.0, 1.0)

        val_out.extend(adv_data.cpu().numpy())

    print('Out', len(train_out),len(val_out))

    train_lr_data = []
    train_lr_label = []
    train_lr_data.extend(train_in)
    train_lr_label.extend(np.zeros(m))
    train_lr_data.extend(train_out)
    train_lr_label.extend(np.ones(m))
    train_lr_data = torch.tensor(train_lr_data)
    train_lr_label = torch.tensor(train_lr_label)

    best_fpr = 1.1
    best_magnitude = 0.0

    for magnitude in [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]:
        train_lr_Mahalanobis = []
        total = 0
        for data_index in range(int(np.floor(train_lr_data.size(0) / args.batch_size))):
            data = train_lr_data[total : total + args.batch_size]
	    #data = train_lr_data[total : total + args.batch_size].cuda()
            total += args.batch_size
            Mahalanobis_scores = get_Mahalanobis_score(data, model, num_classes, sample_mean, precision, num_output, magnitude)
            train_lr_Mahalanobis.extend(Mahalanobis_scores)

        train_lr_Mahalanobis = np.asarray(train_lr_Mahalanobis, dtype=np.float32)
        regressor = LogisticRegressionCV(n_jobs=-1).fit(train_lr_Mahalanobis, train_lr_label)

        print('Logistic Regressor params:', regressor.coef_, regressor.intercept_)

        t0 = time.time()
        f1 = open(os.path.join(save_dir, "confidence_mahalanobis_In.txt"), 'w')
        f2 = open(os.path.join(save_dir, "confidence_mahalanobis_Out.txt"), 'w')

    ########################################In-distribution###########################################
        print("Processing in-distribution images")

        count = 0
        for i in range(int(m/args.batch_size) + 1):
            if i * args.batch_size >= m:
                break
            images = torch.tensor(val_in[i * args.batch_size : min((i+1) * args.batch_size, m)])
	    #images = torch.tensor(val_in[i * args.batch_size : min((i+1) * args.batch_size, m)]).cuda()
            # if j<1000: continue
            batch_size = images.shape[0]
            Mahalanobis_scores = get_Mahalanobis_score(images, model, num_classes, sample_mean, precision, num_output, magnitude)
            confidence_scores= regressor.predict_proba(Mahalanobis_scores)[:, 1]

            for k in range(batch_size):
                f1.write("{}\n".format(-confidence_scores[k]))

            count += batch_size
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, m, time.time()-t0))
            t0 = time.time()

    ###################################Out-of-Distributions#####################################
        t0 = time.time()
        print("Processing out-of-distribution images")
        count = 0

        for i in range(int(m/args.batch_size) + 1):
            if i * args.batch_size >= m:
                break
            images = torch.tensor(val_out[i * args.batch_size : min((i+1) * args.batch_size, m)])
	    #images = torch.tensor(val_out[i * args.batch_size : min((i+1) * args.batch_size, m)]).cuda()
            # if j<1000: continue
            batch_size = images.shape[0]

            Mahalanobis_scores = get_Mahalanobis_score(images, model, num_classes, sample_mean, precision, num_output, magnitude)

            confidence_scores= regressor.predict_proba(Mahalanobis_scores)[:, 1]

            for k in range(batch_size):
                f2.write("{}\n".format(-confidence_scores[k]))

            count += batch_size
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, m, time.time()-t0))
            t0 = time.time()

        f1.close()
        f2.close()

        results = metric(save_dir, stypes)
        print_results(results, stypes)
        fpr = results['mahalanobis']['FPR']
        if fpr < best_fpr:
            best_fpr = fpr
            best_magnitude = magnitude
            best_regressor = regressor

    print('Best Logistic Regressor params:', best_regressor.coef_, best_regressor.intercept_)
    print('Best magnitude', best_magnitude)

    return sample_mean, precision, best_regressor, best_magnitude

if __name__ == '__main__':
    sample_mean, precision, best_regressor, best_magnitude = tune_mahalanobis_hyperparams()
    print('saving results...')
    save_dir = os.path.join('output/mahalanobis_hyperparams/', args.in_dataset, args.name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, 'results'), np.array([sample_mean, precision, best_regressor.coef_, best_regressor.intercept_, best_magnitude]))
