from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=W0611
import types

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import pretrainedmodels
import os
import ast
#203094
class Attention(nn.Module):
    def __init__(self, num_classes=137133, cnn='resnet34', attention_size=8):
        super().__init__()
        self.num_classes = num_classes
        self.cnn = globals().get('get_' + cnn)()
        self.attention_size = attention_size

        self.avgpool = nn.AdaptiveAvgPool2d(self.attention_size)

        in_features = self.cnn.last_linear.in_features
        self.last_linear = nn.Conv2d(in_features, self.num_classes, kernel_size=1, padding=0)
        self.attention = nn.Conv2d(in_features, self.num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        features = self.cnn.features(x)
        if self.attention_size != features.size(-1):
            features = self.avgpool(features)

        logits = self.last_linear(features)
        assert logits.size(1) == self.num_classes and \
               logits.size(2) == self.attention_size and \
               logits.size(3) == self.attention_size

        logits_attention = self.attention(features)
        assert logits_attention.size(1) == self.num_classes and \
               logits_attention.size(2) == self.attention_size and \
               logits_attention.size(3) == self.attention_size
        logits_attention = logits_attention.view(-1, self.num_classes, self.attention_size * self.attention_size)
        attention = F.softmax(logits_attention, dim=2)
        attention = attention.view(-1, self.num_classes, self.attention_size, self.attention_size)

        logits = logits * attention
        return logits.view(-1, self.num_classes, self.attention_size * self.attention_size).sum(2).view(-1, self.num_classes)


class AttentionInceptionV3(nn.Module):
    def __init__(self, num_classes=28, attention_size=8, aux_attention_size=8):
        super().__init__()
        print('AttentionInceptionV3 num_classes  ----', num_classes)
        self.num_classes = num_classes
        self.cnn = torchvision.models.inception_v3(pretrained=True)
        self.attention_size = attention_size
        self.aux_attention_size = aux_attention_size

        conv = self.cnn.Conv2d_1a_3x3.conv
        self.cnn.Conv2d_1a_3x3.conv = nn.Conv2d(in_channels=3,
                                                out_channels=conv.out_channels,
                                                kernel_size=conv.kernel_size,
                                                stride=conv.stride,
                                                padding=conv.padding,
                                                bias=conv.bias)

        # copy pretrained weights
      #  self.cnn.Conv2d_1a_3x3.conv.weight.data[:,:3,:,:] = conv.weight.data
      #  self.cnn.Conv2d_1a_3x3.conv.weight.data[:,3:,:,:] = conv.weight.data[:,:1,:,:]

        self.features_a = nn.Sequential(
            self.cnn.Conv2d_1a_3x3,
            self.cnn.Conv2d_2a_3x3,
            self.cnn.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.cnn.Conv2d_3b_1x1,
            self.cnn.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.cnn.Mixed_5b,
            self.cnn.Mixed_5c,
            self.cnn.Mixed_5d,
            self.cnn.Mixed_6a,
            self.cnn.Mixed_6b,
            self.cnn.Mixed_6c,
            self.cnn.Mixed_6d,
            self.cnn.Mixed_6e,
        )

        self.features_b = nn.Sequential(
            self.cnn.Mixed_7a,
            self.cnn.Mixed_7b,
            self.cnn.Mixed_7c,
        )

        self.aux_avgpool = nn.AdaptiveAvgPool2d(self.aux_attention_size)
        aux_in_features = self.cnn.AuxLogits.fc.in_features
        self.aux_linear = nn.Conv2d(aux_in_features, self.num_classes, kernel_size=1, padding=0)
        self.aux_attention = nn.Conv2d(aux_in_features, self.num_classes, kernel_size=1, padding=0)

        self.avgpool = nn.AdaptiveAvgPool2d(self.attention_size)
        in_features = self.cnn.fc.in_features
        self.last_linear = nn.Conv2d(in_features, self.num_classes, kernel_size=1, padding=0)
        self.attention = nn.Conv2d(in_features, self.num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        features_a = self.features_a(x)
     #   print('features_a ', features_a.shape)
        if self.training:
            if self.aux_attention_size != features_a.size(-1):
                aux_features = self.aux_avgpool(features_a)
      #          print('aux_features ', aux_features.shape)
            else:
                aux_features = features_a
            aux_logits = self.aux_linear(aux_features)
      #      print('aux_logits ', aux_logits.shape)
            assert aux_logits.size(1) == self.num_classes and \
                   aux_logits.size(2) == self.aux_attention_size and \
                   aux_logits.size(3) == self.aux_attention_size
            aux_logits_attention = self.aux_attention(aux_features)
      #      print('aux_logits_attention ', aux_logits_attention.shape)
            assert aux_logits_attention.size(1) == self.num_classes and \
                   aux_logits_attention.size(2) == self.aux_attention_size and \
                   aux_logits_attention.size(3) == self.aux_attention_size
            aux_logits_attention = aux_logits_attention.view(
                -1, self.num_classes,
                self.aux_attention_size * self.aux_attention_size)
      #      print('aux_logits_attention-2 ', aux_logits_attention.shape)

            aux_attention = F.softmax(aux_logits_attention, dim=2)
      #      print('aux_attention-1 ', aux_attention.shape)
            aux_attention = aux_attention.view(
                -1, self.num_classes, self.aux_attention_size, self.aux_attention_size)
      #      print('aux_attention-2 ', aux_attention.shape)
            aux_logits = aux_logits * aux_attention
     #       print('aux_logits-1 ', aux_logits.shape)
            aux_logits = aux_logits.view(
                -1, self.num_classes,
                self.aux_attention_size * self.aux_attention_size)
     #       print('aux_logits-2 ', aux_logits.shape)
            aux_logits = aux_logits.sum(2)
     #       print('aux_logits-3 ', aux_logits.shape)
            aux_logits = aux_logits.view(-1, self.num_classes)
     #       print('aux_logits-4 ', aux_logits.shape)

        features_b = self.features_b(features_a)
    #    print('features_b ', features_b.shape)
        if self.aux_attention_size != features_b.size(-1):
            features_b = self.avgpool(features_b)
    #        print('features_b-1 ', features_b.shape)
        logits = self.last_linear(features_b)
    #    print('logits ', logits.shape)
        assert logits.size(1) == self.num_classes and \
               logits.size(2) == self.attention_size and \
               logits.size(3) == self.attention_size

        logits_attention = self.attention(features_b)
    #    print('logits_attention ', logits_attention.shape)
        assert logits_attention.size(1) == self.num_classes and \
               logits_attention.size(2) == self.attention_size and \
               logits_attention.size(3) == self.attention_size
        logits_attention = logits_attention.view(-1, self.num_classes, self.attention_size * self.attention_size)
    #    print('logits_attention-1 ', logits_attention.shape)
        attention = F.softmax(logits_attention, dim=2)
    #    print('attention ', attention.shape)
        attention = attention.view(-1, self.num_classes, self.attention_size, self.attention_size)
    #    print('attention-1 ', attention.shape)

        logits = logits * attention
    #    print('logits-1 ', logits.shape)
        logits = logits.view(-1, self.num_classes, self.attention_size * self.attention_size).sum(2).view(-1, self.num_classes)
    #    print('logits-2 ', logits.shape)
        if self.training:
            return logits, aux_logits
        return logits


def get_attention_inceptionv3(num_classes=1002, **kwargs):
    return AttentionInceptionV3(num_classes=num_classes, **kwargs)


def get_attention(num_classes=137133, **kwargs):
    return Attention(num_classes=num_classes, **kwargs)


def get_resnet34(num_classes=137133, **_):
    model_name = 'resnet34'
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    conv1 = model.conv1
    model.conv1 = nn.Conv2d(in_channels=3,
                            out_channels=conv1.out_channels,
                            kernel_size=conv1.kernel_size,
                            stride=conv1.stride,
                            padding=conv1.padding,
                            bias=conv1.bias)

    # copy pretrained weights
   # model.conv1.weight.data[:,:3,:,:] = conv1.weight.data
   # model.conv1.weight.data[:,3:,:,:] = conv1.weight.data[:,:1,:,:]

    model.avgpool = nn.AdaptiveAvgPool2d(1)
    in_features = model.last_linear.in_features
    model.last_linear = nn.Linear(in_features, num_classes)
    return model


def get_resnet18(num_classes=203094, **_):
    model_name = 'resnet18'
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    print(model)
    conv1 = model.conv1
    model.conv1 = nn.Conv2d(in_channels=3,
                            out_channels=conv1.out_channels,
                            kernel_size=conv1.kernel_size,
                            stride=conv1.stride,
                            padding=conv1.padding,
                            bias=conv1.bias)

    # copy pretrained weights
  #  model.conv1.weight.data[:,:3,:,:] = conv1.weight.data
  #  model.conv1.weight.data[:,3:,:,:] = conv1.weight.data[:,:1,:,:]

    model.avgpool = nn.AdaptiveAvgPool2d(1)
    in_features = model.last_linear.in_features
    model.last_linear = nn.Linear(in_features, num_classes)
    return model


def get_senet(model_name='se_resnext50', num_classes=203094, **_):
    model = pretrainedmodels.__dict__[model_name](num_classes=1000)
    conv1 = model.layer0.conv1
    model.layer0.conv1 = nn.Conv2d(in_channels=3,
                                   out_channels=conv1.out_channels,
                                   kernel_size=conv1.kernel_size,
                                   stride=conv1.stride,
                                   padding=conv1.padding,
                                   bias=conv1.bias)

    # copy pretrained weights
  #  model.layer0.conv1.weight.data[:,:3,:,:] = conv1.weight.data
  #  model.layer0.conv1.weight.data[:,3:,:,:] = conv1.weight.data[:,:1,:,:]

    model.avgpool = nn.AdaptiveAvgPool2d(1)
    in_features = model.last_linear.in_features
    model.last_linear = nn.Linear(in_features, num_classes)
    return model


def get_se_resnext50(num_classes=203094, **kwargs):
    return get_senet('se_resnext50_32x4d', num_classes=num_classes, **kwargs)


def get_model(config, gi):
    print('model name:', config.model.name)
    f = globals().get('get_' + config.model.name)
    
    class_num = 5000
    with open(os.path.join('./data/', 'key_groups.txt'), 'r') as f: 
        key_group_list = ast.literal_eval(f.read())
        class_num =  len(key_group_list[gi])
        
  #  print('parm ',  config.model.params)
    print('class_num ', class_num)
    if config.model.params is None:
        return f(class_num)
    else:
        #config.model.params['num_classes'] = class_num
        return f(**config.model.params )


if __name__ == '__main__':
    print('main')
    model = get_resnet34()
