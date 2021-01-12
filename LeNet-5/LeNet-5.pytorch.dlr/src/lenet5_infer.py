#!/usr/bin/env python
"""
This file contains LeNet-5 inferencing script.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#-------------------------------------------------------------------------------
__author__     = "Ando Ki"
__copyright__  = "Copyright 2020, Future Design Systems"
__credits__    = ["none", "some"]
__license__    = "FUTURE DESIGN SYSTEMS SOFTWARE END-USER LICENSE AGREEMENT"
__version__    = "0"
__revision__   = "1"
__maintainer__ = "Ando Ki"
__email__      = "contact@future-ds.com"
__status__     = "Development"
__date__       = "2020.10.01"
__description__= "LeNet-5 network model inferencing script"

#-------------------------------------------------------------------------------
import argparse
import os
import sys

import numpy as np

from PIL import Image
from PIL import ImageOps
import matplotlib.pyplot as plt

import torch
import torchvision as tv
import torch.nn.functional as F
import torchvision.transforms as tf

#-------------------------------------------------------------------------------
sys.path.append("../LeNet-5.pytorch/src")
from lenet5_model import Lenet5Model

#sys.path.append("../../..")
import python.modules.dlr_common as dlr_common
import python.torch as dlr

#-------------------------------------------------------------------------------
def lenet5_infer( input
                , c1_kernel, c1_bias
                , c2_kernel, c2_bias
                , f1_weight, f1_bias
                , f2_weight, f2_bias
                , f3_weight, f3_bias
                , softmax=True
                , pkg=F # can be F (torch.nn.functional) or dlr (Deep Learning Routine)
                , rigor=False):
    """ LeNet-5 inference network """
    if pkg is dlr:
        z = pkg.conv2d( input=input, weight=c1_kernel, bias=c1_bias, stride=1
                      , padding=0, dilation=1, groups=1, rigor=rigor )
        z = pkg.relu( input=z, rigor=rigor )
        z = pkg.max_pool2d( input=z, kernel_size=2, stride=2, padding=0, ceil_mode=False, rigor=rigor )

        z = pkg.conv2d( input=z, weight=c2_kernel, bias=c2_bias, stride=1
                    , padding=0, dilation=1, groups=1, rigor=rigor )
        z = pkg.relu( input=z, rigor=rigor )
        z = pkg.max_pool2d( input=z, kernel_size=2, stride=2, padding=0, ceil_mode=False, rigor=rigor )
        z = z.view(z.shape[0], -1) # do not use torch.flatten() to keep minibatch
        z = pkg.linear( input=z, weight=f1_weight, bias=f1_bias, rigor=rigor )
        z = pkg.relu( input=z, rigor=rigor )
        z = pkg.linear( input=z, weight=f2_weight, bias=f2_bias, rigor=rigor )
        z = pkg.relu( input=z, rigor=rigor )
        z = pkg.linear( input=z, weight=f3_weight, bias=f3_bias, rigor=rigor )
        if softmax: z = F.softmax( input=z, dim=1)
    else:
        z = pkg.conv2d( input=input, weight=c1_kernel, bias=c1_bias, stride=1
                      , padding=0, dilation=1, groups=1 )
        z = pkg.relu( input=z )
        z = pkg.max_pool2d( input=z, kernel_size=2, stride=2, padding=0, ceil_mode=False )

        z = pkg.conv2d( input=z, weight=c2_kernel, bias=c2_bias, stride=1
                    , padding=0, dilation=1, groups=1 )
        z = pkg.relu( input=z )
        z = pkg.max_pool2d( input=z, kernel_size=2, stride=2, padding=0, ceil_mode=False )
        z = z.view(z.shape[0], -1) # do not use torch.flatten() to keep minibatch
        z = pkg.linear( input=z, weight=f1_weight, bias=f1_bias )
        z = pkg.relu( input=z )
        z = pkg.linear( input=z, weight=f2_weight, bias=f2_bias )
        z = pkg.relu( input=z )
        z = pkg.linear( input=z, weight=f3_weight, bias=f3_bias )
        if softmax: z = F.softmax( input=z, dim=1)
    return z

#-------------------------------------------------------------------------------
def run_lenet5( args, model, image, verbose=False ):
    """
    Runs 'lenet5_infer' for a given image.
    """
    #---------------------------------------------------------------------------
    # using forward method of PyTorch model
    resultA = model.infer(image, softmax=args.softmax)
    resultA = resultA.view(10)
    if verbose:
        print("PyTorch Lenet5Model used in training stage");
        for idx in range(10):
            print(f"A {idx}: {resultA[idx]:.5f}")
        print("")

    #---------------------------------------------------------------------------
    # using PyTorch model using torch.nn.functional.
    if True:
        resultB = lenet5_infer( input=image
                              , c1_kernel=lenet_params['model.0.weight' ],c1_bias=lenet_params['model.0.bias' ]
                              , c2_kernel=lenet_params['model.3.weight' ],c2_bias=lenet_params['model.3.bias' ]
                              , f1_weight=lenet_params['model.7.weight' ],f1_bias=lenet_params['model.7.bias' ]
                              , f2_weight=lenet_params['model.9.weight' ],f2_bias=lenet_params['model.9.bias' ]
                              , f3_weight=lenet_params['model.11.weight'],f3_bias=lenet_params['model.11.bias']
                              , softmax=args.softmax
                              , pkg=F)
    else:
        resultB = lenet5_infer( input=image
                              , c1_kernel=lenet_params['conv1.weight'],c1_bias=lenet_params['conv1.bias']
                              , c2_kernel=lenet_params['conv2.weight'],c2_bias=lenet_params['conv2.bias']
                              , f1_weight=lenet_params['fc1.weight']  ,f1_bias=lenet_params['fc1.bias']
                              , f2_weight=lenet_params['fc2.weight']  ,f2_bias=lenet_params['fc2.bias']
                              , f3_weight=lenet_params['fc3.weight']  ,f3_bias=lenet_params['fc3.bias']
                              , softmax=args.softmax
                              , pkg=F)
    resultB = resultB.view(10)
    if verbose:
        print("PyTorch Lenet5Model for inferencing");
        for idx in range(10):
            print(f"B {idx}: {resultB[idx]:.5f}")
        print("")

    #---------------------------------------------------------------------------
    # using DLR C model through PyTorch wrapper.
    #dlr_common.set_rigor(args.rigor)
    if True:
        resultC = lenet5_infer( input=image
                              , c1_kernel=lenet_params['model.0.weight' ],c1_bias=lenet_params['model.0.bias' ]
                              , c2_kernel=lenet_params['model.3.weight' ],c2_bias=lenet_params['model.3.bias' ]
                              , f1_weight=lenet_params['model.7.weight' ],f1_bias=lenet_params['model.7.bias' ]
                              , f2_weight=lenet_params['model.9.weight' ],f2_bias=lenet_params['model.9.bias' ]
                              , f3_weight=lenet_params['model.11.weight'],f3_bias=lenet_params['model.11.bias']
                              , softmax=args.softmax
                              , pkg=dlr
                              , rigor=args.rigor)
    else:
        resultC = lenet5_infer( input=image
                              , c1_kernel=lenet_params['conv1.weight'],c1_bias=lenet_params['conv1.bias']
                              , c2_kernel=lenet_params['conv2.weight'],c2_bias=lenet_params['conv2.bias']
                              , f1_weight=lenet_params['fc1.weight']  ,f1_bias=lenet_params['fc1.bias']
                              , f2_weight=lenet_params['fc2.weight']  ,f2_bias=lenet_params['fc2.bias']
                              , f3_weight=lenet_params['fc3.weight']  ,f3_bias=lenet_params['fc3.bias']
                              , softmax=args.softmax
                              , pkg=dlr
                              , rigor=args.rigor)
    resultC = resultC.view(10)
    if verbose:
        print("DLR Lenet5Model for inferencing");
        for idx in range(10):
            print(f"C {idx}: {resultC[idx]:.5f}")

    #---------------------------------------------------------------------------
    return resultA, resultB, resultC

#-------------------------------------------------------------------------------
if __name__ == '__main__':
    def get_args():
        parser = argparse.ArgumentParser(description='PyTorch LeNet-5')
        parser.add_argument('-i', '--input_channels', type=int, default=1,
                            metavar='input_channels',
                            help='input channel size (default: 1)')
        parser.add_argument('-s', '--softmax', action='store_true',
                            help='use softmax at the end of inference (default: False)')
        parser.add_argument('-c', '--checkpoint', type=str, default="checkpoints/mnist_final.pth",
                            metavar='file_name',
                            help='model path_file_name for checkpoint (default: checkpoints/mnist_final.pth)')
        parser.add_argument('-r', '--rigor', action='store_true',
                            help='Set rigor (default: False)')
        parser.add_argument('-v', '--verbose', action='store_true',
                            help='make verbose (default: False)')
        parser.add_argument('images', type=str, nargs='+',
                            help='one or multiple image path_file_name to infer')
        args = parser.parse_args()
        return args

#-------------------------------------------------------------------------------
if __name__ == '__main__':
    args = get_args()

    if not os.path.exists(args.checkpoint):
        print(args.checkpoint, "not exist")
        quit()

    model = Lenet5Model(args.input_channels)
    if args.verbose:
        print(model)

    extension = os.path.splitext(args.checkpoint)[1]
    if extension == '.pkl':
        model = torch.load(args.checkpoint)
    elif extension == '.pth':
        model.load_state_dict(torch.load(args.checkpoint))
        model.eval()
    elif extension == '.onnx':
        model = torch.onnx.load(args.checkpoint)
        torch.onnix.checker.check_model(model)
    else:
        print("un-known weights file: ", args.checkpoint);

    # getting weights and biases
    lenet_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            lenet_params[name] = param.data

    # run inference over image
    for image in args.images:
        # getting images
        if not os.path.exists(image):
            print(image, "not exist")
            quit()
        img = Image.open(image)
        img = img.resize((32,32), Image.ANTIALIAS)
        img.show() #img.save('x.png')
        if img.mode != 'L': # Not Luminance
            print("Convert to grayscale")
            img = img.convert('L') # get luminance using Pillow convert()
            img.show() #img.save('x.png')
            img = ImageOps.invert(img)
            img.show() #img.save('x.png')
        if args.input_channels != 1:
            if args.verbose: print(f"need to convert 1-channel to 3-channel")
            img = np.stack([img]*args.input_channels, axis=-1)
        data = tv.transforms.ToTensor()(img)
        data = data.view(-1,args.input_channels,32,32) # (minibatch,channel,height,width)

        RA, RB, RC = run_lenet5( args, model, data, args.verbose )

        print("Origin model / PyTorch model / DLR model")
        for idx in range(10):
            print(f"{idx}: {RA[idx]:10.5f} / {RB[idx]:10.5f} / {RC[idx]:10.5f}")

#===============================================================================
# Revision history:
#
# 2020.10.01: Started by Ando Ki (adki@future-ds.com)
#===============================================================================
