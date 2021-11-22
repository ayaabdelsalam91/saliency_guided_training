


import matplotlib.pyplot as plt
import numpy as np


import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from torchvision import models

from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz


import Helper

import argparse
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet

import time
import torch.nn.functional as F

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))


parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)

parser.add_argument('--ProgressiveMasking', default=False, action="store_true")
parser.add_argument('--featuresDroped', type=float, default=0.1)
parser.add_argument('--NotKL', default=False, action="store_true",
                    help='resume from checkpoint')
parser.add_argument('--WithNorm', default=False, action="store_true",
                    help='resume from checkpoint')


parser.add_argument('--RandomMasking', default=False, action="store_true",
                    help='resume from checkpoint')
best_prec1 = 0


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor



def attribute_image_features(algorithm, input, **kwargs):
    net.zero_grad()
    tensor_attributions = algorithm.attribute(input,
                                              target=labels[ind],
                                              **kwargs
                                             )
    
    return tensor_attributions


models=["CIFAR_regular1_model","CIFAR_interpretablefeaturesDroped_0.5_RandomMasking_1_model"]


def main():
    global args, best_prec1
    args = parser.parse_args()


    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)





    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    trainloader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data/CIFAR_data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    testloader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data/CIFAR_data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=100, shuffle=False,
        num_workers=args.workers, pin_memory=True)
        

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    saliencyMethod="Grad"




    dataiter = iter(testloader)
    images, labels = dataiter.next()




    for model in models:
      net = torch.nn.DataParallel(resnet.__dict__[args.arch]())
      

      checkpoint = torch.load('./models/'+model+'.th')
      net.load_state_dict(checkpoint['state_dict'])


      net.eval()


      net.cuda()
      saliencyValues,labels_,allData = Helper.getSaliency(net,testloader,saliencyMethod,10000,abs=False)

      np.save("outputs/SaliencyValues/Mean_"+saliencyMethod+"_"+model+".npy", saliencyValues.mean(axis=0))
      np.save("outputs/SaliencyValues/All_"+saliencyMethod+"_"+model+".npy", saliencyValues)



      for ind in range(100):



        input = images[ind].unsqueeze(0)
        input.requires_grad = True




        saliency = Saliency(net)
        grads = saliency.attribute(input, target=labels[ind].item(),abs=False)
        grads_=grads.cpu().detach().numpy()


        np.save('outputs/SaliencyValues/Grad_'+model+"_"+str(ind)+".npy", grads_)
        grads=grads.abs()
        grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))





        unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        original_image=unorm(images[ind].cpu().detach()).numpy()
        original_image = np.transpose(original_image, (1, 2, 0))



        figure, axis = viz.visualize_image_attr(None, original_image, 
                              method="original_image")
        # plt.savefig()
        plt.savefig('./outputs/Graphs/original_image_'+str(ind)+".png" , bbox_inches = 'tight',
            pad_inches = 0)
        plt.show()





        _ = viz.visualize_image_attr(grads, original_image, method="heat_map", sign="absolute_value")

        plt.savefig('./outputs/Graphs/Grad_'+model+"_"+str(ind)+"1.png" , bbox_inches = 'tight',
        pad_inches = 0)
        plt.show()

        _ = viz.visualize_image_attr(grads, original_image, method="blended_heat_map", sign="absolute_value")

        plt.savefig('./outputs/Graphs/Grad_'+model+"_"+str(ind)+"2.png" , bbox_inches = 'tight',
        pad_inches = 0)
        plt.show()


        _ = viz.visualize_image_attr(grads, original_image, method="masked_image", sign="absolute_value")

        plt.savefig('./outputs/Graphs/Grad_'+model+"_"+str(ind)+"3.png" , bbox_inches = 'tight',
        pad_inches = 0)
        plt.show()



        _ = viz.visualize_image_attr(grads, original_image, method="alpha_scaling", sign="absolute_value")

        plt.savefig('./outputs/Graphs/Grad_'+model+"_"+str(ind)+"4.png" , bbox_inches = 'tight',
        pad_inches = 0)
        plt.show()

     



if __name__ == '__main__':
    main()

