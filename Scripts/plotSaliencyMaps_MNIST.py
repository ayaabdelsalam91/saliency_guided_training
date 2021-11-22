
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
import random
import os
import argparse
import numpy as np
import importlib
from cnn import Net
import heatmaps
# from utils import progress_bar
import copy 
import torch.optim as optim
from torchvision import datasets, transforms
import time
import Helper
import torch.utils.data as data_utils


from regular import train as train_regular
from regular import test as test_regular


from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    InputXGradient,
    Saliency,
    NoiseTunnel
)




parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--trainingType', default='regular', type=str)
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--saveAll', default=False, action="store_true",
                    help='resume from checkpoint')

parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')

parser.add_argument('--featuresDroped', type=float, default=0.1)

parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
args = parser.parse_args()



train_kwargs = {'batch_size': args.batch_size}
test_kwargs = {'batch_size': args.test_batch_size}


# Data
print('==> Preparing data..')

transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
dataset1 = datasets.MNIST('./data/MNIST_data', train=True, download=True,
                   transform=transform)
dataset2 = datasets.MNIST('./data/MNIST_data', train=False,
                   transform=transform)
trainloader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
testloader = torch.utils.data.DataLoader(dataset2, **test_kwargs)





use_cuda =  torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")


if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                   'pin_memory': True,
                   'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)


trainingTypes=["MNIST_regular1_model","MNIST_interpretablefeaturesDroped_0.7_RandomMasking_1_model","MNIST_interpretablefeaturesDroped_0.5_RandomMasking_1_model"]


for  t in range (len(trainingTypes)):

    print('==> Loading',trainingTypes[t],"...")


    modelToTest = Net().to(device)


    checkpoint = torch.load('./models/'+trainingTypes[t]+'.th')
    modelToTest.load_state_dict(checkpoint['state_dict'])


    if device == 'cuda':
        modelToTest = torch.nn.DataParallel(modelToTest)
        cudnn.benchmark = True
    modelToTest.eval()

    saliencyMethods=["Grad","IG" ,"DL","GS","DLS","SG"]


    for saliencyMethod in saliencyMethods:



        saliencyValues,labels,allData = Helper.getSaliency(modelToTest,testloader,saliencyMethod,10000,abs=False)
        

        saliencyValues=saliencyValues.detach().cpu().numpy()
        labels=labels.detach().cpu().numpy()

        allData=allData.detach().cpu().numpy()

        for i in range(50):
            heatmaps.plotExampleWise(np.abs(saliencyValues[i,0,:,:]),'saliencyValues_'+saliencyMethod+"_"+trainingTypes[t]+"_"+str(i))

            heatmaps.plotExampleWise(allData[i,0,:,:],'TestData_'+str(i))

            np.save("outputs/SaliencyValues/"+saliencyMethod+"_"+trainingTypes[t]+"_"+str(i)+".npy", saliencyValues[i,0,:,:])

        np.save("outputs/SaliencyValues/Mean_"+saliencyMethod+"_"+trainingTypes[t]+".npy", saliencyValues[:,0,:,:].mean(axis=0))
        np.save("outputs/SaliencyValues/All_"+saliencyMethod+"_"+trainingTypes[t]+".npy", saliencyValues[:,0,:,:])

        print('./models/'+trainingTypes[t]+'.th',saliencyValues.max(),saliencyValues.min())



