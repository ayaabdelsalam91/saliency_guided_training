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
import torch.nn.functional as F

import resnet



import Helper
import regular
import interpretable 
from Helper import save_checkpoint


parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                    help='model architecture: (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
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
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)

parser.add_argument('--trainingType', default='regular', type=str)
parser.add_argument('--featuresDroped', type=float, default=0.1)
parser.add_argument('--RandomMasking', default=False, action="store_true", help='Random Masking while interpretable training')
parser.add_argument('--TrainWithAugmented', default=False, action="store_true",help='Use augmentation while training')
parser.add_argument('--abs', default=False, action="store_true",help='take abs value of saliency while interpretable training')
parser.add_argument('--maskType', type=str, default="meanMask")
parser.add_argument('--maskPercentageRandom', default=False, action="store_true",help='% of masking in augmentation')
parser.add_argument('--save-dir', dest='save_dir',help='The directory used to save the trained models',default='models', type=str)
parser.add_argument('--maskPercentage', type=float, default=0.1)
parser.add_argument('--mu', type=float, default=0)
parser.add_argument('--isMNIST', default=False, action="store_true",help='Dataset is MNIST')
parser.add_argument('--isCIFAR', default=True, action="store_true",help='Dataset is CIFAR')
parser.add_argument('--isBIRD', default=False, action="store_true",help='Dataset is BIRD')
parser.add_argument('--append', default='1', type=str)



def main():

    args = parser.parse_args()
    start_epoch = 0 

    print('==> Preparing data..')


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
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)
        
        
        
    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    model.cuda()



    cudnn.benchmark = True




    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    criterionKDL=torch.nn.KLDivLoss(reduction = 'batchmean')

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=- 1)

    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1


    if args.evaluate:
        validate(val_loader, model, criterion)
        return


    saveFile= "CIFAR_"+args.trainingType

    if(args.TrainWithAugmented):
        if(args.maskPercentageRandom):
            saveFile=saveFile+"TrainWithAugmented_"+args.maskType+"_maskPercentageRandom_"
        else:
            saveFile=saveFile+"TrainWithAugmented_"+args.maskType+"_maskPercentage"+str(args.maskPercentage)+"_"

        if(args.maskType=="constantMask" or args.maskType=="customMask"):
            saveFile=saveFile + "mu"+str(args.mu)+"_"

    if("interpretable" in args.trainingType):
        saveFile=saveFile+"featuresDroped_"+str(args.featuresDroped)+"_"
        if(args.RandomMasking):
            saveFile=saveFile+"RandomMasking_"        
        if(args.abs):
            saveFile=saveFile+"abs_"

    saveFile=saveFile+args.append+"_"


    best_prec1=0
    best_epoch=0
    NoChange=0
    start_training=time.time()
    if(args.resume):

        filename=os.path.join(args.save_dir, saveFile+'model.th')
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch=checkpoint['epoch']
        best_prec1=checkpoint['best_prec1']
    for epoch in range(start_epoch,args.epochs ):
        start_epoch_time=time.time()

      
        if(args.trainingType=="regular"):
            model= regular.train(args,epoch,model,trainloader,optimizer,criterion,saveFile)
            if(args.TrainWithAugmented):
                acc ,  prec1 = regular.test(args,epoch,model,testloader,criterion,best_prec1,best_epoch,returnMaskedAcc=True)
            else:
                prec1 =regular.test(args,epoch,model,testloader,criterion,best_prec1,best_epoch)

        elif(args.trainingType=="interpretable"):
            model , train_Model_loss , train_Kl_loss = interpretable.train(args,epoch,model,trainloader,optimizer,criterion,criterionKDL,Name=saveFile)
            if(args.TrainWithAugmented):
                acc, prec1 ,test_Kl_loss  = interpretable.test(args,epoch,model,testloader,criterion,criterionKDL,best_prec1,best_epoch,returnMaskedAcc=True)
            else:
                prec1 ,test_Kl_loss  = interpretable.test(args,epoch,model,testloader,criterion,criterionKDL,best_prec1,best_epoch)

        lr_scheduler.step()
      

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)



        end_epoch_time=time.time()
        print("epoch time:",end_epoch_time-start_epoch_time,"No Change Flag",NoChange)

        
        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, saveFile+'checkpoint.th'))
        if(is_best):
            best_epoch=epoch
            save_checkpoint({
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'epoch': epoch,
            }, is_best, filename=os.path.join(args.save_dir, saveFile+'model.th'))
            NoChange=0
        else:
            NoChange+=1

        if(epoch+1 ==args.epochs):
            best_epoch=epoch
            save_checkpoint({
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'epoch': epoch,
            }, is_best, filename=os.path.join(args.save_dir, saveFile+'Last_model.th'))
        


  
        if(NoChange>=50):
            best_epoch=epoch
            save_checkpoint({
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'epoch': epoch,
            }, is_best, filename=os.path.join(args.save_dir, saveFile+'Last_model.th'))
            break



    end_training=time.time()
    print("Trainig time", end_training- start_training)

if __name__ == '__main__':
    main()
