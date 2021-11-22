import os
import cv2
import time
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torch.utils.data import RandomSampler
import torchvision.models as models
from dataset_BIRD import BirdDataset,get_transform
import Helper
import regular
import interpretable 
from Helper import save_checkpoint







parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=48, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=5)
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
parser.add_argument('--isCIFAR', default=False, action="store_true",help='Dataset is CIFAR')
parser.add_argument('--isBIRD', default=True, action="store_true",help='Dataset is BIRD')
parser.add_argument('--append', default='1', type=str)

def main():

    args = parser.parse_args()
    start_epoch = 0 


    DIR_TRAIN = "./data/bird260/train/"
    DIR_VALID = "./data/bird260/valid/"
    DIR_TEST = "./data/bird260/test/"

    classes = os.listdir(DIR_TRAIN)
    print("Total Classes: ",len(classes))


    train_count = 0
    valid_count = 0
    test_count = 0
    for _class in classes:
        train_count += len(os.listdir(DIR_TRAIN + _class))
        valid_count += len(os.listdir(DIR_VALID + _class))
        test_count += len(os.listdir(DIR_TEST + _class))

    print("Total train images: ",train_count)
    print("Total valid images: ",valid_count)
    print("Total test images: ",test_count)



    ### Creating a list of all images : DIR_TRAIN/class_folder/img.jpg - FOR METHOD 2 of data loading
    #   A dict for mapping class labels to index

    train_imgs = []
    valid_imgs = []
    test_imgs = []

    for _class in classes:
        
        for img in os.listdir(DIR_TRAIN + _class):
            train_imgs.append(DIR_TRAIN + _class + "/" + img)
        
        for img in os.listdir(DIR_VALID + _class):
            valid_imgs.append(DIR_VALID + _class + "/" + img)
            
        for img in os.listdir(DIR_TEST + _class):
            test_imgs.append(DIR_TEST + _class + "/" + img)

    class_to_int = {classes[i] : i for i in range(len(classes))}




    train_dataset = BirdDataset(train_imgs, class_to_int, get_transform())
    valid_dataset = BirdDataset(valid_imgs, class_to_int, get_transform())
    test_dataset = BirdDataset(test_imgs, class_to_int, get_transform())

    #Data Loader  -  using Sampler (YT Video)
    train_random_sampler = RandomSampler(train_dataset)
    valid_random_sampler = RandomSampler(valid_dataset)
    test_random_sampler = RandomSampler(test_dataset)

    #Shuffle Argument is mutually exclusive with Sampler!
    trainloader = DataLoader(
        dataset = train_dataset,
        batch_size = args.batch_size,
        sampler = train_random_sampler,
        num_workers = 4,
    )

    testloader = DataLoader(
        dataset = valid_dataset,
        batch_size = args.batch_size,
        sampler = valid_random_sampler,
        num_workers = 4,
    )

    test_data_loader = DataLoader(
        dataset = test_dataset,
        batch_size = args.batch_size,
        sampler = test_random_sampler,
        num_workers = 4,
    )

    ### Define model
    model = models.vgg16(pretrained = True)

    ### Modifying last few layers and no of classes
    # NOTE: cross_entropy loss takes unnormalized op (logits), then function itself applies softmax and calculates loss, so no need to include softmax here
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4048, bias = True),
        nn.ReLU(inplace = True),
        nn.Dropout(0.4),
        nn.Linear(4048, 2048, bias = True),
        nn.ReLU(inplace = True),
        nn.Dropout(0.4),
        nn.Linear(2048,260)
    )


    ### Get device

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()

    model.to(device)

    ### Training Details

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.75)
    criterion = nn.CrossEntropyLoss()
    criterionKDL=torch.nn.KLDivLoss(reduction = 'batchmean')




    saveFile= "BIRD_"+args.trainingType

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

    for epoch in range(start_epoch,args.epochs ):
        start_epoch_time=time.time()
        

        if(args.trainingType=="regular"):
            model= regular.train(args,epoch,model,trainloader,optimizer,criterion,saveFile)
            prec1 =regular.test(args,epoch,model,testloader,criterion,best_prec1,best_epoch)
        elif(args.trainingType=="interpretable"):
            model , train_Model_loss , train_Kl_loss = interpretable.train(args,epoch,model,trainloader,optimizer,criterion,criterionKDL,Name=saveFile)

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
        



        if(NoChange>=10):
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
