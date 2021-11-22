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

import torchvision.transforms as T
import torchvision.models as TVmodels
from torchvision.utils import make_grid
from dataset_BIRD import BirdDataset,get_transform
import Helper
import regular
import interpretable 
from matplotlib import pyplot as plt
import copy

from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

from Helper import save_checkpoint




parser = argparse.ArgumentParser()
parser.add_argument('--RandomMasking', default=False, action="store_true")
parser.add_argument('--trainingType', default='regular', type=str)
parser.add_argument('--featuresDroped', type=float, default=0.5)
args = parser.parse_args()
def main():



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
        batch_size = 10,
        sampler = train_random_sampler,
        num_workers = 4,
    )

    testloader = DataLoader(
        dataset = valid_dataset,
        batch_size = 100,
        sampler = valid_random_sampler,
        num_workers = 4,
    )

    test_data_loader = DataLoader(
        dataset = test_dataset,
        batch_size = 10,
        sampler = test_random_sampler,
        num_workers = 4,
    )

    ### Define model
    BaseModel = TVmodels.vgg16(pretrained = True)

    ### Modifying last few layers and no of classes
    # NOTE: cross_entropy loss takes unnormalized op (logits), then function itself applies softmax and calculates loss, so no need to include softmax here
    BaseModel.classifier = nn.Sequential(
        nn.Linear(25088, 4048, bias = True),
        nn.ReLU(inplace = True),
        nn.Dropout(0.4),
        nn.Linear(4048, 2048, bias = True),
        nn.ReLU(inplace = True),
        nn.Dropout(0.4),
        nn.Linear(2048,260)
    )






    models=["BIRD_regular1_model","BIRD_interpretablefeaturesDroped_0.5_RandomMasking_1_model"]

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    saliencyMethod="Grad"
    for model in models:

      net =copy.deepcopy(BaseModel)

      checkpoint = torch.load('./models/'+model+'.th')
      net.load_state_dict(checkpoint['state_dict'])


      net.eval()

      outputs = net(images)

      _, predicted = torch.max(outputs, 1)


      for ind in range(100):

        input = images[ind].unsqueeze(0)
        input.requires_grad = True


        saliency = Saliency(net)
        grads = saliency.attribute(input, target=labels[ind].item(),abs=False)
        grads_=grads.cpu().detach().numpy()
        np.save('outputs/SaliencyValues/Grad_'+model+"_"+str(ind)+".npy", grads_)
        grads=grads.abs()
        grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))


        original_image=images[ind].cpu().detach().numpy()
        original_image = np.transpose(original_image, (1, 2, 0))


        figure, axis = viz.visualize_image_attr(None, original_image, 
                              method="original_image")
        # plt.savefig()
        plt.savefig('./outputs/Graphs/BIRD_original_image_'+str(ind)+".png" , bbox_inches = 'tight',
            pad_inches = 0)
        plt.show()





        print(grads.shape)
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


    net.cuda()
    saliencyValues,labels_,allData = Helper.getSaliency(net,testloader,saliencyMethod,1300,abs=False)

    np.save("outputs/SaliencyValues/Mean_"+saliencyMethod+"_"+model+".npy", saliencyValues.mean(axis=0))
    np.save("outputs/SaliencyValues/All_"+saliencyMethod+"_"+model+".npy", saliencyValues)

     



if __name__ == '__main__':
    main()

