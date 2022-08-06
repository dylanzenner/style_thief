from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image

import torchvision.transforms as transforms
import torchvision.models as models

import copy



def resize_images(img_1, img_2, image_size = 128):

    device = torch.device('cpu')

    loader = transforms.Compose([
        transforms.Resize(image_size), 
        transforms.ToTensor()
        ])
    
    resized_img_1, resized_img_2 = loader(img_1).unsqueeze(0), loader(img_2).unsqueeze(0)
    return (resized_img_1.to(device, torch.float), resized_img_2.to(device, torch.float))

def content_loss():
    pass

def style_loss():
    pass

def gram_matrix():
    pass

def style_transfer():
    pass


    

