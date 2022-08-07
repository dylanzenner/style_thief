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
        transforms.Resize((image_size, image_size)), 
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    resized_img_1, resized_img_2 = loader(img_1).unsqueeze(0), loader(img_2).unsqueeze(0)
    return (resized_img_1.to(device), resized_img_2.to(device))


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.chosen_features = ["0", "5", "10", "19", "28"]
        self.model = models.vgg19(pretrained=True).features[:29]


    def forward(self, x):

        features = []

        # Go through each layer in model, if the layer is in the chosen_features,
        # store it in features. At the end we'll just return all the activations
        # for the specific layers we have in chosen_features
        for layer_num, layer in enumerate(self.model):
            x = layer(x)

            if str(layer_num) in self.chosen_features:
                features.append(x)

        return features




    


    

